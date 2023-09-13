"""
github repo
https://github.com/xlxwalex/FCGEC
updated to Apr 5, 2023 commit
"""
# Import Libs
import torch
from torch import nn
from  tqdm import tqdm
import numpy as np
import time
import logging
import os
import json
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import classification_report
import scipy.io as sio

import numpy as np
from argparse import Namespace
import operator as op

from utils.JointSTG import TAGGER_MAP, INSERT_TAG, MODIFY_TAG, KEEP_TAG
from utils.JointSTG import padding, attention_mask, save_model, clip_maxgenerate, reconstruct_switch, reconstruct_tagger, reconstruct_tagger_V2, softmax_logits, SwitchSearch, fillin_tokens, report_pipeline_output, convert_spmap2tokens, convert_spmap_tg, convert_spmap_sw
from dataset_provider.FCGEC import SwitchDataset, TaggerDataset, GeneratorDataset, collate_fn_base, collate_fn_tagger
from trainers.base import Trainer

logger = logging.getLogger(__name__)

## Loss Part

class SwitchLoss(nn.Module):
    def __init__(self, args : Namespace, device : torch.device, criterion, size_average : bool = True, denomitor : float = 1e-8):
        super(SwitchLoss, self).__init__()
        self.args = args
        self.device = device
        self.criterion = criterion
        self.denomitor = denomitor
        self.average = size_average
        try:
            self.gamma = args.swloss_gamma
        except:
            self.gamma = 1e-2

    def forward(self, logits : torch.Tensor, gts : torch.Tensor, masks : torch.Tensor = None) -> torch.Tensor:
        label_loss = self.criterion(logits, gts)
        if masks is not None:
            #mask_logits = logits * masks
            mask_logits = softmax_logits(logits) * masks
            order_logits = torch.cat([torch.diag_embed(torch.diag(mask_logits[ins], -1), offset=-1).unsqueeze(0) for ins in range(mask_logits.shape[0])], dim = 0)
            irorder_logits = mask_logits - order_logits
            order_loss =  torch.sum(torch.exp(irorder_logits), dim = [1, 2]) / (torch.sum(torch.exp(order_logits), dim=[1, 2]) + self.denomitor)
            if self.average:
                order_loss = torch.mean(order_loss)
            else:
                order_loss = torch.sum(order_loss)
            combine_loss = label_loss + order_loss
        else:
            combine_loss = label_loss
        return combine_loss


class TaggerLoss(nn.Module):
    def __init__(self, args : Namespace, device : torch.device, criterion, size_average : bool = True):
        super(TaggerLoss, self).__init__()
        self.args = args
        self.device = device
        self.criterion = criterion
        self.tagger_criterion = criterion[0]
        self.insmod_criterion = criterion[1]
        self.average = size_average
        self.max_gen = args.max_generate

    def forward(self, tagger_logits : torch.Tensor, comb_logits : torch.Tensor,
                tagger_gts : torch.Tensor, comb_gts :torch.Tensor) -> torch.Tensor:
        tagger_logits = tagger_logits.permute(0, 2, 1)
        comb_logits = comb_logits.permute(0, 2, 1)
        tagger_loss = self.tagger_criterion(tagger_logits, tagger_gts)
        combine_loss = tagger_loss
        # if torch.max(ins_gts) > 0:
        insmod_loss = self.insmod_criterion(comb_logits, comb_gts)
        combine_loss += insmod_loss
        return combine_loss


class GeneratorLoss(nn.Module):
    def __init__(self, args : Namespace, device : torch.device, criterion, size_average : bool = True):
        super(GeneratorLoss, self).__init__()
        self.args = args
        self.device = device
        self.criterion = criterion
        self.average = size_average
        self.max_gen = args.max_generate
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, mlm_logits : torch.Tensor, mlm_tgts : torch.Tensor) -> torch.Tensor:
        output_mlm = self.softmax(mlm_logits)
        loss_mlm = self.criterion(output_mlm, mlm_tgts)
        return loss_mlm


## Metric Part

class Metric(object):
    '''
    Base Modules of Metric Calculation
    '''
    def __init__(self, args : Namespace):
        self.args = args
        self.denom = 1e-8

    def __call__(self, gts, preds, mask : list) -> dict:
        raise NotImplementedError

    def _cal_token_level(self, gts, preds, mask: list):
        raise NotImplementedError

    def _cal_sentence_level(self, gts, preds, mask : list):
        raise NotImplementedError

class SwitchMetric_Spec():
    def __init__(self, args : Namespace, mode = 'all'):
        super(SwitchMetric_Spec, self).__init__()
        self.args = args
        self.mode = mode
        self.amax = args.padding_size
        self.use_lm = args.language_model

    def __call__(self, wd_idxs : list, gts: list, preds: list, mask: list, out_put: bool = False) -> tuple:
        assert len(gts) == len(preds)
        pred_token = self._apply_switch_operator(wd_idxs, preds)
        truth_token = self._apply_switch_operator(wd_idxs, gts)
        token_acc = self._cal_token_level(truth_token, pred_token, mask)
        sentence_acc = self._cal_sentence_level(truth_token, pred_token, mask)
        if out_put is not True:
            return token_acc, sentence_acc
        else:
            return token_acc, sentence_acc, pred_token, truth_token

    def _apply_switch_operator(self, wd_idxs : list, switch_ops: list) -> list:
        res = []
        for lidx in range(len(wd_idxs)):
            post_token = [101]
            switch_pred = switch_ops[lidx]
            sw_pidx = switch_pred[0]
            wd_idx = wd_idxs[lidx]
            while sw_pidx not in [0, -1]:
                post_token.append(wd_idx[sw_pidx])
                sw_pidx = switch_pred[sw_pidx]
                if wd_idx[sw_pidx] == 102: switch_pred[sw_pidx] = 0
            # assert len(post_token) == np.sum(ori_token > 0)
            res.append(post_token)
        return res

    def _cal_token_level(self, gts : list, preds : list, mask : list) -> float:
        size = len(gts)
        assert len(preds) == size
        total_token = np.sum(np.array(mask))
        correct_token = 0
        for idx in range(size):
            for lidx in range(min(len(gts[idx]), len(preds[idx]))):
                if gts[idx][lidx] == preds[idx][lidx]: correct_token += 1
        return correct_token * 1. / total_token

    def _cal_sentence_level(self, gts : list, preds : list, mask : list) -> float:
        size = len(gts)
        assert len(preds) == size
        correct = 0
        for idx in range(size):
            if op.eq(gts[idx], preds[idx]):correct += 1
        return correct * 1. / size


class SwitchMetric(Metric):
    def __init__(self, args : Namespace, mode = 'all'):
        super(SwitchMetric, self).__init__(args)
        self.args = args
        self.mode = mode
        self.amax = args.padding_size
        self.use_lm = args.language_model

    def __call__(self, gts : list, preds : list, mask : list) -> dict:
        '''
        Calculate Switch Metric
        :param gts: groud truth
        :param preds: preds label
        :param mask: attention mask utilized in PLM
        :return: metric (token accuracy / sentence accuracy) based on mode
        '''
        ret = {}
        assert len(gts) == len(preds)
        preds = self._repairlm_sep(preds, mask)
        if self.mode == 'all':
            ret['token'] = self._cal_token_level(gts, preds, mask)
            ret['sentence'] = self._cal_sentence_level(gts, preds)
        elif self.mode == 'token':
            ret['token'] = self._cal_token_level(gts, preds, mask)
        elif self.mode == 'sentence':
            ret['sentence'] = self._cal_sentence_level(gts, preds)
        else:
            raise Exception('SwitchMetric.__call__ occure some errors, invalid params `mode`.')
        return ret

    def _repairlm_sep(self, preds : list, mask : list):
        if self.use_lm:
            seq_lens = [np.where(mk == 0)[0][0] - 1 if mk[-2] != 1 else len(mk) - 2 if mk[-1] != 1 else len(mk) - 1 for mk in mask]
            for insid in range(len(seq_lens)): preds[insid][seq_lens[insid]] = 0
        return preds

    def _cal_token_level(self, gts : list, preds : list, mask : list) -> float:
        gts = np.clip(gts, a_min=0, a_max=self.amax)
        total_token = len(np.where(gts > 0)[0])
        externel_token = np.array(mask).size - total_token
        correct_token = np.sum(np.array(gts) == np.array(preds)) - externel_token
        return correct_token * 1. / total_token

    def _cal_sentence_level(self, gts : list, preds : list, mask : list = None) -> float:
        total_sentence = len(gts)
        gts = np.clip(gts, a_min=0, a_max=self.amax)
        correct_sentence = sum([1 if op.eq(gts[ins_idx].tolist(), preds[ins_idx].tolist()) else 0 for ins_idx in range(len(gts))])
        return correct_sentence * 1. / total_sentence

class TaggerMetric(Metric):
    def __init__(self, args : Namespace, mode = 'all', level = 'normal'):
        super(TaggerMetric, self).__init__(args)
        self.args = args
        self.mode = mode
        self.level = level # high/normal (whether ignore element not in MI & I)
        self.amax = args.padding_size
        self.use_lm = args.language_model
        self.ins_tag = TAGGER_MAP[INSERT_TAG]
        self.mod_tag = TAGGER_MAP[MODIFY_TAG]
        assert level in ['normal', 'high']

    def __call__(self, gts : dict, preds : dict, mask : list) -> dict:
        '''
        Calculate Tagger Metric
        :param gts: groud truth (dict)
        :param preds: preds label (dict)
        :param mask: attention mask utilized in PLM
        :return: metric (token accuracy / sentence accuracy) based on mode
        '''
        ret = {}
        token_ret = self._cal_token_level(gts, preds, mask, self.level)
        ret['token'] = token_ret
        sentence_ret = self._cal_sentence_level(gts, preds, mask, self.level)
        ret['sentence'] = sentence_ret
        return ret

    def _cal_token_level(self, gts : dict, preds : dict, mask : list,  level: str = 'normal') -> dict:
        # Unpack
        tagger_gts = np.clip(gts['tagger'], a_min=0, a_max=self.amax)
        ins_gts = np.clip(gts['insert'], a_min=0, a_max=np.max(gts['insert']))
        mod_gts = np.clip(gts['modify'], a_min=0, a_max=np.max(gts['modify']))
        tagger_preds = preds['tagger']
        ins_preds = preds['insert']
        mod_preds = preds['modify']
        # Calculate
        total_token = np.sum(np.array(mask))
        externel_token = np.array(mask).size - total_token
        batch_size = len(tagger_preds)
        # | - Tagger
        if level == 'normal':
            correct_tagger = sum([np.sum(np.array(tagger_gts[ins])[tagger_gts[ins] > 0] == np.array(tagger_preds[ins])[tagger_gts[ins] > 0]) for ins in range(batch_size)])
            tagger_acc = correct_tagger * 1. / total_token
        else:
            correct_tagger = sum([np.sum(np.array(tagger_gts[ins])[tagger_preds[ins] > TAGGER_MAP[KEEP_TAG]] == np.array(tagger_preds[ins])[tagger_preds[ins] > TAGGER_MAP[KEEP_TAG]]) for ins in range(batch_size)])
            tagger_acc = correct_tagger * 1. / (len(np.where(np.array(tagger_preds) > TAGGER_MAP[KEEP_TAG])[0]) + self.denom)
        # | - Insert
        if level == 'normal':
            insert_index   = np.array(tagger_gts) == self.ins_tag
            correct_insert = np.sum(np.array(ins_gts)[insert_index] == np.array(ins_preds)[insert_index])
            total_insert   = len(np.where(tagger_gts == self.ins_tag)[0]) + self.denom
        else:
            insert_index = np.array(tagger_gts) == self.ins_tag
            correct_insert = np.sum(np.array(ins_gts)[insert_index] == np.array(ins_preds)[insert_index])
            total_insert = len(np.where(tagger_gts == self.ins_tag)[0]) + self.denom
            # correct_insert = np.sum(np.array(ins_gts) == self.ins_tag) - externel_token
            # total_insert   = total_token
        insert_acc = correct_insert * 1. / total_insert
        # | - Modify
        if level == 'normal':
            modify_index   = np.array(tagger_gts) == self.mod_tag
            correct_modify = np.sum(np.array(mod_gts)[modify_index] == np.array(mod_preds)[modify_index])
            total_modify   = len(np.where(tagger_gts == self.mod_tag)[0]) + self.denom
        else:
            modify_index = np.array(tagger_gts) == self.mod_tag
            correct_modify = np.sum(np.array(mod_gts)[modify_index] == np.array(mod_preds)[modify_index])
            total_modify = len(np.where(tagger_gts == self.mod_tag)[0]) + self.denom
            # correct_modify = np.sum(np.array(mod_gts) == np.array(mod_preds)) - externel_token
            # total_modify   = total_token
        modify_acc = correct_modify * 1. / total_modify
        return {'tagger' : tagger_acc, 'insert' : insert_acc, 'modify' : modify_acc}

    def _cal_sentence_level(self, gts : dict, preds : dict, mask : list, level :str = 'normal') -> dict:
        # Unpack
        tagger_gts = np.clip(gts['tagger'], a_min=0, a_max=self.amax)
        ins_gts = np.clip(gts['insert'], a_min=0, a_max=np.max(gts['insert']))
        mod_gts = np.clip(gts['modify'], a_min=0, a_max=np.max(gts['modify']))
        tagger_preds = preds['tagger']
        ins_preds = preds['insert']
        mod_preds = preds['modify']
        # Calculate
        total_sentence = len(gts['tagger'])
        # | - Tagger
        correct_tagger= sum([1 if op.eq(tagger_gts[ins_idx][tagger_gts[ins_idx]>0].tolist(), tagger_preds[ins_idx][tagger_gts[ins_idx]>0].tolist()) else 0 for ins_idx in range(total_sentence)])
        tagger_acc = correct_tagger * 1. / total_sentence
        # | - Insert
        if level == 'normal':
            insert_index = np.array(tagger_gts) == self.ins_tag
            correct_insert = sum([1 if op.eq(ins_gts[ins_idx][insert_index[ins_idx]].tolist(), ins_preds[ins_idx][insert_index[ins_idx]].tolist()) else 0 for ins_idx in range(total_sentence)])
        else:
            insert_index = np.array(tagger_gts) == self.ins_tag
            correct_insert = sum([1 if op.eq(ins_gts[ins_idx][insert_index[ins_idx]].tolist(), ins_preds[ins_idx][insert_index[ins_idx]].tolist()) else 0 for ins_idx in range(total_sentence)])
            #correct_insert = sum([1 if op.eq(ins_gts[ins_idx][ins_gts>0].tolist(), ins_preds[ins_idx][ins_gts>0].tolist()) else 0 for ins_idx in range(total_sentence)])
        non_insert = sum([1 if np.max(ins_gts[ins_idx]) < 1 else 0 for ins_idx in range(total_sentence)])
        insert_acc = (correct_insert - non_insert) * 1. / (total_sentence - non_insert + self.denom) if non_insert != total_sentence else 1.0
        # | - Modify
        if level == 'normal':
            modify_index = np.array(tagger_gts) == self.mod_tag
            correct_modify = sum([1 if op.eq(mod_gts[ins_idx][modify_index[ins_idx]].tolist(), mod_preds[ins_idx][modify_index[ins_idx]].tolist()) else 0 for ins_idx in range(total_sentence)])
        else:
            modify_index = np.array(tagger_gts) == self.mod_tag
            correct_modify = sum([1 if op.eq(mod_gts[ins_idx][modify_index[ins_idx]].tolist(), mod_preds[ins_idx][modify_index[ins_idx]].tolist()) else 0 for ins_idx in range(total_sentence)])
            #correct_modify = sum([1 if op.eq(mod_gts[ins_idx][mod_gts > 0].tolist(), mod_preds[ins_idx][mod_gts > 0].tolist()) else 0 for ins_idx in range(total_sentence)])
        non_modify = sum([1 if np.max(mod_gts[ins_idx]) < 1 else 0 for ins_idx in range(total_sentence)])
        modify_acc = (correct_modify - non_modify) * 1. / (total_sentence - non_modify + self.denom) if non_modify != total_sentence else 1.0
        return {'tagger' : tagger_acc, 'insert' : insert_acc, 'modify' : modify_acc}

class TaggerMetricV2(Metric):
    def __init__(self, args : Namespace, mode = 'all', level = 'normal'):
        super(TaggerMetricV2, self).__init__(args)
        self.args = args
        self.mode = mode
        self.level = level # high/normal (whether ignore element not in MI & I)
        self.amax = args.padding_size
        self.use_lm = args.language_model
        self.ins_tag = TAGGER_MAP[INSERT_TAG]
        self.mod_tag = TAGGER_MAP[MODIFY_TAG]
        assert level in ['normal', 'high']

    def __call__(self, gts : dict, preds : dict, mask : list) -> dict:
        '''
        Calculate Tagger Metric
        :param gts: groud truth (dict)
        :param preds: preds label (dict)
        :param mask: attention mask utilized in PLM
        :return: metric (token accuracy / sentence accuracy) based on mode
        '''
        ret = {}
        token_ret = self._cal_token_level(gts, preds, mask, self.level)
        ret['token'] = token_ret
        sentence_ret = self._cal_sentence_level(gts, preds, mask, self.level)
        ret['sentence'] = sentence_ret
        return ret

    def _cal_token_level(self, gts : dict, preds : dict, mask : list,  level: str = 'normal') -> dict:
        # Unpack
        tagger_gts = np.clip(gts['tagger'], a_min=0, a_max=self.amax)
        insmod_gts = np.clip(gts['insmod'], a_min=0, a_max=np.max(gts['insmod']))
        tagger_preds = preds['tagger']
        insmod_preds = preds['insmod']
        # Calculate
        total_token = np.sum(np.array(mask))
        externel_token = np.array(mask).size - total_token
        batch_size = len(tagger_preds)
        # | - Tagger
        if level == 'normal':
            correct_tagger = sum([np.sum(np.array(tagger_gts[ins])[tagger_gts[ins] > 0] == np.array(tagger_preds[ins])[tagger_gts[ins] > 0]) for ins in range(batch_size)])
            tagger_acc = correct_tagger * 1. / total_token
        else:
            correct_tagger = sum([np.sum(np.array(tagger_gts[ins])[tagger_preds[ins] > TAGGER_MAP[KEEP_TAG]] == np.array(tagger_preds[ins])[tagger_preds[ins] > TAGGER_MAP[KEEP_TAG]]) for ins in range(batch_size)])
            tagger_acc = correct_tagger * 1. / (len(np.where(np.array(tagger_preds) > TAGGER_MAP[KEEP_TAG])[0]) + self.denom)
        # | - InsMod
        if level == 'normal':
            insert_index   = np.array(tagger_gts) == self.ins_tag
            modify_index   = np.array(tagger_gts) == self.mod_tag
            correct_ins = np.sum(np.array(insmod_gts)[insert_index] == np.array(insmod_preds)[insert_index])
            correct_mod = np.sum(np.array(insmod_gts)[modify_index] == np.array(insmod_preds)[modify_index])
            correct_comb = correct_ins + correct_mod
            total_comb   = len(np.where(tagger_gts == self.ins_tag)[0])  + len(np.where(tagger_gts == self.mod_tag)[0]) + self.denom
        else:
            insert_index = np.array(tagger_gts) == self.ins_tag
            modify_index = np.array(tagger_gts) == self.mod_tag
            correct_ins  = np.sum(np.array(insmod_gts)[insert_index] == np.array(insmod_preds)[insert_index])
            correct_mod  = np.sum(np.array(insmod_gts)[modify_index] == np.array(insmod_preds)[modify_index])
            correct_comb = correct_ins + correct_mod
            total_comb   = len(np.where(tagger_gts == self.ins_tag)[0])  + len(np.where(tagger_gts == self.mod_tag)[0]) + self.denom
            # correct_insert = np.sum(np.array(ins_gts) == self.ins_tag) - externel_token
            # total_insert   = total_token
        comb_acc = correct_comb * 1. / total_comb
        return {'tagger' : tagger_acc, 'comb' : comb_acc}

    def _cal_sentence_level(self, gts : dict, preds : dict, mask : list, level :str = 'normal') -> dict:
        # Unpack
        tagger_gts = np.clip(gts['tagger'], a_min=0, a_max=self.amax)
        insmod_gts = np.clip(gts['insmod'], a_min=0, a_max=np.max(gts['insmod']))
        tagger_preds = preds['tagger']
        insmod_preds = preds['insmod']
        # Calculate
        total_sentence = len(gts['tagger'])
        # | - Tagger
        correct_tagger= sum([1 if op.eq(tagger_gts[ins_idx][tagger_gts[ins_idx]>0].tolist(), tagger_preds[ins_idx][tagger_gts[ins_idx]>0].tolist()) else 0 for ins_idx in range(total_sentence)])
        tagger_acc = correct_tagger * 1. / total_sentence
        # | - InsMod
        if level == 'normal':
            insert_index = np.array(tagger_gts) == self.ins_tag
            modify_index = np.array(tagger_gts) == self.mod_tag
            insmod_index = insert_index + modify_index
            correct_insmod = sum([1 if op.eq(insmod_gts[ins_idx][insmod_index[ins_idx]].tolist(), insmod_preds[ins_idx][insmod_index[ins_idx]].tolist()) else 0 for ins_idx in range(total_sentence)])
        else:
            insert_index = np.array(tagger_gts) == self.ins_tag
            modify_index = np.array(tagger_gts) == self.mod_tag
            insmod_index = insert_index + modify_index
            correct_insmod = sum([1 if op.eq(insmod_gts[idx][insmod_index[idx]].tolist(), insmod_preds[idx][insmod_index[idx]].tolist()) else 0 for idx in range(total_sentence)])
            #correct_insert = sum([1 if op.eq(ins_gts[ins_idx][ins_gts>0].tolist(), ins_preds[ins_idx][ins_gts>0].tolist()) else 0 for ins_idx in range(total_sentence)])
        non_insmod = sum([1 if np.max(insmod_gts[idx]) < 1 else 0 for idx in range(total_sentence)])
        insmod_acc = (correct_insmod - non_insmod) * 1. / (total_sentence - non_insmod + self.denom) if non_insmod != total_sentence else 1.0
        return {'tagger' : tagger_acc, 'comb' : insmod_acc}

class GeneratorMetric(Metric):
    def __init__(self, args : Namespace, mode = 'all'):
        super(GeneratorMetric, self).__init__(args)
        self.args = args
        self.mode = mode
        self.amax = args.padding_size
        self.use_lm = args.language_model

    def __call__(self, gts : list, preds : list, mask : list) -> dict:
        '''
        Calculate Switch Metric
        :param gts: groud truth
        :param preds: preds label
        :param mask: attention mask utilized in PLM
        :return: metric (token accuracy / sentence accuracy) based on mode
        '''
        ret = {}
        assert len(gts) == len(preds)
        if self.mode == 'all':
            ret['token'] = self._cal_token_level(gts, preds, mask)
            ret['sentence'] = self._cal_sentence_level(gts, preds, mask)
        elif self.mode == 'token':
            ret['token'] = self._cal_token_level(gts, preds, mask)
        elif self.mode == 'sentence':
            ret['sentence'] = self._cal_sentence_level(gts, preds, mask)
        else:
            raise Exception('GeneratorMetric.__call__ occure some errors, invalid params `mode`.')
        return ret

    def _cal_token_level(self, gts : list, preds : list, mask : list) -> float:
        #gts_token = np.clip(np.array(gts), a_min=0, a_max=1)
        total_token = np.array(gts).size
        correct_token = np.sum(np.array(gts) == np.array(preds))
        return correct_token * 1. / total_token

    def _cal_sentence_level(self, gts : list, preds : list, mask : list = None) -> float:
        mask = np.clip(np.array(mask), 0, 1)
        mask_length = np.sum(mask, axis=1).tolist()
        index = 0
        total_sentence = len(mask)
        correct_sentence = 0
        for elen in range(len(mask)):
            if op.eq(gts[index:index+mask_length[elen]], preds[index:index+mask_length[elen]]):
                correct_sentence += 1
            index += mask_length[elen]
        return correct_sentence * 1. / total_sentence

## Trainer

class JointTrainer(Trainer):
    def __init__(self, args, config, model):
        super(JointTrainer, self).__init__(args, config, model)
        self.global_args = args
        self.args        = config
        self.infer_export= "stg_joint_test.xlsx"
        # Training Component
        self.model       = model
        self.model.to(args.device)
        self.criterion   = {
            'sw'  : torch.nn.CrossEntropyLoss(ignore_index=config.ignore_val).to(args.device),
            'gen' : torch.nn.NLLLoss().to(args.device),
            'tag' : (torch.nn.CrossEntropyLoss(reduction = 'sum').to(args.device),
                    torch.nn.CrossEntropyLoss(reduction = 'sum', ignore_index=config.ignore_val).to(args.device))

        }
        self.optimizer   = None  # init while training
        self.scheduler   = None
        self.device      = args.device
        # Training Params
        self.epoch       = 0
        self.step        = 0
        self.best        = - float('inf')
        self.eval_inform = []
        self.train_loss  = []
        self.eval_loss   = []
        self.all_acc     = []
        self.all_f1      = []
        # Loss Function
        if self.criterion is not None:
            self.sw_loss_fn  = SwitchLoss(config, self.device, self.criterion['sw'])
            self.tag_loss_fn = TaggerLoss(config, self.device, self.criterion['tag'])
            self.gen_loss_fn = GeneratorLoss(config, self.device, self.criterion['gen'])
        # Switch Decoder
        self.decoder     = SwitchSearch(config, config.sw_mode)
        # Metric Function
        self.sw_metric   = SwitchMetric(config, mode='all')
        self.tag_metric  = TaggerMetricV2(config , mode='all', level='normal')
        self.gen_metric  = GeneratorMetric(config, mode='all')
        self.spec_metric = SwitchMetric_Spec(config, mode='all')
        self.eval_inform = {'eval_inform' : [], 'metric_info' : [], 'loss_info' : []}

    def collect_metric(self, swicths : tuple, tags : tuple, generates : tuple, desc : str = 'train'):
        ret = {'switch' : {}, 'tagger' : {}, 'generate' : {}}
        # Unpack Data
        switch_gts, switch_preds, switch_masks = swicths
        tagger_gts, tagger_preds, tagger_masks = tags
        gen_gts, gen_preds, gen_masks = generates
        # Switch
        switch_met = self.sw_metric(switch_gts, switch_preds, switch_masks)
        sw_acc_token, sw_acc_sentence = switch_met['token'], switch_met['sentence']
        ret['switch']['acc_token'] = sw_acc_token
        ret['switch']['acc_sent'] = sw_acc_sentence
        # Tagger
        tagger_met = self.tag_metric(tagger_gts, tagger_preds, tagger_masks)
        ret['tagger'] = tagger_met
        # Generate
        generate_met = self.gen_metric(gen_gts, gen_preds, gen_masks)
        ret['generate'] = generate_met
        train_acc_refer = [ret['switch']['acc_sent'], ret['tagger']['sentence']['tagger'], ret['tagger']['sentence']['comb'], ret['generate']['token']]
        # Process Desc for return
        if desc in ['train']:
            return sum(train_acc_refer) / len(train_acc_refer)
        else:
            return ret, sum(train_acc_refer) / len(train_acc_refer)

    # Train Model
    def do_train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 1e-2},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer   = AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.optimizer.zero_grad()
        self.step = 0
        switch_preds, switch_masks, switch_truths = [], [], []
        tag_preds, insmod_preds, tag_masks, tag_truths, insmod_truths = [], [], [], [], []
        mlm_preds, mlm_masks, mlm_truths = [], [], []
        for epoch in tqdm(range(self.args.epoch), desc='Training Epoch'):
            for step, batch_data in enumerate(train_dataloader):
                st_time = time.time()
                self.model.train()
                # Process Data
                token_collection, tag_collection, label_collection = batch_data
                ori_tokens, tag_tokens, gen_tokens  = token_collection
                tag_label, insmod_label = tag_collection
                sw_label, mlm_label = label_collection
                # Token Padded
                padded_orig = padding(ori_tokens, self.args.padding_size, self.args.padding_val)
                padded_tags = padding(tag_tokens, self.args.padding_size, self.args.padding_val)
                padded_gens = padding(gen_tokens, self.args.padding_size, self.args.padding_val)
                # Attention Masks
                ori_attn_mask = attention_mask(padded_orig, self.args.padding_val).to(self.device)
                tag_attn_mask = attention_mask(padded_tags, self.args.padding_val).to(self.device)
                gen_attn_mask = attention_mask(padded_gens, self.args.padding_val).to(self.device)
                # Tensor Trans
                ori_token_tensor = torch.from_numpy(padded_orig).to(self.device)
                tag_token_tensor = torch.from_numpy(padded_tags).to(self.device)
                gen_token_tensor = torch.from_numpy(padded_gens).to(self.device)
                # MLM Label
                padded_mlm     = padding(mlm_label, self.args.padding_size, self.args.padding_val)
                tgt_mlm_tensor = torch.from_numpy(padded_mlm).to(self.device)
                # Label Tensor
                pad_swlabel    = padding(sw_label, self.args.padding_size, self.args.ignore_val)
                swlabel_tensor = torch.from_numpy(pad_swlabel).to(self.device)
                padded_insmod  = padding(insmod_label, self.args.padding_size, self.args.ignore_val)
                insmod_label   = clip_maxgenerate(torch.from_numpy(padded_insmod), self.args.max_generate).to(self.device)
                padded_tagger  = padding(tag_label, self.args.padding_size, self.args.padding_val)
                tagger_label   = torch.from_numpy(padded_tagger).to(self.device)
                # Pack Data
                token_tensor   = (ori_token_tensor, tag_token_tensor, gen_token_tensor)
                attnmask_tensor= (ori_attn_mask, tag_attn_mask, gen_attn_mask)
                tagger_gts     = (tagger_label, insmod_label)
                # NetVal
                pointer_ret, tagger_logits, gen_logits = self.model(token_tensor, tgt_mlm_tensor, attnmask_tensor)
                # Unpack
                poi_ret, poi_mask = pointer_ret
                mlm_logits, mlm_tgts, _gen_logits = gen_logits
                tagger_logits, comb_logits = tagger_logits
                # Loss Process
                loss_switch = self.sw_loss_fn(poi_ret, swlabel_tensor, poi_mask)
                loss_tagger = self.tag_loss_fn(tagger_logits, comb_logits, tagger_label, insmod_label)
                loss_generator = self.gen_loss_fn(mlm_logits, mlm_tgts)
                loss = loss_switch + 0.01* loss_tagger + loss_generator
                loss.backward()
                self.train_loss.append(loss.item())
                self.optimizer.step()
                self.scheduler.step() if self.scheduler is not None else None
                self.train_loss.append(loss.item())
                # Clear Gradient
                self.optimizer.zero_grad()
                # Collect Result
                # | - Switch
                switch_pred = self.decoder(poi_ret.detach().cpu(), ori_attn_mask.detach().cpu().numpy())
                switch_preds.extend(switch_pred)
                switch_truths.extend(pad_swlabel)
                switch_masks.extend(ori_attn_mask.detach().cpu().numpy())
                # | - Tagger
                tag_pred = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
                insmod_pred = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
                tag_preds.extend(tag_pred)
                insmod_preds.extend(insmod_pred)
                tag_truths.extend(padded_tagger)
                insmod_truths.extend(padded_insmod)
                tag_masks.extend(tag_attn_mask.detach().cpu().numpy())
                # | - Generator
                mlm_pred = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
                mlm_preds.extend(mlm_pred)
                mlm_truths.extend(mlm_tgts.detach().cpu().numpy())
                mlm_masks.extend(padded_mlm)
                # Pack Result
                switch_res = (switch_truths, switch_preds, switch_masks)
                tagger_res = ({'tagger' : tag_truths, 'insmod' : insmod_truths}, {'tagger' : tag_preds, 'insmod' : insmod_preds}, tag_masks)
                gen_res    = (mlm_truths, mlm_preds, mlm_masks)
                if (self.step + 1) % self.args.print_step == 0:
                    metric_info = self.collect_metric(switch_res, tagger_res, gen_res)
                    logger.info("step: %s, ave loss = %.4f, refer acc = %.4f, ..switch_loss = %.3f, tag_loss = %.3f, gen_loss = %.3f" %
                          (self.step + 1, loss.item(), metric_info, loss_switch, loss_tagger, loss_generator))
                    switch_preds, switch_masks, switch_truths = [], [], []
                    tag_preds, insmod_preds, tag_masks, tag_truths, insmod_truths = [], [], [], [], []
                    mlm_preds, mlm_masks, mlm_truths = [], [], []
                # Process Step
                self.step += 1
                # Eval Data
                if (self.step + 1) % self.args.eval_step == 0:
                    eval_time = time.time()
                    valid_loss, valid_metric, refer = self.do_test(val_dataloader, 'VAL')
                    eval_inform = eval_information(valid_loss, valid_metric)
                    logger.info('Final validation result: step: %d, speed: %f s/total' % (self.step + 1, 1 / (time.time() - eval_time)))
                    logger.info(eval_inform)
                    self.eval_inform['eval_inform'].append(eval_inform)
                    self.eval_inform['metric_info'].append(valid_metric)
                    self.eval_inform['loss_info'].append(valid_loss)
                    self.eval_loss.append(valid_loss)
                    # Save Checkpoints For Best Model
                    if self.best < refer:
                        self.save(save_dir=self.global_args.save_dir)
                        logger.info(">>>>>>> Model Reached Best Performance, Save To Check_points")
                        # Save Model To ./checkpoints
                        self.best = refer
        return self.best  # best refer acc

    # Valid Model
    def do_test(self, dataloader: DataLoader, mode: str = 'VAL') -> tuple:
        if mode == 'VAL' or mode == 'TEST':
            self.model.eval()
            switch_preds, switch_masks, switch_truths = [], [], []
            tag_preds, insmod_preds, tag_masks, tag_truths, insmod_truths = [], [], [], [], []
            mlm_preds, mlm_masks, mlm_truths = [], [], []
            eval_loss = {'overall' : [], 'switch' : [], 'tagger' : [], 'generate' : []}
            for step, batch_data in enumerate(dataloader):
                # Process Data
                token_collection, tag_collection, label_collection = batch_data
                ori_tokens, tag_tokens, gen_tokens = token_collection
                tag_label, insmod_label = tag_collection
                sw_label, mlm_label = label_collection
                # Token Padded
                padded_orig = padding(ori_tokens, self.args.padding_size, self.args.padding_val)
                padded_tags = padding(tag_tokens, self.args.padding_size, self.args.padding_val)
                padded_gens = padding(gen_tokens, self.args.padding_size, self.args.padding_val)
                # Attention Masks
                ori_attn_mask = attention_mask(padded_orig, self.args.padding_val).to(self.device)
                tag_attn_mask = attention_mask(padded_tags, self.args.padding_val).to(self.device)
                gen_attn_mask = attention_mask(padded_gens, self.args.padding_val).to(self.device)
                # Tensor Trans
                ori_token_tensor = torch.from_numpy(padded_orig).to(self.device)
                tag_token_tensor = torch.from_numpy(padded_tags).to(self.device)
                gen_token_tensor = torch.from_numpy(padded_gens).to(self.device)
                # MLM Label
                padded_mlm = padding(mlm_label, self.args.padding_size, self.args.padding_val)
                tgt_mlm_tensor = torch.from_numpy(padded_mlm).to(self.device)
                # Label Tensor
                pad_swlabel = padding(sw_label, self.args.padding_size, self.args.ignore_val)
                swlabel_tensor = torch.from_numpy(pad_swlabel).to(self.device)
                padded_insmod = padding(insmod_label, self.args.padding_size, self.args.ignore_val)
                insmod_label = clip_maxgenerate(torch.from_numpy(padded_insmod), self.args.max_generate).to(self.device)
                padded_tagger = padding(tag_label, self.args.padding_size, self.args.padding_val)
                tagger_label = torch.from_numpy(padded_tagger).to(self.device)
                # Pack Data
                token_tensor = (ori_token_tensor, tag_token_tensor, gen_token_tensor)
                attnmask_tensor = (ori_attn_mask, tag_attn_mask, gen_attn_mask)
                tagger_gts = (tagger_label, insmod_label)
                # NetVal
                with torch.no_grad():
                    # NetVal
                    pointer_ret, tagger_logits, gen_logits = self.model(token_tensor, tgt_mlm_tensor, attnmask_tensor)
                    # Unpack
                    poi_ret, poi_mask = pointer_ret
                    mlm_logits, mlm_tgts, _gen_logits = gen_logits
                    tagger_logits, comb_logits = tagger_logits
                    # Loss Process
                    loss_switch = self.sw_loss_fn(poi_ret, swlabel_tensor, poi_mask)
                    loss_tagger = self.tag_loss_fn(tagger_logits, comb_logits, tagger_label, insmod_label)
                    loss_generator = self.gen_loss_fn(mlm_logits, mlm_tgts)
                    loss = loss_switch +0.01* loss_tagger + loss_generator
                    eval_loss['overall'].append(loss.item())
                    eval_loss['switch'].append(loss_switch.item())
                    eval_loss['tagger'].append(0.01*loss_tagger.item())
                    eval_loss['generate'].append(loss_generator.item())
                # Collect Result
                # | - Switch
                switch_pred = self.decoder(poi_ret.detach().cpu(), ori_attn_mask.detach().cpu().numpy())
                switch_preds.extend(switch_pred)
                switch_truths.extend(pad_swlabel)
                switch_masks.extend(ori_attn_mask.detach().cpu().numpy())
                # | - Tagger
                tag_pred = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
                insmod_pred = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
                tag_preds.extend(tag_pred)
                insmod_preds.extend(insmod_pred)
                tag_truths.extend(padded_tagger)
                insmod_truths.extend(padded_insmod)
                tag_masks.extend(tag_attn_mask.detach().cpu().numpy())
                # | - Generator
                mlm_pred = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
                mlm_preds.extend(mlm_pred)
                mlm_truths.extend(mlm_tgts.detach().cpu().numpy())
                mlm_masks.extend(padded_mlm)
            # Pack Result
            switch_res = (switch_truths, switch_preds, switch_masks)
            tagger_res = (
                {'tagger': tag_truths, 'insmod': insmod_truths}, {'tagger': tag_preds, 'insmod': insmod_preds}, tag_masks)
            gen_res = (mlm_truths, mlm_preds, mlm_masks)
            loss_info = gather_loss(eval_loss)
            metric_info, refer_acc = self.collect_metric(switch_res, tagger_res, gen_res, desc='valid')
            if mode == 'VAL':
                return loss_info, metric_info, refer_acc
            else:
                return {"Switch_ACC_Sent": metric_info['switch']['acc_sent'], 
                        "Tagger_Sent": metric_info['tagger']['sentence']['tagger'],
                        "Tagger_Sent_Comb": metric_info['tagger']['sentence']['comb'], 
                        "Generator_Token": metric_info['generate']['token']}
        
        elif mode == 'INFER':
            self.model.eval()
            binary_preds, type_preds, switch_preds, switch_masks, binary_truths, type_truths, switch_truths = [], None, [], [], [], None, []
            tag_preds, ins_preds, mod_preds, tag_masks, tag_truths, ins_truths, mod_truths = [], [], [], [], [], [], []
            mlm_preds, mlm_masks, mlm_truths = [], [], []
            for step, batch_data in enumerate(tqdm(dataloader, desc='Inferencing')):
                # Process Data
                token_collection, tag_collection, label_collection = batch_data
                ori_tokens, tag_tokens, gen_tokens = token_collection
                tag_label, ins_label, mod_label = tag_collection
                binary_label, type_label, sw_label, mlm_label = label_collection
                # Token Padded
                padded_orig = padding(ori_tokens, self.args.padding_size, self.args.padding_val)
                padded_tags = padding(tag_tokens, self.args.padding_size, self.args.padding_val)
                # Attention Masks
                ori_attn_mask = attention_mask(padded_orig, self.args.padding_val).to(self.device)
                tag_attn_mask = attention_mask(padded_tags, self.args.padding_val).to(self.device)
                # Tensor Trans
                ori_token_tensor = torch.from_numpy(padded_orig).to(self.device)
                # MLM Label
                padded_mlm = padding(mlm_label, self.args.padding_size, self.args.padding_val)
                # tgt_mlm_tensor = torch.from_numpy(padded_mlm).to(self.device)
                # Label Tensor
                pad_swlabel = padding(sw_label, self.args.padding_size, self.args.ignore_val)
                padded_insert = padding(ins_label, self.args.padding_size, self.args.ignore_val)
                padded_modify = padding(mod_label, self.args.padding_size, self.args.ignore_val)
                padded_tagger = padding(tag_label, self.args.padding_size, self.args.padding_val)
                # Stage 1 : Binary + Type + Pointer
                with torch.no_grad():
                    bi_logits, cls_logits, pointer_ret = self.model(ori_token_tensor, 'tag_before', ori_attn_mask, need_mask=True)
                    poi_ret, poi_mask = pointer_ret
                # Processing 4 Tagger & Generator
                binary_pred = np.argmax(bi_logits.detach().cpu().numpy(), axis=1).astype('int32')
                binary_preds.extend(binary_pred)
                binary_truths.extend(binary_label)
                type_pred = np.argmax(cls_logits.detach().cpu().numpy(), axis=2).astype('int32').T
                type_preds = np.vstack((type_preds, type_pred)) if type_preds is not None else type_pred
                type_truths = np.vstack((type_truths, np.array(type_label))) if type_truths is not None else np.array(type_label)
                switch_pred = self.decoder(poi_ret.detach().cpu(), ori_attn_mask.detach().cpu().numpy())
                switch_preds.extend(switch_pred)
                switch_truths.extend(pad_swlabel)
                switch_masks.extend(ori_attn_mask.detach().cpu().numpy())
                switch_tokens = reconstruct_switch(padded_orig, switch_pred)
                tag_token_tensor = torch.from_numpy(switch_tokens).to(self.device)
                # Stage 2 : Tagger
                with torch.no_grad():
                    tagger_logits = self.model(tag_token_tensor, 'tagger', tag_attn_mask)
                    tag_logits, ins_logits, mod_logits = tagger_logits
                # | - Tagger
                tag_pred = np.argmax(tag_logits.detach().cpu().numpy(), axis=2).astype('int32')
                ins_pred = np.argmax(ins_logits.detach().cpu().numpy(), axis=2).astype('int32')
                mod_pred = np.argmax(mod_logits.detach().cpu().numpy(), axis=2).astype('int32')
                tag_preds.extend(tag_pred)
                ins_preds.extend(ins_pred)
                mod_preds.extend(mod_pred)
                tag_truths.extend(padded_tagger)
                ins_truths.extend(padded_insert)
                mod_truths.extend(padded_modify)
                tag_masks.extend(tag_attn_mask.detach().cpu().numpy())
                # Obtain generate tokens
                tag_construct = (tag_pred, ins_pred, mod_pred)
                tag_tokens, mlm_tgt_masks = reconstruct_tagger(switch_tokens, tag_construct)
                padded_gens = padding(tag_tokens, self.args.padding_size, self.args.padding_val)
                gen_attn_mask = attention_mask(padded_gens, self.args.padding_val).to(self.device)
                gen_token_tensor = torch.from_numpy(padded_gens).to(self.device)
                padded_mlm_tgt_mask = padding(mlm_tgt_masks, self.args.padding_size, self.args.padding_val)
                tgt_mlm_tensor = torch.from_numpy(padded_mlm_tgt_mask).to(self.device)
                # Stage 3 : Generator
                with torch.no_grad():
                    gen_logits = self.model(gen_token_tensor, 'generator', gen_attn_mask, tgt_mlm_tensor)
                    mlm_logits, mlm_tgts, _gen_logits = gen_logits
                # | - Generator
                mlm_pred = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
                mlm_preds.extend(mlm_pred)
                mlm_truths.extend(mlm_tgts.detach().cpu().numpy())
                mlm_masks.extend(padded_mlm_tgt_mask)
            # Pack Result
            binary_res = (binary_preds, binary_truths)
            types_res = (type_preds, type_truths)
            switch_res = (switch_truths, switch_preds, switch_masks)
            tagger_res = ({'tagger': tag_truths, 'insert': ins_truths, 'modify': mod_truths}, {'tagger': tag_preds, 'insert': ins_preds, 'modify': mod_preds}, tag_masks)
            gen_res = (mlm_truths, mlm_preds, mlm_masks)
            metric_info, refer_acc = self.collect_metric(binary_res, types_res, switch_res, tagger_res, gen_res, desc='valid')
            logger.info('>> Binary:\n' + classification_report(binary_truths, binary_preds) + '\n')
            logger.info('>> Types:\n' + classification_report(type_truths, type_preds))
            tagger_res = ((tag_truths, ins_truths, mod_truths), (tag_preds, ins_preds, mod_preds), tag_masks)
            return metric_info, refer_acc, eval_information(None, metric_info), (binary_res, types_res, switch_res, tagger_res, gen_res)
        else:
            raise NotImplementedError()

    def _apply_switch_operator(self, wd_idxs: list, switch_ops: list) -> list:
        res = []
        for lidx in range(len(wd_idxs)):
            post_token = [101]
            switch_pred = switch_ops[lidx]
            sw_pidx = switch_pred[0]
            wd_idx = wd_idxs[lidx]
            while sw_pidx not in [0, -1]:
                post_token.append(wd_idx[sw_pidx])
                sw_pidx = switch_pred[sw_pidx]
                if wd_idx[sw_pidx] == 102: switch_pred[sw_pidx] = 0
            res.append(post_token)
        return res

    def do_infer(self, dataloader: DataLoader, mode: str = 'TEST') -> tuple:
        logger.info("Inferring on test.csv...")
        assert dataloader is None, "Now only support directly load test.csv to infer."
        # Switch
        test_dir = os.path.join(self.args.data_base_dir, 'test.csv')
        if os.path.exists(os.path.join(self.args.data_base_dir, 'uuid.mat')):
            uuid = sio.loadmat(os.path.join(self.args.data_base_dir, 'uuid.mat'))['uuid']
        else:
            uuid = None

        switch_test = SwitchDataset(self.args, test_dir, 'test')
        TestLoader = DataLoader(switch_test, batch_size=self.args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_base)
        decoder = SwitchSearch(self.args, self.args.sw_mode)
        self.model.to(self.device)
        self.model.eval()

        pred_logits, truth_label, met_masks, tokens_ls, switch_preds, switch_gts, switch_tokens = None, [], [], [], [], [], []
        for step, batch_data in enumerate(tqdm(TestLoader, desc='Processing')):
            # Process Data
            tokens, labels = batch_data
            tokens_ls.extend(tokens)
            padded = padding(tokens, self.args.padding_size, self.args.padding_val)
            attn_mask = attention_mask(padded, self.args.padding_val).to(self.device)
            padded = torch.from_numpy(padded).to(self.device)
            # Model Value
            with torch.no_grad():
                pointer_logits = self.model.switch(padded, attn_mask)
            truths = padding(labels, self.args.padding_size, self.args.padding_val)
            pred_logits = torch.cat((pred_logits, pointer_logits), dim=0) if pred_logits is not None else pointer_logits
            met_masks.extend(attn_mask.detach().cpu().numpy())
            truth_label.extend(truths)
        pred_label    = decoder(pred_logits.detach().cpu(), met_masks)
        switch_tokens = self._apply_switch_operator(tokens_ls, pred_label)
        switch_gts    = self._apply_switch_operator(tokens_ls, truth_label)

        # Special map
        if self.args.sp_map:
            sp_maps = [pt.spmap for pt in switch_test.point_seq]
            sw_spmaps = convert_spmap_sw(sp_maps, pred_label)
        else: sw_spmaps = None

        logger.info('Construct Tagger Data')
        tagger_test = TaggerDataset(self.args, test_dir, 'test', switch_tokens)
        TestLoader = DataLoader(tagger_test, batch_size=self.args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_tagger)
        tagger_gts_dataset = TaggerDataset(self.args, test_dir, 'trains')  # EM
        gts_tagger, gts_comb = padding(tagger_gts_dataset.tagger_idx, self.args.padding_size, self.args.padding_val), padding(tagger_gts_dataset.comb_label, self.args.padding_size, self.args.padding_val)  # EM
        tag_construct_gts = (gts_tagger, gts_comb)  # EM

        pred_tagger, pred_comb, met_masks, tagger_tokens = [], [], [], []
        for step, batch_data in enumerate(tqdm(TestLoader, desc='Processing Tagger')):
            # Process Data
            tokens = batch_data
            padded_token = padding(tokens, self.args.padding_size, self.args.padding_val)
            tagger_tokens.extend(padded_token.tolist())
            attn_mask = attention_mask(padded_token, self.args.padding_val).to(self.device)
            token_padded = torch.from_numpy(padded_token).to(self.device)
            # Model Value
            with torch.no_grad():
                tagger_logits, comb_logits = self.model.tagger(token_padded, attn_mask)
            tagger_preds = np.argmax(tagger_logits.detach().cpu().numpy(), axis=2).astype('int32')
            comb_preds = np.argmax(comb_logits.detach().cpu().numpy(), axis=2).astype('int32')
            pred_tagger.extend(tagger_preds)
            pred_comb.extend(comb_preds)
            met_masks.extend(attn_mask.detach().cpu().numpy())

        logger.info('Construct Generator Data')
        tag_construct = (pred_tagger, pred_comb)
        tag_tokens, mlm_tgt_masks, tg_mapper = reconstruct_tagger_V2(np.array(tagger_tokens), tag_construct)
        tag_gts_tokens, _, _ = reconstruct_tagger_V2(padding(switch_gts, self.args.padding_size, self.args.padding_val),  tag_construct_gts)  # EM
        generator_test = GeneratorDataset(self.args, test_dir, 'test', tag_tokens, mlm_tgt_masks)
        TestLoader = DataLoader(generator_test, batch_size=self.args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn_base)

        # Special map
        if self.args.sp_map:
            tg_spmaps = convert_spmap_tg(sw_spmaps, tg_mapper)
        else:
            tg_spmaps = None

        pred_mlm, truth_mlm, met_masks = [], [], []
        for step, batch_data in enumerate(tqdm(TestLoader, desc='Processing Generator')):
            # Process Data
            tokens, label = batch_data
            padded_gens = padding(tokens, self.args.padding_size, self.args.padding_val)
            gen_attn_mask = attention_mask(padded_gens, self.args.padding_val).to(self.device)
            gen_token_tensor = torch.from_numpy(padded_gens).to(self.device)
            padded_mlm_tgt_mask = padding(label, self.args.padding_size, self.args.padding_val)
            tgt_mlm_tensor = torch.from_numpy(padded_mlm_tgt_mask).to(self.device)
            # Model Value
            with torch.no_grad():
                mlm_logits, tgt_mlm, _ = self.model.generator(gen_token_tensor, tgt_mlm_tensor, gen_attn_mask)
            # Preds
            token_preds = np.argmax(mlm_logits.detach().cpu().numpy(), axis=1).astype('int32')
            token_truth = tgt_mlm.detach().cpu().numpy()
            pred_mlm.extend(token_preds)
            truth_mlm.extend(token_truth)
            met_masks.extend(gen_attn_mask.detach().cpu().numpy())

        print('>>> Start to constrcut final output')
        logger.info('>>> Start to constrcut final output')
        outputs = fillin_tokens(tag_tokens, mlm_tgt_masks, pred_mlm)
        switch_tokens = [''.join(convert_spmap2tokens(switch_test.tokenizer.convert_ids_to_tokens(ele[1:-1]), sw_spmaps[i])).replace('##', '').replace('[PAD]', '').replace('[UNK]', '') for i, ele in enumerate(switch_tokens)] \
            if self.args.sp_map else  [''.join(switch_test.tokenizer.convert_ids_to_tokens(ele[1:-1])).replace('##', '').replace('[UNK]', '"').replace('[PAD]', '') for ele in switch_tokens]
        tagger_tokens = [''.join(convert_spmap2tokens(switch_test.tokenizer.convert_ids_to_tokens(ele[1:-1]), tg_spmaps[i])).replace('##', '').replace('[PAD]', '').replace('[UNK]', '') for i, ele in enumerate(tag_tokens)] \
            if self.args.sp_map else [''.join(tagger_test.tokenizer.convert_ids_to_tokens(ele[1:-1])).replace('##', '').replace('[UNK]', '"').replace('[PAD]', '') for ele in tag_tokens]
        generate_tokens = [''.join(convert_spmap2tokens(switch_test.tokenizer.convert_ids_to_tokens(ele[1:-1]), tg_spmaps[i])).replace('##', '').replace('[PAD]', '').replace('[UNK]', '') for i, ele in enumerate(outputs)] \
            if self.args.sp_map else [''.join(generator_test.tokenizer.convert_ids_to_tokens(ele[1:-1])).replace('##', '').replace('[UNK]', '"').replace('[PAD]', '') for ele in outputs]
        report_pipeline_output(os.path.join(self.global_args.save_dir, self.infer_export), 
                               [switch_test.sentences[sent_id] for sent_id in switch_test.final_sentences_ids], 
                               switch_test.label, switch_tokens, tagger_tokens, generate_tokens, uuid=uuid)
        logger.info('Final output saved at %s' % os.path.join(self.global_args.save_dir, self.infer_export))

        # if there was error in sentence switching, copy the original sentence as output
        sentence_num = len(switch_test.sentences)
        final_predict = []
        last_sent_id = 0
        for idx, sent_id in enumerate(switch_test.final_sentences_ids):
            final_predict.extend(switch_test.sentences[last_sent_id:sent_id])
            final_predict.append(generate_tokens[idx])
            last_sent_id = sent_id + 1
        if last_sent_id != sentence_num:
            final_predict.extend(switch_test.sentences[last_sent_id:sentence_num])
        
        # load id
        test_id_file = os.path.join(self.args.data_base_dir, 'test.id.json')
        with open(test_id_file) as f:
            test_ids = json.load(f)

        assert sentence_num == len(final_predict) == len(test_ids)
        json_result = []
        for i in range(sentence_num):
            json_result.append({'id': test_ids[i], 'src': switch_test.sentences[i], 'predict': final_predict[i]})
        
        return json_result

    # Generate Checkpoints
    def _generate_checkp(self) -> dict:
        checkpoints = {
            'model': self.model.state_dict(),
            'optim': self.optimizer,
            'metric': self.eval_inform,
            'args': self.args,
            'epoch': self.epoch,
            'train_loss' : self.train_loss,
            'eval_loss' : self.eval_loss
        }
        return checkpoints
    
    def save(self, save_dir):
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
        torch.save(self._generate_checkp(), checkpoint_path)

    def load(self, save_dir):
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
        params = torch.load(checkpoint_path)["model"]
        self.model.load_state_dict(params)


def gather_loss(loss_info : dict):
    gathered = {}
    for lokey in loss_info.keys(): gathered[lokey] = np.mean(loss_info[lokey])
    return gathered

def eval_information(loss_info : dict, metric_info : dict) -> str:
    if loss_info is not None:
        inform = ('>>' * 80 + '\n')
        inform += '> Loss Info: \n'
        for lokey in loss_info.keys(): inform += (lokey + 'loss = %.2f ' % loss_info[lokey])
        inform += '\n'
    else: inform = ''
    inform += '> Metric Info:\n'
    for metkey in metric_info.keys():
        inform += (metkey + ': ')
        sub_metric = metric_info[metkey]
        for skey in sub_metric.keys():
            if metkey != 'tagger': inform += (skey + ': %.4f ' % sub_metric[skey])
            else:
                inform += skey + ':['
                for sskey in sub_metric[skey]: inform += (sskey + ': %.4f ' % sub_metric[skey][sskey])
                inform+= '] '
        inform += '\n'
    inform += '>>' * 80
    return inform
