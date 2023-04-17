import numpy as np
import torch
from scipy.special import logsumexp
from argparse import Namespace
from copy import copy
import xlsxwriter
import json

'''
Definitions of Consts
'''

# Attention Mask Const (-inf)
MASK_LOGIT_CONST = 1e9

# Tagger Map
INSERT_TAG = 'I'
MODIFY_TAG = 'MI'
KEEP_TAG = 'K'
DELETE_TAG = 'D'
MOIFY_ONLY_TAG = 'M'
MODIFY_DELETE_TAG = 'MD'
TAGGER_MAP = {'PAD' : 0, KEEP_TAG : 1, DELETE_TAG : 2, INSERT_TAG : 3, MOIFY_ONLY_TAG: 4, MODIFY_TAG : 5, MODIFY_DELETE_TAG : 6}
TAGGER_MAP_INV = ['PAD', KEEP_TAG, DELETE_TAG, INSERT_TAG, MOIFY_ONLY_TAG, MODIFY_TAG, MODIFY_DELETE_TAG]

# Generators
GEN_KEEP_LABEL = 0
MASK_SYMBOL = '[MASK]'
MASK_LM_ID = 103

# Type Map
TYPE_MAP = {'IWO' : 0, 'IWC' : 1, 'SC' : 2, 'ILL' : 3, 'CM' : 4, 'CR' : 5, 'AM' :6}
TYPE_MAP_INV = ['语序不当', '搭配不当', '结构混乱', '不合逻辑', '成分残缺', '成分赘余', '表意不明']
TYPE_MAP_INV_NEW =  {'语序不当' : 'IWO', '搭配不当' : 'IWC', '结构混乱' :'SC', '不合逻辑' : 'ILL', '成分残缺' : 'CM', '成分赘余' : 'CR', '表意不明' : 'AM'}

## functions
def padding(inputs : list, paddings : int, pad_val : int) -> np.ndarray:
    doc = np.array([
        np.pad(x[0:paddings], ( 0, paddings - len(x[0:paddings])),
               'constant', constant_values=pad_val)
        for x in inputs
    ]).astype('int64')
    return doc

def attention_mask(padded : np.ndarray, pad_val : int) -> torch.Tensor:
    np_mask = (padded != pad_val).astype('int32')
    return torch.from_numpy(np_mask)

def save_model(path : str, checkp : dict) -> None:
    torch.save(checkp, path)

def normalize_logits(logits):
    numerator = logits
    denominator = logsumexp(logits)
    return numerator - denominator

def softmax_logits(logits :torch.Tensor, dim : int = 1):
    return torch.softmax(logits, dim=dim)

def clip_maxgenerate(gts :torch.Tensor, maxgen : int, sub_num : int = -2):
    if sub_num < -1:
        gts[gts >maxgen] = maxgen
    else:
        gts[gts > maxgen] = sub_num
    return gts

def reconstruct_switch(ori_tokens : np.array, switch_preds : np.array, wopad : bool = False):
    post_tokens = []
    batch_size, seq_len = ori_tokens.shape
    for lidx in range(batch_size):
        ori_token = ori_tokens[lidx]
        post_token = [101]
        switch_pred = switch_preds[lidx]
        sw_pidx = switch_pred[0]
        while sw_pidx not in [0, -1] :
            post_token.append(ori_token[sw_pidx])
            sw_pidx = switch_pred[sw_pidx]
            if ori_token[sw_pidx] == 102: switch_pred[sw_pidx] = 0
        assert len(post_token) == np.sum(ori_token > 0)
        post_tokens.append(post_token)
    if wopad is not True:
        return padding(post_tokens, seq_len, 0)
    else:
        return post_tokens, padding(post_tokens, seq_len, 0)

def reconstruct_tagger(tag_tokens : np.array, tag_preds : tuple) -> tuple:
    post_tokens, mlm_tgt_masks = [], []
    tagger, insert, modify = tag_preds
    batch_size, seq_len = tag_tokens.shape
    for lidx in range(batch_size):
        post_token, mlm_mask = [], []
        tag_cur = tagger[lidx]
        ins_cur = insert[lidx]
        mod_cur = modify[lidx]
        token_cur = tag_tokens[lidx]
        for cidx in range(seq_len):
            if tag_cur[cidx] == TAGGER_MAP['PAD']: break   # Pad ignore
            elif tag_cur[cidx] == TAGGER_MAP[KEEP_TAG]:
                mlm_mask.append(0)
                post_token.append(token_cur[cidx])
            elif tag_cur[cidx] in [TAGGER_MAP[DELETE_TAG], TAGGER_MAP[MODIFY_DELETE_TAG]]:
                continue
            elif tag_cur[cidx] == TAGGER_MAP[INSERT_TAG]:
                insert_num = ins_cur[cidx]
                if (insert_num < 1): continue
                post_token.append(token_cur[cidx])
                mlm_mask.append(0)
                post_token.extend([MASK_LM_ID] * insert_num)
                mlm_mask.extend([1] * insert_num)
            elif tag_cur[cidx] == TAGGER_MAP[MOIFY_ONLY_TAG]:
                mlm_mask.append(1)
                post_token.append(MASK_LM_ID)
            elif tag_cur[cidx] == TAGGER_MAP[MODIFY_TAG]:
                modify_num = mod_cur[cidx]
                post_token.append(MASK_LM_ID)
                mlm_mask.append(1)
                if (modify_num < 1): continue
                post_token.extend([MASK_LM_ID] * modify_num)
                mlm_mask.extend([1] * modify_num)
        post_tokens.append(post_token)
        mlm_tgt_masks.append(mlm_mask)
    return post_tokens, mlm_tgt_masks

def reconstruct_tagger_V2(tag_tokens : np.array, tag_preds : tuple, return_flag : bool = False) -> tuple:
    post_tokens, mlm_tgt_masks, op_flag = [], [], []
    tagger, insmod = tag_preds
    batch_size, seq_len = tag_tokens.shape
    for lidx in range(batch_size):
        post_token, mlm_mask = [], []
        tag_cur = tagger[lidx]
        insmod_cur = insmod[lidx]
        token_cur = tag_tokens[lidx]
        flag = False
        for cidx in range(seq_len):
            if tag_cur[cidx] == TAGGER_MAP['PAD']: break   # Pad ignore
            elif tag_cur[cidx] == TAGGER_MAP[KEEP_TAG]:
                mlm_mask.append(0)
                post_token.append(token_cur[cidx])
            elif tag_cur[cidx] in [TAGGER_MAP[DELETE_TAG], TAGGER_MAP[MODIFY_DELETE_TAG]]:
                flag = True
                continue
            elif tag_cur[cidx] == TAGGER_MAP[INSERT_TAG]:
                flag = True
                insert_num = insmod_cur[cidx]
                if (insert_num < 1): continue
                post_token.append(token_cur[cidx])
                mlm_mask.append(0)
                post_token.extend([MASK_LM_ID] * insert_num)
                mlm_mask.extend([1] * insert_num)
            elif tag_cur[cidx] == TAGGER_MAP[MOIFY_ONLY_TAG]:
                flag = True
                mlm_mask.append(1)
                post_token.append(MASK_LM_ID)
            elif tag_cur[cidx] == TAGGER_MAP[MODIFY_TAG]:
                flag = True
                modify_num = insmod_cur[cidx]
                post_token.append(MASK_LM_ID)
                mlm_mask.append(1)
                if (modify_num < 1): continue
                post_token.extend([MASK_LM_ID] * modify_num)
                mlm_mask.extend([1] * modify_num)
        post_tokens.append(post_token)
        mlm_tgt_masks.append(mlm_mask)
        op_flag.append(flag)
    if return_flag:
        return post_tokens, mlm_tgt_masks, op_flag
    else:
        return post_tokens, mlm_tgt_masks

class SwitchSearch(object):
    def __init__(self, args : Namespace, mode : str = 'vabm', pad : bool = True):
        self.args = args
        self.beamw = args.beam_width
        self.mode = mode
        self.max_length = args.padding_size
        self.use_lm = args.language_model
        self.pad = pad
        self.padval = args.padding_val

    def __call__(self, logits : torch.Tensor, masks : list = None) -> np.array:
        if self.mode == 'vags':
            # Vanilla Greedy Search
            return self._vanilla_gready(logits)
        elif self.mode == 'rsgs':
            # Restricted Greedy Search
            return self._restrict_greedy(logits, masks)
        elif self.mode == 'rsbm':
            # Restricted Beam Search
            seqlen = logits.shape[1]
            endids = self._get_sepend_id(masks)
            sepids = [[id] for id in endids]
            legals = [list(range(id + 1)) for id in endids]
            seq_result = self._res_beamsearch_batch(logits, endids, sepids, legals)
            if self.pad:
                seq_result = padding(seq_result, seqlen, self.padval)
            return seq_result

    def _get_sepend_id(self, mask : list) -> list:
        sep_idxs = []
        if self.use_lm:
            sep_idxs = [np.where(mk == 0)[0][0] - 1 if mk[-2] != 1 else len(mk) - 2 if mk[-1] != 1 else len(mk) - 1 for mk in mask]
        return sep_idxs

    def _restrict_greedy(self, logits : torch.Tensor, masks : list) -> np.array:
        endids = self._get_sepend_id(masks)
        sequence = [self._rs_greedy_single(logits[sid], endids[sid]) for sid in range(logits.shape[0])]
        return padding(sequence, self.args.padding_size, self.padval)

    def _rs_greedy_single(self, logit: torch.Tensor, endid : int) -> list:
        seq = []
        logit = logit.numpy()
        logit[0, :endid+1] = -float("inf")
        for sid in range(endid + 1):
            if len(seq) > 0 : logit[np.array(seq), sid] = -float("inf")
            if sid < endid - 1: logit[endid, sid] = -float("inf")
            argmax_id = np.argmax(logit[:, sid])
            seq.append(argmax_id)
        assert len(seq) == endid + 1
        return seq

    def _rs_greedy_single_v2(self, logit: torch.Tensor, endid : int) -> list:
        seq, seq_map = [], {}
        back_logit = copy(logit).numpy()
        #logit = softmax_logits(logit)
        logit[0, :endid+1] = 0
        pidx, sid = 0, 0
        while sid <= endid:
            if len(seq) > 0: logit[np.array(seq), pidx] = 0
            if sid < endid - 1: logit[endid, pidx] = 0
            argmax_id = np.argmax(softmax_logits(logit[:, pidx], dim=0).numpy())
            if pidx == 0 and argmax_id == 0: argmax_id = 1
            seq.append(argmax_id)
            seq_map[pidx] = argmax_id
            pidx = argmax_id
            sid += 1
        post_seq =[]
        for i in range(endid + 1):
            if i not in seq_map.keys():
                post_seq.append(endid)
            else:
                post_seq.append(seq_map[i])
        return post_seq

    def _rs_greedy_singleV3(self, logit: torch.Tensor, endid : int) -> list:
        seq = []
        logit = logit.numpy()
        logit[0, :endid+1] = 0
        for sid in range(endid + 1):
            if len(seq) > 0 : logit[np.array(seq), sid] = 0
            if sid < endid - 1: logit[endid, sid] = 0
            argmax_id = np.argmax(logit[:, sid])
            seq.append(argmax_id)
        assert len(seq) == endid + 1
        return seq

    def _vanilla_gready(self, logits :torch.Tensor) -> np.array:
        return np.argmax(logits.numpy(), axis=1).astype('int32')

    def _res_beamsearch_batch(self, logits : torch.Tensor, endid : list, sepids : list, legals : list):
        tensor_shape = logits.shape
        logits = logits.permute(0, 2, 1)
        if len(tensor_shape) < 3:
            if isinstance(endid, list):
                endid = endid[0]
            return self._res_beamsearch_single(logits, endid, sepids, legals)
        else:
            beam_seqs = []
            batch_size = tensor_shape[0]
            for index in range(batch_size):
                logit = logits[index]
                endidx = endid[index]
                sepid = sepids[index]
                legal = legals[index]
                beam_seqs.append(self._res_beamsearch_single(logit, endidx, sepid, legal))
            return beam_seqs

    def _res_beamsearch_single(self, logits : torch.Tensor, endid : int, sepids : list, legals : list) -> list:
        beamw = copy(self.beamw)
        predicted_points =  -1 * softmax_logits(logits/2)
        sequences = [[0]]
        scores = [0]
        finished_sequences = []
        finished_scores = []
        for _ in range(self.max_length):
            assert len(sequences) == len(scores)
            candidate_scores = []
            candidate_sequences_reconstructor = []
            for j, (sequence, score) in enumerate(zip(sequences, scores)):
                sequence_set = set(sequence)
                next_scores = predicted_points[sequence[-1]]
                for index in range(endid + 1):
                    if index in sequence_set:
                        continue
                    if index not in legals:
                        continue
                    if len(sequence) == len(legals) - 1:
                        if index not in sepids:
                            continue
                    elif index in sepids and len(sepids) == 1:
                        continue

                    candidate_scores.append(score + next_scores[index])
                    candidate_sequences_reconstructor.append((j, index))

            if not candidate_scores:
                break

            if beamw < 1:
                break
            if beamw >= len(candidate_scores):
                top_n_indexes = list(range(len(candidate_scores)))
            else:
                top_n_indexes = np.argpartition(candidate_scores, beamw)[:beamw]

            new_sequences = []
            new_scores = []

            for top_n_index in top_n_indexes:
                sequence_index, token_index = candidate_sequences_reconstructor[
                    top_n_index]
                new_sequence = sequences[sequence_index] + [token_index]
                new_score = candidate_scores[top_n_index]
                if len(new_sequence) == len(legals):
                    finished_sequences.append(new_sequence)
                    finished_scores.append(-1 * new_score / len(new_sequence))
                    beamw -= 1
                else:
                    new_sequences.append(new_sequence)
                    new_scores.append(new_score)

            sequences = new_sequences
            scores = new_scores
            if beamw < 1:
                break
        if not finished_sequences:
            return None

        return finished_sequences[np.argmax(finished_scores)][1:]


def report_pipeline_output(out_path, sentences, labels, switchs, taggers, outputs, tgt_tokens = None, eq_label = None, uuid = None):
    workbook = xlsxwriter.Workbook(out_path)
    worksheet = workbook.add_worksheet('data')
    if tgt_tokens is not None:
        header = ['Sentence', 'Label', 'Switch', 'Tagger', 'Output', 'tgt', 'Eq']
    else:
        header = ['Sentence', 'Label', 'Switch', 'Tagger', 'Output']
    if uuid is not None:
        header = ['UUID'] + header
    worksheet.write_row(0, 0, header)
    row_id = 1
    size = len(sentences)
    for idx in range(size):
        sentence = sentences[idx]
        label = labels[idx]
        switch = switchs[idx]
        tagger = taggers[idx]
        output = outputs[idx]
        eqlab  = eq_label[idx] if eq_label is not None else None
        tgt = tgt_tokens[idx] if tgt_tokens is not None else None
        collection = [sentence, json.dumps(label, ensure_ascii=False), switch, tagger, output, tgt, eqlab] if tgt_tokens is not None else [sentence, json.dumps(label, ensure_ascii=False), switch, tagger, output]
        if uuid is not None:
            collection = [uuid[idx]] + collection
        worksheet.write_row(row_id, 0, collection)
        row_id += 1
    workbook.close()


def fillin_tokens(generator_tokens, mlm_masks, mlm_tgts):
    size = len(generator_tokens)
    data_out, tgt_counter = [], 0
    for lidx in range(size):
        tokens = generator_tokens[lidx]
        masks = mlm_masks[lidx]
        posts = []
        length = len(tokens)
        assert length == len(masks)
        for idx in range(length):
            if masks[idx] == 1 and tokens[idx] == 103:
                posts.append(mlm_tgts[tgt_counter])
                tgt_counter += 1
            elif (masks[idx] == 1 and tokens[idx] != 103) and (masks[idx] == 0 and tokens[idx] == 103):
                raise Exception('Error Instance.')
            else:
                 posts.append(tokens[idx])
        data_out.append(posts)
    return data_out


def obtain_uuid(file_path : str):
    with open(file_path, 'r', encoding='utf-8') as fp:
        test_data = json.load(fp)
    fp.close()
    return list(test_data.keys())
