from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import json
import traceback
import torch
from copy import copy, deepcopy
from utils.JointSTG import *

'''
Definitions of Consts
'''

def compare_iner_operate(sentence : str, ops : list) -> dict:
    op_num = []
    def iner_same_compare(sentence : str, opis : list) -> dict:
        oper_num = []
        for op in opis:
            tmp_opnum = 0
            if 'Delete' in op.keys(): tmp_opnum += len(op['Delete'])
            if 'Insert' in op.keys():
                for ins in op['Insert']: tmp_opnum += len(ins['label'])
            if 'Modify' in op.keys():
                for ins in op['Modify']: tmp_opnum += len(ins['label'])
            oper_num.append(tmp_opnum)
        argmin = oper_num.index(min(oper_num))
        if 0 in [oele - min(oper_num) for oele in oper_num]:
            mini_ops = [opis[i] for i, x in enumerate(oper_num) if x == min(oper_num)]
            return mini_ops[0]
        else: return opis[argmin]
    for opidx in range(len(ops)):
        op = ops[opidx]
        op_num.append(len(op.keys()))
    argmin = op_num.index(min(op_num))
    if 0 in [oele - min(op_num) for oele in op_num]:
        min_ops = [ops[i] for i, x in enumerate(op_num) if x == min(op_num)]
        return iner_same_compare(sentence, min_ops)
    else: return ops[argmin]


def data_filter(sentences : list, operates : list):
    filter_operates = []
    print('>>> Select Operate Mode')
    for oidx in range(len(operates)):
        operate = operates[oidx]
        sentence = sentences[oidx]
        if len(operate) < 1: filter_operates.append({})
        elif len(operate) < 2: filter_operates.append(operate[0])
        elif len(operate) > 1: filter_operates.append(compare_iner_operate(sentence, operate))
    return filter_operates


def operate_filter(sentence : str, operates : list) -> list:
    filter_operate = []
    if len(operates) < 1: filter_operate.append({})
    elif len(operates) < 2: filter_operate.append(operates[0])
    elif len(operates) > 1: filter_operate.append(compare_iner_operate(sentence, operates))
    return filter_operate


def map_unk2word(tokens : list, sentence : str):
    map_dict, sentid = {}, 0
    for idx in range(1, len(tokens)-1):
        if sentence[sentid] == tokens[idx]:
            sentid += 1
        else:
            if tokens[idx].startswith('#'):
                tmptok = tokens[idx].replace('#', '')
                sentid += len(tmptok)
            elif tokens[idx] == '[UNK]':
                map_dict[idx] = sentence[sentid]
                sentid+=1

    return map_dict


def combine_insert_modify(insert_label, modify_label):
    comb_label = []
    assert len(insert_label) == len(modify_label)
    size = len(insert_label)
    for idx in range(size):
        if insert_label[idx] != -1 and modify_label[idx] != -1: raise Exception("Error combinition.")
        elif insert_label[idx] != -1:
            comb_label.append(insert_label[idx])
        elif modify_label[idx] != -1:
            comb_label.append(modify_label[idx])
        else: comb_label.append(-1)
    return comb_label


def convert_tagger2generator(tokens : list, tagger : list, mask_label : dict) -> tuple:
    '''
    Convert Tag (I / MI) To Target Sequence with [Mask] Symbol
    :param tokens: tokens from tokenizer (list)
    :param tagger: tag_sequence list (From TaggerConvertor Label)
    :param mask_label: mask tgt labels (list) from tokenizer
    :return: tokens (list), tgt_labels (list)
    '''
    post_sequence, post_label = [], []
    for index in range(len(tagger)):
        if tagger[index] == KEEP_TAG:
            post_sequence.append(tokens[index])
            post_label.append(GEN_KEEP_LABEL)
        elif tagger[index] in [DELETE_TAG, MODIFY_DELETE_TAG]:
            continue
        elif tagger[index] == INSERT_TAG:
            post_sequence.append(tokens[index])
            post_label.append(GEN_KEEP_LABEL)
            assert index in mask_label.keys()
            post_sequence.extend([MASK_SYMBOL] * len(mask_label[index]))
            post_label.extend(mask_label[index])
        elif tagger[index] == MODIFY_TAG:
            assert index in mask_label.keys()
            post_sequence.extend([MASK_SYMBOL] * len(mask_label[index]))
            post_label.extend(mask_label[index])
        elif tagger[index] == MOIFY_ONLY_TAG:
            post_sequence.append(MASK_SYMBOL)
            assert index in mask_label.keys() and isinstance(mask_label[index], int)
            post_label.append(mask_label[index])
    return post_sequence, post_label


def switch_convertor(sentence : str, switch_op : list):
    assert len(sentence) == len(switch_op)
    convert_tokens = [sentence[index] for index in switch_op]
    return ''.join(convert_tokens)

def collate_fn_base(batch):
    dim = len(batch[0].keys())
    if dim == 2:  # Train DataLoader
        tokens    = [item['token'] for item in batch]
        labels = [item['label'] for item in batch]
        return (tokens, labels)
    elif dim == 1: # Test DataLoader
        tokens    = [item['token'] for item in batch]
        return (tokens)
    else:
        raise Exception('Error Batch Input, Please Check.')

# Collate_fn of DataLoader(TaggerDdataset)
def collate_fn_tagger(batch):
    dim = len(batch[0].keys())
    if dim == 4:  # Train DataLoader
        tokens = [item['token'] for item in batch]
        tagger = [item['tagger'] for item in batch]
        ins    = [item['ins'] for item in batch]
        mod    = [item['mod'] for item in batch]
        return (tokens, tagger, ins, mod)
    elif dim == 1: # Test DataLoader
        tokens = [item['token'] for item in batch]
        return (tokens)
    else:
        raise Exception('Error Batch Input, Please Check.')

INER_PUNCT = {'''“''' : '''"''', '''”“''' : '''"''', '''‘''' : "'", '''’''' : "'", '：' : ':',
              '''”''' : '''"''', '℅' : '%'}
import re
ENG_PATTERN = re.compile(r'[A-Za-z]', re.S)

class TextWash(object):
    @staticmethod
    def punc_wash(sentence : str, puncls = None, lower = True):
        if puncls is None:
            punc_ls = INER_PUNCT
        else:
            punc_ls = puncls
        if lower:
            sentence = sentence.lower()
        for ele in punc_ls:
            sentence = sentence.replace(ele, punc_ls[ele])
        return sentence
    
    @staticmethod
    def punc_wash_res(sentence : str, puncls = None, lower = True):
        map_special_element, post_sentence = {}, ''
        if puncls is None:
            punc_ls = INER_PUNCT
        else:
            punc_ls = puncls

        for i, item in enumerate(sentence):
            if (ord(item) > 96 and ord(item) < 123) or (ord(item) > 64 and ord(item) < 91):
                map_special_element[i] = item
                if lower:
                    post_sentence += item.lower()
                else: post_sentence += item
            elif item in punc_ls:
                post_sentence += punc_ls[item]
                map_special_element[i] = item
            else:
                post_sentence += item
        return post_sentence, map_special_element

class Point(object):
    """Point Class"""

    def __init__(self, index : int, token : str, offset : int = 0):
        self.point_index = index
        self.token = token
        self.offset = offset

    def __str__(self):
        return "(%d:%s[%d])" % (self.point_index, self.token, self.offset)

    def __repr__(self):
        return str(self)

class Converter(object):
    """Base Converter Class"""

    def __init__(self, args):
        '''
        Base Converter Class
        :param args: Work Prams
        '''
        self.origins = []
        self.labels = []

    def convert_point(self, sentence : str, ops : dict, **kwargs):
        '''
        Convert Operators 2 Labels
        :param ops: operator dict (json format)
        :return:
        '''
        raise NotImplementedError

    def convert_tagger(self, sentence :str, ops : dict, **kwargs):
        raise NotImplementedError

    def get_ordersum(self):
        raise NotImplementedError

    def __len__(self):
        '''
        Get Length of Converter
        :return: Pointer Length
        '''
        return len(self.labels)

    def __repr__(self):
        '''
        Print Operator descriptions
        '''
        raise NotImplementedError

    def getlabel(self, types = "list"):
        raise NotImplementedError

    def _getlabel(self, types = "list"):
        '''
        Return Labels of Converter
        :param types: label type ["list", "numpy", "tensor"]
        :return: labels with specific format
        '''
        if len(self.labels) == 0:
            raise Exception("You should run Convertor.convert_xxx first.")
        if types == "list":
            return self.labels
        elif types == "numpy":
            try:
                return np.array(self.labels)
            except:
                raise ValueError('Label should be an legal, not {}'.format(type(self.labels)))
        elif types == "tensor":
            try:
                return torch.from_numpy(np.array(self.labels))
            except:
                raise ValueError('Label should be an legal, not {}'.format(type(self.labels)))
        else:
            raise Exception('params types %s not exist, please check.' % types)

class PointConverter(Converter):
    """Converter from training target Switch Ops into pointor format."""

    def __init__(self, args, auto : bool = False, spmap :bool = False, **kwargs):
        super(PointConverter, self).__init__(args)
        self.point_sequence = []
        self.post_sentence = ""
        self.origin_sentence = ""
        self.use_lm = False
        self.p2next = True
        self.spmap = spmap
        try:
            if args.language_model: self.use_lm = True
        except: self.use_lm = False
        try:
            if args.p2next is not True: self.p2next = False
        except: self.use_lm = False
        if auto:
            if 'sentence' in kwargs.keys() and 'ops' in kwargs.keys():
                kwargs_copy = copy(kwargs)
                del kwargs_copy['sentence']
                del kwargs_copy['ops']
                self.convert_point(sentence=kwargs['sentence'], ops=kwargs['ops'], **kwargs_copy)
            else:
                raise Exception("param `auto` is true, but param `sentence` or `ops` not found")

    def _convert_point_lm(self, sentence : str, ops : dict, token : list) -> tuple:
        '''
        Convert PLM Mechanism 2 PointConvertor Format
        :param sentence: original sentence (str)
        :param ops: operator description (dict)
        :param token: tokens (list)
        :return: post_sentence (list), post_ops (dict)
        '''
        self.origin_sentence = sentence if isinstance(sentence, str) else ''.join(sentence)
        post_sentence, post_ops = [], {'Switch' : []}
        pos_map = {-1 :-1}
        tpidx = 0
        for eidx in range(len(token)):
            if token[eidx] == '[CLS]' or token[eidx] == '[SEP]':
                continue
            if sentence[tpidx] == token[eidx] or token[eidx] == '[UNK]':
                post_sentence.append(token[eidx])
                pos_map[tpidx] = eidx
                tpidx += 1
            else:
                tmptoken = token[eidx]
                if tmptoken.startswith("#"):
                    tmptoken = token[eidx].replace('#', '')
                if tmptoken == sentence[tpidx] or tmptoken == '[UNK]':
                    post_sentence.append(token[eidx])
                    pos_map[tpidx] = eidx
                    tpidx += 1
                else:
                    tlen = len(tmptoken)
                    for cidx in range(tlen):
                        if tmptoken[cidx] == sentence[tpidx + cidx]:
                            pos_map[tpidx + cidx] = eidx
                        else:
                            raise Exception("The sample can not be convert to token case.")
                    tpidx += tlen
                    post_sentence.append(tmptoken)
        switch_ops = [-1] + ops['Switch']
        try:
            post_ops['Switch'] = [pos_map[switch_ops[eidx]] for eidx in range(1, len(switch_ops)) if pos_map[switch_ops[eidx]] != pos_map[switch_ops[eidx - 1]]]
        except: 
            traceback.print_exc()
            print('[{}] Sentence Error.'.format(sentence))
        assert len(token) == len(post_ops['Switch'])
        return post_sentence, post_ops

    def alignment_spmap(self, token : list, sentence : str, spmap : dict):
        new_spmap = {}
        tpidx = 0
        pos_map = {}
        sentence = sentence.replace(' ', '')
        for eidx in range(len(token)):
            if token[eidx] == '[CLS]' or token[eidx] == '[SEP]':
                continue
            if sentence[tpidx] == token[eidx] or token[eidx] == '[UNK]':
                if token[eidx] == '[UNK]' and sentence[tpidx] not in spmap.values():
                    spmap[tpidx] = sentence[tpidx]
                pos_map[tpidx] = eidx
                tpidx += 1
            else:
                tmptoken = token[eidx]
                if tmptoken.startswith("#"):
                    tmptoken = token[eidx].replace('#', '')
                if tmptoken == sentence[tpidx] or tmptoken == '[UNK]':
                    pos_map[tpidx] = eidx
                    tpidx += 1
                else:
                    tlen = len(tmptoken)
                    for cidx in range(tlen):
                        if tmptoken[cidx] == sentence[tpidx + cidx]:
                            pos_map[tpidx + cidx] = eidx
                        else:
                            raise Exception("The sample can not be convert to token case.")
                    tpidx += tlen
        for trans in spmap:
            if pos_map[trans] not in new_spmap: new_spmap[pos_map[trans]] = spmap[trans]
            else: new_spmap[pos_map[trans]] += spmap[trans]
        return new_spmap

    def convert_point(self, sentence : str, ops : dict, **kwargs):
        if self.spmap: sentence, specials = sentence
        else: specials = None
        self.origins = list(range(len(sentence)))
        if 'Switch' not in ops.keys():
            if self.use_lm:
                self.point_sequence.append(Point(0, '[CLS]'))
                if 'token' in kwargs.keys():
                    if specials: self.spmap = self.alignment_spmap(kwargs['token'], sentence, specials)
                    sentence = kwargs['token']
                self.origins = list(range(len(sentence)))
                self.point_sequence += [Point(ele+1, sentence[ele]) for ele in self.origins]
                self.point_sequence.append(Point(len(self.point_sequence), '[SEP]'))
            else:
                self.point_sequence = [Point(ele, sentence[ele]) for ele in self.origins]
        else:
            if 'token' in kwargs.keys():
                sentence, ops = self._convert_point_lm(sentence, ops, kwargs['token'])
            self.point_sequence = self._convert_point(sentence, ops['Switch'])
        self.post_sentence = ''.join([ele.token for ele in self.point_sequence])
        self.labels = [(ele.point_index, ele.offset) for ele in self.point_sequence]

    def _convert_p2next_label(self, ori_labels : list):
        '''
        Convert labels to p2next version
        :param ori_labels: origin label format
        :return: next version label
        '''
        fl_map = {}
        labels = [-1] * len(ori_labels)
        for eidx in range(len(ori_labels) - 1):
            fl_map[ori_labels[eidx]] = ori_labels[eidx + 1]
        fl_map[len(ori_labels) - 1] = -1
        for eidx in range(len(ori_labels)):
            labels[eidx] = fl_map[eidx]
        return labels

    def _convert_point(self, sentence : str, op : list) -> list:
        """
        Convert Switch ops 2 Labels
        :param sentence: sentence of sample
        :param op: Switch Op List
        :return: point_sequence
        """
        index_map_inv = dict(zip(op, list(range(len(sentence)))))
        sequence = []
        if self.use_lm:
            sequence.append(Point(0, '[CLS]', 0))
            for ele in op:
                sequence.append(Point(ele + 1, sentence[ele], abs(ele - index_map_inv[ele])))
            sequence.append(Point(len(sequence), '[SEP]', 0))
        else:
            for ele in op:
                sequence.append(Point(ele, sentence[ele], abs(ele - index_map_inv[ele])))
        return sequence

    def getlabel(self, types = "list", offset : bool = True):
        if len(self.point_sequence) == 0:
            raise Exception("You should run Convertor.convert_xxx first.")
        if offset:
            if types != "list":
                raise Exception("PointConvertor not support %s type when you need offset." % types)
            return self.labels
        else:
            # Pointer2Next Version
            label = [ele[0] for ele in self.labels]
            if self.p2next and self.use_lm:
                label = self._convert_p2next_label(label)
            if types == "list":
                return label
            elif types == "numpy":
                return np.array(label)
            elif types == "tensor":
                return torch.from_numpy(np.array(label))
            else:
                raise Exception("PointConvertor only support ['list', 'numpy', 'tensor'] type")


    def convert_tagger(self, sentence :str, ops : dict, **kwargs):
        raise Exception("PointConvertor not support %s method" % "convert_tagger")

    def get_ordersum(self, need_seq : bool = False):
        '''
        Get Order for Regularization
        :param need_seq:  return order_seq or not
        :return: sum of order_seq
        '''
        order_2nd = []
        if len(self.labels) == 0:
            raise Exception("You should run Convertor.convert_xxx first.")
        for idx in range(1, len(self.labels)):
            order_2nd.append(abs(self.labels[idx][-1] - self.labels[idx-1][-1]))
        if need_seq:
            return sum(order_2nd), order_2nd
        else:
            return sum(order_2nd)

    def __repr__(self):
        obreturn = ''
        obreturn += ('>>> PointConvertor Elements : ' + '\n')
        if len(self.labels) == 0 or len(self.point_sequence) == 0:
            obreturn += 'Null object'
            return obreturn
        obreturn += 'Original sentence : {}\n'.format(self.origin_sentence)
        obreturn += ('> position index :' + '\n')
        obreturn += (', '.join([str(ele) for ele in self.getlabel(offset=False)]) + '\n')
        obreturn += ('> convert sentence :' + '\n')
        obreturn += (''.join([ele.token for ele in self.point_sequence]) + '\n')
        obreturn += ('> index offset :' + '\n')
        obreturn += (', '.join([str(ele[1]) for ele in self.labels]) + '\n')
        obreturn += ('> Sum offset : {}\n'.format(sum([ele[1] for ele in self.labels])))
        order_sum, order_seq = self.get_ordersum(need_seq=True)
        obreturn += ('>> Order Sequence :\n' + ', '.join([str(ele) for ele in order_seq]) + '\n')
        obreturn += ('> Sum 2nd-order (Regular): {}'.format(order_sum))
        return obreturn

class TaggerConverter(Converter):
    """Converter from training target into Tagger format."""

    def __init__(self, args, auto : bool = False, **kwargs):
        super(TaggerConverter, self).__init__(args)
        self.tagger_sequence = []
        self.mask_label = {}
        self.ins_label = []
        self.mod_label = []
        self.post_tokens = []
        self.origin_sentence = ""
        self.use_lm = False
        self.ignore_index = args.ignore_val
        try:
            if args.language_model: self.use_lm = True
        except:
            self.use_lm = False
        if auto:
            if 'sentence' in kwargs.keys() and 'ops' in kwargs.keys():
                kwargs_copy = copy(kwargs)
                del kwargs_copy['sentence']
                del kwargs_copy['ops']
                self.convert_tagger(sentence=kwargs['sentence'], ops=kwargs['ops'], **kwargs_copy)
            else:
                raise Exception("param `auto` is true, but param `sentence` or `ops` not found")

    def convert_point(self, sentence: str, ops: dict, **kwargs):
        raise Exception("PointConvertor not support %s method" % "convert_point")

    def getlabel(self, types="list") -> dict:
        if len(self.labels) == 0:
            raise Exception("You should run Convertor.convert_xxx first.")
        if types != "dict":
            raise Exception("TaggerConverter not support %s type" % types)
        return self.labels

    def _convert_tagger_lm(self, sentence : str, ops : dict, token : list) -> tuple:
        '''
        Convert PLM Mechanism 2 Tagger ConvertorFormat
        :param sentence: original sentence(str)
        :param ops: operator description(dict)
        :param token: tokens(list)
        :return: post_sentence(list), post_ops(dict)
        '''
        self.origin_sentence = sentence if isinstance(sentence, str) else ''.join(sentence)
        post_sentence, post_ops = [], {}
        pos_map, pos_map_inv = {-1 :-1}, {-1 : -1}
        tpidx = 0
        for eidx in range(len(token)):
            if token[eidx] == '[CLS]' or token[eidx] == '[SEP]':
                continue
            if sentence[tpidx] == ' ':
                tpidx += 1
            if sentence[tpidx] == token[eidx] or token[eidx] == '[UNK]':
                post_sentence.append(token[eidx])
                pos_map[tpidx] = eidx
                pos_map_inv[eidx] = [tpidx]
                tpidx += 1
            else:
                tmptoken = token[eidx]
                if tmptoken.startswith("#"):
                    tmptoken = token[eidx].replace('#', '')
                if tmptoken == sentence[tpidx] or tmptoken == '[UNK]':
                    post_sentence.append(token[eidx])
                    pos_map[tpidx] = eidx
                    pos_map_inv[eidx] = [tpidx]
                    tpidx += 1
                else:
                    tlen = len(tmptoken)
                    pos_map_inv[eidx] = [tpidx]
                    tp_cidx = 0
                    for cidx in range(tlen):
                        if tmptoken[cidx] == sentence[tpidx + cidx]:
                            pos_map[tpidx + cidx] = eidx
                            tp_cidx = cidx
                        else:
                            raise Exception("The sample can not be convert to token case.")
                    pos_map_inv[eidx].append(tpidx + tp_cidx)
                    tpidx += tlen
                    post_sentence.append(tmptoken)
        if 'Delete' in ops:
            delete_op = []
            for didx in ops['Delete']:
                delete_op.append(pos_map[didx])
            post_ops['Delete'] = delete_op
        if 'Insert' in ops:
            insert_op = []
            for ins in ops['Insert']:
                nins = {}
                pos = ins['pos']
                nins['pos'] = pos_map[pos]
                nins['tag'] = ins['tag']
                nins['label'] = ins['label']
                nins['label_token'] = ins['label_token']
                insert_op.append(nins)
            post_ops['Insert'] = insert_op
        if 'Modify' in ops:
            modify_op = []
            for mod in ops['Modify']:
                nmod_op = {}
                pos, tag = mod['pos'], mod['tag']
                nmod_op['pos'] = pos_map[pos]
                tag_o = eval(tag.split('+')[0].split('_')[-1])
                label_length = len(mod['label_token'])
                e_pos = pos_map[pos + tag_o - 1]
                s_pos = pos_map[pos]
                cur_len = e_pos - s_pos + 1
                if cur_len == label_length:
                    nmod_op['tag'] = 'MOD_' + str(cur_len)
                else:
                    delta_len = label_length - cur_len
                    if delta_len > 0: nmod_op['tag'] = 'MOD_' + str(cur_len) + '+INS_' + str(delta_len)
                    else: nmod_op['tag'] = 'MOD_' + str(cur_len) + '+DEL_' + str(-delta_len)
                nmod_op['label'] = mod['label']
                nmod_op['label_token'] = mod['label_token']
                modify_op.append(nmod_op)
            post_ops['Modify'] = modify_op
        return post_sentence, post_ops

    def apply_tagger(self, tokens:list, tagger : list, mask_label : list):
        post_tokens = []
        for tid in range(len(tokens)):
            if tagger[tid] == 'K':
                post_tokens.append(tokens[tid])
            elif tagger[tid] in ['D', 'MD']:
                continue
            elif tagger[tid] == 'I':
                post_tokens.append(tokens[tid])
                post_tokens.extend(mask_label[tid])
            elif tagger[tid] == 'MI':
                post_tokens.append(mask_label[tid][0])
                post_tokens.extend(mask_label[tid][1:])
            elif tagger[tid] == 'M':
                post_tokens.append(mask_label[tid])
        return post_tokens

    def convert_tagger(self, sentence :str, ops : dict, **kwargs):
        self.origins = list(range(len(sentence)))
        self.origin_sentence = sentence
        # Preprocess LM
        if 'token' in kwargs.keys():
            sentence, ops = self._convert_tagger_lm(sentence, ops, kwargs['token'])
        self.tagger_sequence, self.mask_label, self.ins_label, self.mod_label = self._convert_tagger(sentence, ops)
        self.labels = {}
        self.labels['tagger'] = self.tagger_sequence
        self.labels['mask_label'] = self.mask_label
        self.labels['ins_label'] = self.ins_label
        self.labels['mod_label'] = self.mod_label
        self.post_tokens = self.apply_tagger(['[CLS]'] + kwargs['token'] + ['[SEP]'], self.tagger_sequence, self.mask_label)

    def _convert_tagger(self, post_sentence: str, ops: dict) -> tuple:
        '''
        Convert Tagger ops 2 Labels
        :param post_sentence: The sentence after Tagger operator
        :param ops: tagger_label, ins_label, mod_label, mask_label (list)
        '''
        if self.use_lm and isinstance(post_sentence, list):
            post_sentence = ['[CLS]'] + post_sentence + ['[SEP]']
        tagger = ['K'] * len(post_sentence)
        ins_label = [self.ignore_index] * len(post_sentence)
        mod_label = [self.ignore_index] * len(post_sentence)
        mask_label = {}
        # Delete Operator
        if 'Delete' in ops.keys():
            for ele in ops['Delete']:
                tagger[ele + 1] = 'D'
        # Insert Operator
        if 'Insert' in ops.keys():
            for inop in ops['Insert']:
                tagger[inop['pos'] + 1] = 'I'
                ins_label[inop['pos'] + 1] = eval(inop['tag'].split('_')[-1])
                mask_label[inop['pos'] + 1] = inop['label_token']
        # Modify Operator
        if 'Modify' in ops.keys():
            for moop in ops["Modify"]:
                oplen = moop['tag']
                if '+' not in oplen:
                    oplen = eval(oplen.split('_')[-1])
                    for idx in range(oplen):
                        tagger[moop['pos'] + idx + 1] = 'M'
                        mask_label[moop['pos'] + idx + 1] = moop['label_token'][idx]
                else:
                    # Contain + in op
                    if 'DEL' in moop['tag']:
                        oplen = len(moop['label_token'])
                        opdel = eval(moop['tag'].split('+')[-1].split('_')[-1])
                        for idx in range(oplen):
                            tagger[moop['pos'] + idx + 1] = 'M'
                            mask_label[moop['pos'] + idx + 1] = moop['label_token'][idx]
                        for idx in range(opdel):
                            tagger[moop['pos'] + oplen + idx + 1] = 'MD'
                    elif 'INS' in moop['tag']:
                        oplen = eval(moop['tag'].split('+')[0].split('_')[-1])
                        opins = eval(moop['tag'].split('+')[-1].split('_')[-1])
                        for idx in range(oplen - 1):
                            tagger[moop['pos'] + idx + 1] = 'M'
                            mask_label[moop['pos'] + idx + 1] = moop['label_token'][idx]
                        tagger[moop['pos'] + oplen] = 'MI'
                        mod_label[moop['pos'] + oplen] = opins
                        mask_label[moop['pos'] + oplen] = moop['label_token'][oplen - 1:]
                    else:
                        raise Exception("Operator contains unkown tag %s" % moop['tag'])
        return tagger, mask_label, ins_label, mod_label

    def __repr__(self):
        obreturn = ''
        obreturn += ('>>> TaggerConverter Elements : ' + '\n')
        if len(self.labels) == 0 or len(self.tagger_sequence) == 0:
            obreturn += 'Null object'
            return obreturn
        obreturn += 'Original sentence : {}\n'.format(self.origin_sentence)
        obreturn += ('> Tagger sequence :\n')
        obreturn += (', '.join([ele for ele in self.tagger_sequence]) + '\n')
        obreturn += ('> convert sentence :' + '\n')
        obreturn += (''.join([ele for ele in self.post_tokens]) + '\n')
        obreturn += ('> Mask label :\n')
        obreturn += (', '.join(['({}, {})'.format(ele, self.mask_label[ele]) for ele in self.mask_label.keys()]) + '\n')
        obreturn += ('> Insert label :\n')
        obreturn += (', '.join(['{}'.format(ele, self.ins_label[ele]) for ele in self.ins_label]) + '\n')
        obreturn += ('> Modify label :\n')
        obreturn += (', '.join(['{}'.format(ele, self.mod_label[ele]) for ele in self.mod_label]) + '\n')
        return obreturn

class SwitchDataset(Dataset):
    def __init__(self, args, path : str, desc : str):
        super(SwitchDataset, self).__init__()
        # INITIALIZE
        self.args          = args
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.tokenizer     = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
        self.padding_size  = args.padding_size
        self.desc = desc
        self.spmap = args.sp_map
        # DATA PROCESSER
        self.error_number  = 0
        self.sentences, self.label   = self._read_csv(path)
        self.origin_label = deepcopy(self.label)
        self.label = data_filter(self.sentences, self.label)
        self.final_sentences_ids = []
        self.point_seq, self.token, self.wd_idx, self.label = self._process_switch(self.sentences, self.label)

    def _read_csv(self, path):
        sentences, labels = [], []
        data = np.array(pd.read_csv(path))
        for ele in data:
            sentences.append(ele[0])
            labels.append(json.loads(ele[1]))
        return sentences, labels

    def _unpack(self, sentences, labels):
        '''
        Unpack multi-operator samples to sigle-operator samples (Expand)
        :param sentence: sentence list
        :param label: label list
        :return: expanded sentence, label
        '''
        assert len(sentences) == len(labels)
        unpack_sentences, unpack_labels = [], []
        for idx in range(len(sentences)):
            sentence = sentences[idx]
            label = labels[idx]
            for ele in label:
                unpack_sentences.append(sentence)
                unpack_labels.append(ele)
        return unpack_sentences, unpack_labels

    def _process_switch(self, sentences, labels):
        '''
        Process Switch Labels
        :param sentences: sentence list
        :param labels: label list
        :return: point list, token list, label
        '''
        point_seqs, wd_collect, post_labels, token_collection = [], [], [], []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset'):
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]) if not self.spmap else TextWash.punc_wash_res(sentences[idx])[0])
            kwargs = {
                'sentence' : TextWash.punc_wash(sentences[idx]) if not self.spmap else TextWash.punc_wash_res(sentences[idx]),
                'ops' : labels[idx],
                'token' : token
            }
            tokens = [self.CLS] + token + [self.SEP]
            try:
                pointer = PointConverter(self.args, auto=True, spmap=self.spmap, **kwargs)
                wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            except:
                import traceback
                print(traceback.print_exc())
                self.error_number += 1
                print(sentences[idx])
                continue
            if len(pointer.labels) > self.args.padding_size:
                self.error_number += 1
                continue
            self.final_sentences_ids.append(idx)
            point_seqs.append(pointer)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            post_labels.append(pointer.getlabel(offset=False))
        return  point_seqs, token_collection, wd_collect, post_labels

    def __getitem__(self, item):
        wid, label = self.wd_idx[item], self.label[item]
        ret = {
            'token' : wid,
            'label' : label
        }
        return ret

    def __len__(self):
        return len(self.label)

class TaggerDataset(Dataset):
    def __init__(self, args, path : str, desc : str, token_ext : list = None):
        super(TaggerDataset, self).__init__()
        # INITIALIZE
        self.args          = args
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.tokenizer     = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
        self.padding_size  = args.padding_size
        self.error_number  = 0
        self.desc = desc
        self.sentences, self.label = self._read_csv(path)
        if self.desc != 'test' or token_ext is None:
            # DATA PROCESSER
            # self.sentences, self.label = self._unpack(sentences, label)
            self.label = data_filter(self.sentences, self.label)
            self.tagger_seq, self.token, self.wd_idx, self.tagger_label, self.comb_label = self._process_tagger(self.sentences, self.label)
            self.tagger_idx = self._tag2idx(self.tagger_label)
        else:
            self.wd_idx = token_ext

    def _read_csv(self, path):
        sentences, labels = [], []
        data = np.array(pd.read_csv(path))
        for ele in data:
            sentences.append(ele[0])
            labels.append(json.loads(ele[1]))
        return sentences, labels

    def _tag2idx(self, tagger_labels : list):
        '''
        Convert Tagger Labels 2 Index based on Defines
        :param tagger_labels: Tagger labels (list)
        :return: Tagger Labels(index map) (list)
        '''
        tagidxs = [[TAGGER_MAP[ele] for ele in ins] for ins in tagger_labels]
        return tagidxs

    def _unpack(self, sentences, labels):
        '''
        Unpack multi-operator samples to sigle-operator samples (Expand)
        :param sentence: sentence list
        :param label: label list
        :return: expanded sentence, label
        '''
        assert len(sentences) == len(labels)
        unpack_sentences, unpack_labels = [], []
        for idx in range(len(sentences)):
            sentence = sentences[idx]
            label = labels[idx]
            for ele in label:
                unpack_sentences.append(sentence)
                unpack_labels.append(ele)
        return unpack_sentences, unpack_labels

    def _preprocess_modify(self, ops : dict):
        '''
        Pre-tokenize modify labels for convertor
        :param ops: operator (dict)
        :return: processed operator (dict)
        '''
        if 'Modify' not in ops.keys() and 'Insert' not in ops.keys():
            return ops
        nop = copy(ops)
        if 'Modify' in ops.keys():
            nmod = []
            for mod in nop['Modify']:
                if isinstance(mod['label'], list):
                    labstr = mod['label'][0]
                else:
                    labstr = mod['label']
                mod['label_token'] = self.tokenizer.tokenize(labstr)
                nmod.append(mod)
            nop['Modify'] = nmod
        if 'Insert' in ops.keys():
            nins = []
            for ins in nop['Insert']:
                if isinstance(ins['label'], list):
                    labstr = ins['label'][0]
                else:
                    labstr = ins['label']
                ins['label_token'] = self.tokenizer.tokenize(labstr)
                nins.append(ins)
            nop['Insert'] = nins
        return nop

    def _process_tagger(self, sentences : list, labels : list, token_ext : list=None) -> tuple:
        '''
        Process Tagger Labels
        :param sentences: sentence list
        :param labels: label list
        :return: Tagger list, token list, label list
        '''
        tagger_seqs, wd_collect, token_collection, tagger_label, comb_labels = [], [], [], [], []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset'):
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]))
            kwargs = {
                'sentence' : TextWash.punc_wash(sentences[idx]),
                'ops' : self._preprocess_modify(labels[idx]),
                'token' : token
            }
            tokens = [self.CLS] + token + [self.SEP]
            try:
                tagger = TaggerConverter(self.args, auto=True, **kwargs)
                label_comb = tagger.getlabel(types='dict')
                comb_label = combine_insert_modify(label_comb['ins_label'], label_comb['mod_label'])
                # if max(comb_label) > 10:
                #     self.error_number += 1
                #     continue
            except:
                self.error_number += 1
                print(sentences[idx])
                continue
            # label_comb = tagger.getlabel(types='dict')
            # comb_label = combine_insert_modify(label_comb['ins_label'], label_comb['mod_label'])
            wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            tagger_seqs.append(tagger)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            tagger_label.append(label_comb['tagger'])
            comb_labels.append(comb_label)
        return tagger_seqs, token_collection, wd_collect, tagger_label, comb_labels

    def __getitem__(self, item):
        if self.desc != 'test':
            wid, tagger, comb = self.wd_idx[item], self.tagger_idx[item], self.comb_label[item]
            ret = {
                'token'  : wid,
                'tagger' : tagger,
                'comb'   : comb
            }
        else:
            wid = self.wd_idx[item]
            ret = {
                'token' : wid
            }
        return ret

    def __len__(self):
        return len(self.wd_idx)


class GeneratorDataset(Dataset):
    def __init__(self, args, path : str, desc : str, token_ext : list = None, label_ext : list = None):
        super(GeneratorDataset, self).__init__()
        # INITIALIZE
        self.args          = args
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.tokenizer     = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
        self.padding_size  = args.padding_size
        self.desc = desc
        self.error_number = 0
        # DATA PROCESSER
        sentences, labels  = self._read_csv(path)
        sentences, labels = self._unpack(sentences, labels)
        self.sentences, self.labels = self._extract_gendata(sentences, labels)
        if desc != 'test' and token_ext is None and label_ext is None:
            self.tagger_seq, self.token, self.wd_idx, self.mlm_labels = self._process_generator(self.sentences, self.labels)
        else:
            self.wd_idx = token_ext
            self.mlm_labels = label_ext
            self.labels = label_ext

    def _read_csv(self, path):
        type_flag = False
        sentences, labels = [], []
        data = np.array(pd.read_csv(path))
        if data.shape[1] == 3: type_flag = True
        for ele in data:
            sentences.append(ele[0])
            labels.append(json.loads(ele[1])) if not type_flag else labels.append(json.loads(ele[2]))
        return sentences, labels

    def _unpack(self, sentences, labels):
        '''
        Unpack multi-operator samples to sigle-operator samples (Expand)
        :param sentence: sentence list
        :param label: label list
        :return: expanded sentence, label
        '''
        assert len(sentences) == len(labels)
        unpack_sentences, unpack_labels = [], []
        for idx in range(len(sentences)):
            sentence = sentences[idx]
            label = labels[idx]
            for ele in label:
                unpack_sentences.append(sentence)
                unpack_labels.append(ele)
        return unpack_sentences, unpack_labels

    def _extract_gendata(self, sentences : list, labels : list) -> tuple:
        '''
        Extract Data that Contains I / MI / M
        :param sentences: sentences collection list
        :param labels: labels list
        :return: extracted sentence, label
        '''
        post_sentences, post_labels = [], []
        for index in range(len(sentences)):
            sentence, label = sentences[index], labels[index]
            if 'Switch' in label:
                try:
                    sentence = switch_convertor(sentence, label['Switch'])
                except:
                    print(sentence)
            if 'Insert' in label or 'Modify' in label:
                if 'Modify' in label:
                    mod_op = label['Modify']
                    MI_flags = [True if 'INS' in mod['tag'] or '+' not in mod['tag'] else False for mod in mod_op]
                    if True not in MI_flags:
                        continue
                post_sentences.append(sentence)
                post_labels.append(label)
            else:
                continue
        return post_sentences, post_labels

    def _preprocess_insert(self, ops : dict):
        '''
        Pre-tokenize modify labels for convertor
        :param ops: operator (dict)
        :return: processed operator (dict)
        '''
        if 'Modify' not in ops.keys() and 'Insert' not in ops.keys():
            return ops
        nop = copy(ops)
        if 'Modify' in ops.keys():
            nmod = []
            for mod in nop['Modify']:
                if isinstance(mod['label'], list):
                    labstr = mod['label'][0]
                else:
                    labstr = mod['label']
                mod['label_token'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(labstr))
                nmod.append(mod)
            nop['Modify'] = nmod
        if 'Insert' in ops.keys():
            nins = []
            for ins in nop['Insert']:
                if isinstance(ins['label'], list):
                    labstr = ins['label'][0]
                else:
                    labstr = ins['label']
                ins['label_token'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(labstr))
                nins.append(ins)
            nop['Insert'] = nins
        return nop

    def _process_generator(self, sentences : list, labels : list) -> tuple:
        '''
        Process Generator Labels
        :param sentences: sentence list
        :param labels: label list
        :return: Tagger list, token list, label list
        '''
        tagger_seqs, wd_collect, token_collection, tgt_mlms = [], [], [], []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset'):
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]))
            try:
                kwargs = {
                    'sentence' : TextWash.punc_wash(sentences[idx]),
                    'ops' : self._preprocess_insert(labels[idx]),
                    'token' : token
                }
            except:
                print(sentences[idx])
            tokens = [self.CLS] + token + [self.SEP]
            try:
                tagger = TaggerConverter(self.args, auto=True, **kwargs)
                label_comb = tagger.getlabel(types='dict')
                tags, mask_label = label_comb['tagger'], label_comb['mask_label']
                tokens, label = convert_tagger2generator(tokens, tags, mask_label)
                wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            except:
                print(sentences[idx])
                self.error_number += 1
                continue
            tagger_seqs.append(tagger)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            tgt_mlms.append(label)
        return tagger_seqs, token_collection, wd_collect, tgt_mlms

    def __getitem__(self, item):
        wid, mlm_label = self.wd_idx[item], self.mlm_labels[item]
        ret = {
            'token'  : wid,
            'label'  : mlm_label
        }
        return ret

    def __len__(self):
        return len(self.wd_idx)


class JointDataset(Dataset):
    def __init__(self, args, path : str, desc : str):
        super(JointDataset, self).__init__()
        self.args          = args
        # INITIALIZE
        self.CLS           = "[CLS]"
        self.SEP           = "[SEP]"
        self.tokenizer     = BertTokenizer.from_pretrained(args.lm_path, cache_dir='./.cache')
        self.num_classes   = args.num_classes
        self.padding_size  = args.padding_size
        self.desc = desc
        # DATA PROCESSER
        self.sentences, self.label   = self._read_csv(path)
        if self.desc in ['train', 'valid', 'test']:
            self.operates = data_filter(self.sentences, self.label)
        # Switch Data
        self.error_ids = []
        self.point_seq, self.token, self.wd_idx, self.sw_label, self.unk_map = self._process_switch(self.sentences, self.operates)
        # Tagger Data & Generate Data
        self.tag_token, self.tagwd_idx = self._switch_tokens(self.point_seq, self.operates, self.token)
        self.tagger_seq, self.tokens, self.tagger_tokens, self.tagger_label, self.comb_labels, \
            self.gen_token, self.genwd_idx, self.tgt_mlm = self._process_tagger(self.sentences, self.operates)
        self.tagger_idx = self._tag2idx(self.tagger_label)
        print('>> Joint {} Dataset Has Been Processed'.format(desc))

    def _read_csv(self, path):
        sentences, labels = [], []
        data = np.array(pd.read_csv(path))
        for ele in data:
            sentences.append(ele[0])
            try:
                labels.append(json.loads(ele[1]))
            except:
                print(ele[0])
        return sentences, labels

    def _process_switch(self, sentences, labels):
        '''
        Process Switch Labels
        :param sentences: sentence list
        :param labels: label list
        :return: point list, token list, label
        '''
        point_seqs, wd_collect, post_labels, token_collection = [], [], [], []
        unk_map = []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset[Switch Part]'):
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]))
            kwargs = {
                'sentence' : TextWash.punc_wash(sentences[idx]),
                'ops' : labels[idx],
                'token' : token
            }
            tokens = [self.CLS] + token + [self.SEP]
            unk_map.append(map_unk2word(tokens, sentences[idx]))
            try:
                pointer = PointConverter(self.args, auto=True, **kwargs)
            except:
                traceback.print_exc()
                print(sentences[idx])
            lab = pointer.getlabel(offset=False)
            if max(lab) >= self.args.padding_size:
                # print('Sentence length exceeds the limit: {}'.format(sentences[idx]))
                self.error_ids.append(idx)
                continue
            wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            point_seqs.append(pointer)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            post_labels.append(pointer.getlabel(offset=False))
        return  point_seqs, token_collection, wd_collect, post_labels, unk_map

    def _switch_tokens(self, pointer_seqs :list, operates : list, token_ls : list):
        tag_tokens, tagwd_idxs = [], []
        for idx in tqdm(range(len(pointer_seqs)), desc='Processing ' + self.desc + ' Dataset[Switch Trans_Tokens]'):
            pointer = pointer_seqs[idx]
            operate = operates[idx]
            tokens  = token_ls[idx]
            if 'Switch' not in operate:
                tag_tokens.append(tokens)
            else:
                tag_tokens.append([ele.token for ele in pointer.point_sequence])
            wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            tagwd_idxs.append(wd_idxs)
        return tag_tokens, tagwd_idxs

    def _tag2idx(self, tagger_labels : list):
        '''
        Convert Tagger Labels 2 Index based on Defines
        :param tagger_labels: Tagger labels (list)
        :return: Tagger Labels(index map) (list)
        '''
        tagidxs = [[TAGGER_MAP[ele] for ele in ins] for ins in tagger_labels]
        return tagidxs

    def _preprocess_modify(self, ops: dict):
        '''
        Pre-tokenize modify labels for convertor
        :param ops: operator (dict)
        :return: processed operator (dict)
        '''
        if 'Modify' not in ops.keys() and 'Insert' not in ops.keys():
            return ops
        nop = copy(ops)
        if 'Modify' in ops.keys():
            nmod = []
            for mod in nop['Modify']:
                if isinstance(mod['label'], list):
                    labstr = mod['label'][0]
                else:
                    labstr = mod['label']
                mod['label_token'] = self.tokenizer.tokenize(labstr)
                nmod.append(mod)
            nop['Modify'] = nmod
        if 'Insert' in ops.keys():
            nins = []
            for ins in nop['Insert']:
                if isinstance(ins['label'], list):
                    labstr = ins['label'][0]
                else:
                    labstr = ins['label']
                ins['label_token'] = self.tokenizer.tokenize(labstr)
                nins.append(ins)
            nop['Insert'] = nins
        return nop

    def _preprocess_gendata(self, ops: dict):
        '''
        Pre-tokenize modify labels and insert labels for convertor
        :param ops: operator (dict)
        :return: processed operator (dict)
        '''
        if 'Modify' not in ops.keys() and 'Insert' not in ops.keys():
            return ops
        nop = copy(ops)
        if 'Modify' in ops.keys():
            nmod = []
            for mod in nop['Modify']:
                if isinstance(mod['label'], list):
                    labstr = mod['label'][0]
                else:
                    labstr = mod['label']
                mod['label_token'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(labstr))
                nmod.append(mod)
            nop['Modify'] = nmod
        if 'Insert' in ops.keys():
            nins = []
            for ins in nop['Insert']:
                if isinstance(ins['label'], list):
                    labstr = ins['label'][0]
                else:
                    labstr = ins['label']
                ins['label_token'] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(labstr))
                nins.append(ins)
            nop['Insert'] = nins
        return nop

    def _process_tagger(self, sentences : list, labels : list) -> tuple:
        '''
        Process Tagger Labels
        :param sentences: sentence list
        :param labels: label list
        :return: Tagger list, token list, label list
        '''
        tagger_seqs, wd_collect, token_collection, tagger_label, comb_labels = [], [], [], [], []
        gen_tokens, genwd_idxs, tgt_mlms = [], [], []
        for idx in tqdm(range(len(sentences)), desc='Processing ' + self.desc + ' Dataset[Tagger/Gen Part]'):
            if idx in self.error_ids:
                continue
            token = self.tokenizer.tokenize(TextWash.punc_wash(sentences[idx]))
            kwargs = {
                'sentence' : TextWash.punc_wash(sentences[idx]),
                'ops' : self._preprocess_gendata(labels[idx]),
                'token' : token
            }
            tokens = [self.CLS] + token + [self.SEP]
            try:
                tagger = TaggerConverter(self.args, auto=True, **kwargs)
            except:
                traceback.print_exc()
                print(sentences[idx])
            label_comb = tagger.getlabel(types='dict')
            wd_idxs = self.tokenizer.convert_tokens_to_ids(tokens)
            tagger_seqs.append(tagger)
            token_collection.append(tokens)
            wd_collect.append(wd_idxs)
            comb_label = combine_insert_modify(label_comb['ins_label'], label_comb['mod_label'])
            tagger_label.append(label_comb['tagger'])
            gen_token, gen_label = convert_tagger2generator(tokens, label_comb['tagger'], label_comb['mask_label'])
            comb_labels.append(comb_label)
            #genwd_idxs.append(wd_idxs)
            genwd_idxs.append(self.tokenizer.convert_tokens_to_ids(gen_token))
            gen_tokens.append(gen_token)
            tgt_mlms.append(gen_label)
        return tagger_seqs, token_collection, wd_collect, tagger_label, comb_labels, gen_tokens, genwd_idxs, tgt_mlms

    def __getitem__(self, item):
        wid, wid_tag, wid_gen = self.wd_idx[item], self.tagwd_idx[item], self.genwd_idx[item]
        mlm_label = self.tgt_mlm[item]
        tag_label, comb_label = self.tagger_idx[item], self.comb_labels[item]
        sw_label = self.sw_label[item]
        ret = {
            'wid_ori'   : wid,
            'wid_tag'   : wid_tag,
            'wid_gen'   : wid_gen,

            'tag_label' : tag_label,
            'comb_label': comb_label,

            'swlabel'   : sw_label,
            'mlmlabel'  : mlm_label
        }
        return ret

    def __len__(self):
        return len(self.sw_label)