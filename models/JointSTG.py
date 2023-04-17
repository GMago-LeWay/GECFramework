"""
Github Repo
"""

import torch
from torch import nn
import argparse
from argparse import Namespace
import math
from transformers import BertForMaskedLM, BertModel
from transformers import AutoTokenizer

## Layer

class Linear(nn.Module):
    def __init__(self, Text_InFeature, Text_OutFeature):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=Text_InFeature, out_features=Text_OutFeature)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, args : Namespace, device : torch.device, hidden_size : int = 768, eps : float =1e-6):
        super(LayerNorm, self).__init__()
        self.eps    = eps
        self.args   = args
        self.device = device
        self.gamma  = nn.Parameter(torch.ones(hidden_size))
        self.beta   = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        hidden_states =  self.gamma * (x-mean) / (std + self.eps)
        return hidden_states + self.beta

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PointerNetwork(nn.Module):
    '''
    Pointer Network Module
    '''
    def __init__(self, args : argparse.Namespace, device : torch.device):
        super(PointerNetwork, self).__init__()
        self.args = args
        self.device = device
        # Attention Layer
        self._attn_score = AttetionScore(args, device)
        # Dense
        self._query_embedding = Linear(args.lm_hidden_size, args.lm_hidden_size)
        self._key_embedding   = Linear(args.lm_hidden_size, args.lm_hidden_size)

    def forward(self, inputs : torch.Tensor, masks : torch.Tensor = None, need_mask : bool = False):
        query  = self._query_embedding(inputs)
        key    = self._key_embedding(inputs)
        attn_ret = self._attn_score(query, key, masks, need_mask = need_mask)
        if need_mask:
            scores, mask = attn_ret
            return scores, mask
        else:
            scores = attn_ret
            return scores

# Attention Mask Const (-inf)
MASK_LOGIT_CONST = 1e9
def logits_mask(inputs :torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
    mask_add =  -MASK_LOGIT_CONST * (1. - mask)
    scores = inputs * mask + mask_add
    return scores

class SelfAttentionMask(object):
    '''
    Create Attention Mask From 2-D Mask Tensor
    '''
    def __call__(self, inputs : torch.Tensor, mask : torch.Tensor = None, diag_mask : bool = True, need_mask : bool = False) -> torch.Tensor:
        '''
        Create 3D Tensor of Mask For PointerNetwork
        :param inputs: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        :param mask: int32 Tensor of shape [batch_size, to_seq_length]
        :param diag_mask: whether mask diag (To prevent self-loop)
        :return: float Tensor of shape [batch_size, seq_length, seq_length].
        '''
        if isinstance(inputs, list) and mask is None:
            mask = inputs[1]
            inputs = inputs[0]
        mask = mask.type_as(inputs)
        # BroadCast
        mask = mask.unsqueeze(1) * mask.unsqueeze(-1)
        # Diagonal Mask Operator
        if diag_mask:
            diag = torch.cat([torch.diag_embed(torch.diag(ins)).unsqueeze(0) for ins in mask], dim=0)
            mask = mask - diag
        return mask

AttnMask = SelfAttentionMask()

class AttetionScore(nn.Module):
    '''
    AttentionScore is used to calculate attention scores for Pointer Network
    '''
    def __init__(self, args : argparse.Namespace, device: torch.device = None):
        super(AttetionScore, self).__init__()
        self.device = device
        self.args = args
        try:
            self.scale_attn = args.scale_attn
        except:
            self.scale_attn = False

    def forward(self, query : torch.Tensor, key : torch.Tensor, mask : torch.Tensor = None, need_mask : bool = False):
        '''
        Calculate Attention Scores  For PointerNetwork
        :param query: Query tensor of shape `[batch_size, sequence_length, hidden_size]`.
        :param key: Key tensor of shape `[batch_size, sequence_length, hidden_size]`.
        :param mask: mask tensor of shape `[batch_size, sequence_length]`.
        :return: Tensor of shape `[batch_size, sequence_length, sequence_length]`.
        '''
        scores = torch.matmul(query, key.permute(0, 2, 1))
        if self.scale_attn:
            scores = scores * (1 / (query.shape[-1] ** (1/2)))
        if mask is not None:
            mask = AttnMask(scores, mask, diag_mask=False).to(self.device) if self.device else AttnMask(scores, mask, diag_mask=False).cuda()
            scores = logits_mask(scores, mask)
        if need_mask:
            return scores, mask
        else:
            return scores

## PLM

class PLM(nn.Module):
    def __init__(self, args : argparse.Namespace, device : torch.device, use_encoder : bool = False, pooler_output : bool = True, all_output : bool = False):
        super(PLM, self).__init__()
        self.modelid       = "bert_baseline"
        self.args          = args
        self.device        = device
        self.use_encoder   = use_encoder
        self.pooler_output = pooler_output
        self.all_output    = all_output
        self._bert         = BertModel.from_pretrained(args.lm_path, cache_dir='.cache/')
        # Finetune Or Freeze
        if args.finetune is not True:
            for param in self._bert.base_model.parameters():
                param.requires_grad = False
        # Modify BertModel - output_hidden_states
        self._bert_output = args.output_hidden_states
        self._bert.config.output_hidden_states = self._bert_output
        # Linear
        self._fc = Linear(args.lm_hidden_size, args.num_classes)

    def forward(self, inputs : torch.Tensor, attention_mask : torch.Tensor = None) -> tuple:
        encode_output = self._bert(inputs, attention_mask=attention_mask)
        # Pooler or Hidden
        if self.pooler_output:
            encoded = encode_output.pooler_output
        else:
            encoded = encode_output.last_hidden_state
        # Whether to apply dense
        if self.use_encoder is not True:
            output  = self._fc(encoded)
        else:
            if self.all_output:
                output = (encode_output.pooler_output, encode_output.last_hidden_state)
            else:
                output  = encoded
        return output

## Sub-model

class SwitchModel(nn.Module):
    def __init__(self, args : Namespace, device : torch.device):
        super(SwitchModel, self).__init__()
        self.modelid = "switch_baseline"
        self.args = args
        self.device = device
        # Encoder
        self._encoder = PLM(args, device, use_encoder=True, pooler_output=False)
        # Pointer Network
        self._pointer = PointerNetwork(args, device)
        # Dropout
        self._lm_dropout = nn.Dropout(args.dropout)

    def forward(self, input : torch.Tensor, attention_mask : torch.Tensor = None, need_mask : bool = False):
        # Encoder
        encoded = self._encoder(input, attention_mask = attention_mask)
        encoded = self._lm_dropout(encoded)
        # Pointer Network
        pointer_ret = self._pointer(encoded, attention_mask, need_mask)
        if need_mask:
            pointer_logits, masks = pointer_ret
            return pointer_logits, masks
        else:
            pointer_logits = pointer_ret
            return pointer_logits


class TaggerModel(nn.Module):
    def __init__(self, args : Namespace, device : torch.device):
        super(TaggerModel, self).__init__()
        self.modelid = "tagger_baseline"
        self.args = args
        self.device = device
        self.max_token = args.max_generate + 1
        # Encoder
        self._encoder = PLM(args, device, use_encoder=True, pooler_output=False)
        # Solution A
        # | - Dense
        self._hidden2tag = Linear(args.lm_hidden_size, args.tagger_classes)
        self._hidden2t = Linear(args.lm_hidden_size, self.max_token)
        # | - Dropout
        self._lm_dropout = nn.Dropout(args.dropout)

    def forward(self, input : torch.Tensor, attention_mask : torch.Tensor = None):
        # Encoder
        encoded = self._encoder(input, attention_mask=attention_mask)
        encoded = self._lm_dropout(encoded)
        # Tagger
        tagger_logits = self._hidden2tag(encoded)
        # Generate Token
        t_logits = self._hidden2t(encoded)
        return tagger_logits, t_logits


class GeneratorModel(nn.Module):
    def __init__(self, args : Namespace, device : torch.device):
        super(GeneratorModel, self).__init__()
        self.modelid = "generator_baseline"
        self.args = args
        self.device = device
        self._lmodel = BertForMaskedLM.from_pretrained(args.lm_path, cache_dir='.cache/')
        self.lmconfig = self._lmodel.config
        self.vocab_size = self.lmconfig.vocab_size
        # Finetune Or Freeze
        if args.finetune is not True:
            for param in self._bert.base_model.parameters():
                param.requires_grad = False
        # Mlm Linear Layers
        if self.args.factorized_embedding:
            self._mlm_fc_1 = Linear(args.lm_hidden_size, args.lm_emb_size)
            self._lnorm = LayerNorm(args, device, args.emb_size)
            self._mlm_fc_2 = Linear(args.lm_emb_size, self.vocab_size)
        else:
            self._mlm_fc_1 = Linear(args.lm_hidden_size, args.lm_hidden_size)
            self._lnorm = LayerNorm(args, device, args.lm_hidden_size)
            self._mlm_fc_2 = Linear(args.lm_hidden_size, self.vocab_size)
        # Activate Function
        self._act = gelu

    def forward(self, inputs : torch.Tensor, tgt_mlm : torch.Tensor, attention_mask : torch.Tensor = None) -> tuple:
        # Encoded
        encoded = self._lmodel(inputs, attention_mask=attention_mask).logits
        # Mlm Linear Layer # 1
        # output_mlm = self._act(self._mlm_fc_1(encoded))
        # output_mlm = self._lnorm(output_mlm)
        # if self.factorized_embedding:
        #     output_mlm = output_mlm.contiguous().view(-1, self.args.lm_emb_size)
        # else:
        #     output_mlm = output_mlm.contiguous().view(-1, self.args.lm_hidden_size)
        # Extract Logits & Label
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = encoded.contiguous().view(-1, self.vocab_size)
        output_mlm = output_mlm[tgt_mlm > 0, :]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        #output_mlm = self._mlm_fc_2(output_mlm)
        denominator = torch.tensor(output_mlm.size(0) + 1e-6).cuda()
        return output_mlm, tgt_mlm, denominator


## Model

class LMEncoder(nn.Module):
    def __init__(self, lm_path : str, finetune : bool = True, output_hidden_states : bool = True, dropout : float = 0.1):
        super(LMEncoder, self).__init__()
        self._lm = BertModel.from_pretrained(lm_path, cache_dir='.cache/')
        if finetune is not True:
            for param in self._lm.base_model.parameters():
                param.requires_grad = False
        self._lm_output = output_hidden_states
        self._lm.config.output_hidden_states = self._lm_output
        self._lm_dropout = nn.Dropout(dropout)

    def forward(self, inputs : torch.Tensor, attention_mask : torch.Tensor = None) -> tuple:
        encode_output = self._lm(inputs, attention_mask=attention_mask)
        pooler_output = encode_output.pooler_output
        last_hidden   = encode_output.last_hidden_state
        return pooler_output, last_hidden

class LMGenerator(nn.Module):
    def __init__(self, args : Namespace, lm_path : str, finetune : bool = True, device : torch.device = None):
        super(LMGenerator, self).__init__()
        self.args =args
        self._lmodel = BertForMaskedLM.from_pretrained(lm_path, cache_dir='.cache/')
        self.lmconfig = self._lmodel.config
        self.vocab_size = self.lmconfig.vocab_size
        if finetune is not True:
            for param in self._lm.base_model.parameters():
                param.requires_grad = False
        # Mlm Linear Layers
        if self.args.factorized_embedding:
            self._mlm_fc_1 = Linear(args.lm_hidden_size, args.lm_emb_size)
            self._lnorm = LayerNorm(args, device, args.emb_size)
            self._mlm_fc_2 = Linear(args.lm_emb_size, self.vocab_size)
        else:
            self._mlm_fc_1 = Linear(args.lm_hidden_size, args.lm_hidden_size)
            self._lnorm = LayerNorm(args, device, args.lm_hidden_size)
            self._mlm_fc_2 = Linear(args.lm_hidden_size, self.vocab_size)
        # Activate Function
        self._act = gelu

    def forward(self, inputs : torch.Tensor, tgt_mlm : torch.Tensor, attention_mask : torch.Tensor = None):
        hidden_states = self._lmodel(inputs, attention_mask=attention_mask).logits
        tgt_mlm       = tgt_mlm.contiguous().view(-1)
        output_mlm    = hidden_states.contiguous().view(-1, self.vocab_size)
        output_mlm    = output_mlm[tgt_mlm > 0, :]
        tgt_mlm       = tgt_mlm[tgt_mlm > 0]
        denominator   = torch.tensor(output_mlm.size(0) + 1e-6)
        return output_mlm, tgt_mlm, denominator

class ClassificationLayer(nn.Module):
    def __init__(self, lm_hidden_size : int, num_classes : int):
        super(ClassificationLayer, self).__init__()
        # Binary Module
        self._fc     = Linear(lm_hidden_size, num_classes)
        # Type Module
        self._fc_iwo = Linear(lm_hidden_size, num_classes)
        self._fc_ip  = Linear(lm_hidden_size, num_classes)
        self._fc_sc  = Linear(lm_hidden_size, num_classes)
        self._fc_ill = Linear(lm_hidden_size, num_classes)
        self._fc_cm  = Linear(lm_hidden_size, num_classes)
        self._fc_cr  = Linear(lm_hidden_size, num_classes)
        self._fc_um  = Linear(lm_hidden_size, num_classes)

    def forward(self, pooler_output : torch.Tensor) -> tuple:
        bi_logits = self._fc(pooler_output)
        logit_iwo, logit_ip, logit_sc = self._fc_iwo(pooler_output).unsqueeze(0), self._fc_ip(pooler_output).unsqueeze(0), self._fc_sc(pooler_output).unsqueeze(0)
        logit_ill, logit_cm, logit_cr, logit_um = self._fc_ill(pooler_output).unsqueeze(0), self._fc_cm(pooler_output).unsqueeze(0), self._fc_cr(pooler_output).unsqueeze(0), self._fc_um(pooler_output).unsqueeze(0)
        type_logits = torch.cat((logit_iwo, logit_ip, logit_sc, logit_ill, logit_cm, logit_cr, logit_um), dim=0)
        return bi_logits, type_logits

class JointModel(nn.Module):
    def __init__(self, args, config):
        super(JointModel, self).__init__()
        self.max_token  = config.max_generate + 1
        self.config = config
        self.switch = SwitchModel(config, args.device)
        self.tagger = TaggerModel(config, args.device)
        self.generator = GeneratorModel(config, args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm_path, cache_dir='./.cache')

    def forward(self, inputs : tuple, tgt_mlm : torch.Tensor, attention_mask : torch.Tensor = None):
        sw_inputs, tag_inputs, gen_inputs = inputs
        sw_attnmask, tag_attnmask, generate_attnmask = attention_mask
        switch_logits = self.switch(sw_inputs, sw_attnmask, need_mask=True)
        tagger_logits = self.tagger(tag_inputs, tag_attnmask)
        gen_logits = self.generator(gen_inputs, tgt_mlm, generate_attnmask)
        return switch_logits, tagger_logits, gen_logits


class JointInferenceModel(nn.Module):
    def __init__(self, args : Namespace, device : torch.device):
        super(JointInferenceModel, self).__init__()
        self.max_token = args.max_generate + 1
        # LM Encoder
        self.encoder = LMEncoder(args.lm_path, args.finetune, args.output_hidden_states)
        # Classification Layer (Binary, Types)
        self.cls_layer = ClassificationLayer(args.lm_hidden_size, args.num_classes)
        # Switch Layer
        self.pointer = PointerNetwork(args, device)
        # Tagger Layer + Generate Layer
        self.hidden2tag = Linear(args.lm_hidden_size, args.tagger_classes)
        self.hidden2ins = Linear(args.lm_hidden_size, self.max_token)
        self.hidden2mod = Linear(args.lm_hidden_size, self.max_token)
        self.generator  = LMGenerator(args, args.lm_path, args.finetune, device)

    def forward(self, inputs , stage : str, attention_mask : torch.Tensor = None, tgt_mlm : torch.Tensor = None, need_mask : bool = False):
        if stage == 'tag_before': # With binary & type
            orig_inputs, unify_attnmask = inputs, attention_mask
            pooler_ouptput, hidden_states = self.encoder(orig_inputs, attention_mask=unify_attnmask)
            bi_logits, cls_logits = self.cls_layer(pooler_ouptput)
            if need_mask:
                pointer_ret, masks = self.pointer(hidden_states, unify_attnmask, need_mask)
                return bi_logits, cls_logits, (pointer_ret, masks)
            else:
                pointer_ret = self.pointer(hidden_states, unify_attnmask, need_mask)
                return bi_logits, cls_logits, pointer_ret
        elif stage == 'tagger': # tagger stage
            tag_inputs, tag_attnmask = inputs, attention_mask
            _, tag_hidden_states = self.encoder(tag_inputs, attention_mask=tag_attnmask)
            # Tagger w (ins + mod logots)
            tagger_logits = self.hidden2tag(tag_hidden_states)
            ins_logits = self.hidden2ins(tag_hidden_states)
            mod_logits = self.hidden2mod(tag_hidden_states)
            tagger_logits = (tagger_logits, ins_logits, mod_logits)
            return tagger_logits
        elif stage == 'generator': # generator stage
            gen_inputs, generate_attnmask = inputs, attention_mask
            gen_logits = self.generator(gen_inputs, tgt_mlm, generate_attnmask)
            return gen_logits
        else: raise Exception('Model params `stage` error, please check.')
