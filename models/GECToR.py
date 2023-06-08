#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import torch
from transformers.models.bert import BertModel, BertPreTrainedModel, BertConfig
import os

from torch.nn import CrossEntropyLoss, Module
from dataset_provider.GECToR import CtcTokenizer

class LabelSmoothingLoss(torch.nn.Module):
    """formula
    loss= {
        (1-smoothing) * logP(x), if (x==y)
        (smoothing) / (num_classes-1) * logP(x), if (x!=y)
    }
    Args:
        torch (_type_): _description_
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean', ignore_index: int = -100):
        assert reduction in ('mean', 'sum', 'none')
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self._reduction = reduction
        self._ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        num_classes = pred.size()[-1]
        pred = pred.log_softmax(dim=-1)

        pred = pred[target != self._ignore_index]
        target = target[target != self._ignore_index]

        new_target = torch.zeros_like(pred)
        new_target.fill_(value=self.smoothing / (num_classes - 1))
        new_target.scatter_(dim=1, index=target.data.unsqueeze(1), value=self.confidence)
        loss = -new_target * pred
        if self._reduction == 'mean':
            return torch.mean(torch.sum(loss, -1))
        elif self._reduction == 'sum':
            return torch.sum(loss, -1)
        elif self._reduction == 'none':
            return loss


class ModelingCtcBert(Module):
    def __init__(self, args, config):
        super(ModelingCtcBert, self).__init__()
        self.args = args
        self.config = config
        # get vocab information
        with open(os.path.join(config.ctc_vocab_dir, config.correct_tags_file), "r") as fp:
            vocab_szie = len(fp.read().strip().split("\n"))
        config.correct_vocab_size = vocab_szie

        self.tokenizer = CtcTokenizer.from_pretrained(config.pretrained_model)
        bert_config = BertConfig.from_pretrained(config.pretrained_model)
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)
        self.tag_detect_projection_layer = torch.nn.Linear(
            bert_config.hidden_size, config.detect_vocab_size)
        self.tag_label_projection_layer = torch.nn.Linear(
            bert_config.hidden_size, config.correct_vocab_size)
        self._detect_criterion = CrossEntropyLoss(ignore_index=-100)
        self._correct_criterion = LabelSmoothingLoss(smoothing=0.1, ignore_index=-100)

        self.reward_estimate_projection = torch.nn.Linear(
            bert_config.hidden_size, 1)

    @staticmethod
    def build_dummpy_inputs():
        inputs = {}
        inputs['input_ids'] = torch.LongTensor(
            torch.randint(low=1, high=10, size=(8, 56)))
        inputs['attention_mask'] = torch.ones(size=(8, 56)).long()
        inputs['token_type_ids'] = torch.zeros(size=(8, 56)).long()
        inputs['detect_labels'] = torch.zeros(size=(8, 56)).long()
        inputs['correct_labels'] = torch.zeros(size=(8, 56)).long()
        return inputs

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            detect_labels=None,
            correct_labels=None
    ):

        hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)[0]
        detect_outputs = self.tag_detect_projection_layer(hidden_states)
        correct_outputs = self.tag_label_projection_layer(hidden_states)

        reward_outputs = self.reward_estimate_projection(hidden_states[:, 0, :])

        result = {
            "detect_outputs": detect_outputs,
            "correct_outputs": correct_outputs,
            "reward_outputs": reward_outputs,
            "detect_loss": None,
            "correct_loss": None,
            "loss": None,
        }

        loss = None
        if detect_labels is not None and correct_labels is not None:
            detect_loss = self._detect_criterion(
                detect_outputs.view(-1, self.config.detect_vocab_size), detect_labels.view(-1))
            correct_loss = self._correct_criterion(
                correct_outputs.view(-1, self.config.correct_vocab_size), correct_labels.view(-1))
            loss = detect_loss + correct_loss
            result["detect_loss"] = detect_loss
            result["correct_loss"] = correct_loss

        elif detect_labels is not None:
            loss = self._detect_criterion(
                detect_outputs.view(-1, self.config.detect_vocab_size), detect_labels.view(-1))
        elif correct_labels is not None:
            loss = self._correct_criterion(
                correct_outputs.view(-1, self.config.correct_vocab_size), correct_labels.view(-1))

        result["loss"] = loss
        return result
