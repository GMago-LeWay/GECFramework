import time
import os
import logging

from dataset_provider.MuCGEC import get_token_embedders
from utils.gector.gec_model import Seq2Labels

def get_model(args, config, model_name, vocab, tune_bert=False, predictor_dropout=0,
              label_smoothing=0.0,
              confidence=0,
              model_dir="",
              log=None):
    token_embs = get_token_embedders(model_name, tune_bert=tune_bert)
    model = Seq2Labels(vocab=vocab,
                       text_field_embedder=token_embs,
                       predictor_dropout=predictor_dropout,
                       label_smoothing=label_smoothing,
                       confidence=confidence,
                       model_dir=model_dir,
                       cuda_device=eval(args.device[5:]),
                       dev_file=config.dev_set,
                       logger=log,
                       vocab_path=config.vocab_path,
                       weight_name=config.pretrained_model,
                       save_metric=config.save_metric
                       )
    return model


def Seq2EditModel(args, config):
    return None
