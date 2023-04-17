from trainers.base import Trainer
from trainers.SoftMaskedBert import SoftMaskedBertTrainer
from trainers.MaskLM import MaskLMTrain
from trainers.JointSTG import JointTrainer
from trainers.Seq2Seq import Seq2SeqModelTrainer
from trainers.Seq2Edit import Seq2EditTrainer
from trainers.CausalLM import CausalLMTrain
from trainers.Llama import LlamaTrainer
from trainers.GECToR import GECToRTrainer

def get_train(model) -> Trainer:
    TRAIN_MAP = {
        'bert': MaskLMTrain,
        'softmaskedbert': SoftMaskedBertTrainer,
        'stgjoint': JointTrainer,
        'seq2seq': Seq2SeqModelTrainer,
        'seq2edit': Seq2EditTrainer,
        'gector': GECToRTrainer,
        'llm': CausalLMTrain,
        'llama': LlamaTrainer,
        'llama_quant': LlamaTrainer,
    }

    assert model in TRAIN_MAP.keys(), 'Not support ' + model

    return TRAIN_MAP[model]