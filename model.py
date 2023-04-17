from models.BERT import BERT
from models.SoftMaskedBert import GRUSoftMaskedBERT
from models.JointSTG import JointModel
from models.Seq2Seq import Seq2SeqModel
from models.Seq2Edit import Seq2EditModel
from models.CausalLM import CausalLM
from models.Llama import LlamaNormal7B, LlamaQuant
from models.GECToR import ModelingCtcBert


def get_model(model):
    MODEL_MAP = {
        'bert': BERT,
        'softmaskedbert': GRUSoftMaskedBERT,
        'stgjoint': JointModel,
        'seq2seq': Seq2SeqModel,
        'seq2edit': Seq2EditModel,
        'gector': ModelingCtcBert,
        'llm': CausalLM,
        'llama': LlamaNormal7B,
        'llama_quant': LlamaQuant,
    }

    assert model in MODEL_MAP.keys(), 'Not support ' + model

    return MODEL_MAP[model]