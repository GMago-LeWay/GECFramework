import importlib

MODULE_MAP = {
    'bert': ('trainers.MaskLM', 'MaskLMTrain'),
    'softmaskedbert': ('trainers.SoftMaskedBert', 'SoftMaskedBertTrainer'),
    'stgjoint': ('trainers.JointSTG', 'JointTrainer'),
    'seq2seq': ('trainers.Seq2Seq', 'Seq2SeqModelTrainer'),
    'seq2edit': ('trainers.Seq2Edit', 'Seq2EditTrainer'),
    'gector': ('trainers.GECToR', 'GECToRTrainer'),
    'llm': ('trainers.CausalLM', 'CausalLMTrain'),
    'llama': ('trainers.Llama', 'LlamaTrainer'),
    'llama_quant': ('trainers.Llama', 'LlamaTrainer'),
}


def get_train(model):
    assert model in MODULE_MAP.keys(), 'Not support ' + model
    module_name, class_name = MODULE_MAP[model]
    train_class = getattr(importlib.import_module(module_name), class_name)
    return train_class
