import importlib

MODULE_MAP = {
    'bert': ('models.BERT', 'BERT'),
    'softmaskedbert': ('models.SoftMaskedBert', 'GRUSoftMaskedBERT'),
    'stgjoint': ('models.JointSTG', 'JointModel'),
    'seq2seq': ('models.Seq2Seq', 'Seq2SeqModel'),
    'seq2edit': ('models.Seq2Edit', 'Seq2EditModel'),
    'gector': ('models.GECToR', 'ModelingCtcBert'),
    'llm': ('models.CausalLM', 'CausalLM'),
    'chinese_llama': ('models.CausalLM', 'CausalLMLLAMA'),
    'llama': ('models.Llama', 'LlamaNormal7B'),
    'llama_quant': ('models.Llama', 'LlamaQuant'),
    'chatglm': ('models.ChatGLM', 'ChatGLM'),
    'correctionglm': ('models.CorrectionGLM', 'GLMForGrammaticalCorrection'),
}


def get_model(model):
    assert model in MODULE_MAP.keys(), 'Not support ' + model
    module_name, class_name = MODULE_MAP[model]
    model_class = getattr(importlib.import_module(module_name), class_name)
    return model_class
