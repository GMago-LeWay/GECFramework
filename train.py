import importlib

MODULE_MAP = {
    'bert': ('trainers.MaskLM', 'MaskLMTrain'),
    'softmaskedbert': ('trainers.SoftMaskedBert', 'SoftMaskedBertTrainer'),
    'stgjoint': ('trainers.JointSTG', 'JointTrainer'),
    'seq2seq': ('trainers.Seq2Seq', 'Seq2SeqModelTrainer'),
    'seq2edit': ('trainers.Seq2Edit', 'Seq2EditTrainer'),
    'gector': ('trainers.GECToR', 'GECToRTrainer'),
    'chinese_llama': ('trainers.CausalLM', 'CausalLMTrain'),
    'llm': ('trainers.CausalLM', 'CausalLMTrain'),
    'llama': ('trainers.Llama', 'LlamaTrainer'),
    'llama_quant': ('trainers.Llama', 'LlamaTrainer'),
    'chatglm': ('trainers.ChatGLM', 'ChatGLMTrainer'),
    'correctionglm': ('trainers.CorrectionGLM', 'CorrectionGLMTrainer'),
    'seq2seqbeta': ('trainers.Seq2SeqBeta', 'Seq2SeqBetaTrainer'),
    'seq2span': ('trainers.Seq2Span', 'Seq2SpanTrainer'),
    'openai': ('trainers.OpenAI', 'OpenAIUser'),
}


def get_train(model):
    assert model in MODULE_MAP.keys(), 'Not support ' + model
    module_name, class_name = MODULE_MAP[model]
    train_class = getattr(importlib.import_module(module_name), class_name)
    return train_class
