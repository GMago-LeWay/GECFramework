import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

class CausalLM(torch.nn.Module):
    def __init__(self, args, config) -> None:
        super(CausalLM, self).__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(config.pretrained_model)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.language_model = LlamaForCausalLM.from_pretrained(config.pretrained_model)
        print("LLM Model Loaded.")

    def forward(self, texts, others=None):
        logits = self.language_model(**texts)
        return self.log_softmax(logits)
    
    def generate(self, inputs):
        outputs = self.language_model.generate(**inputs, max_new_tokens=120)
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts
