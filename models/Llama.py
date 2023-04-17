import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import json
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, LLaMA

from llama.hf import LLaMATokenizer
from llama.llama_quant import load_quant


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    return generator


class LlamaNormal7B:
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config
        self.model = load(
            config.ckpt_dir, 
            tokenizer_path=config.tokenizer_path, 
            local_rank=0,
            world_size=1,
            max_seq_len=config.max_seq_len,
            max_batch_size=config.max_batch_size,
        )

    def generate(self, texts, stop_words=["\n"]):
        model_generations = self.model.generate(
            texts, 
            max_gen_len=self.config.max_gen_len, 
            temperature=self.config.temperature, 
            top_p=self.config.top_p,
            stop_words=stop_words
        )
        return model_generations


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=3):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        assert input_ids.shape[0] == 1
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()

        if stop_count >= self.ENCOUNTERS:
            return True
        return False
    

class LlamaQuant:
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config
        self.model = load_quant(config.pretrained_model, config.load, config.wbits, config.max_seq_len)
        self.model.to(args.device)
        self.tokenizer = LLaMATokenizer.from_pretrained(config.pretrained_model)

        self.stopping_criteria = None

    def generate(self, texts, stop_words=["\n"]):
        if self.stopping_criteria == None:
            stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[-1].item() for stop_word in stop_words]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=3)])
        
        model_generations = []
        for text in texts:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.args.device)
            generated_ids = self.model.generate(
                input_ids,
                do_sample=self.config.do_sample,
                min_length=0,
                max_length=self.config.max_seq_len,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                stopping_criteria=self.stopping_criteria,
            )
            model_generations.append(self.tokenizer.decode([el.item() for el in generated_ids[0]]))

        return model_generations
