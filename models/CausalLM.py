import torch
import logging
from peft import  PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

logger = logging.getLogger(__name__)

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

class CausalLM(torch.nn.Module):
    def __init__(self, args, config) -> None:
        super(CausalLM, self).__init__()
        self.args = args
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, trust_remote_code=True)
        try:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("PAD Token WILL be set to EOS token in this model.")
        except Exception as e:
            logger.info("Pad Token would NOT be set to EOS token in this model.")
            
        self.language_model = AutoModelForCausalLM.from_pretrained(config.pretrained_model, trust_remote_code=True, torch_dtype=config.torch_dtype, load_in_8bit=config.load_in_8bit)
        if config.lora_model is not None:
            logger.info("loading peft model")
            self.language_model = PeftModel.from_pretrained(self.language_model, config.lora_model, torch_dtype=config.torch_dtype)
        self.language_model.to(args.device)

        ## Generation Config
        stop_words = ['\n']
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[-1].item() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=3)])
        self.generation_config = config.generation_config
        self.generation_config['stopping_criteria'] = self.stopping_criteria


    def forward(self, **kwargs):
        return self.language_model(**kwargs)
    
    def generate(self, input_texts):
        self.language_model.eval()
        outputs = []
        for text in input_texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)  #add_special_tokens=False ?
            generation_output = self.language_model.generate(
                input_ids = inputs["input_ids"].to(self.args.device), 
                attention_mask = inputs['attention_mask'].to(self.args.device),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                **self.generation_config
            )
            s = generation_output[0]
            output = self.tokenizer.decode(s, skip_special_tokens=True)
            outputs.append(output)
        return outputs

class CausalLMLLAMA(torch.nn.Module):
    def __init__(self, args, config) -> None:
        super(CausalLM, self).__init__()
        self.args = args
        self.config = config
        load_type = torch.float16
        self.tokenizer = LlamaTokenizer.from_pretrained(config.pretrained_model)
        self.language_model = LlamaForCausalLM.from_pretrained(
            config.pretrained_model, 
            load_in_8bit=False,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
        )
        model_vocab_size = self.language_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(self.tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size!=tokenzier_vocab_size:
            assert tokenzier_vocab_size > model_vocab_size
            print("Resize model embeddings to fit tokenizer")
            self.language_model.resize_token_embeddings(tokenzier_vocab_size)
        if config.lora_model is not None:
            print("loading peft model")
            self.language_model = PeftModel.from_pretrained(self.language_model, config.lora_model,torch_dtype=load_type)

        self.language_model.to(args.device)
        self.language_model.eval()

        ## Generation Config
        stop_words = ['\n']
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[-1].item() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=3)])
        self.generation_config = dict(
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.3,
            max_new_tokens=512,
            stopping_criteria=self.stopping_criteria,
        )


    def forward(self, texts, others=None):
        raise NotImplementedError()
    
    def generate(self, input_texts):
        outputs = []
        for text in input_texts:
            inputs = self.tokenizer(text, return_tensors="pt")  #add_special_tokens=False ?
            generation_output = self.language_model.generate(
                input_ids = inputs["input_ids"].to(self.args.device), 
                attention_mask = inputs['attention_mask'].to(self.args.device),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **self.generation_config
            )
            s = generation_output[0]
            output = self.tokenizer.decode(s, skip_special_tokens=True)
            outputs.append(output)
        return outputs