from tqdm import tqdm
import os
import logging

import torch

from utils.tools import dict_to_str, get_time
from trainers.base import Trainer

logger = logging.getLogger(__name__)

class CausalLMTrain(Trainer):
    def __init__(self, args, config, model):
        super(CausalLMTrain, self).__init__(args, config, model)
        self.args = args
        self.config = config
        self.model = model

        logger.info("You have loaded CausalLM Trainer for LLM, but it can only do infer task.")

    def do_train(self, train_dataloader, val_dataloader):
        raise NotImplementedError()


    def do_test(self, dataloader, mode="VAL"):
        """
        do test process, based on ids of every token.
        """
        raise NotImplementedError()


    def do_infer(self, dataloader, mode="INFER"):
        """
        do infer or test process, based on decoded text of sentences' tokens.
        """
        self.model.eval()
        results = []
        for batch_data in tqdm(dataloader):        
            texts = batch_data['raw_texts']
            labels = batch_data['raw_labels']

            batch_size = len(texts)
            for i in range(batch_size):
                single_text_batch = self.model.tokenizer(
                    "请对以下语句进行纠正：" + texts[i] + "纠正结果：",
                    return_tensors="pt", 
                    add_special_tokens=False
                )
                prediction = self.model.generate(single_text_batch)
                if mode=="TEST":
                    results.append({"src": texts[i], "predict": prediction[0], "tgt": labels[i]})
                elif mode=="INFER":
                    results.append({"src": texts[i], "predict": prediction[0]})
                else:
                    raise NotImplementedError()


        return results

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass
