from trainers.base import TrainerBeta
from typing import Dict, List
from datasets import Dataset
import logging
import os
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class OpenAIUser(TrainerBeta):
    def __init__(self, args, settings, model, dataset: Dict[str, Dataset]) -> None:
        super().__init__(args, settings, model, dataset)
    

    def train_dataset_transform(self):
        """
        Do not support in OpenAI.
        """
        raise NotImplementedError()

    def do_train(self):
        """
        Do not support in OpenAI.
        """
        raise NotImplementedError()

    def do_eval(self):
        """
        Do not support in OpenAI.
        """
        raise NotImplementedError()

    def infer_on_dataset(self, split):

        # save settings
        save_dir = os.path.join(self.args.save_dir, split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # prompt select
        if self.args.dataset in ['fangzhenggrammar', 'fangzhengspell', 'mucgec', 'fcgec']:
            prompt = self.settings.cn_prompt
            logger.info("Using Chinese Prompt to do GEC.")
        else:
            prompt = self.settings.en_prompt
            logger.info("Using English Prompt to do GEC.")

        ## save real-time result one by one
        f = open(os.path.join(save_dir, 'real-time-results.json'), 'w')
        results = []

        # generate
        assert self.args.task_mode == 'infer', "For openai, Only support infer mode."
        for item in tqdm(self.dataset[split]):
            input_text = prompt.replace('[TEXT]', item['text'])
            output = self.model.conversation(message=input_text, history=[]).strip()

            if 'label' in item:
                res = {'id': item['id'], 'src': item['text'], 'tgt': item['label'], 'predict': output}
            else:
                res = {'id': item['id'], 'src': item['text'], 'predict': output}
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

        f.close()

        return results


    def do_infer(self):
        """
        do infer on inputs.
        """
        logger.info("Inferring on test dataset...")
        test_infer_res = self.infer_on_dataset('test')
        return test_infer_res


    def save(self, save_dir):
        raise NotImplementedError()

    def load(self, save_dir):
        raise NotImplementedError()
    
    def get_best_checkpoint_dir(self):
        raise NotImplementedError()