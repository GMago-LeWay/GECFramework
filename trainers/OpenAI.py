from trainers.base import TrainerBeta
from typing import Dict, List
from datasets import Dataset
import logging
import threading
import time
import os
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


class OpenAIUser(TrainerBeta):
    def __init__(self, args, settings, model, dataset: Dict[str, Dataset]) -> None:
        super().__init__(args, settings, model, dataset)
        self.semaphore = threading.Semaphore(16)
        self.lock = threading.Lock()

    def train_dataset_transform(self):
        """
        Do not support in OpenAI.
        """
        raise NotImplementedError()

    def test_dataset_transform(self):
        """
        For loading detection resutls of CorrectionGLM
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
        # test api
        logger.info("Testing the api...")
        logger.info(self.model.conversation(message='你是谁？', history=[]))

        # record id -- index of order map
        order = {}
        for i, item in enumerate(self.dataset[split]):
            order[item['id']] = i

        # save settings
        save_dir = os.path.join(self.args.save_dir, split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    

        # load detection results (if exists)
        if self.settings.detection_results:
            logger.info(f"loading first stage detection results from {self.settings.detection_results}")
            detection_results = json.load(open(self.settings.detection_results[split]))
            # test_dataset is self.dataset['split'], infer train mode: all split will be inferred, infer mode: split=test.
            # check id compatible
            assert len(detection_results) == len(self.dataset[split]), f"Uncompatible detection results from {self.settings.detection_results[split]}"
            for i in range(len(detection_results)):
                assert detection_results[i]["id"] == self.dataset[split][i]["id"], f"Uncompatible detection results from {self.settings.detection_results[split]}"

            # load detections
            # self.test_dataset_transform(split)
            # assert self.settings.detection_load_way == "masked_text":
            for i in range(len(detection_results)):
                self.dataset[split][i]["masked_text"] = detection_results[i]["masked_text"]

            # prompt select
            if self.args.dataset in ['fangzhenggrammar', 'fangzhengspell', 'mucgec', 'fcgec']:
                prompt = self.settings.cn_assisted_prompt
                logger.info("Using Assisted Chinese Prompt to do GEC.")
            else:
                prompt = self.settings.en_assisted_prompt
                logger.info("Using Assisted English Prompt to do GEC.")
        else:
            # prompt select
            if self.args.dataset in ['fangzhenggrammar', 'fangzhengspell', 'mucgec', 'fcgec']:
                prompt = self.settings.cn_prompt
                logger.info("Using Chinese Prompt to do GEC.")
            else:
                prompt = self.settings.en_prompt
                logger.info("Using English Prompt to do GEC.")
        
        logger.info(f"Prompt: {prompt}")

        ## save real-time result one by one
        f = open(os.path.join(save_dir, 'real-time-results.json'), 'w')
        results = []

        def single_generate(item):
            input_text = prompt.replace('[TEXT]', item['text'])
            if 'masked_text' in item:
                input_text = input_text.replace('[MASKED_TEXT]', item['masked_text'])
            print(input_text)
            output = self.model.conversation(message=input_text, history=[]).strip()
            self.semaphore.release()

            if 'label' in item:
                res = {'id': item['id'], 'src': item['text'], 'tgt': item['label'], 'predict': output}
            else:
                res = {'id': item['id'], 'src': item['text'], 'predict': output}
            
            # save results
            self.lock.acquire()
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            f.flush()
            self.lock.release()

        # parallel generate
        assert self.args.task_mode == 'infer', "For openai, Only support infer mode."
        ts = []
        logger.info("Inferring by GPT-4")
        for item in tqdm(self.dataset[split]):
            t = threading.Thread(target=single_generate, args=[item])
            ts.append(t)
        for t in tqdm(ts):
            t.start()
            time.sleep(0.01)
            self.semaphore.acquire()
        for t in tqdm(ts):
            t.join()

        f.close()

        # reorder results
        ordered_results = [None] * len(self.dataset[split])
        for item in results:
            ordered_results[order[item['id']]] = item
        return ordered_results


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
        logger.info("OpenAI API, nothing to load.")
    
    def get_best_checkpoint_dir(self):
        raise NotImplementedError()