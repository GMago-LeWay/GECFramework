'''
Main entry for this project.
@GMago123
'''

## base import 
import random
import json
import logging
import argparse
import time
import os
import importlib
import traceback
from tqdm import tqdm
import zipfile
import codecs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gector',
                        help='bert/softmaskedbert/stgjoint/seq2seq/seq2edit/gector/llm/chinese_llama/llama/llama_quant/chatglm')    
    parser.add_argument('--task_mode', type=str, default='train',
                        help='train/tune/test/infer/augmentation')  
    parser.add_argument('--dataset', type=str, default='mucgec',
                        help='hybridset/nlpcc2018task2/fangzhengspell/fangzhenggrammar/guangming/peopledaily/augment/fangzhengaugment/fangzhengdapei/fcgec/mucgec')  
    parser.add_argument('--pre_save_dir', type=str, default='results',
                        help='root path to save results.')
    parser.add_argument('--devices', type=str, default='0,1,2,3',
                        help='GPU Environment.')    
    parser.add_argument('--device', type=int, default=2,
                        help='GPU id.')
    parser.add_argument('--load', type=str, default=None,
                        help='model save directory to be loaded in infer task.')
    parser.add_argument('--seed', type=int, default=111,
                        help='random seed.')
    parser.add_argument('--data_save_dir', type=str, default=None,
                        help='only use in independent augmentation task, identify the save directory.')

    return parser.parse_args()

## Set devices before pytorch imported
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from utils import *
from data import get_data
from train import get_train
from model import get_model
from config import Config
from trainers.base import Trainer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_log(args):
    file_name = os.path.join(args.save_dir, 'log.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 由于每调用一次set_logger函数，就会创建一个handler，会造成重复打印的问题，因此需要判断root logger中是否已有该handler
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(file_name)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class ExperimentsOfGEC:
    def __init__(self, args) -> None:
        self.args = args

    def run(self, config):
        dataset_ = get_data(self.args.dataset, self.args.model)(args=self.args, config=config)
        model_to_be_init = get_model(model=self.args.model)
        if self.args.model in ['seq2seq', 'seq2edit', 'chatglm']:
            model = model_to_be_init(config=config, args=self.args)
        else:
            model = model_to_be_init(config=config, args=self.args).to(self.args.device)
        train_to_be_init = get_train(model=self.args.model)
        train: Trainer = train_to_be_init(args=self.args, config=config, model=model)
        if self.args.model in ['seq2seq', 'seq2edit', 'chatglm']:
            train_loader, val_loader, test_loader = dataset_.train_val_test_data()
        else:
            train_loader, val_loader, test_loader = dataset_.get_train_val_dataloader(model.tokenizer)
        # do train
        if self.args.load:
            train.load(self.args.load)
        best_score = train.do_train(train_loader, val_loader)
        # save result
        logger.info(get_time() + '本次最优结果：%.4f' % best_score)

        train.load(self.args.save_dir)
        test_results = train.do_test(test_loader, mode="TEST")

        return test_results

    def run_test(self, config):
        # run infer task on labeled test set
        setup_seed(self.args.seed)
        # model settings
        dataset_ = get_data(self.args.dataset, self.args.model)(args=self.args, config=config)
        logger.info(get_time() + f"Test: Use model {config.name} at {self.args.load}, on dataset {self.args.dataset}")
        logger.info(f"Args: {self.args}; Config: {config}")
        model_to_be_init = get_model(model=self.args.model)
        if self.args.model in ['seq2seq', 'seq2edit', 'chatglm']:
            model = model_to_be_init(config=config, args=self.args)
        else:
            model = model_to_be_init(config=config, args=self.args).to(self.args.device)
        train_to_be_init = get_train(model=self.args.model)
        train: Trainer = train_to_be_init(args=self.args, config=config, model=model)
        train.load(self.args.load)

        try:
            tokenizer = model.tokenizer
        except:
            tokenizer = None
        if tokenizer:
            dataloader = dataset_.get_test_dataloader(tokenizer)
        elif 'pretrained_model' in config:
            tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, trust_remote_code=True)
            dataloader = dataset_.get_test_dataloader(tokenizer)
        else:
            raise NotImplementedError()
        json_results = train.do_infer(dataloader, mode='TEST')
        save_path = os.path.join(self.args.save_dir, f'{self.args.model}-{self.args.dataset}-{self.args.task_mode}.json')
        save_txt = os.path.join(self.args.save_dir, f'{self.args.model}-{self.args.dataset}-{self.args.task_mode}.txt')
        with codecs.open(save_path, "w", "utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=4)
        with codecs.open(save_txt, "w", "utf-8") as f:
            for item in json_results:
                f.write("%s\t%s\t%s\n" % (item["src"], item["tgt"], item["predict"]))
        logger.info(get_time() + f"Results have been stored in {save_path}.")
        return json_results

    def run_infer(self, config):
        # only inferring
        setup_seed(self.args.seed)
        # model settings
        dataset_ = get_data(self.args.dataset, self.args.model)(args=self.args, config=config)
        logger.info(get_time() + f"Infer: Use model {config.name} at {self.args.load}, on dataset {self.args.dataset}")
        logger.info(f"Args: {self.args}; Config: {config}")
        model_to_be_init = get_model(model=self.args.model)
        if self.args.model in ['seq2seq', 'seq2edit', 'chatglm']:
            model = model_to_be_init(config=config, args=self.args)
        else:
            model = model_to_be_init(config=config, args=self.args).to(self.args.device)
        train_to_be_init = get_train(model=self.args.model)
        train: Trainer = train_to_be_init(args=self.args, config=config, model=model)
        train.load(self.args.load)

        try:
            tokenizer = model.tokenizer
        except:
            tokenizer = None
        if tokenizer:
            dataloader = dataset_.get_test_dataloader(tokenizer)
        elif 'pretrained_model' in config:
            tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
            dataloader = dataset_.get_test_dataloader(tokenizer)
        else:
            raise NotImplementedError()
        json_results = train.do_infer(dataloader, mode="INFER")
        save_path = os.path.join(self.args.save_dir, f'{self.args.model}-{self.args.dataset}-{self.args.task_mode}.json')
        with codecs.open(save_path, "w", "utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=4)
        logger.info(get_time() + f"Results have been stored in {save_path}.")
        
        ## MuCGEC output
        if self.args.dataset == 'mucgec':
            save_txt = os.path.join(self.args.save_dir, f'MuCGEC_test.txt')
            with codecs.open(save_txt, "w", "utf-8") as f:
                for item in json_results:
                    f.write("%s\t%s\t%s\n" % (item["id"], item["src"], item["predict"]))
            with zipfile.ZipFile(os.path.join(self.args.save_dir, 'submit.zip'), mode='w') as zipf:
                zipf.write(save_txt, 'MuCGEC_test.txt')
        
        ## FCGEC output
        if self.args.dataset == 'fcgec':
            fcgec_json = {}
            for item in json_results:
                error_flag = 1 if item["src"] != item["predict"] else 0
                fcgec_json[item['id']] = {"error_flag": error_flag, "error_type": "IWO", "correction": item["predict"]}
            fcgec_path = os.path.join(self.args.save_dir, 'predict.json')
            with codecs.open(fcgec_path, "w", "utf-8") as f:
                json.dump(fcgec_json, f, ensure_ascii=False, indent=4)      
            with zipfile.ZipFile(os.path.join(self.args.save_dir, 'predict.zip'), mode='w') as zipf:
                zipf.write(fcgec_path, 'predict.json')
        
        return json_results


    def run_task(self, seeds, config, fixed_csv=False):
        logger.info('************************************************************')
        logger.info(get_time() + 'Parameters: ' + str(config))
        logger.info(get_time() + 'Arguments: ' + str(self.args))

        original_result = {}

        if args.seed:
            seeds = [args.seed]
        for seed in seeds:
            args.seed = seed
            setup_seed(seed)
            logger.info(get_time() + 'Seed: %d Train Starts...' % seed)      
            current_res = self.run(config)
            if not original_result:
                for key in current_res:
                    original_result[key] = [current_res[key]]
            else:
                for key in current_res:
                    original_result[key].append(current_res[key])

        # 保存实验结果
        result = {}
        for key in original_result:            
            mean, std = round(np.mean(original_result[key]), 3), round(np.std(original_result[key]), 3)
            result[key] = str(mean)
            result[key + '-std'] = str(std)
        for key in config:
            result[key] = config[key]
        result['Args'] = self.args
        result['Config'] = config
        result['Time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

        logger.info('##Task Results## ' + str(result))
        # logger.info('##Task Results##, Mean: %s, Std: %s' % (result[config.KeyEval], result[config.KeyEval + '-std']))
        
        save_path = os.path.join(self.args.save_dir, f'{self.args.model}-{self.args.dataset}-{self.args.task_mode}.csv')

        # save or append the current result
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            columns = set(df.columns)
            if set(result.keys()) == columns:       # format are identical (csv column's keys equals to result's keys)
                df = df.append(result, ignore_index=True)
                logger.info(get_time() + "Results are appended to %s." % save_path)
            else:
                tmp_result_file = os.path.join(self.args.save_dir, "temp_result.txt")
                with codecs.open(tmp_result_file, 'a', "utf-8") as f:
                    f.write(json.dumps(result) + '\n')
                logger.info(get_time() + 'Warning: Results are saved to temp_result.txt, because the result format can not match with %s.' % save_path)
        else:       # 
            for key in result:
                result[key] = [result[key]]
            df = pd.DataFrame(result)
            logger.info(get_time() + "Results are saved to %s." % save_path)
        
        # update csv
        df.to_csv(save_path, index=None)
        logger.info('************************************************************')


    def run_tune(self, preconfig, seeds, tune_times=50):
        has_debuged = []
        tune_number = 0
        while tune_number < tune_times:
            logger.info('-----------------------------------Tune(%d/%d)----------------------------' % (tune_number+1, tune_times))
            # load previous result
            save_path = os.path.join(self.args.save_dir, f'{self.args.model}-{self.args.dataset}-{self.args.task_mode}.csv')
            if os.path.exists(save_path):
                df = pd.read_csv(save_path)
                for j in range(len(df)):
                    has_debuged.append(df.loc[j, "Config"])

            # only refresh seed for config
            setup_seed(int(time.time()))              # 随机选取种子以初始化随机的config
            config = Config(self.args.model, self.args.dataset, tune=True, preconfig=preconfig).get_config()

            if str(config) in has_debuged:
                logger.info(get_time() + '该参数已经被搜索过.')
                time.sleep(1.)
                continue

            try:
                self.run_task(seeds=seeds, config=config, fixed_csv=True)
                has_debuged.append(str(config))
                tune_number += 1
            except Exception as e:
                msg = traceback.format_exc()
                logger.info(get_time() + 'Error: %s' % str(msg))


    def conduct(self):
        preset_config = {}

        if self.args.task_mode == 'test':
            config = Config(model=self.args.model, dataset=self.args.dataset, preconfig=preset_config).get_config()
            self.run_test(config=config)
        elif self.args.task_mode == 'infer':
            config = Config(model=self.args.model, dataset=self.args.dataset, preconfig=preset_config).get_config()
            predictions = self.run_infer(config=config)                
        elif self.args.task_mode == 'train':
            config = Config(model=self.args.model, dataset=self.args.dataset, preconfig=preset_config).get_config()
            self.run_task(seeds=[111], config=config)
        elif self.args.task_mode == 'tune':
            self.run_tune(preconfig=preset_config, seeds=[111], tune_times=100)
        else:
            raise NotImplementedError()
        # except Exception as e:
        #     msg = traceback.format_exc()
        #     logger.info(get_time() + 'Error: %s' % str(msg))


## LLM model only used for inference. Simplified experiments.
class ExperimentsOfLLM:
    def __init__(self, args) -> None:
        self.args = args

    def run_test(self, config):
        # run infer task on labeled test set
        setup_seed(self.args.seed)
        # model settings
        dataset_ = get_data(self.args.dataset, self.args.model)(args=self.args, config=config)
        logger.info(get_time() + f"Test: Use model {config.name} at {self.args.load}, on dataset {self.args.dataset}")
        logger.info(f"Args: {self.args}; Config: {config}")
        model_to_be_init = get_model(model=self.args.model)
        model = model_to_be_init(config=config, args=self.args)
        train_to_be_init = get_train(model=self.args.model)
        train: Trainer = train_to_be_init(args=self.args, config=config, model=model)
        train.load(self.args.load)

        dataloader = dataset_.get_test_dataloader()
        json_results = train.do_infer(dataloader, mode='TEST')
        save_path = os.path.join(self.args.save_dir, f'{self.args.model}-{self.args.dataset}-{self.args.task_mode}.json')
        save_txt = os.path.join(self.args.save_dir, f'{self.args.model}-{self.args.dataset}-{self.args.task_mode}.txt')
        with codecs.open(save_path, "w", "utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=4)
        with codecs.open(save_txt, "w", "utf-8") as f:
            for item in json_results:
                f.write("%s\t%s\t%s\n" % (item["src"], item["tgt"], item["predict"]))
        logger.info(get_time() + f"Results have been stored in {save_path}.")
        return json_results

    def run_infer(self, config):
        # run infer task on unlabeled test set
        setup_seed(self.args.seed)
        # model settings
        dataset_ = get_data(self.args.dataset, self.args.model)(args=self.args, config=config)
        logger.info(get_time() + f"Test: Use model {config.name} at {self.args.load}, on dataset {self.args.dataset}")
        logger.info(f"Args: {self.args}; Config: {config}")
        model_to_be_init = get_model(model=self.args.model)
        model = model_to_be_init(config=config, args=self.args)
        train_to_be_init = get_train(model=self.args.model)
        train: Trainer = train_to_be_init(args=self.args, config=config, model=model)
        train.load(self.args.load)

        dataloader = dataset_.get_test_dataloader()
        json_results = train.do_infer(dataloader, mode="INFER")
        save_path = os.path.join(self.args.save_dir, f'{self.args.model}-{self.args.dataset}-{self.args.task_mode}.json')
        with codecs.open(save_path, "w", "utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=4)
        logger.info(get_time() + f"Results have been stored in {save_path}.")

        ## MuCGEC output
        if self.args.dataset == 'mucgec':
            save_txt = os.path.join(self.args.save_dir, f'MuCGEC_test.txt')
            with codecs.open(save_txt, "w", "utf-8") as f:
                for item in json_results:
                    f.write("%s\t%s\t%s\n" % (item["id"], item["src"], item["predict"]))
            with zipfile.ZipFile(os.path.join(self.args.save_dir, 'submit.zip'), mode='w') as zipf:
                zipf.write(save_txt, 'MuCGEC_test.txt')
        
        ## FCGEC output
        if self.args.dataset == 'fcgec':
            fcgec_json = {}
            for item in json_results:
                error_flag = 1 if item["src"] != item["predict"] else 0
                fcgec_json[item['id']] = {"error_flag": error_flag, "error_type": "IWO", "correction": item["predict"]}
            fcgec_path = os.path.join(self.args.save_dir, 'predict.json')
            with codecs.open(fcgec_path, "w", "utf-8") as f:
                json.dump(fcgec_json, f, ensure_ascii=False, indent=4)      
            with zipfile.ZipFile(os.path.join(self.args.save_dir, 'predict.zip'), mode='w') as zipf:
                zipf.write(fcgec_path, 'predict.json')
        
        return json_results
    
    def conduct(self):
        preset_config = {}

        if self.args.task_mode == 'test':
            config = Config(model=self.args.model, dataset=self.args.dataset, preconfig=preset_config).get_config()
            self.run_test(config=config)
        elif self.args.task_mode == 'infer':
            config = Config(model=self.args.model, dataset=self.args.dataset, preconfig=preset_config).get_config()
            predictions = self.run_infer(config=config)                
        else:
            raise NotImplementedError()


class ExperimentsOfGECAugmentation:
    def __init__(self, args, augmentor) -> None:
        self.args = args
        self.config = Config(model=self.args.model, dataset=self.args.dataset).get_config()
        self.augmentor = augmentor

    def data_filter(self, data):
        new_data = []
        for item in tqdm(data):
            if len(item) <= 15 or len(item) >= 384:
                continue
            chars = list(item)
            if " " in chars or "\u3000" in chars:
                continue
            chinese_count = 0
            for char in chars:
                if is_chinese(char):
                    chinese_count += 1
            if chinese_count / len(chars) < 0.85:
                continue
            new_data.append(item)
        return new_data
    
    def augment(self):
        setup_seed(20)
        augmentation = importlib.import_module('augmentation')
        dataset_ = get_data(self.args.dataset, self.args.model)(args=self.args, config=self.config)
        print(get_time() + "Load Raw Data...")
        data = dataset_.data()[:1000000]
        print(get_time() + "Data Loaded. Filtering...")
        data = self.data_filter([item['label'] for item in data])

        augmenter = augmentation.get_augmentation('clg')(self.args, self.config)
        json_results = augmenter.static_augment(data)
        save_path = self.args.data_save_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, 'aug_data.json')
        with codecs.open(save_file, "w", "utf-8") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=4)        


EXPERIMENTS = {
    'bert': ExperimentsOfGEC,
    'softmaskedbert': ExperimentsOfGEC,
    'stgjoint': ExperimentsOfGEC,
    'seq2seq': ExperimentsOfGEC,
    'seq2edit': ExperimentsOfGEC,
    'gector': ExperimentsOfGEC,
    'chinese_llama': ExperimentsOfLLM,
    'llm': ExperimentsOfLLM,
    'llama': ExperimentsOfLLM,
    'llama_quant': ExperimentsOfLLM,
    'chatglm': ExperimentsOfGEC,
}

if __name__ == '__main__':
    args.device = 'cuda:'+ str(args.device) if args.device >= 0 else 'cpu'
    # set save directory
    time_str = time.strftime('%Y%m%d-%H%M',time.localtime())
    args.save_dir = os.path.join(args.pre_save_dir, f'{args.model}-{args.dataset}-{args.task_mode}-{time_str}')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    setup_log(args)
    if args.task_mode in ['train', 'tune', 'test', 'infer']:
        experiment = EXPERIMENTS[args.model](args)
        experiment.conduct()
    elif args.task_mode in ['augmentation']:
        augment = ExperimentsOfGECAugmentation(args)
        augment.augment()
    else:
        raise NotImplementedError()
