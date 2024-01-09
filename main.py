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

class TaskMode:
    augmentation = 'augmentation'
    train = 'train'
    eval = 'eval'
    infer = 'infer'
    eval_train = 'eval_train'
    infer_train = 'infer_train'
    train_and_infer = 'train_infer'
    train_and_eval_and_infer = 'train_eval_infer'
    custom = 'custom'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='correctionglm',
                        help='bert/softmaskedbert/stgjoint/seq2seq/seq2edit/gector/llm/chinese_llama/llama/llama_quant/chatglm/correctionglm/seq2seqbeta/seq2span')    
    parser.add_argument('--task_mode', type=str, default='train',
                        help=f'{TaskMode.train}/{TaskMode.eval}/{TaskMode.infer}/{TaskMode.eval_train}/{TaskMode.infer_train}/{TaskMode.train_and_infer}/{TaskMode.train_and_eval_and_infer}/{TaskMode.augmentation}')  
    parser.add_argument('--dataset', type=str, default='mucgec',
                        help='hybridset/nlpcc2018task2/fangzhengspell/fangzhenggrammar/guangming/peopledaily/augment/fangzhengaugment/fangzhengdapei/fcgec/mucgec/pretrain/c4/lang8/clang8/fce/nucle/wilocness/hybrid')  
    parser.add_argument('--save_root_dir', type=str, default='results_glm',
                        help='root path to save results.')
    parser.add_argument('--devices', type=str, default='0',
                        help='GPU Environment.')    
    parser.add_argument('--device', type=int, default=0,
                        help='GPU id.')
    parser.add_argument('--load', type=str, default=None,
                        help='model save directory to be loaded in infer task.')
    parser.add_argument('--resume', type=str, default=None,
                        help='model checkpoint to continue training.')
    # Removed after 240109
    # parser.add_argument('--lora', action="store_true", default=False,
    #                     help='LoRA method, for now only support CorrectionGLM.')
    parser.add_argument('--config', type=str, default='',
                        help='optional for CorrectionGLM model. load config file.')
    parser.add_argument('--seed', type=int, default=111,
                        help='random seed.')
    parser.add_argument('--data_save_dir', type=str, default=None,
                        help='only use in independent augmentation task, identify the save directory.')

    return parser.parse_args()

## Set devices before pytorch imported
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
os.environ["HF_DATASETS_CACHE"] = "/data/liwei/cache/"

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer

# transformers.utils.move_cache('/data/liwei/cache/huggingface/')

from utils import *
from data import get_data, GeneralDataset
from train import get_train
from postprocess import PostProcess
from model import get_model
from config import Storage, Config
from trainers.base import Trainer, TrainerBeta


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


def basic_saving(args, json_results):
    save_dir = args.save_dir
    save_path = os.path.join(save_dir, f'{args.model}-{args.dataset}-{args.task_mode}.json')
    with codecs.open(save_path, "w", "utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)

    save_txt = os.path.join(save_dir, f'{args.model}-{args.dataset}-{args.task_mode}.txt')
    with codecs.open(save_txt, "w", "utf-8") as f:
        for item in json_results:
            if "tgt" in item:
                f.write("%s\t%s\t%s\n" % (item["src"], item["tgt"], item["predict"]))
            else:
                f.write("%s\t%s\n" % (item["src"], item["predict"]))
    logger.info(get_time() + f"Results have been stored in {save_path}.")


def prediction_saving(args, json_results):
    """
    In infer task, some dataset requires a specific version of results to evaluate, this function will do the formatting.
    """
    save_dir = args.save_dir
    ## MuCGEC output
    if args.dataset == 'mucgec':
        save_txt = os.path.join(save_dir, f'MuCGEC_test.txt')
        with codecs.open(save_txt, "w", "utf-8") as f:
            for item in json_results:
                f.write("%s\t%s\t%s\n" % (item["id"], item["src"], item["predict"]))
        with zipfile.ZipFile(os.path.join(save_dir, 'submit.zip'), mode='w') as zipf:
            zipf.write(save_txt, 'MuCGEC_test.txt')
    
    ## FCGEC output
    if args.dataset == 'fcgec':
        fcgec_json = {}
        for item in json_results:
            error_flag = 1 if item["src"] != item["predict"] else 0
            fcgec_json[item['id']] = {"error_flag": error_flag, "error_type": "IWO", "correction": item["predict"]}
        fcgec_path = os.path.join(save_dir, 'predict.json')
        with codecs.open(fcgec_path, "w", "utf-8") as f:
            json.dump(fcgec_json, f, ensure_ascii=False, indent=4)      
        with zipfile.ZipFile(os.path.join(save_dir, 'predict.zip'), mode='w') as zipf:
            zipf.write(fcgec_path, 'predict.json')


class ExperimentsOfGEC:
    def __init__(self, args) -> None:
        self.args = args

    def run(self, config):
        dataset_ = get_data(self.args.dataset, self.args.model)(args=self.args, config=config)
        model_to_be_init = get_model(model=self.args.model)
        model = model_to_be_init(config=config, args=self.args)
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
        model = model_to_be_init(config=config, args=self.args)
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
        model = model_to_be_init(config=config, args=self.args)
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
        
        prediction_saving(args=self.args, json_results=json_results)
        
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


class ExperimentsOfGECBeta:
    def __init__(self, args) -> None:
        self.args = args

    def run(self, config):
        setup_seed(self.args.seed)

        # data settings
        raw_dataset_loader: GeneralDataset = get_data(self.args.dataset, self.args.model)(args=self.args, config=config)
        raw_dataset = raw_dataset_loader.get_dataset_map(split=None)
        logger.info(get_time() + f"Train: Use model {config.name} at {self.args.load}, on dataset {self.args.dataset}")
        logger.info(f"Args: {self.args}; Config: {config}")
        # model settings
        model_to_be_init = get_model(model=self.args.model)
        model = model_to_be_init(args=self.args, settings=config)
        train_to_be_init = get_train(model=self.args.model)
        trainer: TrainerBeta = train_to_be_init(args=self.args, settings=config, model=model, dataset=raw_dataset)
        # do train
        if self.args.load:
            trainer.load(self.args.load)
        best_score = trainer.do_train()
        # save result
        logger.info(get_time() + 'Result：%s' % best_score)
        # # do test
        best_checkpoint = trainer.get_best_checkpoint_dir()
        return best_checkpoint

    def run_infer(self, config):
        # only inferring
        setup_seed(self.args.seed)

        # data settings
        raw_dataset_loader: GeneralDataset = get_data(self.args.dataset, self.args.model)(args=self.args, config=config)
        if self.args.task_mode == 'infer':
            load_key = 'test'
        elif self.args.task_mode == 'eval':
            load_key = 'valid'
        else:
            load_key = None
        raw_dataset = raw_dataset_loader.get_dataset_map(split=load_key)
        logger.info(get_time() + f"Infer: Use model {config.name} at {self.args.load}, on dataset {self.args.dataset}")
        logger.info(f"Args: {self.args}; Config: {config}")
        # model settings
        model_to_be_init = get_model(model=self.args.model)
        model = model_to_be_init(self.args, config)
        train_to_be_init = get_train(model=self.args.model)
        trainer: TrainerBeta = train_to_be_init(args=self.args, settings=config, model=model, dataset=raw_dataset)
        trainer.load(self.args.load)
        logger.info(f"Load Checkpoint from {self.args.load} (Ignore when using lora)")
        if self.args.task_mode in ['infer', 'infer_train']:
            json_results = trainer.do_infer()
            process = PostProcess(self.args, config, json_results, 'test')
            if self.args.task_mode == 'infer':
                process.post_process_and_save()
            else:
                process.basic_saving()
        elif self.args.task_mode in ['eval', 'eval_train']:
            json_results = trainer.do_eval()
            process = PostProcess(self.args, config, json_results, 'valid')
            process.basic_saving()
        else:
            raise NotImplementedError()
        
        return json_results
    
    def run_combine(self, config):
        assert self.args.task_mode in [TaskMode.train_and_infer, TaskMode.train_and_eval_and_infer]
        original_task_mode = str(self.args.task_mode)
        self.args.task_mode = TaskMode.train
        logger.info("All mode. Begin with train mode.")
        best_checkpoint = self.run(config=Storage(config))
        logger.info(f"Training COMPLETE. Best model: {best_checkpoint}")

        if original_task_mode == TaskMode.train_and_eval_and_infer:
            logger.info("Start to evaluate on train set and validation set.")
            self.args.load = best_checkpoint
            self.args.task_mode = TaskMode.infer_train
            json_results = self.run_infer(Storage(config))
            logger.info(f"Evaluation COMPLETE.")

        logger.info("End with infer mode.")
        self.args.load = best_checkpoint
        self.args.task_mode = TaskMode.infer
        json_results = self.run_infer(Storage(config))
        return json_results
    
    def run_custom(self, config):
        assert self.args.model == 'correctionglm' and self.args.dataset in ['wilocness', 'mucgec_dev']
        # threshold experiment
        self.args.task_mode = TaskMode.infer
        original_save_dir = str(self.args.save_dir)

        # keep threshold
        for th in np.arange(0.3, 0.46, 0.02):
            th = round(th, 2)
            config.keep_threshold = th
            logger.info(f"KEEP threshold {th} inference:")
            self.args.save_dir = os.path.join(original_save_dir, f'keep_threshold_{th}')
            if not os.path.exists(self.args.save_dir):
                os.makedirs(self.args.save_dir)
            self.run_infer(config)
        
        # # edit threshold
        # config.keep_threshold = None
        # for th1 in np.arange(0.3, 1.1, 0.1):
        #     for th2 in np.arange(0.3, 1.1, 0.1):
        #         th1, th2 = round(th1, 2), round(th2, 2)
        #         config.error_threshold = th1
        #         config.insert_threshold = th2
        #         logger.info(f"ERROR threshold {th1} INSERT threshold {th2} inference:")
        #         self.args.save_dir = os.path.join(original_save_dir, f'error_{th1}_insert_{th2}')
        #         if not os.path.exists(self.args.save_dir):
        #             os.makedirs(self.args.save_dir)
        #         self.run_infer(config)

        # keep-edit threshold
        for th in np.arange(0.34, 0.42, 0.02):
            th = round(th, 2)
            config.keep_threshold = th
            result_f, result_p, result_r = {}, {}, {}
            th1_list = [round(th1, 2) for th1 in list(np.arange(0.35, 0.8, 0.05))]
            th2_list = [round(th2, 2) for th2 in list(np.arange(0.35, 0.8, 0.05))]
            result_f["INSERT"], result_p["INSERT"], result_r["INSERT"] = th2_list, th2_list, th2_list
            for th1 in th1_list:
                result_f[th1], result_p[th1], result_r[th1] = [], [], []
                for th2 in th2_list:
                    config.error_threshold = th1
                    config.insert_threshold = th2
                    logger.info(f"KEEP threshold {th} ERROR threshold {th1} INSERT threshold {th2} inference:")
                    self.args.save_dir = os.path.join(original_save_dir, f'keep_{th}_error_{th1}_insert_{th2}')
                    if not os.path.exists(self.args.save_dir):
                        os.makedirs(self.args.save_dir)
                    self.run_infer(config)
                    if self.args.dataset == 'wilocness':
                        # read conll14 result
                        evaluation_result_file = os.path.join(self.args.save_dir, 'test', 'conll14_metrics.txt')
                        # print metrics of conll14
                        metrics_lines = open(evaluation_result_file).readlines()
                        precision_name, _, precision = metrics_lines[0].strip().split()
                        recall_name, _, recall = metrics_lines[1].strip().split()
                        f_05_name, _, f_05 = metrics_lines[2].strip().split()
                    elif self.args.dataset == 'mucgec_dev':
                        evaluation_result_file = os.path.join(self.args.save_dir, 'test', 'mucgec_dev_metrics.json')
                        metrics_item = json.load(open(evaluation_result_file))
                        precision = metrics_item['precision']
                        recall = metrics_item['recall']
                        f_05 = metrics_item['f_0.5']
                    else:
                        raise NotImplementedError()
                    result_f[th1].append(f_05)
                    result_p[th1].append(precision)
                    result_r[th1].append(recall)
            pd.DataFrame(result_f).to_csv(os.path.join(original_save_dir, f"keep_{th}_f_05.csv"), index=False)
            pd.DataFrame(result_r).to_csv(os.path.join(original_save_dir, f"keep_{th}_r.csv"), index=False)
            pd.DataFrame(result_p).to_csv(os.path.join(original_save_dir, f"keep_{th}_p.csv"), index=False)

    def conduct(self):
        preset_config = self.args

        if self.args.task_mode in ['infer', 'infer_train', 'eval', 'eval_train']:
            config = Config(model=self.args.model, dataset=self.args.dataset, preconfig=preset_config).get_config()
            self.run_infer(config=config)                
        elif self.args.task_mode == 'train':
            config = Config(model=self.args.model, dataset=self.args.dataset, preconfig=preset_config).get_config()
            self.run(config=config)
        elif self.args.task_mode in [TaskMode.train_and_infer, TaskMode.train_and_eval_and_infer]:
            config = Config(model=self.args.model, dataset=self.args.dataset, preconfig=preset_config).get_config()
            self.run_combine(config=config)      
        elif self.args.task_mode == TaskMode.custom:
            config = Config(model=self.args.model, dataset=self.args.dataset, preconfig=preset_config).get_config()
            self.run_custom(config=config)
        else:
            raise NotImplementedError()


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
        prediction_saving(args=self.args, json_results=json_results)

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
    'correctionglm': ExperimentsOfGECBeta,
    'seq2seqbeta': ExperimentsOfGECBeta,
    'seq2span': ExperimentsOfGECBeta,
    'openai': ExperimentsOfGECBeta,
}

if __name__ == '__main__':
    args.model = args.model.lower()
    args.dataset = args.dataset.lower()
    args.device = 'cuda:'+ str(args.device) if args.device >= 0 else 'cpu'
    # set save directory
    time_str = time.strftime('%Y%m%d-%H%M',time.localtime())
    args.save_dir = os.path.join(args.save_root_dir, f'{args.model}-{args.dataset}-{args.task_mode}-{time_str}')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    setup_log(args)
    if args.task_mode in ['train', 'tune', 'test', 'infer', 'eval', 'eval_train', 'infer_train', TaskMode.train_and_infer, TaskMode.train_and_eval_and_infer, TaskMode.custom]:
        experiment = EXPERIMENTS[args.model](args)
        experiment.conduct()
    elif args.task_mode in ['augmentation']:
        augment = ExperimentsOfGECAugmentation(args)
        augment.augment()
    else:
        raise NotImplementedError()
