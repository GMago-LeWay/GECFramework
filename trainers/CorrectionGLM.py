from typing import Optional, Union
from tqdm import tqdm
import os
import logging
import json
import copy
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
from torch.utils.data.dataloader import DataLoader
import datasets
import transformers
from dataclasses import dataclass, field
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    RobertaTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    BatchEncoding,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from utils.checkpoint_select import get_all_best_checkpoint
from utils.model_utils import count_trainable_parameters

from trainers.base import TrainerBeta
from dataset_provider.CorrectionGLM import GLMDataProcessor
from metrics import DetectionCorrectionMetrics

logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForGLMGEC:

    tokenizer: AutoTokenizer
    loss_ignore_id: int = -100

    def __call__(self, features):
        # print(len(features))
        max_len = max(map(len, [item['input_ids'] for item in features]))
        batch_size = len(features)

        # generate attention masks
        batch_input_ids = []
        batch_target_ids = []
        batch_position_ids = []
        batch_detection_label_ids = []
        batch_attention_masks = []

        # pad input_ids, target ids, detection_labels and position_ids
        # pad mode: max
        for sample in features:
            current_len = len(sample['input_ids'])
            assert current_len == len(sample['target_ids']) == len(sample['detection_labels']) == len(sample['position_ids'][0]) == len(sample['position_ids'][1])
            pad_input_ids = list(sample['input_ids']) + [self.tokenizer.pad_token_id]*(max_len-current_len)
            batch_input_ids.append(pad_input_ids)
            pad_target_ids = list(sample['target_ids']) + [self.loss_ignore_id]*(max_len-current_len)
            batch_target_ids.append(pad_target_ids)
            pad_detection_labels = list(sample['detection_labels']) + [self.loss_ignore_id]*(max_len-current_len)
            batch_detection_label_ids.append(pad_detection_labels)
            pad_position_ids = list(sample['position_ids'][0]) + [sample['position_ids'][0][-1]]*(max_len-current_len)
            pad_block_ids = list(sample['position_ids'][1]) + list(range(sample['position_ids'][1][-1]+1, sample['position_ids'][1][-1]+max_len-current_len+1))
            batch_position_ids.append([pad_position_ids, pad_block_ids])
            attention_mask = np.tril(np.ones((current_len, current_len), dtype=int))
            attention_mask[:sample['prefix_length'], :sample['prefix_length']] = 1
            attention_mask[sample['prefix_length']:sample['source_length'], :sample['source_length']] = 1
            pad_attention_mask = np.pad(attention_mask, ((0, max_len-current_len), (0, max_len-current_len)), 'constant', constant_values=0)
            batch_attention_masks.append(pad_attention_mask)

        # batch encoding
        batch = BatchEncoding({
            'input_ids': torch.LongTensor(batch_input_ids),
            'target_ids': torch.LongTensor(batch_target_ids),
            'position_ids': torch.LongTensor(batch_position_ids),
            'detection_labels': torch.LongTensor(batch_detection_label_ids),
            'attention_mask': torch.LongTensor(np.array(batch_attention_masks)).unsqueeze(1),
            'source_length': torch.LongTensor([item['source_length'] for item in features]),
            'prefix_length': torch.LongTensor([item['prefix_length'] for item in features]),
            'prefix_prompt_length': torch.LongTensor([item['prefix_prompt_length'] for item in features])
        })
        return batch

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    result = {}
    glm_pred, detection_pred = None, None
    if 'glm' in logits:
        lm_logits = logits['glm']
        glm_pred = torch.argmax(lm_logits, dim=-1)
        result['glm'] = glm_pred
    if 'detection' in logits:
        detection_logits = logits['detection']
        detection_pred = torch.argmax(detection_logits, dim=-1)
        result['detection'] = detection_pred
    
    return result, labels


class CorrectionGLMTrainer(TrainerBeta):
    def __init__(self, args, settings, model, dataset):
        super(CorrectionGLMTrainer, self).__init__(args, settings, model, dataset)

        # dataset processor
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.pretrained_model, trust_remote_code=True)
        self.data_processor = GLMDataProcessor(tokenizer=self.tokenizer, args=args, config=settings)
        self.data_collator = DataCollatorForGLMGEC(tokenizer=self.tokenizer, loss_ignore_id=self.data_processor._loss_ignore_id)
        # self.generation_data_collator = DataCollatorForGLMGECGeneration(tokenizer=self.tokenizer, loss_ignore_id=self.data_processor._loss_ignore_id)

        self.callbacks = []
        if self.settings.early_stop:
            self.callbacks.append(EarlyStoppingCallback(self.settings.early_stop))
        
        # check limitation of config
        assert self.settings.detection_load_way == 'detections', "Sorry, for now only detecition predictions can be loaded, other modes have BUG. (tokenizer Unconsistency)"

        # initialize config for transformers trainer
        self.training_args = TrainingArguments(
            seed=args.seed,
            do_train=True,
            do_eval=True,
            do_predict=False,
            deepspeed=settings.ds_config,
            output_dir=args.save_dir,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            # include_inputs_for_metrics=True,
            label_names=['target_ids', 'detection_labels'],
            per_device_train_batch_size=settings.train_batch_size,
            per_device_eval_batch_size=settings.eval_batch_size,
            num_train_epochs=settings.epoch,
            evaluation_strategy='steps',
            eval_steps=settings.eval_step,
            max_steps=settings.max_steps if self.settings.streaming else -1,
            # eval_accumulation_steps=20,
            save_steps=settings.save_step,
            logging_steps=settings.logging_steps,
            # log_level='info',
            learning_rate=settings.lr,
            bf16=settings.bf16,
            group_by_length=False,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            # lr_scheduler_type=settings.lr_scheduler,
            metric_for_best_model=settings.eval_key,
            load_best_model_at_end=(self.settings.early_stop != None),
            resume_from_checkpoint=args.resume,
        )
        logger.info(self.training_args)

        
        logger.info("Model Trainable parameters: " + str(count_trainable_parameters(self.model)))


    def _hash_data_settings(self):
        pretrained_model = self.settings.pretrained_model
        model_type = str(self.settings.model_type)
        num_labels = str(self.settings.num_labels)
        prompt = str(self.settings.prompt)
        detection_results = str(self.settings.detection_results['train']) + '_' + str(self.settings.detection_results['valid']) + '_' + str(self.settings.detection_results['test']) 
        data_dir = self.settings.data_dir
        settings_list = [pretrained_model, model_type, num_labels, prompt, detection_results, data_dir]
        settings_str = '_'.join(settings_list)
        hash_value = hashlib.md5(settings_str.encode('utf-8')).hexdigest()
        hash_str = str(hash_value)
        logger.info(f'Common Hash Code Key of Readed Dataset: {hash_str}')
        return hash_str
    
    def _get_data_cache_name(self, correctionglm_data_category: str):
        name = self.args.model.lower() + '_' + self.args.dataset.lower()
        file_name =  '_'.join([name, correctionglm_data_category, self._hash_data_settings()]) + ".arrow"
        return os.path.join(self.settings.cache_dir, file_name)
    
    def _get_train_preprocess_function(self, for_validation=False, detection_model=False):
        def _preprocess(examples):
            processed = {}
            for i in range(len(examples['text'])):
                src, tgt = examples['text'][i], examples['label'][i]
                result = self.data_processor.convert_gec_sentence_pair_to_example(src, tgt, 
                            self.settings.max_eval_source_length if for_validation else self.settings.max_train_source_length)
                if not processed:
                    for key in result:
                        processed[key] = []
                for key in result:
                    processed[key].append(result[key])
            return processed
        
        def _preprocess_detection_model(examples):
            processed = {}
            for i in range(len(examples['text'])):
                src, tgt = examples['text'][i], examples['label'][i]
                result = self.data_processor.convert_gec_sentence_pair_to_detection_example(src, tgt, 
                            self.settings.max_eval_source_length if for_validation else self.settings.max_train_source_length)
                if not processed:
                    for key in result:
                        processed[key] = []
                for key in result:
                    processed[key].append(result[key])
            return processed
        
        return _preprocess_detection_model if detection_model else _preprocess
    
    def _get_train_preprocess_function_using_predictions(self, for_validation=False):
        # using predictions in form of detection labels, when tokenizer are same
        def _preprocess(examples):
            processed = {}
            for i in range(len(examples['text'])):
                src, tgt = examples['text'][i], examples['label'][i]
                ## id check
                assert examples["id"][i] == examples["check_id"][i]
                result = self.data_processor.convert_gec_sentence_pair_to_example_using_detections(src, tgt, examples["detections"][i],
                            self.settings.max_eval_source_length if for_validation else self.settings.max_train_source_length)
                if not processed:
                    for key in result:
                        processed[key] = []
                for key in result:
                    processed[key].append(result[key])
            return processed
        # using predictions in form of masked text, when tokenizer are not same
        def _preprocess_with_masked_text(examples):
            processed = {}
            for i in range(len(examples['text'])):
                src, tgt = examples['text'][i], examples['label'][i]
                ## id check
                assert examples["id"][i] == examples["check_id"][i]
                result = self.data_processor.convert_gec_sentence_pair_to_example_using_masked_text(src, tgt, examples["masked_text"][i],
                            self.settings.max_eval_source_length if for_validation else self.settings.max_train_source_length)
                if not processed:
                    for key in result:
                        processed[key] = []
                for key in result:
                    processed[key].append(result[key])
            return processed
        # using predictions in form of masked text, when tokenizer are not same
        def _preprocess_with_masked_words(examples):
            processed = {}
            for i in range(len(examples['text'])):
                src, tgt = examples['text'][i], examples['label'][i]
                ## id check
                assert examples["id"][i] == examples["check_id"][i]
                result = self.data_processor.convert_gec_sentence_pair_to_example_using_masked_text(src, tgt, examples["masked_words"][i],
                            self.settings.max_eval_source_length if for_validation else self.settings.max_train_source_length)
                if not processed:
                    for key in result:
                        processed[key] = []
                for key in result:
                    processed[key].append(result[key])
            return processed
        if self.settings.detection_load_way == "detections":
            return _preprocess
        elif self.settings.detection_load_way == "masked_text":
            return _preprocess_with_masked_text
        elif self.settings.detection_load_way == "masked_words":
            return _preprocess_with_masked_words
    
    def _get_detection_preprocess_function(self, max_sentence_length):
        def _preprocess(examples):
            processed = {}
            for i in range(len(examples['text'])):
                src = examples['text'][i]
                result = self.data_processor.convert_sentence_to_detection_example(src, max_sentence_length, enable_warning=(self.args.task_mode == 'infer'))
                if not processed:
                    for key in result:
                        processed[key] = []
                for key in result:
                    processed[key].append(result[key])
            return processed
        return _preprocess        
    
    def _get_detached_generation_preprocess_function(self, max_sentence_length):
        def _preprocess(examples):
            processed = {}
            for i in range(len(examples['text'])):
                src = examples['text'][i]
                masked_src = examples['masked_text'][i]
                result = self.data_processor.convert_masked_sentence_to_infer_example(src, masked_src, max_sentence_length)
                if not processed:
                    for key in result:
                        processed[key] = []
                for key in result:
                    processed[key].append(result[key])
            return processed
        return _preprocess 
    
    def _get_predict_preprocess_function(self):
        def _preprocess(examples):
            processed = {}
            for i in range(len(examples['text'])):
                src = examples['text'][i]
                edit_label_ids = examples['detection_result'][i]
                result = self.data_processor.convert_detected_sentence_to_infer_example(src=src, edit_label_ids=edit_label_ids)
                if not processed:
                    for key in result:
                        processed[key] = []
                for key in result:
                    processed[key].append(result[key])
            return processed
        return _preprocess
            
    def train_dataset_transform(self):
        """
        Transform (transformers) dataset to meet the requirements of model training.  
        The self.train_dataset and the self.valid_dataset will be transformed from self.dataset['train'] and ['valid'].  
        In load detection results mode, the corresponding results will be loaded.
        """
        if self.args.task_mode[-5:] == "train":
            logger.info("Preparing train dataset transform...")
            if self.settings.detection_results['train']:
                logger.info("Preparing train dataset transform with detection results.")
                assert self.settings.streaming == False, 'Error: Streaming training set is not supported when loading detection results'
                ## Using detections
                assert self.settings.model_type != 'detection', "ModelSettingError: Loading detection results is meaningless when training detection model."
                detection_results = json.load(open(self.settings.detection_results['train']))
                assert len(self.dataset['train']) == len(detection_results), f"Using uncompatible detection results for current training set. {len(self.dataset['train'])}, {len(detection_results)}"
                logger.info(f"Loaded previous detection results from {self.settings.detection_results['train']}")
                ## check if source tokens is matched 
                # check_tokenizer = AutoTokenizer.from_pretrained('../models/glm-roberta-large', trust_remote_code=True)
                # if check_tokenizer:
                #     logger.info(f"Check if source text is matched between dataset and loaded detection results.")
                #     for item, detection_res in tqdm(zip(self.dataset['train'], detection_results), desc="Checking:"):
                #         detection_src_tokens = detection_res["source_tokens"][1:-1]
                #         tokenized_src_tokens = self.tokenizer.encode(item["text"])[1:-1][:len(detection_src_tokens)]
                #         detection_src_text = check_tokenizer.decode(detection_src_tokens)
                #         tokenized_src_text = self.tokenizer.decode(tokenized_src_tokens)
                #         assert detection_src_text == tokenized_src_text, f"Detection input_text: {detection_src_text}, Source text: {tokenized_src_text}" 
                        

                self.dataset['train'] = self.dataset['train'].add_column('check_id', [item['id'] for item in detection_results])
                if self.settings.detection_load_way == "masked_text":
                    self.dataset['train'] = self.dataset['train'].add_column('masked_text', [item['masked_text'] for item in detection_results])
                elif self.settings.detection_load_way == "masked_words":
                    self.dataset['train'] = self.dataset['train'].add_column('masked_words', [item['masked_words'] for item in detection_results])
                else:
                    self.dataset['train'] = self.dataset['train'].add_column('detections', [item['detections'] for item in detection_results])
                logger.info(f"Added detections into train dataset.")
                columns = self.dataset['train'].column_names
                reserved_columns = ['input_ids', 'target_ids', 'position_ids', 'detection_labels', 'source_length', 'prefix_length']
                removed_columns = []
                for column in columns:
                    if column not in reserved_columns:
                        removed_columns.append(column)

                # temp_saving_data = []
                # for i in range(100000):
                #     temp_saving_data.append({"id": self.dataset['train'][i]["id"], "text": self.dataset['train'][i]["text"], "label": self.dataset['train'][i]["label"], "detections": detection_results[i]['detections']})
                self.train_dataset = self.dataset['train'].map(
                    self._get_train_preprocess_function_using_predictions(for_validation=False),
                    batched=True,
                    remove_columns=removed_columns,
                    load_from_cache_file=self.settings.load_cache,
                    cache_file_name=self._get_data_cache_name(f"train_using_pred_{self.settings.detection_load_way}_{self.settings.max_train_source_length}"),
                    num_proc=self.settings.num_proc_trainset,
                    desc="Running preprocessing on train dataset",
                )
                # for i in range(100000):
                #     item = self.train_dataset[i]
                #     temp_saving_data[i]['input_ids'] = item['input_ids']
                #     temp_saving_data[i]['input_text'] = self.tokenizer.decode(temp_saving_data[i]['input_ids'])
                #     temp_saving_data[i]['target_text'] = self.tokenizer.decode(item['target_ids'][item['source_length']:])
                #     temp_saving_data[i]['detection_labels'] = item['detection_labels']
                # json.dump(temp_saving_data, open('results_main/examples/examples_SFT2.json', 'w'), ensure_ascii=False, indent=4)
                
            else:  # Standard train set
                logger.info("Preparing standard train dataset transform.")
                columns = self.dataset['train'].column_names
                reserved_columns = ['input_ids', 'target_ids', 'position_ids', 'detection_labels', 'source_length', 'prefix_length']
                removed_columns = []
                for column in columns:
                    if column not in reserved_columns:
                        removed_columns.append(column)
                
                # temp_saving_data = []
                # for i in range(100000):
                #     temp_saving_data.append({"id": self.dataset['train'][i]["id"], "text": self.dataset['train'][i]["text"], "label": self.dataset['train'][i]["label"]})

                if self.settings.streaming:
                    if self.settings.load_cache:
                        # Standard train set type 1: load cache as streaming dataset
                        logger.info("Mode: Load cache as streaming dataset")
                        cache_file_prefix = os.path.basename(self._get_data_cache_name("train")).replace('.arrow', '')
                        cache_files = os.listdir(self.settings.cache_dir)
                        correspond_cache_files = []
                        for file in cache_files:
                            if cache_file_prefix in file:
                                correspond_cache_files.append(os.path.join(self.settings.cache_dir, file))
                        assert correspond_cache_files, 'No cache file available. If you want to load cache as streaming dataset, please use non-streaming mode to construct the dataset and cache it.'
                        # data_files = {"train": ["path/to/0.arrow", "path/to/1.arrow", ..., "path/to/n.arrow"]}
                        self.train_dataset = datasets.load_dataset("arrow", data_files=correspond_cache_files, streaming=True)['train']
                    else:
                        # Standard train set type 2: Direct use streaming dataset
                        logger.info("Mode: Directly use streaming dataset")
                        assert type(self.dataset['train']) == datasets.iterable_dataset.IterableDataset, 'Error: You are trying to use streaming dataset, but current original train dataset is not an IterableDataset.'
                        logger.info("Warning: Iterable dataset cannot be cached. It will be processed dynamically in training. Note in this mode max_steps must be set.")
                        self.train_dataset = self.dataset['train'].map(
                            self._get_train_preprocess_function(for_validation=False, detection_model=(self.settings.model_type == 'detection')),
                            batched=True,
                            remove_columns=removed_columns,
                        )
                else:
                    # Standard train set type 3: Normal train dataset
                    logger.info("Mode: Full dataset transform")
                    self.train_dataset = self.dataset['train'].map(
                        self._get_train_preprocess_function(for_validation=False, detection_model=(self.settings.model_type == 'detection')),
                        batched=True,
                        remove_columns=removed_columns,
                        load_from_cache_file=self.settings.load_cache,
                        cache_file_name=self._get_data_cache_name(f"train_{self.settings.max_train_source_length}"),
                        num_proc=self.settings.num_proc_trainset,
                        desc="Running preprocessing on train dataset",
                    )
                # for i in range(100000):
                #     item = self.train_dataset[i]
                #     temp_saving_data[i]['input_ids'] = item['input_ids']
                #     temp_saving_data[i]['input_text'] = self.tokenizer.decode(temp_saving_data[i]['input_ids'])
                #     temp_saving_data[i]['target_text'] = self.tokenizer.decode(item['target_ids'][item['source_length']:])
                #     temp_saving_data[i]['detection_labels'] = item['detection_labels']
                # json.dump(temp_saving_data, open('results_main/examples/examples_SFT1.json', 'w'), ensure_ascii=False, indent=4)

        if self.args.task_mode in ["train", "eval"]:
            logger.info("Preparing validation dataset transform...")
            if self.settings.detection_results['valid']:
                ## Using detections
                assert self.settings.model_type != 'detection', "ModelSettingError: Loading detection results is meaningless when evaluating detection model."
                detection_results = json.load(open(self.settings.detection_results['valid']))
                assert len(self.dataset['valid']) == len(detection_results), f"Using uncompatible detection results for current validation set. {len(self.dataset['valid'])}, {len(detection_results)}"
                logger.info(f"Loaded previous detection results from {self.settings.detection_results['valid']}")
                self.dataset['valid'] = self.dataset['valid'].add_column('check_id', [item['id'] for item in detection_results])
                if self.settings.detection_load_way == "masked_text":
                    self.dataset['valid'] = self.dataset['valid'].add_column('masked_text', [item['masked_text'] for item in detection_results])
                elif self.settings.detection_load_way == "masked_words":
                    self.dataset['valid'] = self.dataset['valid'].add_column('masked_words', [item['masked_words'] for item in detection_results])
                else:
                    self.dataset['valid'] = self.dataset['valid'].add_column('detections', [item['detections'] for item in detection_results])
                logger.info(f"Added detections into validation dataset.")
                columns = self.dataset['valid'].column_names
                reserved_columns = ['input_ids', 'target_ids', 'position_ids', 'detection_labels', 'source_length', 'prefix_length']
                removed_columns = []
                for column in columns:
                    if column not in reserved_columns:
                        removed_columns.append(column)
                self.valid_dataset = self.dataset['valid'].map(
                    self._get_train_preprocess_function_using_predictions(for_validation=True),
                    batched=True,
                    remove_columns=removed_columns,
                    load_from_cache_file=self.settings.load_cache,
                    cache_file_name=self._get_data_cache_name(f"valid_using_pred_{self.settings.detection_load_way}_{self.settings.max_eval_source_length}"),
                    desc="Running preprocessing on valid dataset",
                )
            else:
                columns = self.dataset['valid'].column_names
                reserved_columns = ['input_ids', 'target_ids', 'position_ids', 'detection_labels', 'source_length', 'prefix_prompt_length', 'prefix_length']
                removed_columns = []
                for column in columns:
                    if column not in reserved_columns:
                        removed_columns.append(column)
                self.valid_dataset = self.dataset['valid'].map(
                    self._get_train_preprocess_function(for_validation=True, detection_model=(self.settings.model_type == 'detection')),
                    batched=True,
                    remove_columns=removed_columns,
                    load_from_cache_file=self.settings.load_cache,
                    cache_file_name=self._get_data_cache_name(f"valid_{self.settings.max_eval_source_length}"),
                    desc="Running preprocessing on valid dataset"
                )
                # temp save validation set detection labels
                # logger.info("Saving Detection Labels of Dev set...")
                # detection_results = []
                # for original_item, item in zip(self.dataset['valid'], self.valid_dataset):
                #     detection_results.append({
                #         "id": original_item['id'],
                #         "source_tokens": item["input_ids"][item['prefix_prompt_length']:item['prefix_length']],
                #         "masked_text": self.tokenizer.decode(item["input_ids"][item['prefix_length']:item['source_length']][1:-1]),
                #         "detections": item["detection_labels"][item['prefix_prompt_length']:item['prefix_length']],
                #     })
                # json.dump(detection_results, open(os.path.join(self.args.save_dir, 'valid_set_gold_detections.json'), 'w'), ensure_ascii=False, indent=4)


    def test_dataset_transform(self, data_split='test'):
        """
        Transform (transformers) dataset to meet the requirements of model inference of detection task.
        The self.test_dataset will be transformed from self.dataset[split]
        """
        if self.args.task_mode == 'interactive':
            load_cache = False
        else:
            load_cache = self.settings.load_cache

        self.test_dataset = self.dataset[data_split].map(
            self._get_detection_preprocess_function(max_sentence_length=self.settings.max_infer_source_length),
            batched=True,
            remove_columns=[],
            load_from_cache_file=load_cache,
            cache_file_name=self._get_data_cache_name(f"{data_split}_for_detection_{self.settings.max_infer_source_length}_split{self.settings.pre_split_length_for_infer}"),
            desc=f"Running detection preprocessing on {data_split} dataset"
        )

    def generation_test_dataset_transform(self, data_split='test'):
        """
        Transform (transformers) dataset to meet the requirements of model inference of generation task.
        The self.test_dataset will be transformed from self.dataset[split]
        NOTE: can only be used when masked text input is provided.
        """
        self.test_dataset = self.dataset[data_split].map(
            self._get_detached_generation_preprocess_function(self.settings.max_infer_source_length),
            batched=True,
            remove_columns=[],
            load_from_cache_file=self.settings.load_cache,
            cache_file_name=self._get_data_cache_name(f"{data_split}_for_generation_{self.settings.max_infer_source_length}_split{self.settings.pre_split_length_for_infer}"),
            desc=f"Running generation preprocessing on {data_split} dataset"
        )

    def do_train(self):
        """
        do training, using transformers trainer.
        """
        training_args = self.training_args

        # dataset
        self.train_dataset_transform()

        # set logger
        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}\n"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {self.training_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Initialize our Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.valid_dataset if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=DetectionCorrectionMetrics(self.settings.model_type, self.settings.num_labels, self.settings.loss_ignore_id).metrics_func(),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=self.callbacks,
        )
        # # for i, batch in enumerate(trainer.get_eval_dataloader()):
        # #     print(i, batch['input_ids'].size())
        # metrics = trainer.evaluate()
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            self.save(save_dir=self.args.save_dir)
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            self.save(save_dir=self.args.save_dir)

        get_all_best_checkpoint(self.args.save_dir)        
        return metrics

    def do_test(self):
        """
        do test process on labeled dataset.
        """
        raise NotImplementedError()
    
    def construct_inference_model(self):
        logger.info("Start to construct generation model for inference:")
        if self.settings.use_lora:
            logger.info("Peft model will be merged and saved, and reload as seq2seq model.")
            self.model = self.model.merge_and_unload()
            model_file = os.path.join(self.args.load, 'merged_model.bin')
            torch.save(self.model.state_dict(), model_file)
            del self.model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.pretrained_model, trust_remote_code=True, torch_dtype=self.settings.torch_dtype)
            match_information = self.model.load_state_dict(torch.load(model_file), strict=False)
            logger.info(f"load checkpoint from {model_file}, with {match_information}")
        else:
            del self.model
            model_file = os.path.join(self.args.load, 'pytorch_model.bin')
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.pretrained_model, trust_remote_code=True, torch_dtype=self.settings.torch_dtype)
            match_information = self.model.load_state_dict(torch.load(model_file), strict=False)
            logger.info(f"load checkpoint from {model_file}, with {match_information}")

    def infer_on_dataset(self, split):

        # save settings
        save_dir = os.path.join(self.args.save_dir, split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.settings.model_type == "generate":
            assert self.settings.detection_results[split], "Since it is a pure generative model, you should load detection results."
        
        # detection results load (2-stage method)
        if self.settings.detection_results[split]:
            logger.info(f"loading first stage detection results from {self.settings.detection_results[split]}")
            detection_results = json.load(open(self.settings.detection_results[split]))
            # test_dataset is self.dataset['split'], infer train mode: all split will be inferred, infer mode: split=test.
            # check id compatible
            assert len(detection_results) == len(self.dataset[split]), f"Uncompatible detection results from {self.settings.detection_results[split]}"
            for i in range(len(detection_results)):
                assert detection_results[i]["id"] == self.dataset[split][i]["id"], f"Uncompatible detection results from {self.settings.detection_results[split]}"

            # load detections
            self.test_dataset_transform(split)
            if self.settings.detection_load_way == 'detections':
                self.test_dataset = self.test_dataset.add_column('detection_predictions', [item["detections"] for item in detection_results])
            elif self.settings.detection_load_way == "masked_text":
                self.test_dataset = self.test_dataset.add_column('masked_text', [item["masked_text"] for item in detection_results])
            else:
                self.test_dataset = self.test_dataset.add_column('masked_words', [item["masked_words"] for item in detection_results])
            test_dataset_for_generation = []
            reserved_columns = ['id', 'text', 'label']
            for i, item in tqdm(enumerate(self.test_dataset)):
                src_tokens = item['input_ids'][item['prefix_prompt_length']:item['prefix_length']]
                if self.settings.detection_load_way == 'detections':
                    result = self.data_processor.convert_detected_sentence_to_infer_example(src_tokens, item['detection_predictions'])
                elif self.settings.detection_load_way == "masked_text":
                    # TODO: check, adjust mode to masked_text (BUG)
                    result = self.data_processor.convert_masked_sentence_to_infer_example(item['text'], item['masked_text'], self.settings.max_infer_source_length)
                elif self.settings.detection_load_way == "masked_words":
                    # TODO: check, adjust mode to masked_words (BUG)
                    result = self.data_processor.convert_masked_sentence_to_infer_example(item['text'], item['masked_words'], self.settings.max_infer_source_length)
                else:
                    raise NotImplementedError()
                for key in reserved_columns:
                    if key in item:
                        result[key] = item[key]
                result["complete_src_tokens"] = self.tokenizer.encode(result["text"])
                test_dataset_for_generation.append(result)
        
        # detection process (if no detection results are arranged)
        else:
            self.test_dataset_transform(split)
            test_data_loader = DataLoader(self.test_dataset, batch_size=self.settings.detection_batch_size, collate_fn=self.data_collator)
            edit_label_predictions = []
            logger.info("Error Detection:")
            keep_label_id = self.data_processor.edit_label_map['$KEEP']
            error_label_id = self.data_processor.edit_label_map['$ERROR']
            insert_label_id = self.data_processor.edit_label_map['$INSERT']
            for test_data in tqdm(test_data_loader):
                test_data.to(self.args.device)
                detection_logits = self.model(**test_data).logits['detection']
                detection_probs = F.softmax(detection_logits, -1)
                if self.settings.keep_threshold != None:
                    keep_mask = (detection_probs[:, :, keep_label_id] > self.settings.keep_threshold)*1.
                    detection_probs[:, :, keep_label_id] = keep_mask
                if self.settings.error_threshold != None:
                    non_error_mask = (detection_probs[:, :, error_label_id] < self.settings.error_threshold)*1.
                    detection_probs[:, :, error_label_id] -= non_error_mask
                if self.settings.num_labels >= 3 and self.settings.insert_threshold != None:
                    non_insert_mask = (detection_probs[:, :, insert_label_id] < self.settings.insert_threshold)*1.
                    detection_probs[:, :, insert_label_id] -= non_insert_mask
                detection_predictions = detection_probs.argmax(-1).tolist()
                prefix_length = test_data['prefix_length'].tolist()
                prefix_prompt_length = test_data['prefix_prompt_length'].tolist()
                batch_size = len(prefix_length)
                edit_label_predictions.extend([detection_predictions[i][prefix_prompt_length[i]:prefix_length[i]] for i in range(batch_size)])
            self.test_dataset = self.test_dataset.add_column('detection_predictions', edit_label_predictions)

            # In two situations, the mode will be set to detection only:
            # 1. model is only detection model
            if self.settings.model_type == 'detection':
                logger.info("Warning: Because the detection model mode is chosen, The inference will be done under detection-only mode")
                self.settings.detection_only = True
            # 2. in eval_train mode, CorrectionGLM will only output detection in current settings. If needed, this limit can be canceled.
            if self.args.task_mode == 'infer_train':
                logger.info("Warning: In current settings, The inference of train and valid set will be done under detection-only mode")
                self.settings.detection_only = True

            ## detection-only mode
            if self.settings.detection_only:
                logger.info("Attention: Detection-only mode. Saving Detections And Masked text for next stage...")
                save_items = []
                for i, item in enumerate(self.test_dataset):
                    src_tokens = item['input_ids'][item['prefix_prompt_length']:item['prefix_length']]
                    masked_example = self.data_processor.from_edit_label_to_masked_example([self.data_processor.edit_label_id_map[idx] for idx in item['detection_predictions']], src_tokens)
                    # post process to get pure masked text, ensure that encode(masked_text) equals to input_ids of masked_text (token ids can be recovered from text)
                    masked_ids = list(masked_example["input_ids"])
                    if masked_ids[0] in self.tokenizer.all_special_ids:
                        masked_ids = masked_ids[1:]
                    if masked_ids[-1] in self.tokenizer.all_special_ids:
                        masked_ids = masked_ids[:-1]
                    # in glm-roberta-large GLM tokenizer, [MASK] will be decoded with an extra blank, maybe a bug of GLM modeling?
                    if os.path.basename(self.settings.pretrained_model) == 'glm-roberta-large':
                        masked_text = self.tokenizer.decode(masked_ids, clean_up_tokenization_spaces=False)
                        masked_text = masked_text.replace('[MASK] ', '[MASK]')
                    else:
                        masked_text = self.tokenizer.decode(masked_ids)
                    # REMOVED: check if can be recovered (BUG)
                    # try:
                    #     assert self.tokenizer.encode(masked_text) == list(masked_example['input_ids']), (self.tokenizer.encode(masked_text), list(masked_example['input_ids']))
                    # except Exception as e:
                    #     logger.info(f"ERROR while checking: {masked_text} NOT EQUAL TO {self.tokenizer.decode(masked_example['input_ids'])}")

                    example_id = item['id']
                    save_items.append({"id": example_id, "masked_text": masked_text, "masked_words": self.tokenizer.convert_ids_to_tokens(masked_ids), "source_tokens": src_tokens, "detections": edit_label_predictions[i]})
                json.dump(save_items, open(os.path.join(save_dir, 'detection_results.json'), 'w'), indent=4, ensure_ascii=False)
                return []
            
            # transform test dataset for generation
            test_dataset_for_generation = []
            reserved_columns = ['id', 'text', 'label']
            for i, item in tqdm(enumerate(self.test_dataset)):
                src_tokens = item['input_ids'][item['prefix_prompt_length']:item['prefix_length']]
                result = self.data_processor.convert_detected_sentence_to_infer_example(src_tokens, item['detection_predictions'])
                for key in reserved_columns:
                    if key in item:
                        result[key] = item[key]
                result["complete_src_tokens"] = self.tokenizer.encode(result["text"])
                test_dataset_for_generation.append(result)

        ## After detection, Preparing data for generation
        
        logger.info("Using Detection results to generate dataset for mask generation:")
        logger.info(f"Change the model to GLMforConditionalGeneration, load GLM model by {self.args.load}")
        self.construct_inference_model()
        self.model.to(self.args.device)
        self.model.eval()

        # generation settings 
        max_gen_len = self.settings.max_new_tokens

        ## save real-time result one by one
        f = open(os.path.join(save_dir, 'real-time-results.json'), 'w')
        results = []

        ## model do the generation at every [MASK]
        logging.info("GLM model do the generation:")
        for test_data in tqdm(test_dataset_for_generation):
            # check if the text is truncated.
            complete_source_tokens = test_data["complete_src_tokens"]
            length_input = test_data['prefix_length'] - test_data['prefix_prompt_length']
            assert length_input <= len(complete_source_tokens), "Error: input ids can only be truncated, but found source tokens shorter than input tokens"
            length_exceed_flag = (length_input < len(complete_source_tokens))
            # record mask positions for generation
            mask_positions = []
            for idx, id in enumerate(test_data['input_ids']):
                if id == self.tokenizer.mask_token_id:
                    mask_positions.append(idx)
            # generate until all mask is generated
            # print(type(test_data['input_ids']))
            while (test_data['input_ids'] == self.tokenizer.mask_token_id).sum() != (test_data['input_ids'] == self.tokenizer.eop_token_id).sum():
                input_ids_length = len(test_data['input_ids'])
                input_ids = torch.LongTensor(test_data['input_ids']).to(self.args.device).unsqueeze(0)
                current_len = len(test_data['input_ids']) 
                pad_position_ids = list(test_data['position_ids'][0]) + [test_data['position_ids'][0][-1]]*max_gen_len
                pad_block_ids = list(test_data['position_ids'][1]) + list(range(test_data['position_ids'][1][-1]+1, test_data['position_ids'][1][-1]+max_gen_len+1))
                position_ids = torch.stack([torch.LongTensor(pad_position_ids), torch.LongTensor(pad_block_ids)]).to(self.args.device).unsqueeze(0)
                attention_mask = np.tril(np.ones((current_len+max_gen_len, current_len+max_gen_len), dtype=int))
                attention_mask[:test_data['prefix_length'], :test_data['prefix_length']] = 1
                attention_mask[test_data['prefix_length']:test_data['source_length'], :test_data['source_length']] = 1
                attention_mask = torch.LongTensor(attention_mask).to(self.args.device).unsqueeze(0).unsqueeze(0)
                try:
                    outputs = self.model.generate(
                        input_ids=input_ids, position_ids=position_ids, generation_attention_mask=attention_mask,
                        max_new_tokens=max_gen_len, 
                        eos_token_id=self.tokenizer.eop_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_beams=self.settings.num_beams,
                    )
                except Exception as e:
                    print(e)
                    print(mask_positions)
                    print(self.tokenizer.decode(input_ids[0].tolist()))
                    print(input_ids.shape, input_ids)
                    print(position_ids.shape, position_ids)
                    exit()
                new_input_ids = list(test_data['input_ids']) + outputs[0].tolist()[input_ids_length:]
                if new_input_ids[-1] != self.tokenizer.eop_token_id:
                    new_input_ids.append(self.tokenizer.eop_token_id)
                new_input_len = len(new_input_ids)
                new_position_ids = pad_position_ids[:new_input_len]
                new_block_ids = pad_block_ids[:new_input_len]
                if new_input_ids.count(self.tokenizer.mask_token_id) != new_input_ids.count(self.tokenizer.eop_token_id):
                    new_input_ids.append(self.tokenizer.sop_token_id)
                    new_position_ids.append(mask_positions[new_input_ids.count(self.tokenizer.eop_token_id)])
                    new_block_ids.append(1)
                test_data['input_ids'] = np.array(new_input_ids, dtype=int)
                test_data['position_ids'] = np.array([new_position_ids, new_block_ids], dtype=int)

            # end generation
            # mask position replace
            generation_part = list(test_data['input_ids'][test_data['source_length']:])
            source_part = list(test_data['input_ids'][test_data['prefix_length']:test_data['source_length']])
            while source_part.count(self.tokenizer.mask_token_id) > 0:
                first_mask_pos = source_part.index(self.tokenizer.mask_token_id)
                first_sop_pos = generation_part.index(self.tokenizer.sop_token_id)
                first_eop_pos = generation_part.index(self.tokenizer.eop_token_id)
                source_part = source_part[:first_mask_pos] + generation_part[first_sop_pos+1: first_eop_pos] + source_part[first_mask_pos+1:]
                generation_part = generation_part[first_eop_pos+1:]
            
            # truncate text, concate the original rear text
            if length_exceed_flag:
                source_part = source_part[:-1] + complete_source_tokens[length_input-1:]
                logger.info(
                    f"Warning: Found truncation in current example. " \
                    f"\nTruncated: {self.tokenizer.decode(test_data['input_ids'][test_data['prefix_prompt_length']:test_data['prefix_length']])}" \
                    f"\nOriginal: {self.tokenizer.decode(complete_source_tokens)}" \
                    f"\nFinal: {self.tokenizer.decode(source_part)}"
                )
            # post process
            generation_res = self.tokenizer.decode(source_part, skip_special_tokens=True)
            
            if 'label' in test_data:
                res = {'id':test_data['id'], 'src': test_data['text'], 'tgt': test_data['label'], 'predict': generation_res}
            else:
                res = {'id':test_data['id'], 'src': test_data['text'], 'predict': generation_res}
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

        f.close()

        return results
    
    def do_infer(self):
        """
        do infer on inputs.
        """
        ## detection on test dataset

        # prepare model
        self.model.to(self.args.device)
        self.model.eval()
        if self.args.task_mode == "infer_train":
            logger.info("Infer Train Mode, inferring on train dataset...")
            train_infer_res = self.infer_on_dataset('train')
            logger.info("Infer Train Mode, inferring on validation dataset...")
            valid_infer_res = self.infer_on_dataset('valid')
        logger.info("Inferring on test dataset...")
        test_infer_res = self.infer_on_dataset('test')
        return test_infer_res   

    
    def eval_on_dataset(self, split):
        """
        self.test_dataset is set by dataset['spilt'] for evaluation in this function.
        """
        # save directory
        save_dir = os.path.join(self.args.save_dir, split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # dataset and model prepare
        self.test_dataset_transform(split)
        test_data_loader = DataLoader(self.test_dataset, batch_size=self.settings.detection_batch_size, collate_fn=self.data_collator)
        edit_label_probs = []
        edit_label_predictions = []
        # detection
        logger.info("Error Detection:")
        keep_label_id = self.data_processor.edit_label_map['$KEEP']
        error_label_id = self.data_processor.edit_label_map['$ERROR']
        insert_label_id = self.data_processor.edit_label_map['$INSERT']
        for test_data in tqdm(test_data_loader):
            test_data.to(self.args.device)
            detection_logits = self.model(**test_data).logits['detection']
            detection_probs = F.softmax(detection_logits, -1)
            detection_probs_copy = detection_probs.tolist()
            edit_label_probs.extend(detection_probs_copy)
            if self.settings.keep_threshold:
                keep_mask = (detection_probs[:, :, keep_label_id] > self.settings.keep_threshold)*1.
                detection_probs[:, :, keep_label_id] = keep_mask
            if self.settings.error_threshold:
                non_error_mask = (detection_probs[:, :, error_label_id] < self.settings.error_threshold)*1.
                detection_probs[:, :, error_label_id] -= non_error_mask
            if self.settings.num_labels >= 3 and self.settings.insert_threshold:
                non_insert_mask = (detection_probs[:, :, insert_label_id] < self.settings.insert_threshold)*1.
                detection_probs[:, :, insert_label_id] -= non_insert_mask
            detection_predictions = detection_probs.argmax(-1).tolist()
            prefix_length = test_data['prefix_length'].tolist()
            prefix_prompt_length = test_data['prefix_prompt_length'].tolist()
            batch_size = len(prefix_length)
            edit_label_predictions.extend([detection_predictions[i][prefix_prompt_length[i]:prefix_length[i]] for i in range(batch_size)])
        self.test_dataset = self.test_dataset.add_column('detection_predictions', edit_label_predictions)

        # In two situations, the mode will be set to detection only:
        # 1. model is only detection model
        if self.settings.model_type == 'detection':
            logger.info("Warning: Because the detection model mode is chosen, The evaluation will be done under detection-only mode")
            self.settings.detection_only = True
        # 2. in eval_train mode, CorrectionGLM will only output detection in current settings. If needed, this limit can be canceled.
        if self.args.task_mode == 'eval_train':
            logger.info("Warning: In current settings, The evaluation of train and valid set will be done under detection-only mode")
            self.settings.detection_only = True

        if self.settings.detection_only:
            logger.info("Attention: Detection-only mode. Saving Detections... (if masked text is required, please use eval mode.)")
            save_items = []
            for i, item in enumerate(self.test_dataset):
                src_tokens = item['input_ids'][item['prefix_prompt_length']:item['prefix_length']]
                example_id = item['id']
                save_items.append({"id": example_id, "source_tokens": src_tokens, "detections": edit_label_predictions[i]})
            json.dump(save_items, open(os.path.join(save_dir, 'detection_results.json'), 'w'), indent=4, ensure_ascii=False)
            return []
        
        logger.info("Using Detection results to generate dataset for mask generation:")
        logger.info(f"Change the model to GLMforConditionalGeneration, load GLM model by {self.args.load}")
        self.construct_inference_model()
        self.model.to(self.args.device)
        self.model.eval()

        ## prepare dataset for generation
        test_dataset_for_generation = []
        reserved_columns = ['id', 'text', 'label']
        # self.valid_dataset = self.test_dataset
        for i, item in tqdm(enumerate(self.test_dataset)):
            # find correspond labeled data example, save relavant information
            if split == "train":
                valid_data_item = self.train_dataset[i]
            elif split == "valid":
                valid_data_item = self.valid_dataset[i]
            else:
                raise NotImplementedError()
            src_tokens = item['input_ids'][item['prefix_prompt_length']:item['prefix_length']]
            result = self.data_processor.convert_detected_sentence_to_infer_example(src_tokens, item['detection_predictions'])
            std_result = self.data_processor.convert_detected_sentence_to_infer_example(
                src_tokens, 
                valid_data_item['detection_labels'][valid_data_item['prefix_prompt_length']:valid_data_item['prefix_length']]
            )
            for key in reserved_columns:
                if key in item:
                    result[key] = item[key]
            result["complete_src_tokens"] = self.tokenizer.encode(result["text"])
            result['golden_masked_src_token_ids'] = std_result["input_ids"][std_result["prefix_length"]:std_result["source_length"]]
            test_dataset_for_generation.append(result)

        max_gen_len = self.settings.max_new_tokens

        ## save result one by one
        f = open(os.path.join(save_dir, 'real-time-results.json'), 'w')
        results = []

        logging.info("GLM model do the generation:")
        for i, test_data in tqdm(enumerate(test_dataset_for_generation)):
            # check if the text is truncated.
            complete_source_tokens = test_data["complete_src_tokens"]
            length_input = test_data['prefix_length'] - test_data['prefix_prompt_length']
            assert length_input <= len(complete_source_tokens), "Error: input ids can only be truncated, but found source tokens shorter than input tokens"
            length_exceed_flag = (length_input < len(complete_source_tokens))
            # record mask positions for generation
            mask_positions = []
            original_item = self.test_dataset[i]
            # find correspond labeled data example, save relavant information
            if split == "train":
                valid_item = self.train_dataset[i]
            elif split == "valid":
                valid_item = self.valid_dataset[i]
            else:
                raise NotImplementedError()
            
            # get mask position from masked text
            src_tokens = original_item['input_ids'][original_item['prefix_prompt_length']:original_item['prefix_length']]
            detections = original_item['detection_predictions']
            for idx, id in enumerate(test_data['input_ids']):
                if id == self.tokenizer.mask_token_id:
                    mask_positions.append(idx)
            # generate until all mask is generated
            while (test_data['input_ids'] == self.tokenizer.mask_token_id).sum() != (test_data['input_ids'] == self.tokenizer.eop_token_id).sum():
                input_ids_length = len(test_data['input_ids'])
                input_ids = torch.LongTensor(test_data['input_ids']).to(self.args.device).unsqueeze(0)
                current_len = len(test_data['input_ids']) 
                pad_position_ids = list(test_data['position_ids'][0]) + [test_data['position_ids'][0][-1]]*max_gen_len
                pad_block_ids = list(test_data['position_ids'][1]) + list(range(test_data['position_ids'][1][-1]+1, test_data['position_ids'][1][-1]+max_gen_len+1))
                position_ids = torch.stack([torch.LongTensor(pad_position_ids), torch.LongTensor(pad_block_ids)]).to(self.args.device).unsqueeze(0)
                attention_mask = np.tril(np.ones((current_len+max_gen_len, current_len+max_gen_len), dtype=int))
                attention_mask[:test_data['prefix_length'], :test_data['prefix_length']] = 1
                attention_mask[test_data['prefix_length']:test_data['source_length'], :test_data['source_length']] = 1
                attention_mask = torch.LongTensor(attention_mask).to(self.args.device).unsqueeze(0).unsqueeze(0)
                outputs = self.model.generate(
                    input_ids=input_ids, position_ids=position_ids, generation_attention_mask=attention_mask,
                    max_new_tokens=max_gen_len, 
                    eos_token_id=self.tokenizer.eop_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_beams=self.settings.num_beams,
                )
                new_input_ids = list(test_data['input_ids']) + outputs[0].tolist()[input_ids_length:]
                if new_input_ids[-1] != self.tokenizer.eop_token_id:
                    new_input_ids.append(self.tokenizer.eop_token_id)
                new_input_len = len(new_input_ids)
                new_position_ids = pad_position_ids[:new_input_len]
                new_block_ids = pad_block_ids[:new_input_len]
                if new_input_ids.count(self.tokenizer.mask_token_id) != new_input_ids.count(self.tokenizer.eop_token_id):
                    new_input_ids.append(self.tokenizer.sop_token_id)
                    new_position_ids.append(mask_positions[new_input_ids.count(self.tokenizer.eop_token_id)])
                    new_block_ids.append(1)
                test_data['input_ids'] = np.array(new_input_ids, dtype=int)
                test_data['position_ids'] = np.array([new_position_ids, new_block_ids], dtype=int)

            # end generation
            # mask position replace
            generation_part = list(test_data['input_ids'][test_data['source_length']:])
            generation_part_copy = copy.deepcopy(generation_part)
            source_part = list(test_data['input_ids'][test_data['prefix_length']:test_data['source_length']])
            source_part_copy = copy.deepcopy(source_part)
            while source_part.count(self.tokenizer.mask_token_id) > 0:
                first_mask_pos = source_part.index(self.tokenizer.mask_token_id)
                first_sop_pos = generation_part.index(self.tokenizer.sop_token_id)
                first_eop_pos = generation_part.index(self.tokenizer.eop_token_id)
                source_part = source_part[:first_mask_pos] + generation_part[first_sop_pos+1: first_eop_pos] + source_part[first_mask_pos+1:]
                generation_part = generation_part[first_eop_pos+1:]

            # truncate text, concate the original rear text
            if length_exceed_flag:
                source_part = source_part[:-1] + complete_source_tokens[length_input-1:]
                logger.info(
                    f"Warning: Found truncation in current example. " \
                    f"\nTruncated: {self.tokenizer.decode(test_data['input_ids'][test_data['prefix_prompt_length']:test_data['prefix_length']])}" \
                    f"\nOriginal: {self.tokenizer.decode(complete_source_tokens)}" \
                    f"\nFinal: {self.tokenizer.decode(source_part)}"
                )
            # post process
            generation_res = self.tokenizer.decode(source_part, skip_special_tokens=True)
            
            assert 'label' in test_data, "Validation set without targets(labels)"
            res = {
                'id':test_data['id'], 'src': test_data['text'], 'tgt': test_data['label'], 'predict': generation_res,
                'src_tokens': self.tokenizer.convert_ids_to_tokens(src_tokens),
                'detections': [self.data_processor.edit_label_id_map[j] for j in detections],
                # 'detection_logits': edit_label_probs[i],
                'detection_labels': [self.data_processor.edit_label_id_map[j] for j in valid_item['detection_labels'][valid_item['prefix_prompt_length']:valid_item['prefix_length']]],
                'predict_tokens': self.tokenizer.convert_ids_to_tokens(source_part),
                'masked_src_tokens': self.tokenizer.convert_ids_to_tokens(source_part_copy),
                'golden_masked_src_tokens': self.tokenizer.convert_ids_to_tokens(test_data['golden_masked_src_token_ids']),
                'generated_tokens': self.tokenizer.convert_ids_to_tokens(generation_part_copy),
            }
            if 'other_labels' in test_data:
                res['other_labels'] = test_data['other_labels']
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

        f.close()

        return results
    

    def do_eval(self):
        """
        do infer on eval dataset and output as much midium ponents for debugging.
        """
        ## model
        self.model.to(self.args.device)
        self.model.eval()
        ## dataset
        # in eval mode, the true label information are required, so dataset transform is essential
        # no detection results are allowed
        assert self.settings.detection_results['train'] == self.settings.detection_results['test'] == self.settings.detection_results['valid'] == None
        self.train_dataset_transform()
        if self.args.task_mode == "eval_train":
            logger.info("Eval Train Mode, evaluate on train dataset...")
            train_eval_res = self.eval_on_dataset('train')
        logger.info("Evaluate on validation dataset...")
        valid_eval_res = self.eval_on_dataset('valid')
        return valid_eval_res


    def save(self, save_dir):
        if self.args.task_mode == 'train':
            logger.info("Saving manual training settings in config.py...")
            config_file = os.path.join(save_dir, 'presettings.json')
            config_dict = {}
            for key in self.settings:
                content = self.settings[key]
                if key in ['pretrained_model', 'torch_dtype']:
                    config_dict[key] = str(content)
                else:
                    config_dict[key] = content
            config_dict['model'] = self.args.model
            config_dict['dataset'] = self.args.dataset
            config_dict['task_mode'] = self.args.task_mode
            config_dict['seed'] = self.args.seed
            config_dict['load'] = self.args.load
            config_dict['resume'] = self.args.resume

            json.dump(config_dict, open(config_file, 'w'), indent=4, ensure_ascii=False)
        else:    
            raise NotImplementedError()

    def load(self, save_dir: str):
        if self.args.task_mode == 'train':
            if self.settings.use_lora:
                logger.info("Lora model should have been loaded at the model construction.")
                return
            logger.info("Load complete model for CorrectionGLM to continue training.")
            path = os.path.join(save_dir, 'pytorch_model.bin')
            information = self.model.load_state_dict(torch.load(path), strict=False)
            logger.info(information)

        else:
            if save_dir[-1] == '/':
                save_dir = save_dir[:-1]
            checkpoint_root_dir =  os.path.dirname(save_dir) if save_dir.find('checkpoint') != -1 else save_dir
            config_file = os.path.join(checkpoint_root_dir, 'presettings.json')
            logger.info(f"Load manual training settings in checkpoint, include {self.settings.load_config_keys}, from {config_file}")
            presettings = json.load(open(config_file))
            for key in self.settings.load_config_keys:
                self.settings[key] = presettings[key]

            if self.settings.use_lora:
                logger.info("Lora model should have been loaded at the model construction.")
                return
            logger.info("Load complete model for CorrectionGLM.")
            path = os.path.join(save_dir, 'pytorch_model.bin')
            information = self.model.load_state_dict(torch.load(path))
            logger.info(information)
    
    def get_best_checkpoint_dir(self):
        trainer_state = json.load(open(os.path.join(self.args.save_dir, 'trainer_state.json')))
        best_model_checkpoint = trainer_state["best_model_checkpoint"]
        return best_model_checkpoint
