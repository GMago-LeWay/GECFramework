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
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    BatchEncoding,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate

from trainers.base import TrainerBeta
from dataset_provider.CorrectionGLM import GLMDataProcessor

logger = logging.getLogger(__name__)

transformers.utils.move_cache('/data/liwei/cache/huggingface/')

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
            pad_input_ids = sample['input_ids'] + [self.tokenizer.pad_token_id]*(max_len-current_len)
            batch_input_ids.append(pad_input_ids)
            pad_target_ids = sample['target_ids'] + [self.loss_ignore_id]*(max_len-current_len)
            batch_target_ids.append(pad_target_ids)
            pad_detection_labels = sample['detection_labels'] + [self.loss_ignore_id]*(max_len-current_len)
            batch_detection_label_ids.append(pad_detection_labels)
            pad_position_ids = sample['position_ids'][0] + [sample['position_ids'][0][-1]]*(max_len-current_len)
            pad_block_ids = sample['position_ids'][1] + list(range(sample['position_ids'][1][-1]+1, sample['position_ids'][1][-1]+max_len-current_len+1))
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

CN_MARKER_MAP = {
    ',': '，',
    ';': '；',
    ':': '：',
    '(': '（',
    ')': '）',
    '?': '？',
    '!': '！',
}


def postprocess_cn(result: str):
    for key in CN_MARKER_MAP:
        result = result.replace(key, CN_MARKER_MAP[key])
    return result


class CGLMMetrics:
    def __init__(self, model_type, labels_num=3, loss_ignore_id=-100) -> None:
        assert labels_num in [2,3]
        assert model_type in ['all', 'detection', 'generate']
        self.model_type = model_type
        self.labels_num = labels_num
        self.loss_ignore_id = loss_ignore_id
        self.accuracy_metric = evaluate.load("utils/accuracy")

    def accuracy(self, references, predictions, weights):
        if weights.sum() == 0:
            return 0
        else:
            return self.accuracy_metric.compute(references=references, predictions=predictions, sample_weight=weights)['accuracy']
        
    def glm_accuracy(self, glm_pred_ids, glm_labels):
        glm_pred_ids, glm_labels = glm_pred_ids.ravel(), glm_labels.ravel()
        glm_pred_weights = (1 - (glm_labels == -100)*1).ravel()
        glm_accuracy = self.accuracy(references=glm_labels, predictions=glm_pred_ids, weights=glm_pred_weights)
        return glm_accuracy
    
    def detection_metrics(self, detection_pred_ids, detection_labels):
        detection_pred_ids, detection_labels = detection_pred_ids.ravel(), detection_labels.ravel()
        detection_pred_weights = (1 - (detection_labels == -100)*1).ravel()
        keep_pred_weights = ((detection_labels==0)*1).ravel()
        error_pred_weights = ((detection_labels==1)*1).ravel()
        detection_accuracy = self.accuracy(references=detection_labels, predictions=detection_pred_ids, weights=detection_pred_weights)
        keep_accuracy = self.accuracy(references=detection_labels, predictions=detection_pred_ids, weights=keep_pred_weights)
        error_accuracy = self.accuracy(references=detection_labels, predictions=detection_pred_ids, weights=error_pred_weights)

        if self.labels_num == 3:
            insert_pred_weights = ((detection_labels==2)*1).ravel()
            insert_accuracy = self.accuracy(references=detection_labels, predictions=detection_pred_ids, weights=insert_pred_weights)
            detection_geometric_accuracy = ( keep_accuracy*error_accuracy*insert_accuracy ) ** (1/3)
            return {'detection_accuracy': detection_accuracy, 'detection_geometric_accuracy': detection_geometric_accuracy,
                    'keep_accuracy': keep_accuracy, 'error_accuracy': error_accuracy, 'insert_accuracy': insert_accuracy}
        else:
            detection_geometric_accuracy = ( keep_accuracy*error_accuracy ) ** (1/2)
            return {'detection_accuracy': detection_accuracy, 'detection_geometric_accuracy': detection_geometric_accuracy,
                    'keep_accuracy': keep_accuracy, 'error_accuracy': error_accuracy}
        
    def metrics(self, glm_pred_ids, glm_labels, detection_pred_ids, detection_labels):
        if self.model_type == 'generate':
            glm_accuracy = self.glm_accuracy(glm_pred_ids=glm_pred_ids, glm_labels=glm_labels)
            return {'general_accuracy': glm_accuracy, 'glm_accuracy': glm_accuracy}
        else:
            detection_metrics = self.detection_metrics(detection_pred_ids=detection_pred_ids, detection_labels=detection_labels)
            if self.model_type == 'detection':
                detection_metrics['general_accuracy'] = detection_metrics['detection_geometric_accuracy']
                return detection_metrics
            else:
                glm_accuracy = self.glm_accuracy(glm_pred_ids=glm_pred_ids, glm_labels=glm_labels)
                ## avg of glm acc and balanced detecion acc
                general_accuracy = (glm_accuracy*detection_metrics['detection_geometric_accuracy'])**0.5
                ## avg of glm acc and detecion acc
                geometric_accuracy = (glm_accuracy*detection_metrics['detection_accuracy'])**0.5
                metrics = {'general_accuracy': general_accuracy, 'geometric_accuracy': geometric_accuracy, 'glm_accuracy': glm_accuracy,}
                for key in detection_metrics:
                    metrics[key] = detection_metrics[key]
                return metrics

    def metrics_func(self):
        def compute(eval_predictions):
            pred_ids, label_ids = eval_predictions.predictions, eval_predictions.label_ids
            glm_pred_ids, detection_pred_ids = None, None
            if 'glm' in pred_ids[0]:
                glm_pred_ids = pred_ids[0]['glm']
            if 'detection' in pred_ids[0]:
                detection_pred_ids = pred_ids[0]['detection']
            glm_labels, detection_labels = label_ids
            return self.metrics(glm_pred_ids=glm_pred_ids, glm_labels=glm_labels, detection_pred_ids=detection_pred_ids, detection_labels=detection_labels)
        return compute


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

        # initialize config for transformers trainer
        self.training_args = TrainingArguments(
            seed=args.seed,
            do_train=True,
            do_eval=True,
            do_predict=False,
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
            resume_from_checkpoint=args.resume,
        )
        logger.info(self.training_args)

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
        logger.info(f'Hash Code Key of Dataset: {hash_str}')
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
        return _preprocess
    
    def _get_detection_preprocess_function(self):
        def _preprocess(examples):
            processed = {}
            for i in range(len(examples['text'])):
                src = examples['text'][i]
                result = self.data_processor.convert_sentence_to_detection_example(src)
                if not processed:
                    for key in result:
                        processed[key] = []
                for key in result:
                    processed[key].append(result[key])
            return processed
        return _preprocess        
    
    def _get_detached_generation_preprocess_function(self):
        def _preprocess(examples):
            processed = {}
            for i in range(len(examples['text'])):
                src = examples['text'][i]
                masked_src = examples['masked_text'][i]
                result = self.data_processor.convert_masked_sentence_to_infer_example(src, masked_src)
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
                ## Using detections
                assert self.settings.model_type != 'detection', "ModelSettingError: Loading detection results is meaningless when training detection model."
                detection_results = json.load(open(self.settings.detection_results['train']))
                assert len(self.dataset['train']) == len(detection_results), f"Using uncompatible detection results for current training set. {len(self.dataset['train'])}, {len(detection_results)}"
                logger.info(f"Loaded previous detection results from {self.settings.detection_results['train']}")
                self.dataset['train'] = self.dataset['train'].add_column('detections', [item['detections'] for item in detection_results])
                self.dataset['train'] = self.dataset['train'].add_column('check_id', [item['id'] for item in detection_results])
                logger.info(f"Added detections into train dataset.")
                columns = self.dataset['train'].column_names
                reserved_columns = ['input_ids', 'target_ids', 'position_ids', 'detection_labels', 'source_length', 'prefix_length']
                removed_columns = []
                for column in columns:
                    if column not in reserved_columns:
                        removed_columns.append(column)
                self.train_dataset = self.dataset['train'].map(
                    self._get_train_preprocess_function_using_predictions(for_validation=False),
                    batched=True,
                    remove_columns=removed_columns,
                    load_from_cache_file=self.settings.load_cache,
                    cache_file_name=self._get_data_cache_name("train_using_pred"),
                    desc="Running preprocessing on train dataset",
                )
            else:
                columns = self.dataset['train'].column_names
                reserved_columns = ['input_ids', 'target_ids', 'position_ids', 'detection_labels', 'source_length', 'prefix_length']
                removed_columns = []
                for column in columns:
                    if column not in reserved_columns:
                        removed_columns.append(column)
                self.train_dataset = self.dataset['train'].map(
                    self._get_train_preprocess_function(for_validation=False, detection_model=(self.settings.model_type == 'detection')),
                    batched=True,
                    remove_columns=removed_columns,
                    load_from_cache_file=self.settings.load_cache,
                    cache_file_name=self._get_data_cache_name("train"),
                    desc="Running preprocessing on train dataset",
                )
        if self.args.task_mode in ["train", "eval"]:
            logger.info("Preparing validation dataset transform...")
            if self.settings.detection_results['valid']:
                ## Using detections
                assert self.settings.model_type != 'detection', "ModelSettingError: Loading detection results is meaningless when evaluating detection model."
                detection_results = json.load(open(self.settings.detection_results['valid']))
                assert len(self.dataset['valid']) == len(detection_results), f"Using uncompatible detection results for current validation set. {len(self.dataset['valid'])}, {len(detection_results)}"
                logger.info(f"Loaded previous detection results from {self.settings.detection_results['valid']}")
                self.dataset['valid'] = self.dataset['valid'].add_column('detections', [item['detections'] for item in detection_results])
                self.dataset['valid'] = self.dataset['valid'].add_column('check_id', [item['id'] for item in detection_results])
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
                    cache_file_name=self._get_data_cache_name("valid_using_pred"),
                    desc="Running preprocessing on valid dataset",
                )
            else:
                columns = self.dataset['valid'].column_names
                reserved_columns = ['input_ids', 'target_ids', 'position_ids', 'detection_labels', 'source_length', 'prefix_length']
                removed_columns = []
                for column in columns:
                    if column not in reserved_columns:
                        removed_columns.append(column)
                self.valid_dataset = self.dataset['valid'].map(
                    self._get_train_preprocess_function(for_validation=True, detection_model=(self.settings.model_type == 'detection')),
                    batched=True,
                    remove_columns=removed_columns,
                    load_from_cache_file=self.settings.load_cache,
                    cache_file_name=self._get_data_cache_name("valid"),
                    desc="Running preprocessing on valid dataset"
                )

    def test_dataset_transform(self, data_split='test'):
        """
        Transform (transformers) dataset to meet the requirements of model inference of detection task.
        The self.test_dataset will be transformed from self.dataset[split]
        """

        self.test_dataset = self.dataset[data_split].map(
            self._get_detection_preprocess_function(),
            batched=True,
            remove_columns=[],
            load_from_cache_file=self.settings.load_cache,
            cache_file_name=self._get_data_cache_name(f"{data_split}_for_detection"),
            desc="Running detection preprocessing on test dataset"
        )

    def generation_test_dataset_transform(self, data_split='test'):
        """
        Transform (transformers) dataset to meet the requirements of model inference of generation task.
        The self.test_dataset will be transformed from self.dataset[split]
        NOTE: can only be used when masked text input is provided.
        """
        self.test_dataset = self.dataset[data_split].map(
            self._get_detached_generation_preprocess_function(),
            batched=True,
            remove_columns=[],
            load_from_cache_file=self.settings.load_cache,
            cache_file_name=self._get_data_cache_name(f"{data_split}_for_generation"),
            desc="Running generation preprocessing on test dataset"
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
            compute_metrics=CGLMMetrics(self.settings.model_type, self.settings.num_labels, self.settings.loss_ignore_id).metrics_func(),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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

        # # Test
        # if training_args.do_eval:
        #     logger.info("*** Evaluate ***")

        #     metrics = trainer.evaluate(eval_dataset=test_dataset)
        #     metrics["test_samples"] = len(test_dataset)

        #     trainer.log_metrics("test", metrics)
        #     trainer.save_metrics("test", metrics)
        return metrics

    def do_test(self):
        """
        do test process on labeled dataset.
        """
        raise NotImplementedError()
    
    def construct_inference_model(self, model_file):
        del self.model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.pretrained_model, trust_remote_code=True)
        match_information = self.model.load_state_dict(torch.load(model_file), strict=False)
        logger.info(f"load checkpoint from {model_file}, with {match_information}")

    def infer_on_dataset(self, split):

        # save settings
        if split in ['valid', 'test']:
            save_dir = os.path.join(self.args.save_dir, split)
        else:
            save_dir = self.args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # detection results load (2-stage method)
        if self.settings.detection_results[split]:
            logger.info(f"loading first stage detection results from {self.settings.detection_results['infer']}")
            detection_results = json.load(open(self.settings.detection_results[split]))
            # self.test_dataset is self.dataset['split']
            self.test_dataset = self.dataset[split]
            # check id compatible
            assert len(detection_results) == len(self.test_dataset), f"Uncompatible detection results from {self.settings.detection_results['infer']}"
            for i in range(len(detection_results)):
                assert detection_results[i]["id"] == self.test_dataset[i]["id"], f"Uncompatible detection results from {self.settings.detection_results['infer']}"
            # add masked text to test dataset and transform
            self.test_dataset = self.test_dataset.add_column('masked_text', [item["masked_text"] for item in detection_results])
            self.generation_test_dataset_transform()
        
        # detection process (if no detection results are arranged)
        else:
            self.test_dataset_transform(split)
            test_data_loader = DataLoader(self.test_dataset, batch_size=self.settings.detection_batch_size, collate_fn=self.data_collator)
            edit_label_predictions = []
            logger.info("Error Detection:")
            keep_label_id = self.data_processor.edit_label_map['$KEEP']
            for test_data in tqdm(test_data_loader):
                test_data.to(self.args.device)
                detection_logits = self.model(**test_data).logits['detection']
                detection_probs = F.softmax(detection_logits, -1)
                if self.settings.keep_threshold:
                    keep_mask = (detection_probs[:, :, keep_label_id] > self.settings.keep_threshold)*1.
                    detection_probs[:, :, keep_label_id] = keep_mask
                detection_predictions = detection_probs.argmax(-1).tolist()
                prefix_length = test_data['prefix_length'].tolist()
                prefix_prompt_length = test_data['prefix_prompt_length'].tolist()
                batch_size = len(prefix_length)
                edit_label_predictions.extend([detection_predictions[i][prefix_prompt_length[i]:prefix_length[i]] for i in range(batch_size)])
            self.test_dataset = self.test_dataset.add_column('detection_predictions', edit_label_predictions)
            test_dataset_for_generation = self.test_dataset

            if self.settings.model_type == 'detection':
                logger.info("Warning: Because the detection model mode is chosen, The inference will be done under detection-only mode")
                self.settings.detection_only = True

            ## detection-only mode
            if self.settings.detection_only:
                logger.info("Attention: Detection-only mode. Saving Detections And Masked text for next stage...")
                save_items = []
                for i, item in enumerate(self.test_dataset):
                    src_tokens = item['input_ids'][item['prefix_prompt_length']:item['prefix_length']]
                    masked_example = self.data_processor.from_edit_label_to_masked_example([self.data_processor.edit_label_id_map[idx] for idx in item['detection_predictions']], src_tokens)
                    masked_text = self.tokenizer.decode(masked_example['input_ids'])
                    example_id = item['id']
                    save_items.append({"id": example_id, "masked_text": masked_text, "source_tokens": src_tokens, "detections": edit_label_predictions[i]})
                json.dump(save_items, open(os.path.join(save_dir, 'detection_results.json'), 'w'), indent=4, ensure_ascii=False)
                return []

        ## After detection, Preparing data for generation
        
        logger.info("Using Detection results to generate dataset for mask generation:")
        logger.info(f"Change the model to GLMforConditionalGeneration, load GLM model by {self.args.load}")
        self.construct_inference_model(os.path.join(self.args.load, 'pytorch_model.bin'))
        self.model.to(self.args.device)
        self.model.eval()

        test_dataset_for_generation = []
        reserved_columns = ['id', 'text', 'label']
        for i, item in tqdm(enumerate(self.test_dataset)):
            src_tokens = item['input_ids'][item['prefix_prompt_length']:item['prefix_length']]
            result = self.data_processor.convert_detected_sentence_to_infer_example(src_tokens, item['detection_predictions'])
            for key in reserved_columns:
                if key in item:
                    result[key] = item[key]
            test_dataset_for_generation.append(result)

        # generation settings 
        max_gen_len = self.settings.max_new_tokens

        ## save real-time result one by one
        f = open(os.path.join(save_dir, 'real-time-results.json'), 'w')
        results = []

        ## model do the generation at every [MASK]
        logging.info("GLM model do the generation:")
        for test_data in tqdm(test_dataset_for_generation):
            mask_positions = []
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
            source_part = list(test_data['input_ids'][test_data['prefix_length']:test_data['source_length']])
            while source_part.count(self.tokenizer.mask_token_id) > 0:
                first_mask_pos = source_part.index(self.tokenizer.mask_token_id)
                first_sop_pos = generation_part.index(self.tokenizer.sop_token_id)
                first_eop_pos = generation_part.index(self.tokenizer.eop_token_id)
                source_part = source_part[:first_mask_pos] + generation_part[first_sop_pos+1: first_eop_pos] + source_part[first_mask_pos+1:]
                generation_part = generation_part[first_eop_pos+1:]
            
            # post process
            generation_res = self.tokenizer.decode(source_part, skip_special_tokens=True)
            if self.settings.chinese_marker_substitution:
                generation_res = postprocess_cn(generation_res)
            
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
        for test_data in tqdm(test_data_loader):
            test_data.to(self.args.device)
            detection_logits = self.model(**test_data).logits['detection']
            detection_probs = F.softmax(detection_logits, -1)
            detection_probs_copy = detection_probs.tolist()
            edit_label_probs.extend(detection_probs_copy)
            if self.settings.keep_threshold:
                keep_mask = (detection_probs[:, :, keep_label_id] > self.settings.keep_threshold)*1.
                detection_probs[:, :, keep_label_id] = keep_mask
            detection_predictions = detection_probs.argmax(-1).tolist()
            prefix_length = test_data['prefix_length'].tolist()
            prefix_prompt_length = test_data['prefix_prompt_length'].tolist()
            batch_size = len(prefix_length)
            edit_label_predictions.extend([detection_predictions[i][prefix_prompt_length[i]:prefix_length[i]] for i in range(batch_size)])
        self.test_dataset = self.test_dataset.add_column('detection_predictions', edit_label_predictions)


        if self.settings.model_type == 'detection':
            logger.info("Warning: Because the detection model mode is chosen, The evaluation will be done under detection-only mode")
            self.settings.detection_only = True

        if self.settings.detection_only:
            logger.info("Attention: Detection-only mode. Saving Detections...")
            save_items = []
            for i, item in enumerate(self.test_dataset):
                src_tokens = item['input_ids'][item['prefix_prompt_length']:item['prefix_length']]
                example_id = item['id']
                save_items.append({"id": example_id, "source_tokens": src_tokens, "detections": edit_label_predictions[i]})
            json.dump(save_items, open(os.path.join(save_dir, 'detection_results.json'), 'w'), indent=4, ensure_ascii=False)
            return []
        
        logger.info("Using Detection results to generate dataset for mask generation:")
        logger.info(f"Change the model to GLMforConditionalGeneration, load GLM model by {self.args.load}")
        self.construct_inference_model(os.path.join(self.args.load, 'pytorch_model.bin'))
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
            result['golden_masked_src_token_ids'] = std_result["input_ids"][std_result["prefix_length"]:std_result["source_length"]]
            test_dataset_for_generation.append(result)

        max_gen_len = self.settings.max_new_tokens

        ## save result one by one
        f = open(os.path.join(save_dir, 'real-time-results.json'), 'w')
        results = []

        logging.info("GLM model do the generation:")
        for i, test_data in tqdm(enumerate(test_dataset_for_generation)):
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
            
            # post process
            generation_res = self.tokenizer.decode(source_part, skip_special_tokens=True)
            if self.settings.chinese_marker_substitution:
                generation_res = postprocess_cn(generation_res)
            
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
                if type(content) in [str, int, float, bool, list, dict]:
                    config_dict[key] = content
                else:
                    config_dict[key] = str(content)
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
            logger.info("Load complete model for CorrectionGLM to continue training.")
            path = os.path.join(save_dir, 'pytorch_model.bin')
            self.model.load_state_dict(torch.load(path))
        else:
            if save_dir[-1] == '/':
                save_dir = save_dir[:-1]
            checkpoint_root_dir =  os.path.dirname(save_dir) if save_dir.find('checkpoint') != -1 else save_dir
            config_file = os.path.join(checkpoint_root_dir, 'presettings.json')
            logger.info(f"Load manual training settings in checkpoint, include {self.settings.load_config_keys}, from {config_file}")
            presettings = json.load(open(config_file))
            for key in self.settings.load_config_keys:
                self.settings[key] = presettings[key]
            
            logger.info("Load complete model for CorrectionGLM.")
            path = os.path.join(save_dir, 'pytorch_model.bin')
            self.model.load_state_dict(torch.load(path))
            
