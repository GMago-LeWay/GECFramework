from typing import Optional, Union
from tqdm import tqdm
import os
import logging
import json
import numpy as np
import torch
import torch.nn.functional as F
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

accuracy_metric = evaluate.load("utils/accuracy")
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


# TODO: 2 class metrics
def compute_metrics_2_label(eval_predictions):
    pred_ids, label_ids = eval_predictions.predictions, eval_predictions.label_ids
    glm_pred_ids, detection_pred_ids = pred_ids[0]
    glm_labels, detection_labels = label_ids

    glm_pred_ids, detection_pred_ids, glm_labels, detection_labels = glm_pred_ids.ravel(), detection_pred_ids.ravel(), glm_labels.ravel(), detection_labels.ravel()

    glm_pred_weights = (1 - (glm_labels == -100)*1).ravel()
    detection_pred_weights = (1 - (detection_labels == -100)*1).ravel()
    keep_pred_weights = ((detection_labels==0)*1).ravel()
    error_pred_weights = ((detection_labels==1)*1).ravel()
    glm_accuracy = accuracy_metric.compute(references=glm_labels, predictions=glm_pred_ids, sample_weight=glm_pred_weights)['accuracy']
    detection_accuracy = accuracy_metric.compute(references=detection_labels, predictions=detection_pred_ids, sample_weight=detection_pred_weights)['accuracy']
    keep_accuracy = accuracy_metric.compute(references=detection_labels, predictions=detection_pred_ids, sample_weight=keep_pred_weights)['accuracy']
    error_accuracy = accuracy_metric.compute(references=detection_labels, predictions=detection_pred_ids, sample_weight=error_pred_weights)['accuracy']
    geometric_accuracy = (glm_accuracy*detection_accuracy)**0.5
    detection_geometric_accuracy = ( keep_accuracy*error_accuracy ) ** (1/2)
    general_accuary = (glm_accuracy*detection_geometric_accuracy)**0.5

    return {'general_accuracy': general_accuary, 
            'geometric_accuracy': geometric_accuracy, 
            'glm_accuracy': glm_accuracy, 'detection_accuracy': detection_accuracy, 
            'detection_geometric_accuracy': detection_geometric_accuracy,
            'keep_accuracy': keep_accuracy, 'error_accuracy': error_accuracy}


def compute_metrics_3_label(eval_predictions):
    pred_ids, label_ids = eval_predictions.predictions, eval_predictions.label_ids
    glm_pred_ids, detection_pred_ids = pred_ids[0]
    glm_labels, detection_labels = label_ids

    glm_pred_ids, detection_pred_ids, glm_labels, detection_labels = glm_pred_ids.ravel(), detection_pred_ids.ravel(), glm_labels.ravel(), detection_labels.ravel()

    glm_pred_weights = (1 - (glm_labels == -100)*1).ravel()
    detection_pred_weights = (1 - (detection_labels == -100)*1).ravel()
    keep_pred_weights = ((detection_labels==0)*1).ravel()
    error_pred_weights = ((detection_labels==1)*1).ravel()
    insert_pred_weights = ((detection_labels==2)*1).ravel()
    glm_accuracy = accuracy_metric.compute(references=glm_labels, predictions=glm_pred_ids, sample_weight=glm_pred_weights)['accuracy']
    detection_accuracy = accuracy_metric.compute(references=detection_labels, predictions=detection_pred_ids, sample_weight=detection_pred_weights)['accuracy']
    keep_accuracy = accuracy_metric.compute(references=detection_labels, predictions=detection_pred_ids, sample_weight=keep_pred_weights)['accuracy']
    error_accuracy = accuracy_metric.compute(references=detection_labels, predictions=detection_pred_ids, sample_weight=error_pred_weights)['accuracy']
    insert_accuracy = accuracy_metric.compute(references=detection_labels, predictions=detection_pred_ids, sample_weight=insert_pred_weights)['accuracy']
    geometric_accuracy = (glm_accuracy*detection_accuracy)**0.5
    detection_geometric_accuracy = ( keep_accuracy*error_accuracy*insert_accuracy ) ** (1/3)
    general_accuary = (glm_accuracy*detection_geometric_accuracy)**0.5

    return {'general_accuracy': general_accuary, 
            'geometric_accuracy': geometric_accuracy, 
            'glm_accuracy': glm_accuracy, 'detection_accuracy': detection_accuracy, 
            'detection_geometric_accuracy': detection_geometric_accuracy,
            'keep_accuracy': keep_accuracy, 'error_accuracy': error_accuracy, 'insert_accuracy': insert_accuracy}


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    lm_logits, detection_logits = logits
    detection_pred = torch.argmax(detection_logits, dim=-1)
    glm_pred = torch.argmax(lm_logits, dim=-1)
    return (glm_pred, detection_pred), labels


class CorrectionGLMTrainer(TrainerBeta):
    def __init__(self, args, settings, model, dataset):
        super(CorrectionGLMTrainer, self).__init__(args, settings, model, dataset)

        # dataset processor
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.pretrained_model, trust_remote_code=True)
        self.data_processor = GLMDataProcessor(tokenizer=self.tokenizer, args=args, config=settings)
        self.data_collator = DataCollatorForGLMGEC(tokenizer=self.tokenizer, loss_ignore_id=self.data_processor._loss_ignore_id)
        # self.generation_data_collator = DataCollatorForGLMGECGeneration(tokenizer=self.tokenizer, loss_ignore_id=self.data_processor._loss_ignore_id)

        # data config
        self.overwrite_cache = False

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
            fp16=False,
            group_by_length=False,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            warmup_steps=settings.warmup_steps,
            weight_decay=settings.weight_decay,
            # lr_scheduler_type=settings.lr_scheduler,
            metric_for_best_model='eval_general_accuracy',
        )
        logger.info(self.training_args)
    
    def _get_train_preprocess_function(self, for_validation=False):
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
            
    def dataset_transform(self):
        """
        Transform (transformers) dataset to meet the requirements of model
        """
        columns = self.dataset['train'].column_names
        reserved_columns = ['input_ids', 'target_ids', 'position_ids', 'detection_labels', 'source_length', 'prefix_length']
        removed_columns = []
        for column in columns:
            if column not in reserved_columns:
                removed_columns.append(column)
        self.train_dataset = self.dataset['train'].map(
            self._get_train_preprocess_function(for_validation=False),
            batched=True,
            remove_columns=removed_columns,
            load_from_cache_file=not self.overwrite_cache,
            # cache_file_name=os.path.join(self.settings.cache_dir, 'train_dataset.arrow'),
            desc="Running preprocessing on train dataset",
        )

        self.valid_dataset = self.dataset['valid'].map(
            self._get_train_preprocess_function(for_validation=True),
            batched=True,
            remove_columns=removed_columns,
            load_from_cache_file=not self.overwrite_cache,
            # cache_file_name=os.path.join(self.settings.cache_dir, 'valid_dataset.cache'),
            desc="Running preprocessing on valid dataset"
        )

    def test_dataset_transform(self, use_valid_set_as_test=False):
        dataset_key = 'valid' if use_valid_set_as_test else 'test'
        self.test_dataset = self.dataset[dataset_key].map(
            self._get_detection_preprocess_function(),
            batched=True,
            remove_columns=[],
            load_from_cache_file=not self.overwrite_cache,
            # cache_file_name=os.path.join(self.settings.cache_dir, 'test_dataset.cache'),
            desc="Running detection preprocessing on test dataset"
        )

    def do_train(self):
        """
        do training, using transformers trainer.
        """
        training_args = self.training_args

        # dataset
        self.dataset_transform()

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
            compute_metrics=compute_metrics_3_label if self.settings.num_labels==3 else compute_metrics_2_label,
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


    def do_infer(self):
        """
        do infer on inputs.
        """
        ## detection
        self.test_dataset_transform()
        self.model.to(self.args.device)
        self.model.eval()
        test_data_loader = DataLoader(self.test_dataset, batch_size=32, collate_fn=self.data_collator)
        edit_label_predictions = []
        logger.info("Error Detection:")
        keep_label_id = self.data_processor.edit_label_map['$KEEP']
        for test_data in tqdm(test_data_loader):
            test_data.to(self.args.device)
            detection_logits = self.model(**test_data).logits[1]
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

        # def _transform_to_predict_first_phase(examples):
        #     processed = {}
        #     for i in range(len(examples['input_ids'])):
        #         src_tokens = examples['input_ids'][i][examples['prefix_prompt_length'][i]:examples['prefix_length'][i]]
        #         result = self.data_processor.convert_detected_sentence_to_infer_example(src_tokens, examples['detection_predictions'][i])
        #         if not processed:
        #             for key in result:
        #                 processed[key] = []
        #         for key in result:
        #             processed[key].append(result[key])
        #     return processed
        
        logger.info("Using Detection results to generate dataset for mask generation:")
        logger.info(f"Change the model to GLMforConditionalGeneration, load GLM model by {self.args.save_dir}")
        self.construct_inference_model(os.path.join(self.args.load, 'pytorch_model.bin'))
        self.model.to(self.args.device)
        self.model.eval()

        # test_dataset_for_generation = self.test_dataset.map(
        #     _transform_to_predict_first_phase,
        #     batched=True,
        #     remove_columns=[],
        #     load_from_cache_file=not self.overwrite_cache,
        #     desc="Running first generation preprocessing on test dataset"
        # )
        test_dataset_for_generation = []
        reserved_columns = ['id', 'text', 'label']
        for i, item in tqdm(enumerate(self.test_dataset)):
            src_tokens = item['input_ids'][item['prefix_prompt_length']:item['prefix_length']]
            result = self.data_processor.convert_detected_sentence_to_infer_example(src_tokens, item['detection_predictions'])
            for key in reserved_columns:
                if key in item:
                    result[key] = item[key]
            test_dataset_for_generation.append(result)

        max_gen_len = self.settings.max_new_tokens

        ## save result one by one
        f = open(os.path.join(self.args.save_dir, 'real-time-results.json'), 'w')
        results = []

        logging.info("GLM model do the generation:")
        for test_data in tqdm(test_dataset_for_generation):
            mask_positions = []
            for i, id in enumerate(test_data['input_ids']):
                if id == self.tokenizer.mask_token_id:
                    mask_positions.append(i)
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

    def do_eval(self):
        """
        do infer on eval dataset and output midium ponents.
        """
        ## detection
        self.test_dataset_transform(use_valid_set_as_test=True)
        self.model.to(self.args.device)
        self.model.eval()
        test_data_loader = DataLoader(self.test_dataset, batch_size=32, collate_fn=self.data_collator)
        edit_label_predictions = []
        logger.info("Error Detection:")
        keep_label_id = self.data_processor.edit_label_map['$KEEP']
        for test_data in tqdm(test_data_loader):
            test_data.to(self.args.device)
            detection_logits = self.model(**test_data).logits[1]
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
        
        logger.info("Using Detection results to generate dataset for mask generation:")
        logger.info(f"Change the model to GLMforConditionalGeneration, load GLM model by {self.args.save_dir}")
        self.construct_inference_model(os.path.join(self.args.load, 'pytorch_model.bin'))
        self.model.to(self.args.device)
        self.model.eval()

        # test_dataset_for_generation = self.test_dataset.map(
        #     _transform_to_predict_first_phase,
        #     batched=True,
        #     remove_columns=[],
        #     load_from_cache_file=not self.overwrite_cache,
        #     desc="Running first generation preprocessing on test dataset"
        # )
        test_dataset_for_generation = []
        reserved_columns = ['id', 'text', 'label']
        for i, item in tqdm(enumerate(self.test_dataset)):
            src_tokens = item['input_ids'][item['prefix_prompt_length']:item['prefix_length']]
            result = self.data_processor.convert_detected_sentence_to_infer_example(src_tokens, item['detection_predictions'])
            for key in reserved_columns:
                if key in item:
                    result[key] = item[key]
            test_dataset_for_generation.append(result)

        max_gen_len = self.settings.max_new_tokens

        ## save result one by one
        f = open(os.path.join(self.args.save_dir, 'real-time-results.json'), 'w')
        results = []

        logging.info("GLM model do the generation:")
        for i, test_data in tqdm(enumerate(test_dataset_for_generation)):
            mask_positions = []
            original_item = self.test_dataset[i]
            src_tokens = original_item['input_ids'][original_item['prefix_prompt_length']:original_item['prefix_length']]
            detections = original_item['detection_predictions']
            for i, id in enumerate(test_data['input_ids']):
                if id == self.tokenizer.mask_token_id:
                    mask_positions.append(i)
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
            
            assert 'label' in test_data, "Validation set without targets(labels)"
            res = {
                'id':test_data['id'], 'src': test_data['text'], 'tgt': test_data['label'], 'predict': generation_res,
                'src_tokens': self.tokenizer.convert_ids_to_tokens(src_tokens),
                'detections': [self.data_processor.edit_label_id_map[i] for i in detections],
                'predict_tokens': self.tokenizer.convert_ids_to_tokens(source_part),
            }
            if 'other_labels' in test_data:
                res['other_labels'] = test_data['other_labels']
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

        f.close()

        return results

    def save(self, save_dir):
        if self.args.task_mode == 'train':
            logger.info("Saving manual training settings in config.py...")
            config_file = os.path.join(save_dir, 'presettings.json')
            config_dict = {}
            for key in self.settings:
                content = self.settings[key]
                if type(content) in [str, int, float, bool, list]:
                    config_dict[key] = content
                else:
                    config_dict[key] = str(content)
            config_dict['model'] = self.args.model
            config_dict['dataset'] = self.args.dataset
            config_dict['task_mode'] = self.args.task_mode
            config_dict['load'] = self.args.load

            json.dump(config_dict, open(config_file, 'w'), indent=4, ensure_ascii=False)
        else:    
            raise NotImplementedError()

    def load(self, save_dir):
        if self.args.task_mode == 'train':
            logger.info("Load complete model for CorrectionGLM to continue training.")
            path = os.path.join(save_dir, 'pytorch_model.bin')
            self.model.load_state_dict(torch.load(path))
        else:
            if save_dir[-1] == '/':
                save_dir = save_dir[:-1]
            checkpoint_root_dir =  os.path.dirname(save_dir)
            config_file = os.path.join(checkpoint_root_dir, 'presettings.json')
            logger.info(f"Load manual training settings in checkpoint, include {self.settings.load_config_keys}, from {config_file}")
            presettings = json.load(open(config_file))
            for key in self.settings.load_config_keys:
                self.settings[key] = presettings[key]
            
            logger.info("Load complete model for CorrectionGLM.")
            path = os.path.join(save_dir, 'pytorch_model.bin')
            self.model.load_state_dict(torch.load(path))
            
