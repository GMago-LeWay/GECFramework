from trainers.base import Trainer2

## Arguments

from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    ptuning_checkpoint: str = field(
        default=None, metadata={"help": "Path to p-tuning v2 checkpoints"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    quantization_bit: Optional[int] = field(
        default=None
    )
    pre_seq_len: Optional[int] = field(
        default=None
    )
    prefix_projection: bool = field(
        default=False
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    history_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the history of chat."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from models.ChatGLM import Seq2SeqTrainer

logger = logging.getLogger(__name__)


class ChatGLMTrainer(Trainer2):
    def __init__(self, args, config, model) -> None:
        self.args = args
        self.config = config
        assert model == None

        self.args_list = [
            "--do_train",
            "--train_file", os.path.join(config.data_dir, "train.json"),
            "--validation_file", os.path.join(config.data_dir, "valid.json"),
            "--test_file", os.path.join(config.data_dir, "test.json"),
            "--prompt_column", "text",
            "--response_column", "label",
            "--overwrite_cache",
            "--model_name_or_path", config.pretrained_model,
            "--output_dir", args.save_dir,
            "--overwrite_output_dir",
            "--max_source_length", str(config.max_source_length),
            "--max_target_length", str(config.max_target_length),
            "--per_device_train_batch_size", str(config.per_device_train_batch_size),
            "--per_device_eval_batch_size", str(config.per_device_eval_batch_size),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--predict_with_generate",
            "--num_train_epochs", str(config.num_train_epochs),
            "--logging_steps", str(config.logging_steps),
            "--save_steps", str(config.save_steps),
            "--learning_rate", str(config.learning_rate),
            "--pre_seq_len", str(config.pre_seq_len),
            "--quantization_bit", str(config.quantization_bit)
        ]

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=self.args_list)

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        # datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

        # Set seed before initializing model.
        set_seed(args.seed)

        # Load dataset
        raw_datasets = {}
        train_file = os.path.join(config.data_dir, 'train.json')
        validation_file = os.path.join(config.data_dir, 'valid.json')
        test_file = os.path.join(config.data_dir, 'test.json')

        with open(train_file) as f:
            raw_datasets['train'] = Dataset.from_list(json.load(f))
        with open(validation_file) as f:
            raw_datasets['validation'] = Dataset.from_list(json.load(f))
        with open(test_file) as f:
            raw_datasets["test"] = Dataset.from_list(json.load(f))

        # Load pretrained model and tokenizer
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection

        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

        if args.load is not None:
            # Evaluation
            # Loading extra state dict of prefix encoder
            model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
            prefix_state_dict = torch.load(os.path.join(args.load, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        else:
            model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

        if model_args.quantization_bit is not None:
            print(f"Quantized to {model_args.quantization_bit} bit")
            model = model.quantize(model_args.quantization_bit)
        if model_args.pre_seq_len is not None:
            # P-tuning v2
            model = model.half()
            model.transformer.prefix_encoder.float()
        else:
            # Finetune
            model = model.float()

        prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        train_column_names = raw_datasets["train"].column_names
        valid_column_names = raw_datasets["validation"].column_names
        test_column_names = raw_datasets["test"].column_names


        # Get the column names for input/target.
        prompt_column = data_args.prompt_column
        response_column = data_args.response_column
        history_column = data_args.history_column
        
        # Temporarily set max_target_length for training.
        max_target_length = data_args.max_target_length

        def preprocess_function_eval(examples):
            inputs, targets = [], []
            for i in range(len(examples[prompt_column])):
                if examples[prompt_column][i] and examples[response_column][i]:
                    query = examples[prompt_column][i]
                    if history_column is None or len(examples[history_column][i]) == 0:
                        prompt = query
                    else:
                        prompt = ""
                        history = examples[history_column][i]
                        for turn_idx, (old_query, response) in enumerate(history):
                            prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                    inputs.append(prompt)
                    targets.append(examples[response_column][i])

            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
            labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

            if data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        def preprocess_function_train(examples):
            max_seq_length = data_args.max_source_length + data_args.max_target_length

            model_inputs = {
                "input_ids": [],
                "labels": [],
            }
            for i in range(len(examples[prompt_column])):
                if examples[prompt_column][i] and examples[response_column][i]:
                    query, answer = examples[prompt_column][i], examples[response_column][i]

                    if history_column is None:
                        prompt = query
                    else:
                        prompt = ""
                        history = examples[history_column][i]
                        for turn_idx, (old_query, response) in enumerate(history):
                            prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                    prompt = prefix + prompt
                    a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                    b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                    if len(a_ids) > data_args.max_source_length - 1:
                        a_ids = a_ids[: data_args.max_source_length - 1]

                    if len(b_ids) > data_args.max_target_length - 2:
                        b_ids = b_ids[: data_args.max_target_length - 2]

                    input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                    context_length = input_ids.index(tokenizer.bos_token_id)
                    mask_position = context_length - 1
                    labels = [-100] * context_length + input_ids[mask_position+1:]
                    
                    pad_len = max_seq_length - len(input_ids)
                    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                    labels = labels + [tokenizer.pad_token_id] * pad_len
                    if data_args.ignore_pad_token_for_loss:
                        labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["labels"].append(labels)

            return model_inputs
        
        def print_dataset_example(example):
            print("input_ids",example["input_ids"])
            print("inputs", tokenizer.decode(example["input_ids"]))
            print("label_ids", example["labels"])
            print("labels", tokenizer.decode(example["labels"]))

        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    preprocess_function_train,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=train_column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
            print_dataset_example(train_dataset[0])

        # eval dataset
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=valid_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0])

        # predict dataset
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=test_column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0])
        
        ## save to self.
        self.model_args, self.data_args, self.training_args = model_args, data_args, training_args
        self.tokenizer = tokenizer
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.predict_dataset = predict_dataset

    
    def do_train(self, train_dataset, val_dataset):
        # Data collator
        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
            padding=False
        )

        # Metric
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            if self.data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            score_dict = {
                "rouge-1": [],
                "rouge-2": [],
                "rouge-l": [],
                "bleu-4": []
            }
            for pred, label in zip(decoded_preds, decoded_labels):
                hypothesis = list(jieba.cut(pred))
                reference = list(jieba.cut(label))
                rouge = Rouge()
                scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
                result = scores[0]
                
                for k, v in result.items():
                    score_dict[k].append(round(v["f"] * 100, 4))
                bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
                score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            for k, v in score_dict.items():
                score_dict[k] = float(np.mean(v))
            return score_dict

        # Override the decoding parameters of Seq2SeqTrainer
        self.training_args.generation_max_length = (
            self.training_args.generation_max_length
            if self.training_args.generation_max_length is not None
            else self.data_args.val_max_target_length
        )
        self.training_args.generation_num_beams = (
            self.data_args.num_beams if self.data_args.num_beams is not None else self.training_args.generation_num_beams
        )
        # Initialize our Trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if self.training_args.predict_with_generate else None,
            save_prefixencoder=self.model_args.pre_seq_len is not None
        )

        # Training
        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            # elif last_checkpoint is not None:
            #     checkpoint = last_checkpoint
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            # trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()

        return metrics

    def do_test(self, dataset, mode="VAL"):
        """
        do test process, based on ids of every token.
        """
        # Evaluation
        results = {}
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length + 1
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = self.trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=max_seq_length, temperature=0.95)
            max_eval_samples = self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(self.eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))

            self.trainer.log_metrics("eval", metrics)
            self.trainer.save_metrics("eval", metrics)
        return metrics

    def do_infer(self, dataset, mode="INFER"):
        """
        do infer or test process, based on decoded text of sentences' tokens.
        """
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length + 1
        logger.info("*** Predict ***")
        predict_results = self.trainer.predict(self.predict_dataset, metric_key_prefix="predict", max_length=max_seq_length, do_sample=True, top_p=0.7, temperature=0.95)
        metrics = predict_results.metrics
        max_predict_samples = (
            self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(self.predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(self.predict_dataset))

        self.trainer.log_metrics("predict", metrics)
        self.trainer.save_metrics("predict", metrics)

        results = []

        if self.trainer.is_world_process_zero():
            if self.training_args.predict_with_generate:
                predictions = self.tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                labels = self.tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                for p, l in zip(predictions, labels):
                    results.append({"tgt": l, "predict": p})
        return results

    def save(self, save_dir):
        raise NotImplementedError()

    def load(self, save_dir):
        raise NotImplementedError()
