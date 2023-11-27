import logging
import os
import json
import nltk
from typing import Dict
import numpy as np
import hashlib
import evaluate
import datasets
from datasets import Dataset
from trainers.base import TrainerBeta
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    BertTokenizer, BartTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

transformers.utils.move_cache('/data/liwei/cache/huggingface/')

class Seq2SeqBetaTrainer(TrainerBeta):
    def __init__(self, args, settings, model, dataset: Dict[str, Dataset]) -> None:
        super().__init__(args, settings, model, dataset)
        if 'bart-large-chinese' in settings.pretrained_model:
            self.tokenizer = BertTokenizer.from_pretrained(settings.pretrained_model)
        else:
            self.tokenizer = BartTokenizer.from_pretrained(settings.pretrained_model)

        # initialize config for transformers trainer
        self.training_args = Seq2SeqTrainingArguments(
            seed=args.seed,
            do_train=True,
            do_eval=True,
            do_predict=False,
            deepspeed=settings.ds_config,
            output_dir=args.save_dir,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            # include_inputs_for_metrics=True,
            label_names=[''],
            per_device_train_batch_size=settings.train_batch_size,
            per_device_eval_batch_size=settings.eval_batch_size,
            num_train_epochs=settings.epoch,
            evaluation_strategy='steps',
            eval_steps=settings.eval_step,
            predict_with_generate=settings.predict_with_generate,
            # eval_accumulation_steps=20,
            save_steps=settings.save_step,
            logging_steps=settings.logging_steps,
            # log_level='info',
            learning_rate=settings.lr,
            label_smoothing_factor=settings.label_smoothing_factor,
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
        if self.training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # other settings
        self.prefix = self.settings.source_prefix if self.settings.source_prefix is not None else ""

    def _hash_data_settings(self):
        pretrained_model = self.settings.pretrained_model
        prompt = str(self.settings.source_prefix)
        data_dir = self.settings.data_dir
        padding = self.settings.padding
        ignore_pad_token_for_loss = str(self.settings.ignore_pad_token_for_loss)
        settings_list = [pretrained_model, prompt, data_dir, padding, ignore_pad_token_for_loss]
        settings_str = '_'.join(settings_list)
        hash_value = hashlib.md5(settings_str.encode('utf-8')).hexdigest()
        hash_str = str(hash_value)
        logger.info(f'Hash Code Key of Dataset: {hash_str}')
        return hash_str
    
    def _get_data_cache_name(self, data_category: str):
        name = self.args.model.lower() + '_' + self.args.dataset.lower()
        file_name =  '_'.join([name, data_category, self._hash_data_settings()]) + ".arrow"
        return os.path.join(self.settings.cache_dir, file_name)
        
    def train_dataset_transform(self):
        """
        Transform (transformers) dataset to meet the requirements of model training.  
        The self.train_dataset and the self.valid_dataset will be transformed from self.dataset['train'] and ['valid'].  
        In load detection results mode, the corresponding results will be loaded.
        """
        tokenizer = self.tokenizer
        text_column = 'text'
        summary_column = 'label'
        max_source_length = self.settings.max_train_source_length
        max_target_length = self.settings.max_train_target_length
        padding = self.settings.padding
        def preprocess_function(examples):
            # remove pairs where at least one record is None

            inputs, targets = [], []
            for i in range(len(examples[text_column])):
                if examples[text_column][i] and examples[summary_column][i]:
                    inputs.append(examples[text_column][i])
                    targets.append(examples[summary_column][i])

            inputs = [self.prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

            # Tokenize targets with the `text_target` keyword argument
            labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length":
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        if self.args.task_mode == "train":
            logger.info("Preparing train dataset transform...")
            columns = self.dataset['train'].column_names
            self.train_dataset = self.dataset["train"].map(
                preprocess_function,
                batched=True,
                remove_columns=columns,
                load_from_cache_file=self.settings.load_cache,
                cache_file_name=self._get_data_cache_name("train"),
                desc="Running preprocessing on train dataset",
            )
        if self.args.task_mode in ["train", "eval"]:
            max_source_length = self.settings.max_eval_source_length
            max_target_length = self.settings.max_eval_target_length
            logger.info("Preparing validation dataset transform...")
            columns = self.dataset['valid'].column_names
            self.valid_dataset = self.dataset['valid'].map(
                preprocess_function,
                batched=True,
                remove_columns=columns,
                load_from_cache_file=self.settings.load_cache,
                cache_file_name=self._get_data_cache_name("valid"),
                desc="Running preprocessing on valid dataset"
            )

    def do_train(self):
        # Log on each process the small summary:
        training_args = self.training_args
        model = self.model
        tokenizer = self.tokenizer
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

        if 't5' in self.settings.pretrained_model and self.settings.source_prefix == '':
            logger.warning(
                "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
                "`--source_prefix 'summarize: ' `"
            )

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
        set_seed(training_args.seed)
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        self.train_dataset_transform()
        if 'token_type_ids' in self.train_dataset.column_names:
            self.train_dataset = self.train_dataset.remove_columns('token_type_ids')
            self.valid_dataset = self.valid_dataset.remove_columns('token_type_ids')

        if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

        # Data collator
        label_pad_token_id = -100 if self.settings.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        # Metric
        metric = evaluate.load("utils/rouge")

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

            return preds, labels

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            # Replace -100s used for padding as we can't decode them
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {k: round(v * 100, 4) for k, v in result.items()}
            prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            return result
        
        # Override the decoding parameters of Seq2SeqTrainer
        training_args.generation_max_length = (
            training_args.generation_max_length
            if training_args.generation_max_length is not None
            else self.settings.max_eval_target_length
        )
        training_args.generation_num_beams = (
            self.settings.num_beams if self.settings.num_beams is not None else training_args.generation_num_beams
        )

        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset if training_args.do_train else None,
            eval_dataset=self.valid_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            # max_train_samples = (
            #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            # )
            # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()


    def do_eval(self):
        return super().do_eval()
    
    def do_infer(self):
        return super().do_infer()

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
            if self.settings.use_lora:
                logger.info("Lora model should have been loaded at the model construction.")
                return
            logger.info("Load Seq2Seq model to continue training.")
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
            logger.info("Load Seq2Seq model.")
            path = os.path.join(save_dir, 'pytorch_model.bin')
            information = self.model.load_state_dict(torch.load(path))
            logger.info(information)
