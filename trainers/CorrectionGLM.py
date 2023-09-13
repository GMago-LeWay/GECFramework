from typing import Optional, Union
from tqdm import tqdm
import os
import logging
import numpy as np
import torch
from dataclasses import dataclass, field
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from trainers.base import TrainerBeta
from dataset_provider.CorrectionGLM import GLMDataProcessor

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # model_name_or_path: str = field(
    #     default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    # )
    # ptuning_checkpoint: str = field(
    #     default=None, metadata={"help": "Path to p-tuning v2 checkpoints"}
    # )
    # config_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    # )
    # tokenizer_name: Optional[str] = field(
    #     default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    # )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    # max_source_length: Optional[int] = field(
    #     default=128,
    #     metadata={
    #         "help": (
    #             "The maximum total input sequence length after tokenization. Sequences longer "
    #             "than this will be truncated, sequences shorter will be padded."
    #         )
    #     },
    # )
    # max_target_length: Optional[int] = field(
    #     default=32,
    #     metadata={
    #         "help": (
    #             "The maximum total sequence length for target text after tokenization. Sequences longer "
    #             "than this will be truncated, sequences shorter will be padded."
    #         )
    #     },
    # )
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
    # ignore_pad_token_for_loss: bool = field(
    #     default=True,
    #     metadata={
    #         "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
    #     },
    # )


@dataclass
class DataCollatorForGLMGEC:
    """
    Data collator that will dynamically pad the inputs.
    Candidate masks will be computed to indicate which tokens are candidates.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
    """

    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
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
        batch = {
            'input_ids': torch.LongTensor(batch_input_ids),
            'target_ids': torch.LongTensor(batch_target_ids),
            'position_ids': torch.LongTensor(batch_position_ids),
            'detection_labels': torch.LongTensor(batch_detection_label_ids),
            'attention_mask': torch.LongTensor(np.array(batch_attention_masks)).unsqueeze(1),
            'source_length': torch.LongTensor([item['source_length'] for item in features]),
            'prefix_length': torch.LongTensor([item['prefix_length'] for item in features]),
        }
        return batch


class CorrectionGLMTrainer(TrainerBeta):
    def __init__(self, args, settings, model, dataset):
        super(CorrectionGLMTrainer, self).__init__(args, settings, model, dataset)

        # dataset processor
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.pretrained_model, trust_remote_code=True)
        self.data_processor = GLMDataProcessor(tokenizer=self.tokenizer, args=args, config=settings)
        self.data_collator = DataCollatorForGLMGEC(tokenizer=self.tokenizer, padding=True, loss_ignore_id=self.data_processor._loss_ignore_id)
        self.marker_map = {
            ',': '，',
            ';': '；',
            ':': '：',
            '(': '（',
            ')': '）',
            '?': '？',
            '!': '！',
        }

        # some config
        self.text_cut = self.settings.text_cut

        # initialize config for transformers
        self.args_list = [
            '--do_train',
            '--do_eval',
            # '--do_predict',
            '--output_dir', args.save_dir,
            '--remove_unused_columns', False,
            '--per_device_train_batch_size', str(settings.batch_size),
            '--per_device_eval_batch_size', str(settings.batch_size),
            '--overwrite_output_dir',
            # '--max_source_length', '100',
            '--seed', str(args.seed),
            '--num_train_epochs', str(settings.epoch),
            # '--evaluation_strategy', 'epoch',
            '--eval_steps', str(settings.eval_step),
            # '--save_strategy', settings.save_strategy,
            '--save_steps', str(settings.eval_step),
            '--logging_steps', '1',
            '--learning_rate', str(settings.lr),
            # '--fp16',
            # '--predict_with_generate',
            # '--group_by_length',
            '--gradient_accumulation_steps', str(settings.gradient_accumulation_steps),
            '--warmup_steps', str(settings.warmup_steps),
            '--weight_decay', str(settings.weight_decay),
            '--lr_scheduler', settings.lr_scheduler,
        ]
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        self.model_args, self.data_args, self.training_args = parser.parse_args_into_dataclasses(self.args_list)
        logger.info(self.model_args)
        logger.info(self.data_args)
        logger.info(self.training_args)
    
    def _get_train_preprocess_function(self, for_validation=False):
        def _preprocess(examples):
            processed = {}
            for i in range(len(examples['text'])):
                if for_validation:
                    src, tgt = examples['text'][i], examples['label'][i]
                else:
                    src, tgt = examples['text'][i][:self.text_cut], examples['label'][i][:self.text_cut]
                result = self.data_processor.convert_gec_sentence_pair_to_example(src, tgt, 512 if for_validation else self.text_cut)
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
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Running preprocessing on train dataset",
        )

        self.valid_dataset = self.dataset['valid'].map(
            self._get_train_preprocess_function(for_validation=True),
            batched=True,
            remove_columns=removed_columns,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Running preprocessing on valid dataset"
        )

        self.test_dataset = self.dataset['test'].map(
            self._get_detection_preprocess_function(),
            batched=True,
            remove_columns=[],
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc="Running detection preprocessing on test dataset"
        )

    def _get_metrics_compute_function(self):
        def metrics(eval_predictions):
            print(eval_predictions)
            return {'loss': eval_predictions['loss']}
        return metrics

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
            # compute_metrics=self._get_metrics_compute_function(),
        )
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

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

    def do_infer(self):
        """
        do infer on inputs.
        """
        raise NotImplementedError()

    def save(self, save_dir):
        raise NotImplementedError()

    def load(self, save_dir):
        raise NotImplementedError()
