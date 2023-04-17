import os
import logging
import sys
import re
from tqdm import tqdm

import torch
import transformers
from transformers import (AutoTokenizer, BertForTokenClassification,
                          DataCollatorForTokenClassification, HfArgumentParser, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, TrainerCallback, AutoModelForSeq2SeqLM,BartForConditionalGeneration)
from transformers.trainer_utils import is_main_process
import numpy as np
from opencc import OpenCC

# Metric
from rouge import Rouge 
rouge = Rouge()

from trainers.base import Trainer
from utils.MuCGEC import DataTrainingArguments, ModelArguments, load_json

logger = logging.getLogger(__name__)

length_map={'lcsts':'30','csl':'50','adgen':'128', 'gec': '100'}

class Seq2SeqModelTrainer(Trainer):
    def __init__(self, args, config, model) -> None:
        super(Seq2SeqModelTrainer, self).__init__(args, config, model)
        self.args = args
        self.config = config
        self.model = model
        self.args_list = [
            '--model_name_or_path', config.pretrained_model,
            '--train_file', os.path.join(config.data_dir, 'train.json'),
            '--validation_file', os.path.join(config.data_dir, 'valid.json'),
            '--test_file',os.path.join(config.data_dir, 'test.json'),
            '--output_dir', args.save_dir,
            '--per_device_train_batch_size', str(config.batch_size),
            '--per_device_eval_batch_size', str(config.batch_size),
            '--overwrite_output_dir',
            '--max_source_length=100',
            '--val_max_target_length=' + length_map[config.task],
            '--seed', str(args.seed),
            '--num_train_epochs', str(config.epoch),
            '--evaluation_strategy','epoch',
            '--learning_rate', str(config.lr),
            '--fp16',
            '--label_smoothing_factor', '0.1',
            '--predict_with_generate',
            '--group_by_length',
            '--gradient_accumulation_steps', str(config.gradient_accumulation_steps),
            '--lr_scheduler', config.lr_scheduler,
            # '--save_strategy', 'no',
            '--save_strategy', config.save_strategy,
            '--do_train',
            '--do_eval',
            '--do_predict',
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
        self.model_args, self.data_args, self.training_args = parser.parse_args_into_dataclasses(self.args_list)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logger.setLevel(logging.INFO if is_main_process(self.training_args.local_rank) else logging.WARN)

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if is_main_process(self.training_args.local_rank):
            transformers.utils.logging.set_verbosity_info()
        logger.info("Training/evaluation parameters %s", self.training_args)

        # Data collator
        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        # Initialize our Trainer
        self.trainer = None
        # self.trainer = Seq2SeqTrainer(
        #     model=self.model,
        #     args=self.training_args,
        #     train_dataset=None,
        #     eval_dataset=None,
        #     tokenizer=self.tokenizer,
        #     data_collator=self.data_collator,
        # # if training_args.predict_with_generate else None,
        # )


    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        while '' in preds:
            idx=preds.index('')
            preds[idx]='。'

        return preds, labels

    def get_compute_metrics(self):

        def compute(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            if self.data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
            scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
            for key in scores:
                scores[key]=scores[key]['f']*100

            result=scores

            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result
        
        return compute

    def do_train(self, train_dataset, val_dataset):
        # Re-Initialize our Trainer with dataset
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=val_dataset if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.get_compute_metrics(),
        # if training_args.predict_with_generate else None,
        )
        train_result = self.trainer.train()
          # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        ## TODO : metric return is right?
        return metrics[0]["rouge-l"]

    def do_test(self, dataset, mode="VAL"):
        """
        do test process, based on ids of every token.
        """
        assert self.trainer is not None
        if mode == 'VAL':
            predictions, labels, metrics = self.trainer.predict(dataset, metric_key_prefix="predict")
        elif mode == 'TEST':
            predictions, labels, metrics = self.trainer.predict(dataset, metric_key_prefix="predict")
            test_preds = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True,
            )
            test_preds = ["".join(pred.strip().split()) for pred in test_preds]
            output_test_preds_file = os.path.join(self.args.save_dir, "infer_result.txt")
            with open(output_test_preds_file, "w",encoding='UTF-8') as writer:
                writer.write("\n".join(test_preds))
            # return [{"predict": item} for item in test_preds]
        else:
            raise NotImplementedError()
        
        ## TODO : metric return is right?
        return metrics[0]

    def do_infer(self, test_dataloader, mode="INFER"):
        """
        do infer or test process, based on decoded text of sentences' tokens.
        """
        self.model.to(self.args.device)
        inp_max_len = self.config.text_cut
        num_ret_seqs = 1
        beam = 5
        results = []
        cc = OpenCC("t2s")
        if mode == "TEST":
            for batch in tqdm(test_dataloader):
                texts = batch['texts']
                batch_size = len(texts["input_ids"])
                token_len = texts["input_ids"].size(1)
                with torch.no_grad():
                    generated_ids = self.model.generate(batch['texts']['input_ids'].cuda(),
                                            attention_mask=batch['texts']['attention_mask'].cuda(),
                                            num_beams=beam, num_return_sequences=num_ret_seqs, max_length=inp_max_len)
                _out = self.tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True)

                batch_outs = [ batch["raw_texts"][oidx] if len(self.tokenizer.encode(batch["raw_texts"][oidx])) > token_len \
                    else _out[oidx] for oidx in range(batch_size) ]

                for i in range(batch_size):
                    results.append({'src': batch["raw_texts"][i], 'tgt': batch["raw_labels"][i], 'predict': cc.convert(batch_outs[i]).replace(" ", "")})
            
            with open(os.path.join(self.args.save_dir, "predict.output"), 'w') as outf:
                for res in results:
                    outf.write(res['predict'] + "\n")
            return results
        else:
            for batch in tqdm(test_dataloader):
                texts = batch['texts']
                batch_size = len(texts["input_ids"])
                token_len = texts["input_ids"].size(1)
                with torch.no_grad():
                    generated_ids = self.model.generate(batch['texts']['input_ids'].cuda(),
                                            attention_mask=batch['texts']['attention_mask'].cuda(),
                                            num_beams=beam, num_return_sequences=num_ret_seqs, max_length=inp_max_len)
                _out = self.tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True)

                batch_outs = [ batch["raw_texts"][oidx] if len(self.tokenizer.encode(batch["raw_texts"][oidx])) > token_len \
                    else _out[oidx] for oidx in range(batch_size) ]

                for i in range(batch_size):
                    results.append({'id': batch["ids"][i], 'src': batch["raw_texts"][i], 'predict': cc.convert(batch_outs[i]).replace(" ", "")})
            
            return results

    def save(self, save_dir):
        self.trainer.save_model()

    def load(self, save_dir):
        print(f"*********Loading checkpoint from {save_dir}**********")
        if os.path.exists(os.path.join(save_dir, 'pytorch_model.bin')):
            self.model.from_pretrained(save_dir)
        else:
            paths = os.listdir(save_dir)
            for path in paths:
                max_steps = 0
                if path[:10] == "checkpoint":
                    _, steps = path.split('-')
                    steps = eval(steps)
                    if steps > max_steps:
                        max_steps = steps
            self.model.from_pretrained(os.path.join(save_dir, f"checkpoint-{max_steps}"))
