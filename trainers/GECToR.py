import json
import os
import logging

import torch
import numpy as np
from torch.cuda.random import device_count
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tensorboardX import SummaryWriter

from metrics import GECMetric

from dataset_provider.GECToR import DatasetCTC
from trainers.base import Trainer

logger = logging.getLogger(__name__)
USE_TENSORBOARD = True
if USE_TENSORBOARD:
    writer = SummaryWriter(log_dir='./tensorboard')

def build_optimizer_and_scheduler(args, model, t_total, freeze_embedding=False):
    module = (
        model.module if hasattr(model, "module") else model
    )

    if freeze_embedding:
        embedding_name_list = ('embeddings.word_embeddings.weight',
                               'embeddings.position_embeddings.weight',
                               'embeddings.token_type_embeddings.weight')
        for named_para in model.named_parameters():
            named_para[1].requires_grad = False if named_para[
                                                       0] in embedding_name_list else True

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    model_params = list(module.named_parameters())
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model_params
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            0.01
    }, {
        'params':
            [p for n, p in model_params if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * t_total),
        num_training_steps=t_total
    )

    return optimizer, scheduler

class GECToRTrainer(Trainer):
    def __init__(self,
                 args,
                 config,
                 model,
                 _loss_ignore_id=-100,
                 _keep_id_in_ctag=1):
        self.global_args = args
        self.args = config
        self.model = model
        self.device = args.device
        self.epochs = self.args.epochs
        self._loss_ignore_id = _loss_ignore_id
        self._keep_id_in_ctag = _keep_id_in_ctag

        # get vocab information
        with open(os.path.join(config.ctc_vocab_dir, config.correct_tags_file), "r") as fp:
            vocab_szie = len(fp.read().strip().split("\n"))
        config.correct_vocab_size = vocab_szie
        logger.info(f"Correct Tag Num: {vocab_szie}")

        # get dataset config (by initialize an empty dataset)
        empty_dataset = DatasetCTC(in_model_dir=self.args.pretrained_model,
                            src_texts=[],
                            trg_texts=[],
                            max_seq_len=self.args.text_cut,
                            ctc_label_vocab_dir=self.args.ctc_vocab_dir,
                            correct_tags_file=self.args.correct_tags_file,
                            detect_tags_file=self.args.detect_tags_file,
                            _loss_ignore_id=-100)
        self.id2label = empty_dataset.id2ctag

    def do_train(self, train_dataloader, val_dataloader):
        self.t_total = len(train_dataloader) * self.epochs
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.getfloat("train", "learning_rate"))
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(self.args, self.model, self.t_total)
        # Train
        global_step = 1
        best_d_f1 = 0.
        best_c_f1 = 0.
        self.model.zero_grad()
        eval_step = self.args.eval_step  # 每多少个step打印损失及进行验证
        for epoch in range(1, self.epochs + 1):
            for step, batch_data in enumerate(tqdm(train_dataloader)):
                self.model.train()
                for k, v in batch_data.items():
                    batch_data[k] = v.to(self.device)
                detect_labels = batch_data["d_tags"]
                correct_labels = batch_data["c_tags"]
                # 训练过程可能有些许数据出错，跳过
                try:
                    output = self.model(batch_data["input_ids"],
                                        batch_data["attention_mask"],
                                        batch_data["token_type_ids"],
                                        detect_labels,
                                        correct_labels)
                except Exception as e:
                    logger.error('ignore training step error!!')
                    logger.exception(e)
                    continue
                batch_loss = output["loss"]
                batch_loss = batch_loss.mean()
                batch_detect_loss = output["detect_loss"]
                batch_correct_loss = output["correct_loss"]
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{}/{} step:{}/{} detect_loss:{:.6f} correct_loss:{:.6f} loss:{:.6f}'.format(
                    epoch, self.args.epochs, global_step, self.t_total,
                    batch_detect_loss.item(), batch_correct_loss.item(), batch_loss.item()))

                global_step += 1
                if self.args.use_tensorboard:
                    writer.add_scalar('data/detect_loss', batch_detect_loss.item(), global_step)
                    writer.add_scalar('data/correct_loss', batch_correct_loss.item(), global_step)
                    writer.add_scalar('data/loss', batch_loss.item(), global_step)

                do_val = False
                if eval_step:
                    if global_step % eval_step == 0:
                        do_val = True
                else:         # eval_step = None, do validation every epoch
                    if step == len(train_dataloader) - 1:
                        do_val = True
                
                if do_val:
                    dev_metric = self.do_test(val_dataloader, mode='VAL')
                    d_f1 = dev_metric["d_f1"]
                    c_f1 = dev_metric["c_f1"]
                    logger.info("【dev】 loss:{:.6f} d_precision:{:.4f} d_recall:{:.4f} "
                                "d_f1:{:.4f} c_precision:{:.4f} c_recall:{:.4f} c_f1:{:.4f}".format(
                        dev_metric["loss"], dev_metric["d_precision"], dev_metric["d_recall"],
                        dev_metric["d_f1"], dev_metric["c_precision"], dev_metric["c_recall"],
                        dev_metric["c_f1"]
                    ))
                    if c_f1 > best_c_f1:
                        best_d_f1 = d_f1
                        best_c_f1 = c_f1
                        logger.info("【best】 detect_f1:{:.6f} correct_f1:{:.6f}".format(
                            d_f1, c_f1
                        ))
                        if self.args.use_tensorboard:
                            writer.add_scalar("detect_f1", d_f1)
                            writer.add_scalar("correct_f1", c_f1)

                        # take care of model distributed / parallel training
                        model_to_save = (
                            self.model.module if hasattr(self.model, "module") else self.model
                        )
                        output_dir = self.global_args.save_dir
                        logger.info('Saving model checkpoint to {}'.format(output_dir))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        torch.save(model_to_save.state_dict(),
                                   os.path.join(output_dir, '{}_model.pt'.format(self.args.name)))
        return best_c_f1

    def do_test(self, dataloader, mode="VAL"):
        """
        do test process, based on ids of every token(or shallow results).
        return Dict[str,value] metrics.
        The mode is a marker and does not decide test process. In some situations, TEST mode can save results.
        """
        self.model.eval()
        preds, gold_labels, src = [], [], []
        losses = 0.
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                for k, v in batch_data.items():
                    batch_data[k] = v.to(self.device)
                detect_labels = batch_data["d_tags"]
                correct_labels = batch_data["c_tags"]
                output = self.model(batch_data["input_ids"],
                                    batch_data["attention_mask"],
                                    batch_data["token_type_ids"],
                                    detect_labels,
                                    correct_labels)
                batch_loss = output["loss"]
                batch_gold = correct_labels.view(-1).cpu().numpy()
                correct_output = output["correct_outputs"]
                batch_pred = torch.argmax(correct_output, dim=-1).view(-1).cpu().numpy()
                batch_src = batch_data['input_ids'].view(-1).cpu().numpy()

                seq_true_idx = np.argwhere(batch_gold != self._loss_ignore_id)  # 获取非pad部分的标签
                batch_gold = batch_gold[seq_true_idx].squeeze()
                batch_pred = batch_pred[seq_true_idx].squeeze()
                batch_src = batch_src[seq_true_idx].squeeze()

                src += list(batch_src)
                gold_labels += list(batch_gold)
                preds += list(batch_pred)
                losses += batch_loss.item()
        "因为输出和输入空间不一样，所以计算指标要对应输出空间，原字符对应输出空间的keep"
        src = [self._keep_id_in_ctag] * len(src)

        (d_precision, d_recall, d_f1), (c_precision, c_recall, c_f1) = GECMetric.ctc_f1(
            src_texts=[src], trg_texts=[gold_labels], pred_texts=[preds])
        result = {
            "loss": losses,
            "c_precision": c_precision,
            "c_recall": c_recall,
            "c_f1": c_f1,
            "d_precision": d_precision,
            "d_recall": d_recall,
            "d_f1": d_f1,
        }
        return result

    def do_infer(self, dataloader, mode="INFER"):
        """
        do infer or test process, based on decoded text of sentences' tokens. This function give final results.
        TEST mode means the data has label. if possible, print metrics.
        INFER mode means the data does not have label.
        return json results.
        """
        self.model.eval()
        if mode == 'INFER':
            result = []
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    texts = batch['raw_texts']
                    for idx, raw_text in enumerate(texts):
                        inputs = self.model.tokenizer(raw_text, return_batch=True)
                        text = [i for i in raw_text]
                        real_length = 1 + len(raw_text)
                        input_ids = torch.LongTensor(inputs["input_ids"]).to(self.device)
                        real_lenth = input_ids
                        attention_mask = torch.LongTensor(inputs["attention_mask"]).to(self.device)
                        token_type_ids = torch.LongTensor(inputs["token_type_ids"]).to(self.device)
                        output = self.model(input_ids, attention_mask, token_type_ids)
                        correct_outputs = output["correct_outputs"]
                        correct_outputs = correct_outputs.detach().cpu().numpy()
                        detect_outputs = output["detect_outputs"]
                        detect_outputs = detect_outputs.detach().cpu().numpy()
                        detect_outputs = np.argmax(detect_outputs, axis=-1).squeeze()[:real_length]
                        correct_outputs = np.argmax(correct_outputs, axis=-1).squeeze()[:real_length]
                        # print(detect_outputs)
                        # print(correct_outputs)
                        res = {}
                        pre_text = []
                        for d, c, t in zip(detect_outputs, correct_outputs, ["始"] + text):
                            clabel = self.id2label[c]
                            if "APPEND" in clabel:
                                pre_text.append(clabel)
                                insert = clabel.split("_")[-1]
                                pre_text.append(insert)
                            elif "DELETE" in clabel:
                                continue
                            elif "$REPLACE" in clabel:
                                replace = clabel.split("_")[-1]
                                pre_text.append(replace)
                            else:
                                pre_text.append(t)
                        res["src"] = "".join(text)
                        res["predict"] = "".join(pre_text)[1:]
                        res["id"] = batch['ids'][idx]
                        result.append(res)
            return result
        else:
            raise NotImplementedError()
    
    def load(self, save_dir=None):
        default_path = os.path.join(self.global_args.save_dir, '{}_model.pt'.format(self.args.name))
        self.model.load_state_dict(torch.load(default_path))
