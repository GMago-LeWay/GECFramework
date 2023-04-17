"""
@File : train.py
@Description: 模型训练、推理等功能
@Author: bin.chen
@Contact: LebesgueIntigrade@163.com
@Time: 2021/9/1
@IDE: Pycharm Professional
@REFERENCE: NLLLoss:https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
            BCELoss:https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
"""
import os
import torch
import platform
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm

from utils.tools import dict_to_str, get_time
from metrics import Metrics
from trainers.base import Trainer

logger = logging.getLogger(__name__)

class SoftMaskedBertTrainer(Trainer):
    def __init__(self, args, config, model):
        super(SoftMaskedBertTrainer, self).__init__(args, config, model)
        self.config = config
        self.args = args
        self.model = model

        self.detector_criterion = nn.BCELoss()  # 检测器部分的损失，Binary CrossEntropy
        self.criterion = nn.NLLLoss()  # 整个模型的损失，Negative Loglikelihood

        self.gama = config.gamma
        # 论文里两个模块loss的加权系数，论文里也就说了个大概，大于0.5，因为纠错器部分的学习更困难也更重要些

        # 初始化模型保存点
        self.start_epoch = 0  # 初始化开始训练的轮数，在断点续训时要用到

        self.device = args.device  # 计算设备

        self.metrics = Metrics().get_metrics(config.metrics)

    def do_train(self, train_dataloader, val_dataloader):
        """
        训练模型
        Args:
            resume: 是否继续训练
        """
        # detector_optimizer = torch.optim.Adam(model.detector_model.parameters(), lr=self.config.lr)  # 检测器的优化器
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(self.model.language_model.named_parameters()) if self.config.language_model else []

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        model_params_other = [p for n, p in list(self.model.named_parameters()) if 'language_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.config.weight_decay_bert, 'lr': self.config.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.config.learning_rate_bert},
            {'params': model_params_other, 'weight_decay': self.config.weight_decay_other, 'lr': self.config.learning_rate_other}
        ]
        
        optimizer = torch.optim.Adam(optimizer_grouped_parameters)  # 优化器
        valid_num, best_valid_num = 0, 0
        min_or_max = self.config.scheduler_mode
        best_valid = 1e8 if min_or_max == 'min' else 0

        train_loss = []

        for epoch in range(self.start_epoch, self.config.max_epochs):
            detector_correct = 0  # 检测器准确数
            corrector_correct = 0  # 纠错器准确数
            total_loss = 0  # 总loss
            num_data = 0  # 总数据量，字符粒度
            steps = 0
            error_steps = 0
            for batch_data in tqdm(train_dataloader):
                steps += 1
                self.model.train()
                batch_inp_ids, batch_out_ids, batch_mask = batch_data['texts']['input_ids'], batch_data['labels']['input_ids'], batch_data['texts']['attention_mask']
                
                batch_labels = (batch_inp_ids != batch_out_ids).long().to(self.device)
                batch_inp_ids = batch_inp_ids.to(self.device)
                batch_out_ids = batch_out_ids.to(self.device)
                batch_mask = batch_mask.to(self.device)

                prob, out = self.model(batch_inp_ids, batch_mask)
                
                # loss without padding
                detector_loss = self.detector_criterion(prob.squeeze()*batch_mask, batch_labels.float())
                model_loss = self.criterion((out*batch_mask.unsqueeze(-1)).reshape(-1, out.shape[-1]), batch_out_ids.reshape(-1))
                loss = self.gama * model_loss + (1 - self.gama) * detector_loss  # 联合loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                prob = torch.round(prob)
                detector_correct = detector_correct + sum(
                    [(prob.squeeze() * batch_mask).reshape(-1)[i].equal((batch_labels * batch_mask).reshape(-1)[i])
                     for i in range(len(prob.reshape(-1)))])

                output = torch.argmax(out.detach(), dim=-1)

                corrector_correct = corrector_correct + sum(
                    [(output*batch_mask).reshape(-1)[j].equal((batch_out_ids*batch_mask).reshape(-1)[j])
                     for j in range(len(output.reshape(-1)))])
                # 计算准确率，padding部分均视为正确，下同
                total_loss += loss.item()
                num_data += sum([len(m) for m in batch_mask])
                if steps == len(train_dataloader) or (self.config.eval_step and steps % self.config.eval_step == 0):
                    valid_num += 1
                    # calc data of training
                    train_results = {"loss": total_loss / (steps), "detector_accuracy": detector_correct / num_data, "corrector_accuracy": corrector_correct / num_data}

                    # validation
                    val_results = self.do_test(val_dataloader, mode="VAL")
                    cur_valid = val_results[self.config.KeyEval]

                    # save best model
                    isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
                    # save best model
                    if isBetter:
                        # save model
                        best_valid, best_valid_num = cur_valid, valid_num
                        self.save(self.args.save_dir)

                    # print info
                    logger.info(get_time() + "TRAIN-(%s) (Epoch %d | Step %d | Valid %d | NoImprovement %d) >> %s" % 
                    (self.args.model, epoch, steps, valid_num, valid_num - best_valid_num, dict_to_str(train_results)))

                    # early stop
                    if valid_num - best_valid_num >= self.config.early_stop:
                        return best_valid

            if error_steps != 0:
                logger.info(get_time() + "Warning, there are %d steps when error occurred. These data were not used in training." % error_steps)
                    
        return best_valid


    def do_test(self, dataloader, mode="VAL"):
        self.model.eval()  # 推理模式

        detector_correct = 0  # 检测器准确数
        corrector_correct = 0  # 纠错器准确数
        num_data = 0  # 总数据量，字符粒度
        total_loss = 0  # 总loss

        sentences_ids = []
        for batch_data in tqdm(dataloader):
            batch_inp_ids, batch_out_ids, batch_mask = batch_data['texts']['input_ids'], batch_data['labels']['input_ids'], batch_data['texts']['attention_mask']
            batch_size = batch_inp_ids.shape[0]
            batch_labels = (batch_inp_ids != batch_out_ids).long().to(self.device)
            batch_inp_ids = batch_inp_ids.to(self.device)
            batch_out_ids = batch_out_ids.to(self.device)  # 选择计算设备，下同
            batch_mask = batch_mask.to(self.device)

            prob, out = self.model(batch_inp_ids, batch_mask)

            detector_loss = self.detector_criterion(prob.squeeze() * batch_mask, batch_labels.float())
            model_loss = self.criterion((out * batch_mask.unsqueeze(-1)).reshape(-1, out.shape[-1]),
                                        batch_out_ids.reshape(-1))
            loss = self.gama * model_loss + (1 - self.gama) * detector_loss  # 联合loss
            prob = torch.round(prob)
            detector_correct = detector_correct + sum(
                [(prob.squeeze() * batch_mask).reshape(-1)[i].equal((batch_labels * batch_mask).reshape(-1)[i])
                 for i in range(len(prob.reshape(-1)))])
            # 检测器准确数

            output = torch.argmax(out, dim=-1)
            corrector_correct = corrector_correct + sum(
                [(output * batch_mask).reshape(-1)[j].equal((batch_out_ids * batch_mask).reshape(-1)[j])
                 for j in range(len(output.reshape(-1)))])
            # 模型准确数
            # predict.extend(model.decode(output, batch_mask))
            for i in range(batch_size):
                valid_len = batch_mask[i, :].sum()
                span_left = self.config.tokenize_style[0]
                span_right = valid_len+self.config.tokenize_style[1]
                src = batch_inp_ids[i, span_left:span_right].tolist()
                tgt = batch_out_ids[i, span_left:span_right].tolist()
                predict = output[i, span_left:span_right].tolist()
                sentences_ids.append([src, tgt, predict])

            total_loss += loss.item()
            num_data += sum([len(m) for m in batch_mask])

        eval_results = {'loss': total_loss / len(dataloader), 'detector_accuracy': detector_correct / num_data, 'corrector_accuracy': corrector_correct / num_data}

        metric_results = self.metrics(sentences_ids)
        eval_results = {**eval_results, **metric_results}
        logger.info(get_time() + "%s-(%s) >> %s" % (mode, self.args.model, dict_to_str(eval_results)))
        return eval_results
    

    def do_infer(self, dataloader):
        self.model.eval()
        results = []
        for batch_data in tqdm(dataloader):        
            batch_inp_ids, batch_mask = batch_data['texts']['input_ids'], batch_data['texts']['attention_mask']
            batch_size = batch_inp_ids.shape[0]
            batch_inp_ids = batch_inp_ids.to(self.device)
            batch_mask = batch_mask.to(self.device)
            prob, out = self.model(batch_inp_ids, batch_mask)
            output = torch.argmax(out, dim=-1)

            for i in range(batch_size):
                valid_len = batch_mask[i, :].sum()
                span_left = self.config.tokenize_style[0]
                span_right = valid_len+self.config.tokenize_style[1]
                src = batch_inp_ids[i, span_left:span_right].tolist()
                predict = output[i, span_left:span_right].tolist()
                
                predict_text = self.model.decode_predicted_sentence(batch_data['raw_texts'][i], src, predict)
                results.append(predict_text)
        return results

    def save(self, save_dir):
        save_path = os.path.join(save_dir, 'model.pth')
        torch.save(self.model.cpu().state_dict(), save_path)
        self.model.to(self.args.device)

    def load(self, save_dir):
        save_path = os.path.join(save_dir, 'model.pth')
        self.model.load_state_dict(torch.load(save_path))
