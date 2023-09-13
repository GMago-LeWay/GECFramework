import logging
import numpy as np
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


from utils.tools import dict_to_str, get_time
from metrics import Metrics

from trainers.base import Trainer

logger = logging.getLogger(__name__)

class MaskLMTrain(Trainer):
    def __init__(self, args, config, model):
        super(MaskLMTrain, self).__init__(args, config, model)
        self.args = args
        self.config = config
        self.model = model
        self.model.to(self.args.device)
        self.criterion = nn.NLLLoss()
        self.metrics = Metrics().get_metrics(config.metrics)

    def do_train(self, train_dataloader, val_dataloader):
        # OPTIMIZER: finetune Bert Parameters.
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

        optimizer = optim.Adam(optimizer_grouped_parameters)

        # SCHEDULER
        scheduler = ReduceLROnPlateau(optimizer,
                    mode=self.config.scheduler_mode,
                    factor=self.config.scheduler_factor, patience=self.config.scheduler_patience, verbose=True)
        # initilize results
        epochs = 0
        valid_num, best_valid_num = 0, 0
        min_or_max = self.config.scheduler_mode
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while epochs < self.config.max_epochs: 
            epochs += 1
            # train
            train_loss = []
            steps = 0
            with tqdm(train_dataloader) as td:
                for batch_data in td:
                    steps += 1
                    self.model.train()
                    labels = batch_data['labels'].to(self.args.device)
                    texts = batch_data['texts'].to(self.args.device)

                    if labels["input_ids"].shape[1] != texts["input_ids"].shape[1]:
                        logger.info("Warning: In steps %d at epoch %d, texts len (%d) is not equal to labels len (%d)." % (steps, epochs, texts["input_ids"].shape[1], labels["input_ids"].shape[1]))
                        continue
                    
                    # clear gradient
                    optimizer.zero_grad()
                    # forward
                    outputs = self.model(texts=texts)
                    # compute loss
                    loss = self.criterion((outputs*texts["attention_mask"].unsqueeze(-1)).reshape(-1, outputs.shape[-1]),
                                           labels["input_ids"].reshape(-1))
                    # backward
                    loss.backward()
                    # update
                    optimizer.step()
                    # store results
                    train_loss.append(loss.item())

                    if steps == len(train_dataloader) or (self.config.eval_step and steps % self.config.eval_step == 0):
                        valid_num += 1
                        # calc data of training
                        train_loss_avg = np.mean(train_loss)
                        train_loss = []
                        # pred, true = torch.cat(y_pred), torch.cat(y_true)
                        train_results = {"train_loss": train_loss_avg}

                        # validation
                        val_results = self.do_test(val_dataloader, mode="VAL")
                        cur_valid = val_results[self.config.KeyEval]

                        # scheduler step
                        scheduler.step(cur_valid)
                        # save best model
                        isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
                        # save best model
                        if isBetter:
                            # save model
                            best_valid, best_valid_num = cur_valid, valid_num
                            self.save(self.args.save_dir)

                        # print info
                        logger.info(get_time() + "TRAIN-(%s) (Epoch %d | Step %d | Valid %d | NoImprovement %d) >> %s" % 
                        (self.args.model, epochs, steps, valid_num, valid_num - best_valid_num, dict_to_str(train_results)))

                        # early stop
                        if valid_num - best_valid_num >= self.config.early_stop:
                            return best_valid
        
        return best_valid


    def do_test(self, dataloader, mode="VAL"):
        """
        do test process, based on ids of every token.
        """
        self.model.eval()
        eval_loss = 0.0
        sentences_ids = []
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    labels = batch_data['labels'].to(self.args.device)
                    texts = batch_data['texts'].to(self.args.device)
                    raw_texts = batch_data['raw_texts']
                    raw_labels = batch_data['raw_labels']
                    
                    log_likelihood = self.model(texts=texts)
                    loss = self.criterion((log_likelihood*texts["attention_mask"].unsqueeze(-1)).reshape(-1, log_likelihood.shape[-1]),
                                                            labels["input_ids"].reshape(-1))

                    eval_loss += loss.item()

                    prediction_tokens = torch.argmax(log_likelihood, dim=-1)
                    batch_size = prediction_tokens.shape[0]
                    for i in range(batch_size):
                        valid_len = texts["attention_mask"][i, :].sum()
                        span_left = self.config.tokenize_style[0]
                        span_right = valid_len+self.config.tokenize_style[1]
                        src = texts["input_ids"][i, span_left:span_right].tolist()
                        tgt = labels["input_ids"][i, span_left:span_right].tolist()
                        predict = prediction_tokens[i, span_left:span_right].tolist()
                        sentences_ids.append([src, tgt, predict])


            eval_loss = eval_loss / len(dataloader)

        # metric calc
        eval_results = self.metrics(sentences_ids)
        eval_results["loss"] = round(eval_loss, 4)

        logger.info(get_time() + "%s-(%s) >> %s" % (mode, self.args.model, dict_to_str(eval_results)))
        return eval_results


    def do_infer(self, dataloader, mode="INFER"):
        """
        do infer or test process, based on decoded text of sentences' tokens.
        """
        self.model.eval()
        results = []
        for batch_data in tqdm(dataloader):        
            texts = batch_data['texts'].to(self.args.device)
            prediction = self.model(texts=texts)
            output = torch.argmax(prediction, dim=-1)

            batch_size = len(output)
            for i in range(batch_size):
                valid_len = texts["attention_mask"][i, :].sum().item()
                span_left = self.config.tokenize_style[0]
                span_right = valid_len+self.config.tokenize_style[1]
                src = texts["input_ids"][i, span_left:span_right].tolist()
                predict = output[i, span_left:span_right].tolist()
                
                predict_text = self.model.decode_predicted_sentence(batch_data['raw_texts'][i], src, predict)
                if mode=="TEST":
                    results.append({"src": batch_data['raw_texts'][i], "predict": predict_text, "tgt": batch_data["raw_labels"][i]})
                elif mode=="INFER":
                    results.append({"src": batch_data['raw_texts'][i], "predict": predict_text})
                else:
                    raise NotImplementedError()


        return results

    def save(self, save_dir):
        save_path = os.path.join(save_dir, 'model.pth')
        torch.save(self.model.cpu().state_dict(), save_path)
        self.model.to(self.args.device)

    def load(self, save_dir):
        save_path = os.path.join(save_dir, 'model.pth')
        self.model.load_state_dict(torch.load(save_path))
