import json
import os
import logging

import torch
from torch.nn import MSELoss
import numpy as np
from torch.cuda.random import device_count
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tensorboardX import SummaryWriter
import platform

plat = platform.system().lower()

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

## TODO: 完善模型特定数据集的解耦
class GECToRDatasetProvider:
    def __init__(self, dataset_map, tokenizer, args=None, config=None) -> None:
        self.args = args
        self.config = config
        self.tokenizer = tokenizer      # lazy init from model
        self.processed_dir = os.path.join(config.data_dir, 'GECToR')
        self.dataset = dataset_map
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def split_multi_append(self):
        '''
        multi append sentence can be split into multiple sentences.
        '''
        train_list, val_list, test_list = self.dataset['train'], self.dataset['valid'], self.dataset['test']
        new_train_data_file = os.path.join(self.processed_dir, 'train.json')
        new_valid_data_file = os.path.join(self.processed_dir, 'valid.json')
        new_test_data_file = os.path.join(self.processed_dir, 'test.json')

        def split(data_item: dict):
            src_text, tgt_text = '始' + data_item['text'], '始' + data_item['label']
            id = data_item['id'] if 'id' in data_item else None
            src_text_list = list(src_text)
            src_text_edit_list = DatasetCTC.char_edit_list(src_text, tgt_text)
            edit_num = [len(item) for item in src_text_edit_list]
            max_edit = max(edit_num)
            max_generate = 3
            max_error_num = min(max_edit, max_generate)

            split_sample = []
            if max_error_num <= 1:
                if id != None:
                    split_sample.append({'id': f"{id}_0", 'text': ''.join(src_text_list)[1:], 'label': tgt_text[1:]})
                else:
                    split_sample.append({'text': ''.join(src_text_list)[1:], 'label': tgt_text[1:]})
            else:
                for i in range(max_error_num):
                    if id != None:
                        split_sample.append({'id': f"{id}_{i}", 'text': ''.join(src_text_list)[1:], 'label': tgt_text[1:]})
                    else:
                        split_sample.append({'text': ''.join(src_text_list)[1:], 'label': tgt_text[1:]})
                    ## if this is not the last step of generating, apply edit and re-generate
                    if i < max_error_num - 1:
                        for char_idx in range(len(src_text_list)):
                            if src_text_edit_list[char_idx]:
                                edit_op = src_text_edit_list[char_idx].pop(0)
                                ops = edit_op.split('_')
                                if ops[0] == '$DELETE':
                                    src_text_list[char_idx] = ''
                                elif ops[0] == '$REPLACE':
                                    src_text_list[char_idx] = ops[1]
                                elif ops[0] == '$APPEND':
                                    src_text_list[char_idx] += ops[1]
            return split_sample
        
        new_train_list, new_val_list = [], []
        for item in tqdm(train_list):
            new_train_list.extend(split(item))
        for item in tqdm(val_list):
            new_val_list.extend(split(item))
        with open(new_train_data_file, 'w') as f:
            print(f"Origin GECToR Train Data {len(train_list)} -> After processing {len(new_train_list)}")
            json.dump(new_train_list, f, ensure_ascii=False, indent=4)
        with open(new_valid_data_file, 'w') as f:
            print(f"Origin GECToR Valid Data {len(val_list)} -> After processing {len(new_val_list)}")
            json.dump(new_val_list, f, ensure_ascii=False, indent=4)
        with open(new_test_data_file, 'w') as f:
            print(f"Test Data will be remained.")
            json.dump(test_list, f, ensure_ascii=False, indent=4)

    def get_train_val_dataloader(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train, val, test = self.dataset['train'], self.dataset['valid'], self.dataset[test]

        ## cache config check
        current_config = {
            'use_multi_append': self.config.use_multi_append,
            'pretrained_model': self.config.pretrained_model,
            'ctc_vocab_dir': self.config.ctc_vocab_dir,
            'detect_tags_file': self.config.detect_tags_file,
            'correct_tags_file': self.config.correct_tags_file,
            'reward_estimate': self.config.reward_estimate,
        }
        cache_config = None
        cache_config_file = os.path.join(self.processed_dir, 'config.json')
        if os.path.exists(cache_config_file):
            with open(cache_config_file) as f:
                cache_config = json.load(f)
        # ## cache check
        # train_cache_dir = os.path.join(self.processed_dir, 'train.pt')
        # valid_cache_dir = os.path.join(self.processed_dir, 'valid.pt')
        # if cache_config == current_config:
        #     logger.info(f'loading cached dataset...')
        #     train_dataset = torch.load(train_cache_dir, map_location='cpu')
        #     dev_dataset = torch.load(valid_cache_dir, map_location='cpu')
        # else:
        logger.info(f'construct dataset and cache...')
        train_dataset = DatasetCTC(in_model_dir=self.config.pretrained_model,
                            src_texts=[item['text'] for item in train],
                            trg_texts=[item['label'] for item in train],
                            max_seq_len=self.config.text_cut,
                            ctc_label_vocab_dir=self.config.ctc_vocab_dir,
                            correct_tags_file=self.config.correct_tags_file,
                            detect_tags_file=self.config.detect_tags_file,
                            _loss_ignore_id=-100)
        
        dev_dataset = DatasetCTC(in_model_dir=self.config.pretrained_model,
                            src_texts=[item['text'] for item in val],
                            trg_texts=[item['label'] for item in val],
                            max_seq_len=self.config.text_cut,
                            ctc_label_vocab_dir=self.config.ctc_vocab_dir,
                            correct_tags_file=self.config.correct_tags_file,
                            detect_tags_file=self.config.detect_tags_file,
                            _loss_ignore_id=-100)
            
            # # cache dataset
            # torch.save(train_dataset, os.path.join(self.processed_dir, 'train.pt'))
            # torch.save(dev_dataset, os.path.join(self.processed_dir, 'valid.pt'))
            # with open(cache_config_file, 'w') as f:
            #     json.dump(current_config, f, ensure_ascii=False, indent=4)
            # logger.info(f"Cached dataset in condition of {current_config}")
        
        logger.info("训练集数据：{}条 验证集数据：{}条".format(len(train_dataset), len(dev_dataset)))

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2)
        dev_loader = DataLoader(dev_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2)
        
        return train_loader, dev_loader, dev_loader


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
        self.model.to(self.device)
        self.epochs = self.args.epochs
        self._loss_ignore_id = _loss_ignore_id
        self._keep_id_in_ctag = _keep_id_in_ctag

        # get vocab information
        try:
            self._start_vocab_id = self.model.tokenizer.vocab['[START]']
        except KeyError:
            self._start_vocab_id = self.model.tokenizer.vocab['[unused1]']
        with open(os.path.join(config.ctc_vocab_dir, config.correct_tags_file), "r") as fp:
            vocab_szie = len(fp.read().strip().split("\n"))
        config.correct_vocab_size = vocab_szie
        logger.info(f"Correct Tag Num: {vocab_szie}")

        self.id2dtag, self.d_tag2id, self.id2ctag, self.c_tag2id = self.load_label_dict(
            config.ctc_vocab_dir, config.detect_tags_file, config.correct_tags_file)
        self.id2label = self.id2ctag

        self.infer_prune = False
        self.index_prune = []
        self.id2label_prune = []
        if config.infer_tags != None and config.infer_tags != config.correct_tags_file:
            self.infer_prune = True
            id2dtag, d_tag2id, id2ctag, c_tag2id = self.load_label_dict(
                config.ctc_vocab_dir, config.detect_tags_file, config.infer_tags)
            print(f"Correct Tags pruning while inferring. Raw tags num: {len(self.id2ctag)}, Target tags num: {len(id2ctag)}")
            for tag in id2ctag:
                if tag in self.c_tag2id:
                    self.index_prune.append(self.c_tag2id[tag])
                    self.id2label_prune.append(tag)
            print(f"After Pruning, Tags num {len(self.index_prune)}")


    def load_label_dict(self, ctc_label_vocab_dir: str, detect_tags_file: str, correct_tags_file: str):
        dtag_fp = os.path.join(ctc_label_vocab_dir, detect_tags_file)
        ctag_fp = os.path.join(ctc_label_vocab_dir, correct_tags_file)

        if plat == "windows":
            dtag_fp = dtag_fp.replace("\\", "/")
            ctag_fp = ctag_fp.replace("\\", "/")
        id2dtag = [line.strip() for line in open(dtag_fp, encoding='utf8')]
        d_tag2id = {v: i for i, v in enumerate(id2dtag)}

        id2ctag = [line.strip() for line in open(ctag_fp, encoding='utf8')]
        c_tag2id = {v: i for i, v in enumerate(id2ctag)}
        logger.info('d_tag num: {}, d_tags:{}'.format(len(id2dtag), d_tag2id))
        return id2dtag, d_tag2id, id2ctag, c_tag2id
    

    def do_train(self, train_dataloader, val_dataloader):
        self.t_total = len(train_dataloader) * self.epochs
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.getfloat("train", "learning_rate"))
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(self.args, self.model, self.t_total)
        # RL initialize
        reward_loss = MSELoss()
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
                    if type(v) == list:
                        continue
                    batch_data[k] = v.to(self.device)
                detect_labels = batch_data["d_tags"]
                correct_labels = batch_data["c_tags"]
                batch_size = len(batch_data)
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
                ## RL
                if self.args.reward_estimate:
                    scores = self._get_output_text_score(output=output, batch_texts=batch_data['raw_texts'], batch_labels=batch_data['raw_labels'])['score']
                    reward = [score[self.args.reward_metric] for score in scores]
                    reward = torch.tensor(reward).to(self.global_args.device)
                    batch_reward_loss = reward_loss(output["reward_outputs"].view(-1), reward)
                    batch_loss += (self.args.reward_loss_weight * batch_reward_loss)

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                if global_step % 100 == 0:
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
        
        # a final save
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        output_dir = self.global_args.save_dir
        logger.info('Saving model checkpoint to {}'.format(output_dir))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model_to_save.state_dict(),
                    os.path.join(output_dir, '{}_model_final.pt'.format(self.args.name)))
        return best_c_f1
    
    def _get_output_text_score(self, output, batch_texts, batch_labels):
        results = []
        for idx, raw_text in enumerate(batch_texts):
            text = [i for i in "始"+raw_text]
            real_length = 1 + len(text)
            correct_outputs = output["correct_outputs"][idx]
            correct_outputs = correct_outputs.detach().cpu().numpy()
            detect_outputs = output["detect_outputs"][idx]
            detect_outputs = detect_outputs.detach().cpu().numpy()
            detect_outputs = np.argmax(detect_outputs, axis=-1).squeeze()[1:real_length]
            correct_outputs = np.argmax(correct_outputs, axis=-1).squeeze()[1:real_length]
            pre_text = []
            for d, c, t in zip(detect_outputs, correct_outputs, text):
                if self.infer_prune:
                    clabel = self.id2label_prune[c]
                else:
                    clabel = self.id2label[c]
                if "$APPEND" in clabel:
                    pre_text.append(t)
                    insert = clabel.split("_")[-1]
                    pre_text.append(insert)
                elif "$DELETE" in clabel:
                    continue
                elif "$REPLACE" in clabel:
                    replace = clabel.split("_")[-1]
                    pre_text.append(replace)
                else:
                    pre_text.append(t)
            results.append("".join(pre_text)[1:])

        scores = []
        for i in range(len(batch_texts)):
            scores.append(GECMetric.final_f1_score(src_texts=[batch_texts[i]], trg_texts=[batch_labels[i]], pred_texts=[results[i]]))
        return {'predicion': results, 'score': scores}


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
                    if type(v) == list:
                        continue
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
        result = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_size = len(batch['ids'])
                if 'raw_labels' in batch:
                    batch_results = [{"id": batch['ids'][i], "src": batch['raw_texts'][i], "tgt": batch['raw_labels'][i]} for i in range(batch_size)]
                else:
                    batch_results = [{"id": batch['ids'][i], "src": batch['raw_texts'][i]} for i in range(batch_size)]
                current_inputs = list(batch['raw_texts'])
                for _ in range(self.args.iteration):
                    for idx, raw_text in enumerate(current_inputs):
                        raw_text = '始' + raw_text
                        if len(raw_text) > 500:
                            batch_results[idx]["predict"] = raw_text[1:]
                            continue
                        inputs = self.model.tokenizer(raw_text, return_batch=True)
                        inputs['input_ids'][0][1] = self._start_vocab_id
                        text = [i for i in raw_text]
                        real_length = 1 + len(raw_text)
                        input_ids = torch.LongTensor(inputs["input_ids"]).to(self.device)
                        real_lenth = input_ids
                        attention_mask = torch.LongTensor(inputs["attention_mask"]).to(self.device)
                        token_type_ids = torch.LongTensor(inputs["token_type_ids"]).to(self.device)
                        output = self.model(input_ids, attention_mask, token_type_ids)
                        correct_outputs = output["correct_outputs"]
                        if self.infer_prune:
                            correct_outputs = torch.index_select(correct_outputs, dim=-1, index=torch.tensor(self.index_prune, dtype=int).to(self.device))
                        correct_outputs = correct_outputs.detach().cpu().numpy()
                        detect_outputs = output["detect_outputs"]
                        detect_outputs = detect_outputs.detach().cpu().numpy()
                        detect_outputs = np.argmax(detect_outputs, axis=-1).squeeze()[1:real_length]
                        correct_outputs = np.argmax(correct_outputs, axis=-1).squeeze()[1:real_length]
                        # print(detect_outputs)
                        # print(correct_outputs)
                        pre_text = []
                        for d, c, t in zip(detect_outputs, correct_outputs, text):
                            if self.infer_prune:
                                clabel = self.id2label_prune[c]
                            else:
                                clabel = self.id2label[c]
                            if self.args.fixed_length:
                                if "$REPLACE" in clabel:
                                    replace = clabel.split("_")[-1]
                                    pre_text.append(replace)
                                else:
                                    pre_text.append(t)
                            else:
                                if "$APPEND" in clabel:
                                    pre_text.append(t)
                                    insert = clabel.split("_")[-1]
                                    pre_text.append(insert)
                                elif "$DELETE" in clabel:
                                    continue
                                elif "$REPLACE" in clabel:
                                    replace = clabel.split("_")[-1]
                                    pre_text.append(replace)
                                else:
                                    pre_text.append(t)
                        batch_results[idx]["predict"] = "".join(pre_text)[1:]
                    ## refresh text with the predicted output and re-correct
                    current_inputs = [item['predict'] for item in batch_results]
                
                # iteration ends.
                result.extend(batch_results)
        return result
    
    def load(self, save_dir=None):
        default_path = os.path.join(save_dir, '{}_model.pt'.format(self.args.name))
        self.model.load_state_dict(torch.load(default_path, map_location='cpu'), strict=False)
        logger.info(f"Successfully load weights from {default_path}")
