import os
import json
import logging
import random
import argparse
from copy import deepcopy
import spacy
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from copy import copy
import traceback
import scipy.io as sio

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import HfArgumentParser


from transformers import AutoTokenizer

from config import Config
from config import MODEL_CORR_DATA, DATA_ROOT_DIR, DATA_DIR_NAME, MODEL_ROOT_DIR
from utils.tools import get_time, dict_to_str, is_chinese

from dataset_provider.FCGEC import JointDataset
from dataset_provider.FCGEC_transform import min_dist_opt
from dataset_provider.GECToR import DatasetCTC

from utils.MuCGEC import DataTrainingArguments, load_json, FullTokenizer, convert_to_unicode
from dataset_provider.FCGEC import TextWash, TaggerConverter, combine_insert_modify, convert_tagger2generator

logger = logging.getLogger(__name__)

class TextLabelDataset:
    def __init__(self, args=None, config=None) -> None:
        self.args = args
        self.config = config
        self.file = os.path.join(self.config.data_dir, 'data.json')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model) if 'pretrained_model' in config else None
        except:
            self.tokenizer = None
    
    def data(self):
        assert os.path.exists(self.file)
        with open(self.file, "r") as f:
            data = json.load(f)
        return data

    def train_val_test_data(self):
        train_data_file = os.path.join(self.config.data_dir, 'train.json')
        valid_data_file = os.path.join(self.config.data_dir, 'valid.json')
        test_data_file = os.path.join(self.config.data_dir, 'test.json')
        data_file = os.path.join(self.config.data_dir, 'data.json')
        if os.path.exists(train_data_file) and os.path.exists(valid_data_file) and os.path.exists(test_data_file):
            with open(train_data_file, 'r') as f:
                train_data = json.load(f)
            with open(valid_data_file, 'r') as f:
                valid_data = json.load(f)
            with open(test_data_file, 'r') as f:
                test_data = json.load(f)
            return train_data, valid_data, test_data
        elif os.path.exists(data_file):
            gross_data = self.data()
            test_num = int(self.config.test_percent * len(gross_data))
            val_num = int(self.config.valid_percent * len(gross_data))
            test = gross_data[-test_num:]
            # select data
            train = gross_data[:len(gross_data)-val_num-test_num]
            val = gross_data[len(gross_data)-val_num-test_num: -test_num]
            ## save
            with open(train_data_file, 'w') as f:
                json.dump(train, f, ensure_ascii=False, indent=4)
            with open(valid_data_file, 'w') as f:
                json.dump(val, f, ensure_ascii=False, indent=4)
            with open(test_data_file, 'w') as f:
                for i in range(len(test)):
                    test[i]['id'] = i
                json.dump(test, f, ensure_ascii=False, indent=4)
            return train, val, test    
        else:
            raise FileNotFoundError()

    
    def get_collate_fn(self, tokenizer=None, labeled=True):
        if labeled:
            def collate_fn(batch):
                has_id = "id" in batch[0]
                batch_size = len(batch)
                raw_texts = [batch[i]["text"] for i in range(batch_size)]
                raw_labels = [batch[i]["label"] for i in range(batch_size)]
                texts = tokenizer(raw_texts, padding=True, truncation=True, max_length=self.config.text_cut, return_tensors="pt")
                labels = tokenizer(raw_labels, padding=True, truncation=True, max_length=self.config.text_cut, return_tensors="pt")
                if has_id:
                    ids = [batch[i]["id"] for i in range(batch_size)]
                    return {'texts': texts, 'labels': labels, 'raw_texts': raw_texts, 'raw_labels': raw_labels, "ids": ids}
                return {'texts': texts, 'labels': labels, 'raw_texts': raw_texts, 'raw_labels': raw_labels}
            def collate_fn_text(batch):
                has_id = "id" in batch[0]
                batch_size = len(batch)
                raw_texts = [batch[i]["text"] for i in range(batch_size)]
                raw_labels = [batch[i]["label"] for i in range(batch_size)]
                if has_id:
                    ids = [batch[i]["id"] for i in range(batch_size)]
                    return {'raw_texts': raw_texts, 'raw_labels': raw_labels, "ids": ids}
                return {'raw_texts': raw_texts, 'raw_labels': raw_labels}
            if tokenizer == None:
                return collate_fn_text
            else:
                return collate_fn
        else:
            def collate_fn(batch):
                has_id = "id" in batch[0]
                batch_size = len(batch)
                raw_texts = [batch[i]["text"] for i in range(batch_size)]
                texts = tokenizer(raw_texts, padding=True, truncation=True, max_length=self.config.text_cut, return_tensors="pt")
                if has_id:
                    ids = [batch[i]["id"] for i in range(batch_size)]
                    return {'texts': texts, 'raw_texts': raw_texts, "ids": ids}
                return {'texts': texts, 'raw_texts': raw_texts}
            def collate_fn_text(batch):
                has_id = "id" in batch[0]
                batch_size = len(batch)
                raw_texts = [batch[i]["text"] for i in range(batch_size)]
                if has_id:
                    ids = [batch[i]["id"] for i in range(batch_size)]
                    return {'raw_texts': raw_texts, "ids": ids}
                return {'raw_texts': raw_texts}
            if tokenizer == None:
                return collate_fn_text
            else:
                return collate_fn

    def get_train_val_dataloader(self, tokenizer) -> tuple[DataLoader, DataLoader, DataLoader]:
        train, val, test = self.train_val_test_data()
        logger.info(get_time() + 'Total Train samples: %d, Total Valid samples: %d, Total Test samples: %d' % (len(train), len(val), len(test)))
        return DataLoader(train, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(tokenizer), drop_last=False), \
                DataLoader(val, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(tokenizer), drop_last=False), \
                DataLoader(test, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(tokenizer), drop_last=False)

    def get_test_dataloader(self, tokenizer=None) -> DataLoader:
        test_file = os.path.join(self.config.data_dir, 'test.json')
        assert os.path.exists(test_file)
        with open(test_file) as f:
            data = json.load(f)
        # filter
        # data = []
        # for item in gross_data:
        #     if len(item['text']) < 80 and len(item['label']) < 80:
        #         data.append(item)
        # data = random.sample(data, 10000)
        labeled = 'label' in data[0]
        return DataLoader(data, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(tokenizer=tokenizer, labeled=labeled), drop_last=False)

    def process_data_to_STG_Joint(self):
        joint_save_dir = os.path.join(self.config.data_dir, 'stg_joint')
        if not os.path.exists(joint_save_dir):
            os.makedirs(joint_save_dir)
        
        train_list, valid_list, test_list = self.train_val_test_data()

        data_list = {'train': train_list, 'valid': valid_list, 'test': test_list}

        model_config = Config(model='stgjoint', dataset='fangzhengdapei').get_config()
        check_tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_ROOT_DIR, "chinese-roberta-wwm-ext"))

        ## To check item for TaggerConvertor
        def _preprocess_gendata(ops: dict):
            '''
            Pre-tokenize modify labels and insert labels for convertor
            :param ops: operator (dict)
            :return: processed operator (dict)
            '''
            if 'Modify' not in ops.keys() and 'Insert' not in ops.keys():
                return ops
            nop = copy(ops)
            if 'Modify' in ops.keys():
                nmod = []
                for mod in nop['Modify']:
                    if isinstance(mod['label'], list):
                        labstr = mod['label'][0]
                    else:
                        labstr = mod['label']
                    mod['label_token'] = check_tokenizer.convert_tokens_to_ids(check_tokenizer.tokenize(labstr))
                    nmod.append(mod)
                nop['Modify'] = nmod
            if 'Insert' in ops.keys():
                nins = []
                for ins in nop['Insert']:
                    if isinstance(ins['label'], list):
                        labstr = ins['label'][0]
                    else:
                        labstr = ins['label']
                    ins['label_token'] = check_tokenizer.convert_tokens_to_ids(check_tokenizer.tokenize(labstr))
                    nins.append(ins)
                nop['Insert'] = nins
            return nop
        
        ## convert data, delete data with error for train and valid set.
        ## test data will be reserved.
        for split in data_list:
            Sentence = []
            Label = []   
            Ids = []
            exist_id = "id" in data_list[split][0]
            for item in tqdm(data_list[split]):   
                if split == 'test':
                    Sentence.append(item['text'])
                    Label.append('[]') 
                    if exist_id:
                        Ids.append(item['id'])
                    continue
                ## generate label
                token = check_tokenizer.tokenize(TextWash.punc_wash(item['text'])) 
                sent_recycle_len = len(check_tokenizer.convert_tokens_to_string(token).replace(" ", ""))    
                sent_wash_len = len(TextWash.punc_wash(item['text']))
                if sent_wash_len != sent_recycle_len:
                    continue
                try:
                    opt_edit = min_dist_opt(item['text'], item['label'])  
                    edit_label = [opt_edit]

                    ## Check TaggerConvertor
                    kwargs = {
                        'sentence' : TextWash.punc_wash(item['text']),
                        'ops' : _preprocess_gendata(opt_edit),
                        'token' : token
                    }
                    ## test process
                    tokens = ["[CLS]"] + token + ["[SEP]"]
                    tagger = TaggerConverter(model_config, auto=True, **kwargs)
                    label_comb = tagger.getlabel(types='dict')
                    comb_label = combine_insert_modify(label_comb['ins_label'], label_comb['mod_label'])
                    gen_token, gen_label = convert_tagger2generator(tokens, label_comb['tagger'], label_comb['mask_label'])

                    ## if no error occurred, the data item will be added.
                    Sentence.append(item['text'])
                    Label.append(json.dumps(edit_label, ensure_ascii=False))
                    if exist_id:
                        Ids.append(item['id'])
                except:
                    print("Error While Coverting: %s; %s" % (item['text'], item['label']))
            print(f"Data num {len(data_list[split])} -> {len(Sentence)}")
            pd.DataFrame({"Sentence": Sentence, "Label": Label}).to_csv(os.path.join(joint_save_dir, f'{split}.csv'), index=False, encoding='utf_8_sig')
            if exist_id:  # save ids to .id.json  
                assert len(Ids) == len(Sentence) == len(Label)
                with open(os.path.join(joint_save_dir, f'{split}.id.json'), 'w') as f:
                    json.dump(Ids, f, indent=4)

class MuCGECSeq2SeqDataset:  ## MuCGEC
    def __init__(self, args=None, config=None) -> None:
        self.args = args
        self.config = config
        length_map={'lcsts':'30','csl':'50','adgen':'128', 'gec': '100'}
        args_list = [
            '--train_file', config.train_file,
            '--validation_file', config.validation_file,
            '--test_file', config.test_file,
            '--max_source_length=100',
            '--val_max_target_length=' + length_map['gec'],   
        ]
        parser = HfArgumentParser(DataTrainingArguments)
        self.data_args = parser.parse_args(args_list)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model) if 'pretrained_model' in config else None
        self.padding = False

    def get_preprocess_function(self):
        def preprocess_function(examples):
            assert self.tokenizer is not None
            inputs = examples['text']
            targets = examples['label']
            model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=self.padding, truncation=True)

            # Setup the tokenizer for targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=self.data_args.val_max_target_length, padding=self.padding, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        return preprocess_function

    ## For model with transformers trainer, return dataset
    def train_val_test_data(self) -> tuple[Dataset, Dataset, Dataset]:
        datasets = {}
        data_files = {}
        if self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
        if self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
        if self.data_args.validation_file is not None:
            data_files["test"] = self.data_args.validation_file
        for key in data_files:
            datasets[key] = load_json(data_files[key])

        column_names = datasets["train"].column_names

        # load dataset
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if self.data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(self.data_args.max_train_samples))
        train_dataset = train_dataset.map(
            self.get_preprocess_function(),
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

        max_target_length = self.data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if self.data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(self.data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            self.get_preprocess_function(),
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )
        max_eval_num=30000
        if len(eval_dataset)>max_eval_num:
            eval_dataset=Dataset.from_dict(eval_dataset[:max_eval_num])
        print(len(eval_dataset))

        max_target_length = self.data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if self.data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(self.data_args.max_test_samples))
        test_dataset = test_dataset.map(
            self.get_preprocess_function(),
            batched=True,
            batch_size=32,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

        return train_dataset, eval_dataset, test_dataset

    def get_test_dataloader(self, tokenizer):
        with open(os.path.join(self.config.data_dir, 'test.json'), 'r') as f:
            data = json.load(f)
        def collate_fn(batch):
            has_label = "label" in batch[0]
            batch_size = len(batch)
            ids = [batch[i]["id"] for i in range(batch_size)]
            raw_texts = [batch[i]["text"] for i in range(batch_size)]
            texts = tokenizer(raw_texts, padding=True, truncation=True, max_length=self.config.text_cut, return_tensors="pt")
            if not has_label:
                return {'texts': texts, 'raw_texts': raw_texts, 'ids': ids}
            else:
                raw_labels = [batch[i]["label"] for i in range(batch_size)]
                labels = tokenizer(raw_labels, padding=True, truncation=True, max_length=self.config.text_cut, return_tensors="pt")
                return {'texts': texts, 'raw_texts': raw_texts, 'labels': labels, 'raw_labels': raw_labels, 'ids': ids}
        return DataLoader(data, batch_size=self.config.batch_size, collate_fn=collate_fn, drop_last=False)
        
    def process_raw_file(self):
        ### 训练集仓库中提供的训练数据；验证集、测试集为MuCGEC的原始数据的验证集与测试集
        train_file = os.path.join(self.config.data_dir, 'train/train.para')
        # valid_file = os.path.join(self.config.data_dir, 'valid/valid.para')
        assert os.path.exists(train_file), "This function need to be executed only in MuCGEC."
        with open(train_file, 'r') as f:
            train_data = f.readlines()
        new_train_data = []
        for item in train_data:
            src, tgt = item.split()
            src = src.strip()
            tgt = tgt.strip()
            new_train_data.append({"text": src, "label": tgt})

        valid_file = os.path.join(self.config.data_dir, 'origin', "MuCGEC_dev.txt")
        with open(valid_file, 'r') as f:
            valid_data = f.readlines()        
        new_val_data = []
        for item in valid_data:
            segments = item.split()
            item_id, src, tgt = eval(segments[0].strip()), segments[1].strip(), segments[2].strip()
            valid_data_item = {"id": item_id, "text": src, "label": tgt}
            for i in range(3, len(segments)):
                valid_data_item[f'label{i-1}'] = segments[i].strip()
            new_val_data.append(valid_data_item)

        test_file = os.path.join(self.config.data_dir, 'origin', "MuCGEC_test.txt")
        with open(test_file, 'r') as f:
            test_data = f.readlines()
        new_test_data = []
        for item in test_data:
            segments = item.split()
            item_id, src = eval(segments[0].strip()), segments[1].strip()
            new_test_data.append({"id": item_id, "text": src})
        
        with open(os.path.join(self.config.data_dir, 'train.json'), 'w') as f:
            json.dump(new_train_data, f, ensure_ascii=False, indent=4)
        with open(os.path.join(self.config.data_dir, 'valid.json'), 'w') as f:
            json.dump(new_val_data, f, ensure_ascii=False, indent=4)
        with open(os.path.join(self.config.data_dir, 'test.json'), 'w') as f:
            json.dump(new_test_data, f, ensure_ascii=False, indent=4)


class MuCGECEditDataset(TextLabelDataset):
    def __init__(self, args=None, config=None) -> None:
        super(MuCGECEditDataset, self).__init__(args, config)
        self.args = args
        self.config = config
        self.tokenizer = FullTokenizer(vocab_file=config.vocab_file, do_lower_case=False)
        

    def segment_bert(self, line):
        line = line.strip()
        origin_line = line
        line = line.replace(" ", "")
        line = convert_to_unicode(line)
        if not line:
            raise NotImplementedError()
        tokens = self.tokenizer.tokenize(line)
        return ' '.join(tokens)
    
    def preprocess_data_list(self, data_list, save_path):
        '''
        list item: {'text':..., 'label':...}
        '''
        ## Segment
        for i in range(len(data_list)):
            data_list[i]["text"] = self.segment_bert(data_list[i]["text"])
            data_list[i]["label"] = self.segment_bert(data_list[i]["label"])

        from dataset_provider.MuCGEC import convert_data_from_data_list
        convert_data_from_data_list(data_list, save_path, self.config.vocab_path, 5, False, 32)

    def preprocess_data(self):
        train_list, val_list, test_list = self.train_val_test_raw_data()
        ## Seq2Edit cache dir
        seq2edit_save_dir = os.path.join(self.config.data_dir, 'Seq2Edit')
        if not os.path.exists(seq2edit_save_dir):
            os.makedirs(seq2edit_save_dir)
        self.preprocess_data_list(train_list, os.path.join(seq2edit_save_dir, 'train.label'))
        self.preprocess_data_list(val_list, os.path.join(seq2edit_save_dir, 'valid.label'))

    
    def train_val_test_raw_data(self):
        return super(MuCGECEditDataset, self).train_val_test_data()

    def train_val_test_data(self):
        weights_name = self.config.pretrained_model
        from dataset_provider.MuCGEC import get_data_reader
        reader = get_data_reader(weights_name, self.config.max_len, skip_correct=bool(self.config.skip_correct),
                                skip_complex=self.config.skip_complex,
                                test_mode=False,
                                tag_strategy=self.config.tag_strategy,
                                tn_prob=self.config.tn_prob,
                                tp_prob=self.config.tp_prob)

        print("Data Reader is constructed")
        logger.info("Data Reader is constructed.")
        return [reader, self.config.train_set], [reader, self.config.dev_set], [reader, self.config.dev_set]
    
    def get_test_dataloader(self, tokenizer):
        print("...get test data loader...")
        with open(os.path.join(self.config.data_dir, 'test.json'), 'r') as f:
            data = json.load(f)
        def collate_fn(batch):
            has_label = "label" in batch[0]
            batch_size = len(batch)
            ids = [batch[i]["id"] for i in range(batch_size)]
            raw_texts = [self.segment_bert(batch[i]["text"]) for i in range(batch_size)]
            src_texts = [batch[i]["text"] for i in range(batch_size)]
            texts = tokenizer(raw_texts, padding=True, truncation=True, max_length=self.config.text_cut, return_tensors="pt")
            if not has_label:
                return {'texts': texts, 'raw_texts': raw_texts, 'src_texts': src_texts, 'ids': ids}
            else:
                raw_labels = [batch[i]["label"] for i in range(batch_size)]
                labels = tokenizer(raw_labels, padding=True, truncation=True, max_length=self.config.text_cut, return_tensors="pt")
                return {'texts': texts, 'raw_texts': raw_texts, 'src_texts': src_texts, 'labels': labels, 'raw_labels': raw_labels, 'ids': ids}
        return DataLoader(data, batch_size=self.config.batch_size, collate_fn=collate_fn, drop_last=False)


class GECToRDataset(TextLabelDataset):
    def __init__(self, args=None, config=None) -> None:
        super(GECToRDataset, self).__init__(args, config)
        self.tokenizer = None      # lazy init from model

    def _check_tokenizer(self):
        if self.tokenizer:
            return
        else:
            raise ModuleNotFoundError()
        
    def get_collate_fn(self, **kwargs):
        return super().get_collate_fn(**kwargs)
    
    def get_train_val_dataloader(self, tokenizer) -> tuple[DataLoader, DataLoader, DataLoader]:
        train, val, test = self.train_val_test_data()

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
        
        logger.info("训练集数据：{}条 验证集数据：{}条".format(len(train_dataset), len(dev_dataset)))

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2)
        dev_loader = DataLoader(dev_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2)
        
        return train_loader, dev_loader, dev_loader
    
    def get_test_dataloader(self, tokenizer=None) -> DataLoader:
        return super().get_test_dataloader(tokenizer=None)
    

class NLPCC2018TASK2(TextLabelDataset):
    def __init__(self, args=None, config=None) -> None:
        super(NLPCC2018TASK2, self).__init__(args, config)

class FCGEC:
    def __init__(self, args=None, config=None) -> None:
        self.args = args
        self.config = config
        self.data_base_dir = os.path.join(config.data_dir, config.out_dir)

    def process_train_valid(self, path, desc = 'train', err_only = True, need_type = False) -> pd.DataFrame:
        print('[TASK] processing {} file.'.format(desc))
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        fp.close()
        corrects = []
        for key in tqdm(data, 'Processing'):
            element = data[key]
            error_type = element['error_type']
            sentence = element['sentence']
            operate = element['operation']
            if err_only:
                if error_type != '*': corrects.append([sentence, error_type, operate]) if need_type else corrects.append([sentence, operate])
            else:
                corrects.append([sentence, error_type, operate]) if need_type else corrects.append([sentence, operate])
        if need_type:
            return pd.DataFrame(corrects, columns=['Sentence', 'Type', 'Label'])
        else:
            return pd.DataFrame(corrects, columns=['Sentence', 'Label'])

    def process_test(self, path, output_uuid : bool=False):
        print('[TASK] processing test file.')
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        fp.close()
        corrects = []
        for key in tqdm(data, 'Processing'):
            element = data[key]
            sentence = element['sentence']
            corrects.append([sentence, "[]"])
        if output_uuid:
            return pd.DataFrame(corrects, columns=['Sentence', 'Label']), list(data.keys())
        else:
            return pd.DataFrame(corrects, columns=['Sentence', 'Label'])

    def preprocess_independent(self, config):
        print('=' * 20 + "Preprocess Data for STG Independent" + "=" * 20)
        assert os.path.join(config.data_dir, config.train_file)
        assert os.path.join(config.data_dir, config.valid_file)
        assert config.train_file != ''
        assert config.valid_file != ''
        if config.test_file != '':
            assert os.path.join(config.data_dir, config.test_file)
        if not os.path.exists(os.path.join(config.data_dir, config.out_dir)):
            os.mkdir(os.path.join(config.data_dir, config.out_dir))
        # Process Train
        train_df = self.process_train_valid(os.path.join(config.data_dir, config.train_file), 'train', err_only=config.err_only)
        train_df.to_csv(os.path.join(config.data_dir, config.out_dir, 'train.csv'), index=False, encoding='utf_8_sig')
        print('Processed train dataset, saved at %s' % os.path.join(os.path.join(config.data_dir, config.out_dir, 'train.csv')))
        # Process Valid
        valid_df = self.process_train_valid(os.path.join(config.data_dir, config.valid_file), 'valid', err_only=config.err_only)
        valid_df.to_csv(os.path.join(config.data_dir, config.out_dir, 'valid.csv'), index=False, encoding='utf_8_sig')
        print('Processed valid dataset, saved at %s' % os.path.join(os.path.join(config.data_dir, config.out_dir, 'valid.csv')))
        if config.test_file != '':
            if config.out_uuid:
                test_df, uuid = self.process_test(os.path.join(config.data_dir, config.test_file), output_uuid=True)
                sio.savemat(os.path.join(config.data_dir, config.out_dir, 'uuid.mat'), {'uuid' : uuid})
                print('Save test uuid file at %s' % os.path.join(config.data_dir, config.out_dir, 'uuid.mat'))
            else:
                test_df = self.process_test(os.path.join(config.data_dir, config.test_file))
            test_df.to_csv(os.path.join(config.data_dir, config.out_dir, 'test.csv'), index=False, encoding='utf_8_sig')
            print('Processed test dataset, saved at %s' % os.path.join(os.path.join(config.data_dir, config.out_dir, 'test.csv')))
            
    def process_raw_file(self):
        self.preprocess_independent(self.config)

    def get_collate_fn(self):
        # Collate_fn of DataLoader(JointDataset)
        def collate_fn_jointV2(batch):
            dim = len(batch[0].keys())
            if dim == 7:  # Train DataLoader
                wid_ori   = [item['wid_ori'] for item in batch]
                wid_tag   = [item['wid_tag'] for item in batch]
                wid_gen   = [item['wid_gen'] for item in batch]

                tag_label = [item['tag_label'] for item in batch]
                comb_label = [item['comb_label'] for item in batch]

                sw_label  = [item['swlabel'] for item in batch]
                mlmlabel  = [item['mlmlabel'] for item in batch]

                wid_collection  = (wid_ori, wid_tag, wid_gen)
                tag_collection  = (tag_label, comb_label)
                spec_collection = (sw_label, mlmlabel)

                return wid_collection, tag_collection, spec_collection
            else:
                raise Exception('Error Batch Input, Please Check.')
        return collate_fn_jointV2
        
    def get_train_val_dataloader(self, tokenizer):
        # Dataset
        if os.path.exists(os.path.join(self.data_base_dir, 'train.pt')) is not True:
            train_dir = os.path.join(self.data_base_dir, 'train.csv')
            Trainset = JointDataset(self.config, train_dir, 'train')
            torch.save(Trainset, os.path.join(self.data_base_dir, 'train.pt'))
        else:
            Trainset = torch.load(os.path.join(self.data_base_dir, 'train.pt'))
            logger.info('Direct Load Train Dataset')
        if os.path.exists(os.path.join(self.data_base_dir, 'valid.pt')) is not True:
            valid_dir = os.path.join(self.data_base_dir, 'valid.csv')
            Validset = JointDataset(self.config, valid_dir, 'valid')
            torch.save(Validset, os.path.join(self.data_base_dir, 'valid.pt'))
        else:
            Validset = torch.load(os.path.join(self.data_base_dir, 'valid.pt'))
            logger.info('Direct Load Valid Dataset')
        # DataLoader
        TrainLoader = DataLoader(Trainset, batch_size=self.config.batch_size, shuffle=self.config.shuffle, drop_last=self.config.droplast, collate_fn=self.get_collate_fn())
        ValidLoader = DataLoader(Validset, batch_size=self.config.batch_size, shuffle=self.config.shuffle, drop_last=self.config.droplast, collate_fn=self.get_collate_fn())
        return TrainLoader, ValidLoader, ValidLoader
    
    def get_test_dataloader(self, tokenizer):
        print("Warning: in JointDataset (FCGEC) test dataloader does not exist and will use test.csv as data source.")
        return None
    
    def convert_seq2seq(self):
        data_class = FCGEC_SEQ2SEQ(self.args, None)
        data_class.process_raw_data()


class FCGEC_SEQ2SEQ:
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = Config(model=None, dataset='fcgec_seq2seq').get_config()

    def read_json_data(self, args : argparse.Namespace) -> tuple:
        train_data, valid_data, test_data = None, None, None
        def read_data(path :str) -> list:
            if os.path.exists(path) is not True: return None
            with open(path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            fp.close()
            return data
        if args.train_file != '': train_data = read_data(os.path.join(args.data_dir, args.train_file))
        if args.valid_file != '': valid_data = read_data(os.path.join(args.data_dir, args.valid_file))
        if args.test_file != '': test_data = read_data(os.path.join(args.data_dir, args.test_file))
        return train_data, valid_data, test_data

    def train_valid_processor(self, out_path : str, dataset : dict, uuid : bool=True, out_flag : bool=True, out_type : bool=True, desc='Train'):
        out_path = out_path.replace('.json', '.csv')
        out_data = []

        def convert_operator2seq(sentence : str, operate : list) -> list:

            def unpack(operate : list) -> list:
                operates = []
                version_old_mode = True
                for ops in operate:
                    if version_old_mode and ('Insert' in ops.keys() or 'Modify' in ops.keys()):
                        if 'Insert' in ops.keys() and isinstance(ops['Insert'][0]['label'], list): version_old_mode = False
                        if 'Modify' in ops.keys() and isinstance(ops['Modify'][0]['label'], list): version_old_mode = False
                    if version_old_mode:
                        operates.append(ops)
                        continue
                    unpacks = []
                    def convert_lab(opr : dict, label : str) -> dict:
                        op = deepcopy(opr)
                        op['label'] = label
                        return op
                    if 'Insert' in ops.keys():
                        unpacks.extend([{'Insert' : [convert_lab(op, lab)]} for op in ops['Insert'] for lab in op['label']])
                    if 'Modify' in ops.keys():
                        unpacks.extend([{'Modify' : [convert_lab(op, lab)]} for op in ops['Modify'] for lab in op['label']])
                    operates.extend(unpacks)
                return operates

            def get_postsentence(sentence, operate):
                ret = []
                for op in operate:
                    if 'Switch' in op.keys():
                        sentence = ''.join(np.array([s for s in sentence])[np.array(op['Switch'])])
                    sentag = [[s, 'O', ''] for s in sentence]
                    if 'Delete' in op.keys():
                        for i in op['Delete']:
                            sentag[i][1] = 'D'
                    if 'Insert' in op.keys():
                        for i in op['Insert']:
                            sentag[i['pos']][1] = i['tag']
                            sentag[i['pos']][-1] = i['label']
                    if 'Modify' in op.keys():
                        for i in op['Modify']:
                            sentag[i['pos']][1] = i['tag']
                            sentag[i['pos']][-1] = i['label']
                    ret.append(get_psentence(sentag))
                return ret

            def get_psentence(sentag):
                sent = ''
                cou = 0
                for i in range(len(sentag)):
                    if i < cou:
                        continue
                    if sentag[cou][1] == 'O':
                        sent += sentag[cou][0]
                    elif sentag[cou][1] == 'D':
                        cou += 1
                        continue
                    elif sentag[cou][1].startswith('INS'):
                        sent += sentag[cou][0]
                        sent += sentag[cou][-1]
                    elif sentag[cou][1].startswith('MOD'):
                        modnum = eval(sentag[cou][1].split('+')[0].split('_')[-1])
                        sent += sentag[cou][-1]
                        cou += (modnum - 1)
                    cou += 1
                return sent
            operates = unpack(operate)
            return get_postsentence(sentence, operates)

        for datk in tqdm(dataset.keys(), desc='Processing {} data'.format(desc)):
            outs = []
            if uuid: outs.append(datk)
            outs.append(dataset[datk]['sentence'])
            if out_flag: outs.append(dataset[datk]['error_flag'])
            if out_type: outs.append(dataset[datk]['error_type'])
            post_sentences = convert_operator2seq(dataset[datk]['sentence'], json.loads(dataset[datk]['operation'])) if dataset[datk]['error_flag'] == 1 else dataset[datk]['sentence']
            outs.append('\t'.join(post_sentences))
            out_data.append(outs)

        columns = []
        if uuid: columns.append('UUID')
        columns.append('Sentence')
        if out_flag: columns.append('Binary')
        if out_type: columns.append('Type')
        columns.append('Correction')
        df = pd.DataFrame(out_data, columns=columns)

        json_data = []
        key_transformation = {
            'UUID': 'id',
            'Sentence': 'text',
            'Correction': 'label',
            'Binary': 'binary',
            'Type': 'type'
        }

        # Change key name and save it to json file
        for i, item in df.iterrows():
            json_item = {}
            for key in key_transformation:
                if key in columns:
                    json_item[key_transformation[key]] = item[key]
            json_data.append(json_item)

        with open(out_path, 'w') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        # df.to_csv(out_path, index=False, encoding='utf_8_sig')
        print('%s data has been processed and saved at %s' % (desc, out_path))

    def testdata_processor(self, out_path : str, test_data : dict, uuid : bool=True):
        out_path = out_path.replace('.json', '.csv')
        test_out_data = []
        for datk in tqdm(test_data.keys(), desc='Processing Test data'):
            outs = []
            if uuid: outs.append(datk)
            outs.append(test_data[datk]['sentence'])
            test_out_data.append(outs)
        df = pd.DataFrame(test_out_data, columns=['UUID', 'Sentence'] if uuid else ['Sentence'])

        json_data = []
        key_transformation = {
            'UUID': 'id',
            'Sentence': 'text',
        }

        # Change key name and save it to json file
        for i, item in df.iterrows():
            json_item = {}
            for key in key_transformation:
                if key in df.columns:
                    json_item[key_transformation[key]] = item[key]
            json_data.append(json_item)

        with open(out_path, 'w') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        # df.to_csv(out_path, index=False, encoding='utf_8_sig') if uuid else df.to_csv(out_path, index=False, encoding='utf_8_sig')
        print('Test data has been processed and saved at %s' % out_path)

    def convert_data2seq(self, args : argparse.Namespace):
        train_data, valid_data, test_data = self.read_json_data(args)
        if train_data: self.train_valid_processor(os.path.join(args.data_dir, 'train.json'), train_data, args.out_uuid, args.out_errflag, args.out_errtype)
        if valid_data: self.train_valid_processor(os.path.join(args.data_dir, 'valid.json'), valid_data, args.out_uuid, args.out_errflag, args.out_errtype, 'Valid')
        if test_data: self.testdata_processor(os.path.join(args.data_dir, 'test.json'), test_data, args.out_uuid)
    
    def process_raw_data(self):
        self.convert_data2seq(args=self.config)


class FangZhengTest(TextLabelDataset):
    def __init__(self, args=None, config=None) -> None:
        super().__init__(args, config)
        self.raw_file = os.path.join(config.data_dir, 'test.xls')
        # self.file = os.path.join(config.data_dir, 'all.json')
        self.file = os.path.join(config.data_dir, config.test_file)
        self.args = args
        self.config = config

    def process_raw_file(self, cut='jieba'):
        table = pd.read_excel(self.raw_file)
        data = []
        if cut == "spacy":
            nlp = spacy.load("zh_core_web_trf")
        for i, row in tqdm(table.iterrows()):
            pairs = row["错误词对"].split("##")
            errors = row["错误类型"].split("##")
            assert len(pairs) == len(errors)
            data_item = {}
            data_item["text"] = row["原文"].strip()
            data_item["label"] = row["答案"].strip()
            assert len(data_item["text"]) == len(data_item["label"])
            data_item["errors"] = []
            if cut == "jieba":
                for error_idx in range(len(pairs)):
                    _, wrong_word, correct_word = pairs[error_idx].split('$')
                    # jieba cut again
                    if cut == "jieba":
                        error_item = {"wrong_word": wrong_word, "correct_word": correct_word, "type": errors[error_idx]}
                        data_item["errors"].append(error_item)

            elif cut == "spacy":
                # find the error positions
                wrong_idx = []  # all error indexes
                for i in range(len(data_item["text"])):
                    if data_item["text"][i] != data_item["label"][i]:
                        wrong_idx.append(i)
                
                doc = nlp(data_item["label"])
                accumulate_idx = 0
                for token in doc:
                    # iterate until all error position are removed
                    len_token = len(token.text)
                    hit = False
                    while wrong_idx != [] and (accumulate_idx <= wrong_idx[0] < accumulate_idx+len_token):
                        wrong_idx.remove(wrong_idx[0])
                        hit = True
                    
                    if hit:
                        error_item = {"wrong_word": data_item["text"][accumulate_idx:accumulate_idx+len(token)], 
                                        "correct_word": token.text, "pos": token.pos_, "dep":token.dep_}
                        data_item["errors"].append(error_item)

                    accumulate_idx += len_token
                    if wrong_idx == []:
                        break
            else:
                raise(NotImplementedError())

            data.append(data_item)
        with open(self.file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        return data

    def process_raw_file_xlsx(self):
        self.raw_file = os.path.join(config.data_dir, 'dapei_1129.xlsx')
        save = os.path.join(config.data_dir, 'test_1129.json')
        table = pd.read_excel(self.raw_file)
        data = []
        for i, item in table.iterrows():
            data.append({"text": item["sentence"], "label": item["正确句子"], 
                        "wrong": {
                                    "wrong_syntactics": item["错误的句法成分"], "distance": item["搭配词距离"], 
                                    "correct_w_collocation_times": item["正词搭配次数"], "wrong_w_collocation_times": item["错词搭配次数"],
                                    "correct_w_frequency": item["正词词频"], "wrong_w_frequency": item["错词词频"],
                                    "collocation_type": item["搭配类型"]
                                } 
                        })

        with open(save, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        return data



class HybridSet(TextLabelDataset):
    def __init__(self, args=None, config=None) -> None:
        super().__init__(args, config)
        self.raw_file = os.path.join(config.data_dir, 'corpus/train.sgml')
        self.file = os.path.join(config.data_dir, 'corpus/train.json')
        self.args = args
        self.config = config
        
    def process_raw_file(self):
        with open(self.raw_file, 'r') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        sentences = soup.find_all('sentence')
        data = []
        for sentence in sentences:
            data_item = {"text": None, "errors": []}
            for child in sentence.children:
                if child.name == "text":
                    data_item['text'] = str(child.string)
                elif child.name == "mistake":
                    error_description = {"location": child.location.string, "wrong": child.wrong.string, "correction": child.correction.string}  
                    data_item['errors'].append(error_description)
                else:
                    pass  
            label = list(data_item["text"])
            for error in data_item["errors"]:
                label[eval(error['location'])-1] = error["correction"]
            data_item['label'] = ''.join(label)

            data.append(data_item)
        with open(self.file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        return data

class Corpus(TextLabelDataset):
    def __init__(self, args=None, config=None) -> None:
        super().__init__(args, config)
        self.raw_file = os.path.join(config.data_dir, 'corpus.txt')
        self.file = os.path.join(config.data_dir, 'formatted.json')
        self.args = args
        self.config = config
    
    def process_raw_file(self):
        no_text_data = []
        with open(self.raw_file, 'r') as f:
            for line in f.readlines():
                no_text_data.append({"text": None, "label": line.strip()})
        with open(self.file, "w") as f:
            json.dump(no_text_data, f, ensure_ascii=False, indent=4)
        
        return no_text_data

        

class Augment(TextLabelDataset):
    def __init__(self, args=None, config=None) -> None:
        super().__init__(args, config)
        self.raw_file = os.path.join(config.data_dir, 'aug_data.json')
        self.file = os.path.join(config.data_dir, 'data.json')
        self.args = args
        self.config = config

    def data(self):
        assert os.path.exists(self.file)
        with open(self.file, "r") as f:
            data = json.load(f)
        return data
    
    def process_raw_file(self):
        from augmentation import ErrorType
        type_detector = ErrorType()
        assert os.path.exists(self.raw_file)
        with open(self.raw_file, "r") as f:
            data = json.load(f)
        
        given_error_types = set(["非同音形近音近-前后都在词典-多错误", "非同音形近音近-前后都在词典-单错误", "非同音形近音近-改后在词典", "非同音形近音近-改后在结巴词典", "非同音形近音近-其他"])

        final_data = []
        # data cleaning
        for i in tqdm(range(len(data))):
            assert len(data[i]['text']) == len(data[i]['label'])
            chars = list(data[i]['text'])
            if '\n' in chars or '\t' in chars:
                pass
            else:
                if len(data[i]['text']) <= self.config.text_max_length:
                    if not self.config.filter:
                        final_data.append(data[i])
                    else:
                        error_types = type_detector.type_spelling_check_task(data[i]["text"], data[i]["label"])
                        error_types_set = set(error_types.keys())
                        intersection = error_types_set.intersection(given_error_types)
                        if len(intersection):
                            final_data.append(data[i])

        with open(self.file, "w") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        return final_data        


class FangZhengAugment(TextLabelDataset):
    def __init__(self, args=None, config=None) -> None:
        super().__init__(args, config)
        self.file = os.path.join(config.data_dir, 'data.json')    # data after cleaning
        self.raw_file = os.path.join(config.data_dir, 'aug_data.json')  # data after mixing
        self.mix_data_file = os.path.join(config.mix_data_dir, 'aug_data.json')       # added data, from other augment data
        self.collocation_file = os.path.join(config.data_dir, "nonhgm_train_dapei.txt")
        self.train_file = os.path.join(config.data_dir, "nonhgm_train.txt")
        self.args = args
        self.config = config

    def process_raw_file(self, mix=1.):
        # based on mix_data, the texts number from fangzheng corpus is set to len(mix_data) * mix_coef
        corpus_items = []
        with open(self.collocation_file, 'r') as f:
            for item in f.readlines():
                item_content = item.split()
                if len(item_content[0].strip()) == len(item_content[1].strip()):
                    corpus_items.append({"text": item_content[0].strip(), "label": item_content[1].strip()})

        fangzheng_num = len(corpus_items)
        print(f"{self.collocation_file} has {fangzheng_num} samples.")
        with open(self.mix_data_file, 'r') as f:
            data = json.load(f)
        print(f"{self.mix_data_file} had {len(data)} samples. We sampled {min(int(len(data) * mix), fangzheng_num)} from {self.collocation_file}")
        data.extend(corpus_items[:int(len(data) * mix)])
        random.shuffle(data)

        with open(self.raw_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        return data

    def filter_raw_file(self):
        from augmentation import ErrorType
        type_detector = ErrorType()
        assert os.path.exists(self.raw_file)
        with open(self.raw_file, "r") as f:
            data = json.load(f)
        
        given_error_types = set(["非同音形近音近-前后都在词典-多错误", "非同音形近音近-前后都在词典-单错误", "非同音形近音近-改后在词典", "非同音形近音近-改后在结巴词典", "非同音形近音近-其他"])

        final_data = []
        # data cleaning
        for i in tqdm(range(len(data))):
            assert len(data[i]['text']) == len(data[i]['label'])
            chars = list(data[i]['text'])
            if '\n' in chars or '\t' in chars:
                pass
            else:
                if len(data[i]['text']) <= self.config.text_max_length:
                    if not self.config.filter:
                        final_data.append(data[i])
                    else:
                        error_types = type_detector.type_spelling_check_task(data[i]["text"], data[i]["label"])
                        error_types_set = set(error_types.keys())
                        intersection = error_types_set.intersection(given_error_types)
                        if len(intersection):
                            final_data.append(data[i])

        with open(self.file, "w") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        return final_data         


    def data(self):
        assert os.path.exists(self.file)
        with open(self.file, "r") as f:
            data = json.load(f)
        return data


def get_data(dataset_name: str, model_name: str=None):
    DATA_MAP = {
        'nlpcc2018task2': NLPCC2018TASK2,
        'fcgec': TextLabelDataset,
        'mucgec': TextLabelDataset,
        'hybridset': HybridSet,
        'fangzhengspell': FangZhengTest,
        'fangzhenggrammar': FangZhengTest,
        'guangming': Corpus,
        'peopledaily': Corpus,
        'augment': Augment,
        'fangzhengaugment': FangZhengAugment,
        'fangzhengdapei': TextLabelDataset,
    }

    SPECIAL_DATA_MAP = {
        'mucgec_seq2seq': MuCGECSeq2SeqDataset,
        'mucgec_edit': MuCGECEditDataset,
        'joint': FCGEC,
        'gector_data': GECToRDataset,
    }

    if model_name in MODEL_CORR_DATA:
        dataset_name = MODEL_CORR_DATA[model_name]
        assert dataset_name in SPECIAL_DATA_MAP.keys(), 'Not support ' + dataset_name
        return SPECIAL_DATA_MAP[dataset_name]
    else:
        assert dataset_name in DATA_MAP.keys(), 'Not support ' + dataset_name
        return DATA_MAP[dataset_name]


if __name__ == "__main__":

    dataset_name = 'mucgec'
    config = Config('seq2seq', dataset_name, False).get_config()
    data = get_data(dataset_name, 'seq2seq')(None, config)
    data.process_raw_file()

    # dataset_name = 'mucgec_edit'
    # config = Config(None, dataset_name, False).get_config()
    # data = get_data(dataset_name)(None, config)
    # data.preprocess_data()
