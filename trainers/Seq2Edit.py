import logging
import time
import os
import json

import torch
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.learning_rate_schedulers import ReduceOnPlateauLearningRateScheduler
from allennlp.training import GradientDescentTrainer
from allennlp.common.model_card import Dataset
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.data_loaders.multiprocess_data_loader import MultiProcessDataLoader
from allennlp.data import allennlp_collate

from models.Seq2Edit import get_model
from utils.gector.gec_model import GecBERTModel
from tqdm import tqdm
from utils.MuCGEC import FullTokenizer, convert_to_unicode

from opencc import OpenCC

cc = OpenCC("t2s")

from trainers.base import Trainer2

logger = logging.getLogger(__name__)

class Seq2EditTrainer(Trainer2):
    def __init__(self, args, config, model) -> None:
        super(Seq2EditTrainer, self).__init__(args, config, model)

        assert model is None, "The Seq2Edit Model need to be initialized in trainer."
        self.args = args
        self.config = config

        logger = logging.getLogger(__file__)
        logger.setLevel(level=logging.INFO)
        start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        handler = logging.FileHandler(args.save_dir + '/logs_{:s}.txt'.format(str(start_time)))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)    
        self.logger = logger

        if args.task_mode == 'train':
            self.model = None
        else:
            print(f"**************Load checkpoint from {args.load}***************")
            self.model = GecBERTModel(vocab_path=self.config.vocab_path,
                            model_paths=[os.path.join(args.load, "Best_Model_Stage_2.th")],
                            weights_names=[self.config.pretrained_model],
                            max_len=self.config.max_len, min_len=0,
                            iterations=5,
                            min_error_probability=0.0,
                            min_probability=0.0,
                            log=False,
                            confidence=0.0,
                            is_ensemble=0,
                            weigths=None,
                            cuda_device=eval(self.args.device[5:])
                            )
        self.vocab = None

    def build_data_loaders(
            self,
            reader,
            path,
            batch_size: int,
            num_workers: int,
            shuffle: bool,
            batches_per_epoch = None
    ):
        """
        创建数据载入器
        :param batches_per_epoch:
        :param data_set: 数据集对象
        :param batch_size: batch大小
        :param num_workers: 同时使用多少个线程载入数据
        :param shuffle: 是否打乱训练集
        :return: 训练集、开发集、测试集数据载入器
        """
        return MultiProcessDataLoader(reader=reader, data_path=path, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                collate_fn=allennlp_collate, batches_per_epoch=batches_per_epoch) 

    def single_train(self, lr, batch_size, n_epoch, accumulation_size, train_data, dev_data):

        self.model = self.model.to(self.args.device)
        print("Model is set")
        self.logger.info("Model is set")
        parameters = [
            (n, p)
            for n, p in self.model.named_parameters() if p.requires_grad
        ]
        # 使用Adam算法进行SGD
        optimizer = AdamOptimizer(parameters, lr=lr, betas=(0.9, 0.999))
        scheduler = ReduceOnPlateauLearningRateScheduler(optimizer)

        train_reader, train_set = train_data
        dev_reader, dev_set = dev_data
        train_dataloader = self.build_data_loaders(reader=train_reader, path=train_set, batch_size=batch_size, num_workers=0, shuffle=False, batches_per_epoch=self.config.updates_per_epoch)
        val_dataloader = self.build_data_loaders(reader=dev_reader, path=dev_set, batch_size=batch_size, num_workers=0, shuffle=False)

        train_dataloader.index_with(self.vocab)
        val_dataloader.index_with(self.vocab)

        print("Data is loaded")
        self.logger.info("Data is loaded")

        trainer = GradientDescentTrainer(
            model=self.model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            patience=self.config.patience,
            validation_metric=self.config.save_metric,
            validation_data_loader=val_dataloader,
            num_epochs=n_epoch,
            cuda_device=eval(self.args.device[5:]),
            learning_rate_scheduler=scheduler,
            num_gradient_accumulation_steps=accumulation_size,
            use_amp=True  # 混合精度训练，如果显卡不支持请设为false
        )

        print("Start training")
        print('\nepoch: 0')
        self.logger.info("Start training")
        self.logger.info('epoch: 0')
        result = trainer.train()
        return result['best_validation_labels_accuracy']
    

    def do_train(self, train_dataset, val_dataset):
        ## Stage 1
        default_tokens = [DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN]
        namespaces = ['labels', 'd_tags']
        tokens_to_add = {x: default_tokens for x in namespaces}
        
        # build vocab
        if self.config.vocab_path:
            vocab = Vocabulary.from_files(self.config.vocab_path)
        else:
            vocab = Vocabulary.from_instances(train_dataset,
                                            min_count={"labels": 5},
                                            tokens_to_add=tokens_to_add)
            vocab.save_to_files(self.config.vocab_path)

        # set vocab
        self.vocab = vocab
        
        self.model = get_model(
            args=self.args,
            config=self.config,
            model_name=self.config.pretrained_model,
            vocab=vocab,
            tune_bert=0,
            predictor_dropout=self.config.predictor_dropout,
            label_smoothing=self.config.label_smoothing,
            model_dir=os.path.join(self.args.save_dir, 'Best_Model_Stage_1' + '.th'),
            log=self.logger
        )
        self.single_train(
            lr=self.config.lr1,
            batch_size=self.config.batch_size1,
            n_epoch=self.config.n_epoch1,
            accumulation_size=self.config.accumulation_size1,
            train_data=train_dataset,
            dev_data=val_dataset,
        )

        ## Stage 2

        self.model = get_model(
            args=self.args,
            config=self.config,
            model_name=self.config.pretrained_model,
            vocab=vocab,
            tune_bert=1,
            predictor_dropout=self.config.predictor_dropout,
            label_smoothing=self.config.label_smoothing,
            model_dir=os.path.join(self.args.save_dir, 'Best_Model_Stage_2' + '.th'),
            log=self.logger
        )

        self.load_part(self.args.save_dir)

        final_res = self.single_train(
            lr=self.config.lr2,
            batch_size=self.config.batch_size2,
            n_epoch=self.config.n_epoch2,
            accumulation_size=self.config.accumulation_size2,
            train_data=train_dataset,
            dev_data=val_dataset,
        )

        return final_res
        

    def do_test(self, dataset, mode="VAL"):
        """
        do test process, based on ids of every token.
        """
        return {}

    def do_infer(self, dataloader, mode="INFER"):
        """
        do infer or test process, based on decoded text of sentences' tokens.
        """
        self.logger.info("Generating...")
        print("Generating...")
        if mode == "TEST":
            src = []
            predictions = []
            tgt = []
            ids = []
            cnt_corrections = 0
            for batch in tqdm(dataloader):
                batch_texts = batch["raw_texts"]
                batch_labels = batch["raw_labels"]
                batch_ids = batch["ids"]
                batch_src_text = batch["src_texts"]
                src.extend(batch_src_text)
                ids.extend(batch_ids)
                tgt.extend(batch_labels)
                splited_texts = [text.split() for text in batch_texts]
                preds, cnt = self.model.handle_batch(splited_texts)
                assert len(preds) == len(batch_texts)
                predictions.extend(preds)
                cnt_corrections += cnt
            assert len(src) == len(predictions)
            output_file = os.path.join(self.args.save_dir, "Seq2Edit_infer.output")
            results = []
            with open(output_file, 'w') as f1:
                with open(output_file + ".char", 'w') as f2:
                    for i, ret in enumerate(predictions):
                        ret_new = [tok.lstrip("##") for tok in ret]
                        ret = cc.convert("".join(ret_new))
                        results.append({"id": ids[i], "src": src[i], "tgt": tgt[i], "predict": cc.convert(ret)})
                    tokenizer = FullTokenizer(vocab_file=self.config.vocab_file, do_lower_case=False)
                    for item in results:
                        ret = item["predict"]
                        f1.write(ret + "\n")
                        line = convert_to_unicode(ret)
                        tokens = tokenizer.tokenize(line)
                        f2.write(" ".join(tokens) + "\n")
            output_json_file = os.path.join(self.args.save_dir, "Seq2Edit_infer.json")
            with open(output_json_file, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            return results
        else:
            src = []
            predictions = []
            tgt = []
            ids = []
            cnt_corrections = 0
            for batch in tqdm(dataloader):
                batch_texts = batch["raw_texts"]
                batch_ids = batch["ids"]
                batch_src_text = batch["src_texts"]
                src.extend(batch_src_text)
                ids.extend(batch_ids)
                splited_texts = [text.split() for text in batch_texts]
                preds, cnt = self.model.handle_batch(splited_texts)
                assert len(preds) == len(batch_texts)
                predictions.extend(preds)
                cnt_corrections += cnt
            assert len(src) == len(predictions)
            output_file = os.path.join(self.args.save_dir, "Seq2Edit_infer.output")
            results = []
            with open(output_file, 'w') as f1:
                with open(output_file + ".char", 'w') as f2:
                    for i, ret in enumerate(predictions):
                        ret_new = [tok.lstrip("##") for tok in ret]
                        ret = cc.convert("".join(ret_new))
                        results.append({"id": ids[i], "src": src[i], "predict": cc.convert(ret)})
                    tokenizer = FullTokenizer(vocab_file=self.config.vocab_file, do_lower_case=False)
                    for item in results:
                        ret = item["predict"]
                        f1.write(ret + "\n")
                        line = convert_to_unicode(ret)
                        tokens = tokenizer.tokenize(line)
                        f2.write(" ".join(tokens) + "\n")
            output_json_file = os.path.join(self.args.save_dir, "Seq2Edit_infer.json")
            with open(output_json_file, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            return results

    def save(self, save_dir):
        file = os.path.join(save_dir, 'Best_Model_Stage_2' + '.th')
        torch.save(self.model, file)

    def load(self, save_dir):
        # file = os.path.join(save_dir, 'Best_Model_Stage_2' + '.th')
        # self.model.load_state_dict(torch.load(file))
        pass
    
    def load_part(self, save_dir):
        file = os.path.join(save_dir, 'Temp_Model.th')
        pretrained_dict = torch.load(file, map_location='cpu')
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print('load pretrained model')
        self.logger.info('load pretrained model')
        