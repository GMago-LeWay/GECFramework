import random
import logging
import os
from utils.JointSTG import TAGGER_MAP

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"
    
    def __getstate__(self):
        return None


DATA_DIR_NAME = {
    'nlpcc2018task2': "NLPCC2018_GEC",
    'fcgec': "FCGEC",
    'fcgec_seq2seq': "FCGEC",  # only used in FCGEC
    'mucgec': "MuCGEC",
    'hybridset': "HybridSet",
    'fangzhenggrammar': "FangZhengGrammar",
    'fangzhengspell': "FangZhengSpell",
    'guangming': "Corpus/GuangMing",
    'peopledaily': "Corpus/PeopleDaily",
    'augment': "augment3",
    'fangzhengaugment': "FangZhengAugment",
    'fangzhengdapei': "FangZhengDapei",
    'pretrain': "PreTrainSetLarge",
}

MODEL_CORR_DATA = {
    'stgjoint': 'joint',
    'seq2seq': 'mucgec_seq2seq',
    'seq2edit': 'mucgec_edit',
    'gector': 'gector_data',
    'chatglm': 'transformers',
    'correctionglm': 'gec_glm',
}

DATA_ROOT_DIR = '/home/liwei/workspace/datasets'
MODEL_ROOT_DIR = '/home/liwei/workspace/models'

class Config:
    def __init__(self, model: str, dataset: str, tune=False, preconfig=None) -> None:
        self.model_name = model
        self.dataset = dataset
        self.preconfig = preconfig
        self.tune = tune

        self.MODEL_MAP = {
            'bert': self.__BERT,
            'softmaskedbert': self.__SoftMaskedBERT,
            'stgjoint': self.__STG_Joint,
            'seq2seq': self.__Seq2Seq,
            'seq2edit': self.__Seq2Edit,
            'gector': self.__GECToR,
            'chinese_llama': self.__ChineseLLaMA,
            'llm': self.__LLM,
            'llama': self.__LLAMA,
            'llama_quant': self.__LLAMA_QUANT,
            'chatglm': self.__ChatGLM,
            'correctionglm': self.__CorrectionGLM,
            None: self.__NULL,
        }

        self.DATA_MAP = {
            ## Normal data config
            'nlpcc2018task2': self.__NLPCC2018,
            'fcgec': self.__FCGEC,
            'mucgec': self.__MuCGEC,
            'hybridset': self.__HybridSet,
            'fangzhenggrammar': self.__FangZhengTest,
            'fangzhengspell': self.__FangZhengTest,
            'guangming': self.__UnlabeledCorpus,
            'peopledaily': self.__UnlabeledCorpus,
            'augment': self.__Augment,
            'fangzhengaugment': self.__FangZhengAugment,
            'fangzhengdapei': self.__FangZhengDapei,
            'pretrain': self.__PreTrainSet,

            ## For special
            'fcgec_seq2seq': self.__FCGEC_Seq2Seq,

            ## Model-correlated data config
            'joint': self.__FCGEC,
            'mucgec_seq2seq': self.__MuCGEC,
            'mucgec_edit': self.__MuCGEC_Edit,
            'gector_data': self.__GECToR_Data,
            'transformers': self.__TransformersData,
            'gec_glm': self.__GEC_GLM_Data,

            None: self.__NULL,
        }


    def solve_conflict(self):
        '''
        Solve some confict between dataset args and model args. Call this function after self.args is loaded.
        '''
        return
    
    def get_data_args(self):
        '''
        Get dataset config. Some models require a specific dataset.
        Here the data_dir will be reset to correct directory.
        '''
        self.data_dir = os.path.join(DATA_ROOT_DIR, DATA_DIR_NAME[self.dataset])
        if self.model_name in MODEL_CORR_DATA:
            datasetArgs = self.DATA_MAP[MODEL_CORR_DATA[self.model_name]]()
        else:
            datasetArgs = self.DATA_MAP[self.dataset]()
        datasetArgs = {**{'data_dir': self.data_dir}, **datasetArgs}
        
        return datasetArgs

    def get_model_args(self):
        return self.MODEL_MAP[self.model_name](self.tune)
    
    def get_config(self):
        commonArgs = self.get_model_args()
        dataArgs = self.get_data_args()

        self.args = Storage({**commonArgs, **dataArgs})
        self.solve_conflict()
        return self.args

    def __NULL(self, tune=False):
        # Configuration loaded when no model or data is specified.
        Config = {}
        return Config

    def __UnlabeledCorpus(self):

        dataConfig = {
            'valid_percent': 0.1,
            'test_percent': 0.15,
            'batch_size': 32,
            'text_cut': 384,   
            'eval_step': None,        # steps interval of evaluation, None: 1eval/epoch
        }

        return dataConfig

    def __NLPCC2018(self):

        dataConfig = {
            'valid_percent': 0.1,
            'test_percent': 0.15,
            'batch_size': 32,
            'text_cut': 200,   
            'eval_step': None,        # steps interval of evaluation, None: 1eval/epoch
        }

        return dataConfig

    def __MuCGEC(self):
        dataConfig = {
            'train_file': os.path.join(self.data_dir, 'train.json'),
            'validation_file': os.path.join(self.data_dir, 'valid.json'),
            'test_file': os.path.join(self.data_dir, 'test.json'),
            'text_cut': 200,

            'batch_size': 48,  
            'eval_step': None,        # steps interval of evaluation, None: 1eval/epoch            
        }

        return dataConfig

    def __MuCGEC_Edit(self):
        dataConfig = {
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'chinese-struct-bert-large'), 

            'vocab_file': "knowledge/mucgec_data/vocab.txt",
            'vocab_path': os.path.join(MODEL_ROOT_DIR, "chinese-struct-bert-large/output_vocabulary_chinese_char_hsk+lang8_5"),

            'train_set': os.path.join(self.data_dir, 'Seq2Edit', 'train.label'),
            'dev_set': os.path.join(self.data_dir, 'Seq2Edit', 'valid.label'),
            # 'test_set': os.path.join(self.data_dir, 'Seq2Edit', 'test.label'),

            'skip_correct': 1,
            'skip_complex': 0,
            'tag_strategy': 'keep_one',       # ['keep_one', 'merge_all'],
            'tn_prob': 0,           # 保留正确句子的比例
            'tp_prob': 1,           # 保留错误句子的比例
            
            'batch_size': 256,  
            'max_len': 200,
            'text_cut': 200,
            'target_vocab_size': 1000,
            'eval_step': None,        # steps interval of evaluation, None: 1eval/epoch            
        }

        return dataConfig
    

    def __FCGEC_Seq2Seq(self):

        dataConfig = {
            'train_file': 'FCGEC_train.json',
            'valid_file': 'FCGEC_valid.json',
            'test_file': 'FCGEC_test.json',
            'out_uuid': True,       # 'Output UUID in test file'
            'out_errflag': True,     #  'Whether to output `error_flag`'
            'out_errtype': True,    # 'Whether to output `error_type`'
        }

        return dataConfig
    

    def __HybridSet(self):

        dataConfig = {
            'valid_percent': 0.05,
            'test_percent': 0.20,
            'batch_size': 32,
            'text_cut': 256,   
            'eval_step': 20000,        # steps interval of evaluation, None: 1eval/epoch   
        }

        return dataConfig


    def __Augment(self):

        dataConfig = {
            'valid_percent': 0.05,
            'test_percent': 0.15,
            'batch_size': 128,
            'text_cut': 256, 
            'text_max_length': 64,
            'filter': True,
            'eval_step': 10000,        # steps interval of evaluation, None: 1eval/epoch
        }

        return dataConfig


    def __FangZhengAugment(self):

        dataConfig = {
            'mix_data_dir': os.path.join(DATA_ROOT_DIR, "augment3/"),

            'valid_percent': 0.05,
            'test_percent': 0.20,
            'batch_size': 64,
            'text_cut': 256, 
            'text_max_length': 128,
            'filter': True,
            'eval_step': 15000,        # steps interval of evaluation, None: 1eval/epoch
        }

        return dataConfig
        

    def __FangZhengTest(self):

        dataConfig = {
            'test_file': 'test.json',

            'valid_percent': 0.1,
            'test_percent': 0.15,
            'batch_size': 8,
            'text_cut': 384,   
            'eval_step': None,        # steps interval of evaluation, None: 1eval/epoch
        }

        return dataConfig

    def __FangZhengDapei(self):

        dataConfig = {
            'valid_percent': 0.1,
            'test_percent': 0.15,
            'batch_size': 32,
            'text_cut': 384,   
            'eval_step': None,        # steps interval of evaluation, None: 1eval/epoch
        }

        return dataConfig
    
    def __PreTrainSet(self):
        dataConfig = {
            'text_cut': 128,
            'batch_size': 128,
            'eval_step': 2000,        # steps interval of evaluation, None: 1eval/epoch   
        }
        return dataConfig

    def __BERT(self, tune):

        Config = {
            # identifier
            'name': 'bert',

            # pretrained model
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'chinese-macbert-base'),
            'tokenize_style': [1, -1],      # will add [cls] at front and add [sep] at rear

            # fixed parameters

            # parameters that are able to be tuned

            # learning parameters
            'max_epochs': 20,
            'learning_rate_bert': 0.0001,
            'learning_rate_other': 0.0001,
            'weight_decay_bert': 0.0,
            'weight_decay_other': 0.0,
            'early_stop': 8,

            # evaluation config
            'metrics': 'spelling_check_1',
            'KeyEval': 'loss',
            'scheduler_mode': 'min',
            'scheduler_factor': 0.2,
            'scheduler_patience': 3,
        }

        TuneConfig = {     
            # identifier
            'name': 'bert',

            # pretrained model
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'chinese-macbert-base'),
            'tokenize_style': [1, -1],      # will add [cls] at front and add [sep] at rear

            # fixed parameters

            # parameters that are able to be tuned

            # learning parameters
            'max_epochs': 20,
            'learning_rate_bert': random.choice([0, 1e-05, 5e-5, 5e-4, 1e-3]),
            'learning_rate_other': random.choice([1e-4, 5e-4, 0.001, 0.002]),
            'weight_decay_bert': random.choice([0, 0.01, 0.001, 0.0001]),
            'weight_decay_other': random.choice([0, 0.01, 0.001, 0.0001]),         
            'early_stop': 6,

            # evaluation config
            'KeyEval': 'Loss',
            'scheduler_mode': 'min',
            'scheduler_factor': 0.2,
            'scheduler_patience': 5,
 
        }

        return TuneConfig if tune else Config

    def __SoftMaskedBERT(self, tune):

        Config = {
            # identifier
            'name': 'softmaskedbert',

            # pretrained model
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'chinese-roberta-wwm-ext'),
            'embedding_size': 768,
            'tokenize_style': [1, -1],      # will add [cls] at front and add [sep] at rear

            # fixed parameters

            # parameters that are able to be tuned
            'gamma': 0.7,   # loss weight
            'hidden_size': 50,

            # learning parameters
            'batch_size': 32,
            'max_epochs': 20,
            'learning_rate_bert': 0.0001,
            'learning_rate_other': 0.0001,
            'weight_decay_bert': 0.0,
            'weight_decay_other': 0.0,   
            'early_stop': 6,

            # evaluation config
            'metrics': 'spelling_check_1',
            'KeyEval': 'loss',
            'scheduler_mode': 'min',
 
        }

        TuneConfig = {     
            # identifier
            'name': 'softmaskedbert',

            # pretrained model
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'chinese-roberta-wwm-ext'),

            # fixed parameters

            # parameters that are able to be tuned

            # learning parameters

            # evaluation config
 
        }

        return TuneConfig if tune else Config

    def __Seq2Seq(self, tune):
        Config = {
            'name': 'seq2seq',

            'do_train': True,
            'do_eval': True,
            'do_predict': True,

            # pretrained model
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'bart-large-chinese'),

            'save_path': None,
            'task': 'gec',

            'gradient_accumulation_steps': 4,
            'lr': 3e-6,
            # 'batch_size': 32, dataset config
            'epoch': 10,
            'lr_scheduler': 'polynomial',
            'save_strategy': 'epoch',
            'predict_file': 'results',


        }

        return NotImplementedError() if tune else Config


    def __LLM(self, tune):
        ## chatglm-6b, Llama2-Chinese-7b-Chat
        import torch

        Config = {
            # identifier
            'name': 'llm',

            # pretrained model
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'Llama2-Chinese-7b-Chat'),
            'lora_model': '/home/liwei/workspace/Llama2-Chinese/train/sft/llama2_mucgec_edit/checkpoint-6000',
            'tokenize_style': [1, -1],      # will add [cls] at front and add [sep] at rear

            # model config
            'torch_dtype': torch.float16,
            'load_in_8bit': False,
            'generation_config': dict(
                # temperature=0.2,
                # top_k=10,
                # top_p=0.95,
                # do_sample=True,
                # num_beams=1,
                repetition_penalty=1.3,
                max_new_tokens=128,
            ),

            # fixed parameters

            # parameters that are able to be tuned


            # evaluation config
            'chinese_marker_substitution': False,

            'prompt': '<s>Human: 请对以下句子中可能存在的语法错误进行逐步修改: \n[text]\n</s><s>',         # [text] for substitution.
            'final_prompt': 'Assistant：',   # Used for segmentation.
            'max_len_prop': 2.0,
            'min_len_prop': 0.0,

        }

        return NotImplementedError() if tune else Config


    def __ChineseLLaMA(self, tune):

        Config = {
            # identifier
            'name': 'chinese_llama',

            # pretrained model
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'chinese-llama/alpaca-combined'),
            'lora_model': None,
            'tokenize_style': [1, -1],      # will add [cls] at front and add [sep] at rear

            # fixed parameters

            # parameters that are able to be tuned


            # evaluation config
            'chinese_marker_substitution': False,

            'prompt': '请直接改正原句中的语法错误，如果原句没有错误，直接输出原句：\n原句：[text]\n',         # [text] for substitution.
            'final_prompt': '改正：',   # Used for segmentation.
            'max_len_prop': 1.3,
            'min_len_prop': 0.6,

        }

        return NotImplementedError() if tune else Config
    

    def __LLAMA(self, tune):

        Config = {
            # identifier
            'name': 'llama',

            # pretrained model
            'language_model': False,
            'ckpt_dir': os.path.join(MODEL_ROOT_DIR, 'pyllama_data/7B'),
            'tokenizer_path' : os.path.join(MODEL_ROOT_DIR, 'pyllama_data/tokenizer.model'),

            # evaluation config
            'temperature': 0.,
            'top_p': 1.,
            'max_seq_len': 1024,
            'max_gen_len': 512,
            'max_batch_size': 8,

        }

        return NotImplementedError() if tune else Config
    

    def __LLAMA_QUANT(self, tune):

        Config = {
            # identifier
            'name': 'llama_quant',

            # pretrained model
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'llama-30b-hf'),
            'wbits': 4,
            'load': os.path.join(MODEL_ROOT_DIR, 'pyllama_data/pyllama-30B4b.pt'),
            'tokenizer_path' : os.path.join(MODEL_ROOT_DIR, 'pyllama_data/tokenizer.model'),

            # evaluation config
            'temperature': 0.,
            'top_p': 1.,
            'max_seq_len': 1024,
            'max_gen_len': 512,
            'do_sample': False,

        }

        return NotImplementedError() if tune else Config 
    
    def __ChatGLM(self, tune):
        Config = {
            # identifier
            'name': 'chatglm',

            # pretrained model
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'chatglm-6b'),

            # config
            'max_source_length': 100,
            'max_target_length': 100,
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 8,
            'gradient_accumulation_steps': 1,
            'num_train_epochs': 2,
            'logging_steps': 100,
            'save_steps': 1000,
            'learning_rate': 2e-2,
            'pre_seq_len': 128,
            'quantization_bit': 4,
        }

        return NotImplementedError() if tune else Config 

    def __TransformersData(self):
        dataConfig = {

        }

        return dataConfig


    def __Seq2Edit(self, tune):
        Config = {
            'name': 'seq2edit',

            # pretrained model
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'chinese-struct-bert-large'),   #('weights_name')

            # 'model_name': 'Best_Model_Stage_1',         # Best_Model_Stage_1 & 2

            'n_epoch1': 2,
            'n_epoch2': 20,

            'patience': 3,

            'lr1': 1e-3,
            'lr2': 1e-5,

            'batch_size1': 512,
            'batch_size2': 128,
            'updates_per_epoch': None,

            'predictor_dropout': 0.0,
            'label_smoothing': 0.0,

            'pretrain_folder': None,
            'pretrain': None,
            'accumulation_size1': 1,
            'accumulation_size2': 4,

            'save_metric': "+labels_accuracy",      # 模型保存指标["+labels_accuracy", "+labels_accuracy_except_keep"]

        }

        return NotImplementedError() if tune else Config
    
    def __GECToR(self, tune):
        Config = {
            'name': 'gector',

            # pretrained model and tokenizer
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'chinese-macbert-base'), # chinese-macbert-base, structbert-large-zh
            'tokenize_style': [1, -1],      # will add [cls] at front and add [sep] at rear

            # model label vocab
            'ctc_vocab_dir': os.path.join(MODEL_ROOT_DIR, 'GECToR', 'ctc_vocab'),
            'detect_tags_file': "ctc_detect_tags.txt",
            'correct_tags_file': "ctc_correct_tags.txt", # ctc_correct_tags.txt, ctc_correct_cail2022_tags.txt, mucgec_correct_tags.txt
            'infer_tags': None,
            'detect_vocab_size': 2,
            'correct_vocab_size': None,        # lazy init

            # RL
            'reward_estimate': False,
            'reward_metric': 'token_level:f0.5',
            'reward_loss_weight': 5,

            # training setting
            'warmup_proportion': 0.02,
            'learning_rate': 3e-5,     # 2e-5/32 3e-5/48
            'adam_epsilon': 1e-8,
            'use_tensorboard': False,
            'epochs': 10,
            'max_grad_norm': 1.0,

            # infer setting
            'fixed_length': False,
            'iteration': 3,
        }

        return NotImplementedError() if tune else Config

    def __GECToR_Data(self):
        dataConfig = {
            'use_multi_append': False,      # use data where multi-append situation are split into multiple sentences.
            'text_cut': 200,
            'batch_size': 48,
            'eval_step': 1000,        # steps interval of evaluation, None: 1eval/epoch   
        }

        return dataConfig

    
    def __STG_Joint(self, tune):
        Config = {
            # identifier
            'name': 'stgjoint',

            # pretrained model
            'language_model': True,            # 'use_lm'
            'lm_path': os.path.join(MODEL_ROOT_DIR, 'chinese-roberta-wwm-ext'),       # 'lm_path'
            'lm_hidden_size': 768,
            'output_hidden_states': True,
            'finetune': True,
            'tokenize_style': [1, -1],      # will add [cls] at front and add [sep] at rear

            # convertor setting
            'p2next': True,
            'sp_map': True, # special mapping

            # search params
            'sw_mode': 'rsgs',
            'beam_width': 1,

            # model params
            'model': 'modalnet_origin', # modalnet_origin, Model Selection, Can Choose [modalnet_origin]
            'num_classes': 2,
            'tagger_classes': len(TAGGER_MAP.keys()),
            'max_generate': 5,
            # we actually set max_generate (released checkpoint) to 5 during training phase. However, we calculated the distribution of the data after the rebuttal and thought that 6 would be more appropriate.
            'padding_size': 150,
            'padding_val': 0,
            'ignore_val': -1,
            'dropout': 0.1,
            'scale_attn': True,
            'factorized_embedding': False,
            'lm_emb_size': 768,


            # parameters that are able to be tuned
            'layers_num': 12,
            'layer_init_w': 0.1,

            # learning parameters
            'shuffle': True,
            'droplast': False,
            'optimizer': 'adamW',
            'lr': 1e-5,
            'wd': 1e-2,
            'warmup_steps': 10,
            'epoch': 50,
            'criterion': 'CE',
 
        }

        return NotImplementedError() if tune else Config

    def __FCGEC(self):

        dataConfig = {
            'data_base_dir': f"{self.data_dir}/stg_joint",
            'train_file': 'FCGEC_train.json',
            'valid_file': 'FCGEC_valid.json',
            'test_file': 'FCGEC_test.json',
            'out_dir': 'stg_joint',
            'out_uuid': False,       # 'Output UUID in test file'
            'err_only': True,
            'text_cut': 256,

            # learning params
            'batch_size': 32, 
            'print_step': 50,
            'eval_step': 200,      # steps interval of evaluation, None: 1eval/epoch
        }

        return dataConfig

    def __CorrectionGLM(self, tune):
        import torch

        Config = {
            # identifier
            'name': 'correctionglm',

            # pretrained model
            'language_model': True,
            'pretrained_model': os.path.join(MODEL_ROOT_DIR, 'glm-large-chinese'),
            'use_lora': False,
            'tokenize_style': [1, -1],      # will add [cls] at front and add [sep] at rear

            # model config
            'torch_dtype': None,
            'load_in_8bit': False,
            'loss_ignore_id': -100,
            'loss_detach': False,
            'bf16': False,

            # fixed parameters
            'model_type': 'all',        # model type: all, detection, generate
            'num_labels': 3,    # detection label num, 3 means mode ['$KEEP', '$ERROR', '$INSERT'], 2 means mode ['$KEEP', '$ERROR']
            'output_dropout_prob': 0.2,        # detection head dropout
            'logging_steps': 10,

            # parameters that are able to be tuned
            'prompt': '',    # '请修正以下语句中的语法错误，并在后面给出正确的语句：',
            'detection_loss_weight': 3,
            'gradient_accumulation_steps': 8,
            'lr': 2e-5,
            'weight_decay': 1e-4,
            'epoch': 5,
            'warmup_steps': 1000,           # 之前FCGEC训练为100
            'lr_scheduler': 'polynomial',
            'save_strategy': 'epoch',
            'alpha': [1,2,2],  # [1,2,2], or [1,2]

            # data process parameters
            'cache_dir': '.cache',
            'load_cache': True,
            'detection_results': {
                'train': None,
                'valid': None,
                'test': None
            },
            # detections of current best checkpoint 
            # 'glm_results/correctionglm-fcgec-eval_train-20231025-1407/detection_results.json'
            # 'glm_results/correctionglm-mucgec-eval_train-20231024-2312/detection_results.json',
            'max_train_source_length': 128,
            'max_eval_source_length': 256,
            'train_batch_size': 12,
            'eval_batch_size': 8,
            'detection_batch_size': 8,

            # evaluation config
            'eval_step': 1000,        # steps interval of evaluation, None: 1eval/epoch 
            'save_step': 4000,  
            'eval_key': 'eval_general_accuracy',

            # inference config
            'load_config_keys': ['model_type', 'prompt', 'num_labels', 'alpha'],
            'detection_only': False,
            'keep_threshold': None,
            'chinese_marker_substitution': True,
            'max_new_tokens': 10,

        }

        return NotImplementedError() if tune else Config
    
    def __GEC_GLM_Data(self):
        dataConfig = {
            # 'use_multi_append': False,      # use data where multi-append situation are split into multiple sentences.
        }

        return dataConfig
