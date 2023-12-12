import json
import os
import logging
from tqdm import tqdm
import zipfile
import codecs
from nltk import word_tokenize
import spacy
import re
import codecs

# EN_MODEL = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)

PYTHON2_PATH = '/data/liwei/anaconda3/envs/m2/bin/python'

CN_MARKER_MAP = {
    ',': '，',
    ';': '；',
    ':': '：',
    '(': '（',
    ')': '）',
    '?': '？',
    '!': '！',
}

RETOKENIZATION_RULES = [
    # Remove extra space around single quotes, hyphens, and slashes.
    (" ' (.*?) ' ", " '\\1' "),
    (" - ", "-"),
    (" / ", "/"),
    # Ensure there are spaces around parentheses and brackets.
    (r"([\]\[\(\){}<>])", " \\1 "),
    (r"\s+", " "),
]

CONLL14_M2_FILE = 'utils/m2scorer/official-2014.combined.m2'

class PostProcessManipulator:
    cn_marker = 'cn_marker'
    merge_sample = 'merge_sample'
    en_test = 'en_test'


class PostProcess:
    def __init__(self, args, config, json_results, save_dir_name) -> None:
        '''
        Post process after model generated json-like result list. Must be run in infer mode.
        json_results: List[Dict], like [{'id': str or num, 'src': str, 'predict': str, ('tgt': str)}]
        save_dir: save directory name inside the args.save_dir
        '''
        self.args = args
        self.config = config
        self.results = json_results
        self.save_dir_name = save_dir_name
        # assert 'infer' in args.task_mode

        # set save directory
        self.save_dir = os.path.join(args.save_dir, save_dir_name)

        self.post_process_func = {
            PostProcessManipulator.cn_marker: self._chinese_marker_substitute,
            PostProcessManipulator.merge_sample: self._merge_split_test_sample,
            PostProcessManipulator.en_test: self._en_conll_bea_postprocess,
            # 'spacy_retokenize': self._retokenize,
        }

        self.allowed_dataset = {
            PostProcessManipulator.cn_marker: ['mucgec', 'fcgec', 'pretrain', 'fangzhenggrammar', 'fangzhengspell'],
            PostProcessManipulator.merge_sample: [],
            PostProcessManipulator.en_test: ['c4', 'lang8', 'clang8', 'nucle', 'wilocness', 'hybrid']
        }

    def _chinese_marker_substitute(self):
        for i in range(len(self.results)):
            for key in CN_MARKER_MAP:
                self.results[i]["predict"] = self.results[i]["predict"].replace(key, CN_MARKER_MAP[key])

    def _merge_split_test_sample(self):
        raise NotImplementedError()

    
    @staticmethod
    def conll_postprocess(item):
        # nltk tokenize for conll output
        global RETOKENIZATION_RULES

        item["oripred"] = str(item["predict"])
        tokenized_output = word_tokenize(item["predict"])
        output = ' '.join(tokenized_output)
        # fix tokenization issues for CoNLL
        for rule in RETOKENIZATION_RULES:
            output = re.sub(rule[0], rule[1], output)
        item["predict"] = output
        return item
    
    @staticmethod
    def bea19_postprocess(item):
        # item["oripred"] = str(item["predict"])
        # en_doc = EN_MODEL(item["predict"])
        # item["predict"] = ' '.join([token.text for token in en_doc])
        return item
    
    def bea19_postprocess_by_groups(self, results, original_file, output_file):
        with codecs.open(original_file, 'w', 'utf-8') as f:
            for item in results:
                f.write(item["predict"] + '\n')
        os.system(f"{PYTHON2_PATH} utils/spacy_split.py -i {original_file} -o {output_file}")
    
    def _en_conll_bea_postprocess(self):
        last_number = -1
        bea19_index = None
        test_data = {'conll14': [], 'bea19': []}
        current_dataset = 'conll14'
        logger.info("CoNLL14 test data...")
        for i in range(len(self.results)):
            assert 'conll14' in self.results[i]["id"] or 'bea19' in self.results[i]["id"], "Current test set is not the concatenation of CoNLL14 and BEA19."
            # check current result item belongs to which test set
            number, data_source = self.results[i]["id"].split('_')
            if data_source != current_dataset:
                last_number = -1
                bea19_index = i
                current_dataset = 'bea19'
                logger.info("BEA19 test data...")
            assert eval(number) == last_number + 1
            last_number = eval(number)
            assert data_source in test_data

            # postprocess
            if data_source == 'conll14':
                self.results[i] = PostProcess.conll_postprocess(self.results[i])
            else:
                # self.results[i] = PostProcess.bea19_postprocess(self.results[i])
                pass
            
            test_data[data_source].append(self.results[i])
        
        # bea19 processed by groups (generate file and retokenize by spacy 1.9.0 with en_core_web_sm 1.2.0)
        bea19_ori_file_path = os.path.join(self.save_dir, 'bea19_original.txt')
        bea19_file_name = 'bea19.txt'
        bea19_file_path = os.path.join(self.save_dir, bea19_file_name)
        self.bea19_postprocess_by_groups(test_data['bea19'], original_file=bea19_ori_file_path, output_file=bea19_file_path)
        # load retokenized results
        bea19_results = open(bea19_file_path).readlines()
        assert len(self.results) == len(test_data['conll14']) + len(bea19_results) == 1312 + 4477
        assert len(bea19_results) == len(self.results) - bea19_index == 4477
        for i in range(bea19_index, len(self.results)):
            self.results[i]["oripred"] = str(self.results[i]["predict"]) 
            self.results[i]["predict"] = bea19_results[i-bea19_index].strip()           

        # save file for further evaluation
        # conll14 evaluation
        global CONLL14_M2_FILE
        conll14_file_name = 'predict.conll14'
        conll14_file_path = os.path.join(self.save_dir, conll14_file_name)
        evaluation_result_file = os.path.join(self.save_dir, 'CoNLL14_metrics.txt')
        with open(conll14_file_path, 'w') as f:
            for item in test_data["conll14"]:
                f.write(item["predict"] + '\n')
        os.system(f"{PYTHON2_PATH} utils/m2scorer/scripts/m2scorer.py {conll14_file_path} {CONLL14_M2_FILE} >> {evaluation_result_file}")
        # print metrics of conll14
        metrics_lines = open(evaluation_result_file).readlines()
        precision_name, _, precision = metrics_lines[0].strip().split()
        recall_name, _, recall = metrics_lines[1].strip().split()
        f_05_name, _, f_05 = metrics_lines[2].strip().split()
        logger.info(f"{precision_name}\t{recall_name}\t{f_05_name}")
        logger.info(f"{precision}\t{recall}\t{f_05}")
        
        # pack bea19 output  
        with zipfile.ZipFile(os.path.join(self.save_dir, 'bea19.zip'), mode='w') as zipf:
            zipf.write(bea19_file_path, bea19_file_name)


    def basic_saving(self):
        save_path = os.path.join(self.save_dir, f'{self.args.model}-{self.args.dataset}-{self.args.task_mode}.json')
        with codecs.open(save_path, "w", "utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)

        save_txt = os.path.join(self.save_dir, f'{self.args.model}-{self.args.dataset}-{self.args.task_mode}.txt')
        with codecs.open(save_txt, "w", "utf-8") as f:
            for item in self.results:
                if "tgt" in item:
                    f.write("%s\t%s\t%s\n" % (item["src"], item["tgt"], item["predict"]))
                else:
                    f.write("%s\t%s\n" % (item["src"], item["predict"]))
        logger.info(f"Results have been stored in {save_path}.")


    def prediction_saving(self):
        """
        In infer task, some dataset requires a specific version of results to evaluate, this function will do the formatting.
        """
        self.basic_saving()
        ## MuCGEC output
        if self.args.dataset.lower() == 'mucgec':
            save_txt = os.path.join(self.save_dir, f'MuCGEC_test.txt')
            with codecs.open(save_txt, "w", "utf-8") as f:
                for item in self.results:
                    f.write("%s\t%s\t%s\n" % (item["id"], item["src"], item["predict"]))
            with zipfile.ZipFile(os.path.join(self.save_dir, 'submit.zip'), mode='w') as zipf:
                zipf.write(save_txt, 'MuCGEC_test.txt')
        
        ## FCGEC output
        if self.args.dataset.lower() == 'fcgec':
            fcgec_json = {}
            for item in self.results:
                error_flag = 1 if item["src"] != item["predict"] else 0
                fcgec_json[item['id']] = {"error_flag": error_flag, "error_type": "IWO", "correction": item["predict"]}
            fcgec_path = os.path.join(self.save_dir, 'predict.json')
            with codecs.open(fcgec_path, "w", "utf-8") as f:
                json.dump(fcgec_json, f, ensure_ascii=False, indent=4)      
            with zipfile.ZipFile(os.path.join(self.save_dir, 'predict.zip'), mode='w') as zipf:
                zipf.write(fcgec_path, 'predict.json')


    def post_process_and_save(self):
        if 'post_process' in self.config:
            for name in self.config.post_process:
                # check if it is an allowed processing
                allowed = False
                if name in self.allowed_dataset:
                    if self.args.dataset.lower() in self.allowed_dataset[name]:
                        allowed = True
                
                if allowed:
                    self.post_process_func[name]()
                else:
                    logger.info(f"Error: Unsupported post process of {name} for {self.args.dataset}. Skipped.")
                    
        self.prediction_saving()
