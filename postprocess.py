import json
import os
import logging
from tqdm import tqdm
import zipfile
import codecs
from nltk import word_tokenize
import re
import codecs
from utils.ChERRANT.main import compute_cherrant_with_ref_file, compute_cherrant
import spacy
en = spacy.load("en_core_web_sm")

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
BEA_DEV_M2_FILE = '../datasets/BEA19_dev/ABCN.dev.gold.bea19.m2'
MUCGEC_DEV_M2_FILE = '../datasets/MuCGEC/MuCGEC_dev/valid.gold.m2.char'
FCGEC_DEV_FILE = '../datasets/FCGEC/FCGEC_dev/test.json'

class PostProcessManipulator:
    cn_marker = 'cn_marker'
    mucgec_eval = 'mucgec_dev_eval'
    fcgec_eval = 'fcgec_dev_eval'
    bea_eval = 'bea_eval'
    merge_sample = 'merge_sample'
    en_test = 'en_test'
    en_test_py3 = 'en_test_py3'


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
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.post_process_func = {
            PostProcessManipulator.cn_marker: self._chinese_marker_substitute,
            PostProcessManipulator.merge_sample: self._merge_split_test_sample,
            PostProcessManipulator.en_test: self._en_conll_bea_postprocess_py2,
            PostProcessManipulator.en_test_py3: self._en_conll_bea_postprocess_py3,
            PostProcessManipulator.mucgec_eval: self._mucgec_dev_evaluation,
            PostProcessManipulator.bea_eval: self._bea_dev_evaluation,
            PostProcessManipulator.fcgec_eval: self._fcgec_dev_evaluation,
            # 'spacy_retokenize': self._retokenize,
        }

        self.allowed_dataset = {
            PostProcessManipulator.cn_marker: ['mucgec', 'fcgec', 'pretrain', 'fangzhenggrammar', 'fangzhengspell', 'mucgec_dev', 'fcgec_dev'],
            PostProcessManipulator.merge_sample: ['mucgec', 'fcgec', 'pretrain', 'fangzhenggrammar', 'fangzhengspell', 'c4', 'lang8', 'clang8', 'nucle', 'wilocness', 'hybrid'],
            PostProcessManipulator.en_test: ['c4', 'lang8', 'clang8', 'nucle', 'wilocness', 'hybrid'],
            PostProcessManipulator.en_test_py3: ['c4', 'lang8', 'clang8', 'nucle', 'wilocness', 'hybrid'],
            PostProcessManipulator.mucgec_eval: ['mucgec_dev'],
            PostProcessManipulator.bea_eval: ['bea_dev'],
            PostProcessManipulator.fcgec_eval: ['fcgec_dev'],
        }

    def _chinese_marker_substitute(self):
        for i in range(len(self.results)):
            for key in CN_MARKER_MAP:
                self.results[i]["predict"] = self.results[i]["predict"].replace(key, CN_MARKER_MAP[key])

    def _merge_split_test_sample(self):
        merged_results = []
        assert self.results, "Result Null"
        discourse_index = self.results[0]["id"].split('#')[0]
        source_discourse_buff = ""
        target_discourse_buff = ""
        last_item = None
        cur_item = None
        for item in self.results:
            cur_item = item
            line = item["predict"]
            line = line.strip()
            cur_index, _, end = item["id"].split('#')
            end = end.strip()
            if cur_index == discourse_index:
                source_discourse_buff += item["src"]
                target_discourse_buff += line
                last_item = item
            else:
                if "tgt" in item:
                    merged_results.append({"id": discourse_index, "src": source_discourse_buff, "tgt": last_item["tgt"], "predict": target_discourse_buff})
                else:
                    merged_results.append({"id": discourse_index, "src": source_discourse_buff, "predict": target_discourse_buff})
                discourse_index = cur_index
                source_discourse_buff = item["src"]
                target_discourse_buff = line
            if end != 'P':
                target_discourse_buff = target_discourse_buff[:-1] + end
        else:
            if "tgt" in item:
                merged_results.append({"id": discourse_index, "src": source_discourse_buff, "tgt": last_item["tgt"], "predict": target_discourse_buff})
            else:
                merged_results.append({"id": discourse_index, "src": source_discourse_buff, "predict": target_discourse_buff})
        
        logger.info(f"Results length before merged: {len(self.results)}; After merged: {len(merged_results)}")
        self.results = merged_results

    
    @staticmethod
    def conll_postprocess(item):
        global RETOKENIZATION_RULES, en

        item["oripred"] = str(item["predict"])
        line = str(item["predict"])
        line = re.sub(" '\s?((?:m )|(?:ve )|(?:ll )|(?:s )|(?:d ))",
                        "'\\1", line)
        line = " ".join([t.text for t in en.tokenizer(line)])
        # fix tokenization issues for CoNLL
        for rule in RETOKENIZATION_RULES:
            line = re.sub(rule[0], rule[1], line)
        item["predict"] = line
        return item
    
    @staticmethod
    def bea_postprocess(item):
        global en
        item["oripred"] = str(item["predict"])
        line = str(item["predict"])
        line = re.sub(" '\s?((?:m )|(?:ve )|(?:ll )|(?:s )|(?:d ))",
                        "'\\1", line)
        line = " ".join([t.text for t in en.tokenizer(line)])
        # in spaCy v1.9.0 and the en_core_web_sm-1.2.0 model
        # 80% -> 80%, but in newest ver. 2.3.9', 80% -> 80 %
        # haven't -> haven't, but in newest ver. 2.3.9', haven't -> have n't
        line = re.sub("(?<=\d)\s+%", "%", line)
        line = re.sub("((?:have)|(?:has)) n't", "\\1n't", line)
        line = re.sub("^-", "- ", line)
        line = re.sub(r"\s+", " ", line)
        item["predict"] = line
        return item

    def conll14_postprocess_by_groups(self, results, original_file, output_file):
        with codecs.open(original_file, 'w', 'utf-8') as f:
            for item in results:
                f.write(item["predict"] + '\n')
        os.system(f"{PYTHON2_PATH} utils/spacy_split.py -d conll -i {original_file} -o {output_file}")

    def bea19_postprocess_by_groups(self, results, original_file, output_file):
        with codecs.open(original_file, 'w', 'utf-8') as f:
            for item in results:
                f.write(item["predict"] + '\n')
        os.system(f"{PYTHON2_PATH} utils/spacy_split.py -d bea -i {original_file} -o {output_file}")
    
    def _en_conll_bea_postprocess_py2(self):
        last_number = -1
        bea19_start_index = None
        test_data = {'conll14': [], 'bea19': []}
        current_dataset = 'conll14'
        logger.info("Load CoNLL14 test data...")
        for i in range(len(self.results)):
            assert 'conll14' in self.results[i]["id"] or 'bea19' in self.results[i]["id"], "Current test set is not the concatenation of CoNLL14 and BEA19."
            # check current result item belongs to which test set
            number, data_source = self.results[i]["id"].split('_')
            if data_source != current_dataset:
                last_number = -1
                bea19_start_index = i
                current_dataset = 'bea19'
                logger.info("Load BEA19 test data...")
            assert eval(number) == last_number + 1
            last_number = eval(number)
            assert data_source in test_data
            test_data[data_source].append(self.results[i])

        CONLL14_NUM, BEA19_NUM = 1312, 4477
        # conll14 processed by groups (generate file and retokenize by spacy 1.9.0 with en_core_web_sm 1.2.0)
        conll14_ori_file_path = os.path.join(self.save_dir, 'conll14_original.txt')
        conll14_file_name = 'conll14.txt'
        conll14_file_path = os.path.join(self.save_dir, conll14_file_name)
        self.conll14_postprocess_by_groups(test_data['conll14'], original_file=conll14_ori_file_path, output_file=conll14_file_path)
        # load retokenized results
        conll14_results = open(conll14_file_path).readlines()
        assert len(test_data['conll14']) == len(conll14_results) == bea19_start_index == CONLL14_NUM
        for i in range(CONLL14_NUM):
            self.results[i]["oripred"] = str(self.results[i]["predict"]) 
            self.results[i]["predict"] = conll14_results[i].strip() 
        
        # bea19 processed by groups (generate file and retokenize by spacy 1.9.0 with en_core_web_sm 1.2.0)
        bea19_ori_file_path = os.path.join(self.save_dir, 'bea19_original.txt')
        bea19_file_name = 'bea19.txt'
        bea19_file_path = os.path.join(self.save_dir, bea19_file_name)
        self.bea19_postprocess_by_groups(test_data['bea19'], original_file=bea19_ori_file_path, output_file=bea19_file_path)
        # load retokenized results
        bea19_results = open(bea19_file_path).readlines()
        assert len(self.results) == len(test_data['conll14']) + len(bea19_results) == CONLL14_NUM + BEA19_NUM
        assert len(bea19_results) == len(self.results) - bea19_start_index == BEA19_NUM
        for i in range(bea19_start_index, len(self.results)):
            self.results[i]["oripred"] = str(self.results[i]["predict"]) 
            self.results[i]["predict"] = bea19_results[i-bea19_start_index].strip()           

        # save file for further evaluation
        # conll14 evaluation
        global CONLL14_M2_FILE
        evaluation_result_file = os.path.join(self.save_dir, 'conll14_metrics.txt')
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

    def _en_conll_bea_postprocess_py3(self):
        last_number = -1
        bea19_start_index = None
        test_data = {'conll14': [], 'bea19': []}
        current_dataset = 'conll14'
        logger.info("Postprocessing CoNLL14 test data...")
        for i in range(len(self.results)):
            assert 'conll14' in self.results[i]["id"] or 'bea19' in self.results[i]["id"], "Current test set is not the concatenation of CoNLL14 and BEA19."
            # check current result item belongs to which test set
            number, data_source = self.results[i]["id"].split('_')
            if data_source != current_dataset:
                last_number = -1
                bea19_start_index = i
                current_dataset = 'bea19'
                logger.info("Postprocessing BEA19 test data...")
            assert eval(number) == last_number + 1
            last_number = eval(number)
            assert data_source in test_data
            if current_dataset == 'conll14':
                self.results[i] = PostProcess.conll_postprocess(self.results[i])
            else:
                self.results[i] = PostProcess.bea_postprocess(self.results[i])
            test_data[data_source].append(self.results[i])

        CONLL14_NUM, BEA19_NUM = 1312, 4477
        assert len(test_data["bea19"]) == BEA19_NUM and len(test_data['conll14']) == CONLL14_NUM

        # save file for further evaluation
        # conll14 evaluation
        global CONLL14_M2_FILE
        conll14_file_name = 'conll14.txt'
        conll14_file_path = os.path.join(self.save_dir, conll14_file_name)
        evaluation_result_file = os.path.join(self.save_dir, 'conll14_metrics.txt')
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
        bea19_file_name = 'bea19.txt'
        bea19_file_path = os.path.join(self.save_dir, bea19_file_name)
        with open(bea19_file_path, 'w') as f:
            for item in test_data["bea19"]:
                f.write(item["predict"] + '\n')
        with zipfile.ZipFile(os.path.join(self.save_dir, 'bea19.zip'), mode='w') as zipf:
            zipf.write(bea19_file_path, bea19_file_name)

    def _bea_dev_evaluation(self):
        for i in range(len(self.results)):
            self.results[i] = PostProcess.bea_postprocess(self.results[i])
        # conll14 evaluation
        global BEA_DEV_M2_FILE
        bea_dev_file_name = 'bea19_dev.txt'
        bea_dev_file_path = os.path.join(self.save_dir, bea_dev_file_name)
        evaluation_result_file = os.path.join(self.save_dir, 'bea19_dev_metrics.txt')
        with open(bea_dev_file_path, 'w') as f:
            for item in self.results:
                f.write(item["predict"] + '\n')
        os.system(f"{PYTHON2_PATH} utils/m2scorer/scripts/m2scorer.py {bea_dev_file_path} {BEA_DEV_M2_FILE} >> {evaluation_result_file}")
        # print metrics of bea-19 dev set
        metrics_lines = open(evaluation_result_file).readlines()
        precision_name, _, precision = metrics_lines[0].strip().split()
        recall_name, _, recall = metrics_lines[1].strip().split()
        f_05_name, _, f_05 = metrics_lines[2].strip().split()
        logger.info(f"{precision_name}\t{recall_name}\t{f_05_name}")
        logger.info(f"{precision}\t{recall}\t{f_05}")

    def _mucgec_dev_evaluation(self):
        ids = [item["id"] for item in self.results]
        src_texts = [item["src"] for item in self.results]
        predict_texts = [item["predict"] for item in self.results]
        eval_information, eval_metrics = compute_cherrant_with_ref_file(
            ids=ids,
            src_texts=src_texts,
            predict_texts=predict_texts,
            ref_file=MUCGEC_DEV_M2_FILE,
            device=self.args.device
        )
        evaluation_result_file = os.path.join(self.save_dir, 'mucgec_dev_eval.txt')
        open(evaluation_result_file, 'w').write(eval_information)
        evaluation_metric_file = os.path.join(self.save_dir, 'mucgec_dev_metrics.json')
        json.dump(eval_metrics, open(evaluation_metric_file, 'w'), indent=4, ensure_ascii=False)

    def _fcgec_dev_evaluation(self):
        ids = [item["id"] for item in self.results]
        src_texts = [item["src"] for item in self.results]
        predict_texts = [item["predict"] for item in self.results]
        dev_data = json.load(open(FCGEC_DEV_FILE))
        tgt_texts = [[item['label']] + item['other_labels'] for item in dev_data]

        # check id
        assert len(ids) == len(dev_data)
        for id1, item in zip(ids, dev_data):
            assert id1 == item["id"]

        eval_information, eval_metrics = compute_cherrant(
            ids=ids,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            predict_texts=predict_texts,
            device=self.args.device
        )
        evaluation_result_file = os.path.join(self.save_dir, 'fcgec_dev_eval.txt')
        open(evaluation_result_file, 'w').write(eval_information)
        evaluation_metric_file = os.path.join(self.save_dir, 'fcgec_dev_metrics.json')
        json.dump(eval_metrics, open(evaluation_metric_file, 'w'), indent=4, ensure_ascii=False)

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
            if 'pre_split_length_for_infer' in self.config and self.config.pre_split_length_for_infer and PostProcessManipulator.merge_sample not in self.config.post_process:
                logger.info(f"Auto Set: You enable split_sentence for the test set but you did not include {PostProcessManipulator.merge_sample} as a postprocess. Auto added it in the front.")
                self.config.post_process.insert(0, PostProcessManipulator.merge_sample)
            if self.args.dataset == 'mucgec_dev' and PostProcessManipulator.mucgec_eval not in self.config.post_process:
                logger.info(f"Auto Set: You are using mucgec dev set for the test set but you did not include {PostProcessManipulator.mucgec_eval} as a postprocess. Auto added it in the rear.")
                self.config.post_process.append(PostProcessManipulator.mucgec_eval)      
            if self.args.dataset == 'fcgec_dev' and PostProcessManipulator.fcgec_eval not in self.config.post_process:
                logger.info(f"Auto Set: You are using fcgec dev set for the test set but you did not include {PostProcessManipulator.fcgec_eval} as a postprocess. Auto added it in the rear.")
                self.config.post_process.append(PostProcessManipulator.fcgec_eval) 
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
