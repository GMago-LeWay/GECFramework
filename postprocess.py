import json
import os
import logging
from tqdm import tqdm
import zipfile
import codecs

logger = logging.getLogger(__name__)


CN_MARKER_MAP = {
    ',': '，',
    ';': '；',
    ':': '：',
    '(': '（',
    ')': '）',
    '?': '？',
    '!': '！',
}


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
            'cn_marker': self._chinese_marker_substitute,
            'merge_sample': self._merge_split_test_sample,
            'spacy_retokenize': self._retokenize,
        }

    def _chinese_marker_substitute(self):
        for i in range(len(self.results)):
            for key in CN_MARKER_MAP:
                self.results[i]["predict"] = self.results[i]["predict"].replace(key, CN_MARKER_MAP[key])

    def _merge_split_test_sample(self):
        raise NotImplementedError()

    def _retokenize(self):
        raise NotImplementedError()

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
                self.post_process_func[name]()
        self.prediction_saving()
