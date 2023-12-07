import os
import json
import logging
from tqdm import tqdm
import datasets

logger = logging.getLogger(__name__)


class C4Wrapper:
    def __init__(self, args, config) -> None:
        '''
        C4 only contain large dataset for training, please create valid.json and test.json by yourself
        Because this train set does not want to be evaluated or infered, item id is not provided in train set
        '''
        self.args = args
        self.config = config
        self.valid_data_file = os.path.join(self.config.data_dir, 'valid.json')
        self.test_data_file = os.path.join(self.config.data_dir, 'test.json')
        if 'streaming' in config:
            self.streaming = config.streaming
        else:
            self.streaming = False

    def _load_json_and_formatted(self, file_path):
        data = json.load(open(file_path))
        if type(data) == list:
            assert len(data) != 0
            new_data = {}
            if 'id' not in data[0]:
                new_data['id'] = list(range(0, len(data)))
            for key in data[0]:
                new_data[key] = [item[key] for item in data]
            return new_data
        else:
            raise NotImplementedError()
        
    def get_dataset(self, split=None)-> dict:
        assert split in ['train', 'valid', 'test']
        # print("------------", self.config.data_dir)
        if split == 'train':
            data = datasets.load_dataset(
                'dataset_wrapper/C4Builder.py', 
                data_dir=self.config.data_dir, 
                split='train',
                streaming=self.streaming,
            )
        else:
            file = self.valid_data_file if split == 'valid' else self.test_data_file
            assert os.path.exists(file), 'Valid or test file does not exist.'
            json_data = self._load_json_and_formatted(file)
            data = datasets.Dataset.from_dict(json_data)
        return data
