import os
import json
import logging
import datasets

logger = logging.getLogger(__name__)


class BasicWrapper:
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config

        ## judge the status of datasets
        # well-split, to-be-split, corrupted, raw
        self.train_data_file = os.path.join(self.config.data_dir, 'train.json')
        self.valid_data_file = os.path.join(self.config.data_dir, 'valid.json')
        self.test_data_file = os.path.join(self.config.data_dir, 'test.json')
        self.data_file = os.path.join(self.config.data_dir, 'data.json')
        self.status = None
        if os.path.exists(self.train_data_file) and os.path.exists(self.valid_data_file) and os.path.exists(self.test_data_file):
            self.status = 'well-split'
        elif not (os.path.exists(self.train_data_file) or os.path.exists(self.valid_data_file) or os.path.exists(self.test_data_file)) and os.path.exists(self.data_file):
            self.status = 'to-be-split'
        elif not (os.path.exists(self.train_data_file) or os.path.exists(self.valid_data_file) or os.path.exists(self.test_data_file) or os.path.exists(self.data_file)):
            self.status = 'raw'
            logger.info("Warning: You are trying to construct a raw dataset")
        else:
            self.status = 'corrupted'
            logger.info("Warning: You are trying to construct a corrputed/irregular dataset")

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
    
    def _get_whole_data(self) -> datasets.Dataset:
        assert os.path.exists(self.data_file)
        data = self._load_json_and_formatted(self.data_file)
        dataset = datasets.Dataset.from_dict(data)
        return dataset
    
    def _train_val_test_split(self, shuffle=20):
        gross_data = self._get_whole_data()
        gross_data.shuffle(seed=shuffle)
        test_num = int(self.config.test_percent * len(gross_data))
        val_num = int(self.config.valid_percent * len(gross_data))
        test = gross_data[-test_num:]
        # select data
        train = gross_data[:len(gross_data)-val_num-test_num]
        val = gross_data[len(gross_data)-val_num-test_num: -test_num]
        ## save
        with open(self.train_data_file, 'w') as f:
            json.dump(train, f, ensure_ascii=False, indent=4)
        with open(self.valid_data_file, 'w') as f:
            json.dump(val, f, ensure_ascii=False, indent=4)
        with open(self.test_data_file, 'w') as f:
            json.dump(test, f, ensure_ascii=False, indent=4)
        
    def _split_dataset(self)-> dict:
        assert self.status == 'to-be-split'
        logger.info("You are trying to get one split of an unsplit dataset, so the data will be randomly split and saved.")
        self._train_val_test_split()

    def get_dataset(self, split) -> datasets.Dataset:
        assert self.status == 'well-split'
        assert split in ['train', 'valid', 'test']
        data = self._load_json_and_formatted(os.path.join(self.config.data_dir, split+'.json'))
        dataset = datasets.Dataset.from_dict(data)
        return dataset
