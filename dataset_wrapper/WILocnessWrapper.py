import os
import json
import logging
from tqdm import tqdm
import datasets

logger = logging.getLogger(__name__)


class WILocnessWrapper:
    def __init__(self, args, config) -> None:
        '''
        BEA 19 WI&Locness dataset.
        train, valid split available; test split is EQUAL to validation.
        '''
        self.args = args
        self.config = config

    def get_dataset(self, split=None)-> dict:
        assert split in ['train', 'valid', 'test']
        # print("------------", self.config.data_dir)
        if split in ['valid', 'test']:
            split = 'validation'
        data = datasets.load_dataset(
            'dataset_wrapper/WILocnessBuilder.py', 
            data_dir=self.config.data_dir, 
            name='all', 
            split=split,
        )
        return data
