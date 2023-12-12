import os
import json
import logging
from tqdm import tqdm
import datasets

logger = logging.getLogger(__name__)


class FCEWrapper:
    def __init__(self, args, config) -> None:
        '''
        Complete dataset of FCE with train, valid, test.
        '''
        self.args = args
        self.config = config

    def get_dataset(self, split=None)-> dict:
        assert split in ['train', 'valid', 'test']
        # print("------------", self.config.data_dir)
        if split == 'valid':
            split = 'validation'
        data = datasets.load_dataset(
            'dataset_wrapper/FCEBuilder.py', 
            data_dir=self.config.data_dir, 
            split=split,
        )
        return data
