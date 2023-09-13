import logging
import os
import json

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class BasicDataLoaderTool:
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config

    def get_collate_fn(self, tokenizer, labeled):
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

    def get_dataloader(self, dataset, tokenizer, labeled) -> tuple[DataLoader, DataLoader, DataLoader]:
        logger.info('Total Train samples: %d, Total Valid samples: %d, Total Test samples: %d' % (len(dataset)))
        return DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=self.get_collate_fn(tokenizer, labeled), drop_last=False)
    
    def get_all_dataloader(self, train_set, val_set, test_set, tokenizer):
        logger.info('Total Train samples: %d, Total Valid samples: %d, Total Test samples: %d' % (len(train_set), len(val_set), len(test_set)))
        assert len(train_set) != 0 and len(val_set) != 0 and len(test_set) != 0
        assert 'label' in train_set[0] and 'label' in val_set[0]
        labeled_test_set = 'label' in test_set[0]
        return {
            'train': self.get_dataloader(self, dataset=train_set, tokenizer=tokenizer, labeled=True),
            'valid': self.get_dataloader(self, dataset=val_set, tokenizer=tokenizer, labeled=True),
            'test': self.get_dataloader(self, dataset=test_set, tokenizer=tokenizer, labeled=labeled_test_set),
        }
    
    def get_test_dataloader(self, test_set, tokenizer):
        assert len(test_set) != 0
        labeled_test_set = 'label' in test_set[0]
        return {
            'train': None,
            'valid': None,
            'test': self.get_dataloader(self, dataset=test_set, tokenizer=tokenizer, labeled=labeled_test_set),
        }
