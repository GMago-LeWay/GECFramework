from typing import Dict
from datasets import Dataset
from trainers.base import TrainerBeta

class Seq2SpanTrainer(TrainerBeta):
    def __init__(self, args, settings, model, dataset: Dict[str, Dataset]) -> None:
        super().__init__(args, settings, model, dataset)

    