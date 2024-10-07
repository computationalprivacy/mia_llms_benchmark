from attacks import AbstractAttack
from datasets import Dataset


class LossAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(lambda x: {self.name: -x["nlloss"]})
        return dataset