import zlib

from attacks import AbstractAttack
from datasets import Dataset


def zlib_score(record):
    text = record["text"]
    loss = record["nlloss"]
    zlib_entropy = len(zlib.compress(text.encode()))/len(text)
    zlib_score = -loss / zlib_entropy
    return zlib_score

class ZlibAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(lambda x: {self.name: zlib_score(x)})
        return dataset