import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


class AbstractAttack(ABC):
    @abstractmethod
    def __init__(self, name: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.config = config
        self.name = name

    @abstractmethod
    def run(self, dataset: Dataset) -> Dataset:
        """Run the attack on the input data."""
        pass

    def signature(self, dataset: Dataset):
        config_str = json.dumps(self.config, sort_keys=True)
        encoded = (str(dataset.split) + self.name + config_str).encode()
        hash_obj = hashlib.sha256(encoded)
        return hash_obj.hexdigest()[:32]
