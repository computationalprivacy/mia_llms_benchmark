import importlib
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml
from attacks import AbstractAttack
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)


def load_attack(
    attack_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any]
) -> AbstractAttack:
    try:
        module = importlib.import_module(f"attacks.{config['module']}")

        ret = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, AbstractAttack) and attr is not AbstractAttack:
                if ret is None:
                    ret = attr(
                        name=attack_name,
                        model=model,
                        tokenizer=tokenizer,
                        config=config
                    )
                else:
                    raise ValueError(f"Multiple classes implementing AlgorithmInterface found in {attack_name}")

        if ret is not None:
            return ret
        else:
            raise ValueError(f"No class implementing AlgorithmInterface found in {attack_name}")
    except ImportError as e:
        raise ValueError(f"Failed to import algorithm '{attack_name}': {str(e)}")


def get_available_attacks(config) -> list:
    return set(config.keys()) - {"global"}


def load_mimir_dataset(name: str, split: str) -> Dataset:

    dataset = load_dataset("iamgroot42/mimir", name, split=split)

    assert 'member' in dataset.column_names
    assert 'nonmember' in dataset.column_names

    all_texts = [dataset['member'][k] for k in range(len(dataset))]
    all_labels = [1] * len(dataset)
    all_texts += [dataset['nonmember'][k] for k in range(len(dataset))]
    all_labels += [0] * len(dataset)

    new_dataset = Dataset.from_dict({"text": all_texts, "label": all_labels})

    return new_dataset
