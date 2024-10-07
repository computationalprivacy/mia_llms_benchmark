import numpy as np
import torch
from attacks import AbstractAttack
from attacks.utils import batch_nlloss
from datasets import Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer


class RatioAttack(AbstractAttack):

    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.llama_model, self.llama_tokenizer = self._load_llama(config['device'])
        self.llama_device = config['device']

    def _load_llama(self, device):
        llama_model = LlamaForCausalLM.from_pretrained(self.config['llama_model_path']).to(device)
        llama_tokenizer = LlamaTokenizer.from_pretrained(
            self.config['llama_tokenizer_path'], torch_dtype=torch.float16)
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        return llama_model, llama_tokenizer

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: batch_nlloss(x, self.llama_model, self.llama_tokenizer, self.llama_device, key='llama_nlloss'),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v1",
        )
        dataset = dataset.map(lambda x: {self.name: -x['nlloss'] / x['llama_nlloss']})
        return dataset
