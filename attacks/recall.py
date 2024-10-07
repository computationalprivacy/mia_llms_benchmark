import random

from attacks import AbstractAttack
from attacks.utils import compute_nlloss
from datasets import Dataset, load_dataset


def make_recall_prefix(dataset, n_shots, perplexity_bucket=None):
    prefixes = []
    if perplexity_bucket is not None:
        dataset = dataset.filter(lambda x: x["perplexity_bucket"] == perplexity_bucket)
    
    indices = random.sample(range(len(dataset)), n_shots)
    prefixes = [dataset[i]["text"] for i in indices]

    return " ".join(prefixes)

class RecallAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.extra_non_member_dataset = load_dataset(config['extra_non_member_dataset'], split=config['split'])

    def build_fixed_prefixes(self, target_dataset):
        perplexity_buckets = set(x["perplexity_bucket"] for x in target_dataset)
        if self.config["match_perplexity"]:
            prefixes = {
                ppl: make_recall_prefix(
                    dataset=self.extra_non_member_dataset,
                    n_shots=self.config["n_shots"],
                    perplexity_bucket=ppl
                )
                for ppl in perplexity_buckets
            }
        else:
            prefix = make_recall_prefix(
                dataset=self.extra_non_member_dataset,
                n_shots=self.config["n_shots"],
                perplexity_bucket=None
            )
            prefixes = {ppl: prefix for ppl in perplexity_buckets}

        return prefixes

    def build_one_prefix(self, perplexity_bucket):
        if self.config["match_perplexity"]:
            ppl_bucket = perplexity_bucket
        else:
            ppl_bucket = None
        
        return make_recall_prefix(
            dataset=self.extra_non_member_dataset,
            n_shots=self.config["n_shots"],
            perplexity_bucket=ppl_bucket
        )

    def run(self, dataset: Dataset) -> Dataset:
        if self.config["fixed_prefix"]:
            prefixes = self.build_fixed_prefixes(dataset)
        else:
            prefixes = None

        dataset = dataset.map(
            lambda x: self.recall_nlloss(x, prefixes=prefixes),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v2",
        )
        dataset = dataset.map(lambda x: {self.name: x['recall_nlloss'] / x['nlloss']})
        return dataset
    
    def recall_nlloss(self, batch, prefixes = None):
        it = zip(batch["perplexity_bucket"], batch["text"])
        if prefixes is not None:
            texts = [prefixes[ppl_bucket] + " " + text for ppl_bucket, text in it]
        else:
            texts = [self.build_one_prefix(ppl_bucket) + " " + text for ppl_bucket, text in it]

        tokenized = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding="longest")
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        losses = compute_nlloss(self.model, token_ids, attention_mask)
        return {'recall_nlloss': losses}
