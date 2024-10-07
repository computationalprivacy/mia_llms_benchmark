# Adapted from https://github.com/nlp-titech/samia/tree/main

import zlib
from collections import Counter
from typing import Dict

import numpy as np
import torch
from attacks import AbstractAttack
from datasets import Dataset


def split_text(text: str, prefix_ratio: float) -> str:
    num_words = len(text.split())
    num_prefix_words = int(num_words * prefix_ratio)
    prefix = " ".join(text.split()[:num_prefix_words])
    suffix = " ".join(text.split()[num_prefix_words:])
    return prefix, suffix


def ngrams_paper(sequence, n) -> zip:
    """
    Generates n-grams from a sequence.
    """
    return zip(*[sequence[i:] for i in range(n)])


def rouge_n_paper(candidate: list, reference: list, n=1) -> float:
    """
    Calculates the ROUGE-N score between a candidate and a reference.
    """
    if not candidate or not reference:
        return 0
    candidate_ngrams = list(ngrams_paper(candidate, n))
    reference_ngrams = list(ngrams_paper(reference, n))
    ref_words_count = Counter(reference_ngrams)
    cand_words_count = Counter(candidate_ngrams)
    overlap = ref_words_count & cand_words_count
    recall = sum(overlap.values()) / len(reference)
    precision = sum(overlap.values()) / len(candidate)
    return recall


def count_ngrams(text: str, n: int) -> Dict[str, int]:
    """Count n-grams in the given text."""
    words = text.split()
    return Counter([' '.join(words[i:i+n]) for i in range(len(words) - n + 1)])


def rouge_n(candidate: str, reference: str, n: int) -> float:
    """Calculate ROUGE-N score for a candidate summary against a single reference summary."""
    candidate_ngrams = count_ngrams(candidate, n)
    reference_ngrams = count_ngrams(reference, n)

    matching_ngrams = sum((candidate_ngrams & reference_ngrams).values())
    total_reference_ngrams = sum(reference_ngrams.values())

    if total_reference_ngrams == 0:
        return 0.0

    return matching_ngrams / total_reference_ngrams


class SaMIAAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v4",
        )
        return dataset

    def score(self, batch):
        texts = [split_text(x, self.config['prefix_ratio']) for x in batch['text']]
        prefixes, suffixes = list(zip(*texts))

        tokenized = self.tokenizer.batch_encode_plus(prefixes, return_tensors="pt", padding="longest")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        input_ids_arr = []
        attention_mask_arr = []
        for i in range(len(input_ids)):
            input_ids_arr.append(input_ids[i].repeat(self.config['n_candidates'], 1))
            attention_mask_arr.append(attention_mask[i].repeat(self.config['n_candidates'], 1))

        input_ids_dup = torch.cat(input_ids_arr)
        attention_mask_dup = torch.cat(attention_mask_arr)

        cand_tokens = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=self.config['temperature'],
            max_length=self.config['max_length'],  # input+output
            top_k=self.config['top_k'],
            top_p=self.config['top_p'],
            pad_token_id=self.tokenizer.pad_token_id
        )

        candidates = self.tokenizer.batch_decode(cand_tokens, skip_special_tokens=True)
        candidate_suffixes = [x[len(prefix):] for prefix, x in zip(prefixes, candidates)]

        step = self.config['n_candidates']
        ret = []
        for i in range(len(texts)):
            suffix = suffixes[i]
            candidates = candidate_suffixes[i*step:(i+1)*step]

            n = self.config['n']
            if self.config['rouge_version'] == "paper":
                rouge_scores = [rouge_n_paper(x.split(), suffix.split(), n) for x in candidate_suffixes]
            elif self.config['rouge_version'] == "ours":
                rouge_scores = [rouge_n(x, suffix, n) for x in candidate_suffixes]
            else:
                raise ValueError(f"Invalid rouge_version {self.config['rouge_version']}")

            if self.config['zlib']:
                zlib_scores = [len(zlib.compress(" ".join(suffix_cand).encode('utf-8')))
                               for suffix_cand in candidate_suffixes]
                rouge_scores = [x*y for x, y in zip(rouge_scores, zlib_scores)]

            ret.append(np.mean(rouge_scores))

        return {self.name: np.array(ret)}
