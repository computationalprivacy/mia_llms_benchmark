# Adapted from https://github.com/yyy01/PAC

import random

import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import PreTrainedModel


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def eda(sentence, alpha=0.3, num_aug=5):
    words = sentence.split(' ')
    num_words = len(words)
    augmented_sentences = []

    if (alpha > 0):
        n_rs = max(1, int(alpha*num_words))
        for _ in range(num_aug):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [sentence for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug/len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    return augmented_sentences


def polarized_distance(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    k_min: float,
    k_max: float,
):
    with torch.no_grad():
        labels = token_ids.clone()
        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        # we add minus here, because `F.cross_entropy` is a loss, and we need the log-probability.
        # loss goes down when probability goes up.
        token_logp = -F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
        token_logp = token_logp.view(token_ids.shape[0], -1)
        token_logp = token_logp.detach().cpu().numpy()

        sorted_probas = np.sort(token_logp, axis=1)
        lengths = shift_attention_mask.sum(dim=1).cpu().numpy()
        k_min_probas = []
        k_max_probas = []

        for probas, length in zip(sorted_probas, lengths):
            k_min_probas.append(np.mean(probas[:int(k_min * length)]))
            k_max_probas.append(np.mean(probas[-int(k_max * length):]))

        return np.array(k_max_probas) - np.array(k_min_probas)


class PACAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.pac_score(x),
            batched=False,
            new_fingerprint=f"{self.signature(dataset)}_v3",
        )

        return dataset

    def pac_score(self, x):
        text = x["text"]		
        tokenized = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        
        og_dist = polarized_distance(
        	model=self.model, 
        	token_ids=input_ids, 
        	attention_mask=attention_mask, 
        	k_min=self.config['k_min'], 
        	k_max=self.config['k_max']
        )

        adjacent = eda(text, alpha=self.config['alpha'], num_aug=self.config['num_augmentations'])
        tokenized = self.tokenizer.batch_encode_plus(adjacent, return_tensors="pt", padding="longest")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        adjacent_dist = polarized_distance(
        	model=self.model, 
        	token_ids=input_ids, 
        	attention_mask=attention_mask, 
        	k_min=self.config['k_min'], 
        	k_max=self.config['k_max']
        )

        score = og_dist[0] - np.mean(adjacent_dist)
        return {self.name: -score}
