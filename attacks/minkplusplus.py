import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import PreTrainedModel


def min_k_plusplus(model: PreTrainedModel, token_ids: torch.Tensor, attention_mask: torch.Tensor, k: int = 20):
    with torch.no_grad():
        labels = token_ids.clone()
        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :]
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        token_logp = -F.cross_entropy(shift_logits.contiguous().view(-1, model.config.vocab_size),
                                      shift_targets.contiguous().view(-1), reduction="none")
        token_logp = token_logp.view_as(shift_targets)

        mu = F.log_softmax(shift_logits, dim=2).mean(dim=2)
        sigma = F.log_softmax(shift_logits, dim=2).std(dim=2)

        token_score = (token_logp - mu) / sigma
        token_score[shift_attention_mask == 0] = 100
        token_score = token_score.detach().cpu().numpy()

        sorted_scores = np.sort(token_score, axis=1)
        lengths = shift_attention_mask.sum(dim=1).cpu().numpy()
        k_min_scores = []
        for scores, length in zip(sorted_scores, lengths):
            k_min_scores.append(np.mean(scores[:int(k / 100 * length)]))

    return np.array(k_min_scores)


class MinKplusplusAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v2",
        )
        return dataset

    def score(self, batch):
        texts = [x for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding="longest")
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        k_min_probas = min_k_plusplus(self.model, token_ids, attention_mask, k=self.config['k'])
        return {self.name: k_min_probas}
