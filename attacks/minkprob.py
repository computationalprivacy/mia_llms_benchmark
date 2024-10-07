import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import PreTrainedModel


def min_k_prob(model: PreTrainedModel, token_ids: torch.Tensor, attention_mask: torch.Tensor, k: int = 20):
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
        for probas, length in zip(sorted_probas, lengths):
            k_min_probas.append(np.mean(probas[:int(k / 100 * length)]))

    return np.array(k_min_probas)


class MinKProbAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v3",
        )
        return dataset

    def score(self, batch):
        texts = [x for x in batch['text']]
        tokenized = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding="longest")
        token_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        k_min_probas = min_k_prob(self.model, token_ids, attention_mask, k=self.config['k'])
        return {self.name: k_min_probas}
