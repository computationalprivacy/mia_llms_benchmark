from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def compute_nlloss(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_prefix: Optional[int] = None,
):
    with torch.no_grad():
        labels = token_ids.clone()

        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        loss = F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
        loss = loss.view(token_ids.shape[0], -1)

        if ignore_prefix:
            loss = loss[:, ignore_prefix:]
            shift_attention_mask = shift_attention_mask[:, ignore_prefix:]

        loss = loss.sum(axis=1) / shift_attention_mask.sum(axis=1)

        return loss.detach().cpu().numpy()


def batch_nlloss(batch, model, tokenizer, device, key='nlloss'):
    tokenized = tokenizer.batch_encode_plus(batch['text'], return_tensors='pt', padding="longest")
    token_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    losses = compute_nlloss(model, token_ids, attention_mask)
    return {key: losses}
