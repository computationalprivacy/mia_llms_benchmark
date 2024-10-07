# Adapted from https://github.com/mireshghallah/neighborhood-curvature-mia/

import logging
from heapq import nlargest

import numpy as np
import torch
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


class NeighborhoodAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.mlm_device = config['device']
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(
            config['mlm_model'], torch_dtype=torch.float16).to(self.mlm_device)
        self.mlm_tokenizer = AutoTokenizer.from_pretrained(config['mlm_model'])
        self.n_neighbors = config['n_neighbors']
        self.top_k = config['top_k']
        self.is_scale_embeds = config['is_scale_embeds']

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.neighborhood_score(x["text"]),
            batched=False,
            new_fingerprint=f"{self.signature(dataset)}_v1",
        )
        return dataset

    def neighborhood_score(self, text: str):
        with torch.no_grad():
            original_score = self.get_logprob_batch(text=text)[0]
            neighbors = [x[0] for x in self.generate_neighbors(text=text)]
            neighbor_scores = self.get_logprob_batch(text=neighbors)

            final_score = original_score - np.mean(neighbor_scores)
            return {self.name: final_score}

    def get_logprob_batch(self, text: str):
        tokenized = self.tokenizer(text, padding="longest", return_tensors='pt').input_ids.to(self.device)

        ce_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=self.tokenizer.pad_token_id)
        logits = self.model(tokenized, labels=tokenized).logits[:, :-1, :].transpose(1, 2)
        manual_logprob = - ce_loss(logits, tokenized[:, 1:])
        mask = manual_logprob != 0
        manual_logprob_means = (manual_logprob*mask).sum(dim=1)/mask.sum(dim=1)

        return manual_logprob_means.tolist()

    def generate_neighbors(self, text: str):
        max_len = self.mlm_tokenizer.model_max_length  # Get the model's max length
        tokenized = self.mlm_tokenizer(text, return_tensors='pt', truncation=False).input_ids[0]

        # Check if the sequence exceeds the max length
        if len(tokenized) > max_len:
            logging.warning(
                f"Warning: sample is longer than max input length for MLM ({max_len})? Replacements will be made in the first chunk.")
            # Truncate the tokenized input to the max length
            truncated_tokenized = tokenized[:max_len]
            # Save the remainder for appending later
            remainder_tokenized = tokenized[max_len:]
        else:
            truncated_tokenized = tokenized
            remainder_tokenized = None

        # Generate replacements only on the truncated part
        texts_with_replacements = self._generate_neighbors_for_chunk(truncated_tokenized)

        # Append the remainder (if any) to each generated neighbor
        if remainder_tokenized is not None:
            remainder_text = self.mlm_tokenizer.decode(remainder_tokenized, skip_special_tokens=True)
            texts_with_replacements = [
                (replaced_text + " " + remainder_text, score)
                for replaced_text, score in texts_with_replacements
            ]

        return texts_with_replacements

    def _generate_neighbors_for_chunk(self, tokenized_chunk):
        text_tokenized = tokenized_chunk.unsqueeze(0).to(self.mlm_device)
        replacements = dict()

        for target_token_index in range(1, len(text_tokenized[0])-1):
            target_token = text_tokenized[0, target_token_index]

            embeds = self.mlm_model.roberta.embeddings(text_tokenized)
            embeds = torch.cat((embeds[:, :target_token_index, :],
                                F.dropout(embeds[:, target_token_index, :], p=0.7,
                                          training=self.is_scale_embeds).unsqueeze(dim=0),
                                embeds[:, target_token_index+1:, :]), dim=1)

            token_probs = torch.softmax(self.mlm_model(inputs_embeds=embeds).logits, dim=2)

            token_probs[:, :, self.mlm_tokenizer.bos_token_id] = 0
            token_probs[:, :, self.mlm_tokenizer.eos_token_id] = 0

            original_prob = token_probs[0, target_token_index, target_token]

            top_probabilities, top_candidates = torch.topk(token_probs[:, target_token_index, :], self.top_k, dim=1)

            for cand, prob in zip(top_candidates[0], top_probabilities[0]):
                if not cand == target_token:
                    if original_prob.item() == 1:
                        replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                    else:
                        replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())

        highest_scored = nlargest(self.n_neighbors, replacements, key=replacements.get)

        texts = []
        for single in highest_scored:
            alt = text_tokenized.to("cpu")
            target_token_index, cand = single
            alt = torch.cat((alt[:, 1:target_token_index], torch.LongTensor([cand]).unsqueeze(0),
                            alt[:, target_token_index+1:-1]), dim=1)
            alt_text = self.mlm_tokenizer.batch_decode(alt)[0]
            texts.append((alt_text, replacements[single]))

        return texts
