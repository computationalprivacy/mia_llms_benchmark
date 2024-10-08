# Adapted from https://github.com/zhliu0106/probing-lm-data/

import copy
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from attacks import AbstractAttack
from datasets import Dataset, load_dataset
from sklearn.metrics import auc, roc_curve
from transformers import (AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)


class Hook:
    def __call__(self, _module, _module_inputs, module_outputs):
        self.out = module_outputs[0]


def attach_hooks(model):
    hooks = []
    handles = []
    for layer in model.model.layers:
        hook = Hook()
        handle = layer.register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    return hooks, handles


def detach_hooks(handles):
    for handle in handles:
        handle.remove()


class LRProbe(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 1, bias=False), torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def pred(self, x):
        return self(x).round()

    def score(self, x):
        return self(x)

    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device="cpu"):
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1]).to(device)

        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()

        return probe


class ActDataset:
    def __init__(self, activations, labels, center=True, scale=True):
        self.data = {}
        for layer, acts in activations.items():
            self.data[layer] = acts, labels
        self.center = center
        self.scale = scale

    def get(self, layer, device="cpu"):
        acts, labels = self.data[layer]
        if self.center:
            acts = acts - torch.mean(acts, dim=0)
        if self.scale:
            acts = acts / torch.std(acts, dim=0)
        return acts.to(device), labels.to(device)


def split_array(arr, proportions):
    if len(proportions) != 3 or not np.isclose(sum(proportions), 1):
        raise ValueError("Proportions must be a list of 3 numbers that sum to 1")

    arr = np.array(arr)
    np.random.shuffle(arr)

    split_indices = np.cumsum([int(len(arr) * p) for p in proportions[:-1]])
    return np.split(arr, split_indices)


def compute_auc(prediction, answers, print_result=True):
    fpr, tpr, _ = roc_curve(np.array(answers, dtype=bool), -np.array(prediction))
    auc_score = auc(fpr, tpr)
    return auc_score


def evaluate(probe_model, test_acts, test_labels):
    scores = probe_model.score(test_acts)

    predictions = []
    labels = []
    for score, label in zip(scores, test_labels):
        predictions.append(-score.item())
        labels.append(label.item())

    return compute_auc(predictions, labels)


class ProbeAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.proxy_device = config['ft_device']
        self.probe_device = config['pr_device']

    def run(self, dataset: Dataset) -> Dataset:
        ds_members, ds_non_members = self.load_aux_dataset()
        logging.info(f"Loaded aux: {len(ds_members)} members and {len(ds_non_members)} non-members")

        proxy_model = self.fine_tune_proxy(ds_members)
        probe_model, best_layer = self.train_probe_classifier(proxy_model, ds_members, ds_non_members)

        hook = Hook()
        handle = self.model.model.layers[best_layer].register_forward_hook(hook)

        dataset = dataset.map(
            lambda x: self.probe_score(x, probe_model, hook),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v2",
        )

        handle.remove()
        return dataset

    def probe_score(self, batch, probe_model, hook):
        with torch.no_grad():
            tokenized = self.tokenizer.batch_encode_plus(batch["text"], return_tensors='pt', padding="longest")
            token_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)

            self.model(token_ids, attention_mask=attention_mask)
            activations = hook.out[:, -1].to(self.probe_device)
            scores = probe_model(activations)
            return {self.name: scores}

    def collect_activations(self, proxy_model, ds_members, ds_non_members):
        hooks, handles = attach_hooks(proxy_model)
        activations = defaultdict(list)
        labels = []

        with torch.no_grad():
            for ds, label in [(ds_members, 1), (ds_non_members, 0)]:
                for record in ds:
                    input_ids = torch.tensor(record['input_ids']).unsqueeze(0).to(self.proxy_device)
                    attention_mask = torch.tensor(record['attention_mask']).unsqueeze(0).to(self.proxy_device)
                    proxy_model(input_ids, attention_mask=attention_mask)

                    for i, hook in enumerate(hooks):
                        activations[i].append(hook.out[0, -1].detach().cpu())

                labels.extend([label]*len(ds))

        labels = np.array(labels)
        for layer, act in activations.items():
            activations[layer] = torch.stack(act).float()

        detach_hooks(handles)
        return activations, labels

    def make_act_dataset(self, activations, labels, idxs):
        f_acts = {layer: activations[layer][idxs] for layer in activations}
        f_labels = torch.tensor(labels[idxs], dtype=torch.float32)
        ds = ActDataset(f_acts, f_labels, center=self.config['center'], scale=self.config['scale'])
        return ds

    def train_probe_classifier(self, proxy_model, ds_members, ds_non_members):
        activations, labels = self.collect_activations(proxy_model, ds_members, ds_non_members)
        indices = list(range(len(labels)))
        train_idx, dev_idx, test_idx = split_array(indices, [0.4, 0.3, 0.3])

        train_act_ds = self.make_act_dataset(activations, labels, train_idx)
        dev_act_ds = self.make_act_dataset(activations, labels, dev_idx)
        test_act_ds = self.make_act_dataset(activations, labels, test_idx)

        dev_auc_list = []
        test_auc_list = []
        for layer in activations:
            train_acts, train_labels = train_act_ds.get(layer, device=self.probe_device)
            probe_model = LRProbe.from_data(train_acts, train_labels, device=self.probe_device)

            dev_acts, dev_labels = dev_act_ds.get(layer, device=self.probe_device)
            dev_auc = evaluate(probe_model, dev_acts, dev_labels)
            dev_auc_list.append(dev_auc)

            test_acts, test_labels = test_act_ds.get(layer, device=self.probe_device)
            test_auc = evaluate(probe_model, test_acts, test_labels)
            test_auc_list.append(test_auc)
        dev_best_layer = dev_auc_list.index(max(dev_auc_list))

        logging.info(f"average dev auc: {sum(dev_auc_list)/len(dev_auc_list):.4f}")
        logging.info(f"max dev auc: {max(dev_auc_list):.4f} in layer_{dev_best_layer}")
        logging.info(f"test auc: {test_auc_list[dev_best_layer]:.4f} in layer_{dev_best_layer}")

        return probe_model, dev_best_layer

    def fine_tune_proxy(self, ds_members):
        dirname = f"{self.signature(ds_members)}_v1"
        dirpath = os.path.join(self.config['model_save_dir'], dirname)

        if os.path.exists(dirpath):
            logging.info(f"Loading fine-tuned model from {dirpath}")
            return AutoModelForCausalLM.from_pretrained(dirpath).to(self.proxy_device)

        logging.info("Fine-tuning proxy model")
        proxy_model = copy.deepcopy(self.model)
        proxy_model.config.pad_token_id = proxy_model.config.eos_token_id

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            seed=self.config['seed'],
            per_device_train_batch_size=self.config['ft_batch_size'],
            gradient_accumulation_steps=1,
            deepspeed="",
            output_dir=dirpath,
            overwrite_output_dir=True,
            num_train_epochs=self.config['ft_epochs'],
            learning_rate=self.config['ft_lr'],
            optim="adamw_torch",
            lr_scheduler_type="constant",
            dataloader_drop_last=False,
            bf16=True,
            bf16_full_eval=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            save_strategy="no",
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
        )

        trainer = Trainer(
            model=proxy_model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=ds_members,
        )
        trainer.train()
        trainer.save_model()
        logging.info(f"Fine-tuned model saved to {dirpath}")

        return proxy_model.to(self.proxy_device)

    def load_aux_dataset(self):
        aux_dataset = load_dataset(self.config['aux_non_member_dataset'], split=self.config['split'])
        aux_dataset = aux_dataset.shuffle(seed=self.config['seed']).select(range(self.config['samples']))
        split_dataset = aux_dataset.train_test_split(test_size=0.5, seed=self.config['seed'])

        ds_members, ds_non_members = split_dataset['train'], split_dataset['test']
        ds_members = ds_members.map(lambda x: self.tokenizer(
            x["text"]), batched=True, remove_columns=ds_members.column_names)

        ds_non_members = ds_non_members.map(lambda x: self.tokenizer(
            x["text"]), batched=True, remove_columns=ds_non_members.column_names)

        return ds_members, ds_non_members
