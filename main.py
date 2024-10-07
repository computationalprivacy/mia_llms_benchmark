import argparse
import logging
import pickle
from collections import defaultdict

import numpy as np
import torch
from attacks.utils import batch_nlloss
from datasets import Dataset, load_dataset
from sklearn.metrics import auc, roc_curve
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (get_available_attacks, load_attack, load_config,
                   load_mimir_dataset, set_seed)

logging.basicConfig(level=logging.INFO)


def init_model(model_name, device):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_header(config):
    header = ["MIA", "AUC"]
    for t in config["fpr_thresholds"]:
        header.append(f"TPR@FPR={t}")
    return header


def get_printable_ds_name(ds_info):
    if "name" in ds_info:
        name = ds_info["name"]
    elif "mimir_name" in ds_info:
        name = ds_info["mimir_name"]
    else:
        raise ValueError()
    name = f"{name}/{ds_info['split']}"
    name = name.replace("/", "_")
    return name


def init_dataset(ds_info, model, tokenizer, device, batch_size):

    if "mimir_name" in ds_info:
        if "name" in ds_info:
            raise ValueError("Cannot specify both 'name' and 'mimir_name' in dataset config")
        dataset = load_mimir_dataset(name=ds_info["mimir_name"], split=ds_info["split"])
    elif "name" in ds_info:
        dataset = load_dataset(ds_info["name"], split=ds_info["split"])
    else:
        raise ValueError("Dataset name is missing")

    dataset = dataset.map(
        lambda x: batch_nlloss(x, model, tokenizer, device),
        batched=True,
        batch_size=batch_size,
        new_fingerprint=f"{dataset.split}_croissant_ppl_v3",
    )
    return dataset


def results_with_bootstrapping(y_true, y_pred, fpr_thresholds, n_bootstraps=1000):
    n = len(y_true)
    aucs = []
    tprs = {}
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, n, replace=True)
        fpr, tpr, _ = roc_curve(np.array(y_true)[idx], np.array(y_pred)[idx])
        aucs.append(auc(fpr, tpr))
        for t in fpr_thresholds:
            if t not in tprs.keys():
                tprs[t] = [tpr[np.argmin(np.abs(fpr - t))]]
            else:
                tprs[t].append(tpr[np.argmin(np.abs(fpr - t))])

    results = [f"{np.mean(aucs): .4f} ± {np.std(aucs):.4f}"] + \
        [f"{np.mean(tprs[t]): .4f} ± {np.std(tprs[t]):.4f}" for t in fpr_thresholds]
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run attacks')
    parser.add_argument('-c', '--config', type=str, help='Config path', required=True)
    parser.add_argument('--attacks', nargs='*', type=str, help='Attacks to run.')
    parser.add_argument('--run-all', action='store_true', help='Run all available attacks')
    parser.add_argument('--seed', type=int, help='Random seed', default=None)
    parser.add_argument('--output', type=str, help="File to store attack results", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    config = load_config(args.config)
    logging.debug(config)
    global_config = config['global']
    device = global_config["device"]
    if args.run_all:
        attacks = get_available_attacks(config)
    else:
        attacks = args.attacks

    model, tokenizer = init_model(global_config['target_model'], device)

    results_to_save = defaultdict(dict)
    results_to_print = {}
    for ds_info in global_config['datasets']:
        dataset = init_dataset(
            ds_info=ds_info,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=global_config["batch_size"]
        )
        ds_name = get_printable_ds_name(ds_info)

        results = []
        header = get_header(global_config)

        y_true = [x["label"] for x in dataset]
        results_to_save[ds_name]["label"] = y_true

        for attack_name in sorted(attacks):
            logging.info(f"Running {attack_name} on {ds_name}")

            attack = load_attack(attack_name, model, tokenizer, config[attack_name])
            dataset = attack.run(dataset)
            y = [x[attack_name] for x in dataset]
            results_to_save[ds_name][attack_name] = y

            attack_results = results_with_bootstrapping(y_true, y, fpr_thresholds=global_config["fpr_thresholds"],
                                                        n_bootstraps=global_config["n_bootstrap_samples"])

            results.append([attack_name] + attack_results)
            logging.info(f"AUC {attack_name} on {ds_name}: {attack_results[0]}")

        results_to_print[ds_name] = tabulate(results, headers=header, tablefmt="outline")

    for ds_name, res in results_to_print.items():
        print(f"Dataset: {ds_name}")
        print(res)
        print()

    if args.output is not None:
        with open(args.output, 'wb') as f:
            pickle.dump(results_to_save, f)
