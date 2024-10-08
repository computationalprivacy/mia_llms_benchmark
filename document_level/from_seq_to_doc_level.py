import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import random
from datasets import Dataset, load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, auc, roc_curve

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

logging.basicConfig(level=logging.INFO)

def create_dfs(original_dataset: Dataset, results: dict, doc_label_name: str, test_size: float) -> tuple:
    df = pd.DataFrame.from_dict(results)
    assert doc_label_name in original_dataset.column_names
    df[doc_label_name] = original_dataset[doc_label_name]

    test_docs = random.sample(list(df[doc_label_name].unique()),
                              int(df[doc_label_name].nunique() * test_size))
    test_df = df[df[doc_label_name].isin(test_docs)]
    train_df = df[~df[doc_label_name].isin(test_docs)]

    return train_df, test_df

def determine_tau(df: pd.DataFrame, col: str, n_interpol: int=500) -> tuple:
    vals, seq_labels = df[col].values, df['label'].values

    print("Determining tau")
    thresholds = np.linspace(min(vals), max(vals), n_interpol)
    best_acc = 0
    for tau in tqdm(thresholds):
        y_pred = [1 if val >= tau else 0 for val in vals]
        acc = accuracy_score(seq_labels, y_pred)
        if acc >= best_acc:
            best_tau = tau
            best_acc = acc

    return best_tau, best_acc

def apply_tau(df: pd.DataFrame, col: str, tau: float, doc_label_name: str) -> tuple:
    df['seq_level_pred'] = [1 if df[col].values[i] >= tau else 0 for i in range(len(df))]
    grouped_by_doc = df.groupby(doc_label_name, as_index=False)[['seq_level_pred', 'label']].mean()
    return grouped_by_doc.label, grouped_by_doc.seq_level_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scale up sequence-level MIA to the document-level')
    parser.add_argument('--original_dataset', type=str, help='Path to the original dataset containing'
                                                             'multiple sequences per document.')
    parser.add_argument('--original_dataset_split', default=None, help='Split of original dataset.')
    parser.add_argument('--doc_label_name', type=str, help='Column name of the document label of the original dataset.')
    parser.add_argument('--sequence_results', type=str, help='Path to sequence level results')
    parser.add_argument('--fpr_thresholds', default=[0.01, 0.1], help='List of FPR rates to compute TPR')
    parser.add_argument('--n_bootstraps', default=50, help='Number of bootstraps to compute confidence intervals')
    parser.add_argument('--doc_test_size', default=0.5, help='Number of documents used for evaluation')
    args = parser.parse_args()

    # load the original dataset, needed to combine results to the document-level again
    original_dataset = load_dataset(args.original_dataset, split=args.original_dataset_split)

    # load results
    with open(args.sequence_results, 'rb') as f:
        sequence_results = pickle.load(f)
    # assume only one dataset in the results, and multiple attacks
    sequence_results = list(sequence_results.values())[0]
    attacks = [k for k in sequence_results.keys() if k != 'label']

     # first let's get the results across all sequences
    y_true = sequence_results['label']
    for attack in attacks:
        scores = sequence_results[attack]
        attack_results = results_with_bootstrapping(y_true, scores, fpr_thresholds=args.fpr_thresholds,
                                                       n_bootstraps=args.n_bootstraps)
        logging.info(f"AUC {attack} on all sequences: {attack_results[0]}")
        for i, fpr in enumerate(args.fpr_thresholds):
            logging.info(f"TPR@FPR={fpr} on all sequences: {attack_results[i+1]}")

    # now scale up to the document-level
    train_df, test_df = create_dfs(original_dataset, sequence_results,
                                   doc_label_name = args.doc_label_name, test_size = args.doc_test_size)

    for attack in attacks:
        best_tau, best_acc = determine_tau(train_df, attack, n_interpol=500)
        y_true_doc, y_pred_doc = apply_tau(test_df, attack, best_tau, doc_label_name = args.doc_label_name)
        attack_results = results_with_bootstrapping(y_true_doc, y_pred_doc, fpr_thresholds=args.fpr_thresholds,
                                                       n_bootstraps=args.n_bootstraps)
        logging.info(f"AUC {attack} on the document-level: {attack_results[0]}")
        for i, fpr in enumerate(args.fpr_thresholds):
            logging.info(f"TPR@FPR={fpr} on all sequences: {attack_results[i+1]}")
