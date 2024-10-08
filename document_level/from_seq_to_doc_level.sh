#!/bin/bash

python from_seq_to_doc_level.py \
    --original_dataset="imperial-cpg/pile_arxiv_doc_mia_sequences" \
    --original_dataset_split='train' \
    --doc_label_name='doc_idx' \
    --sequence_results='{SOME_PATH}/pile_arxiv_sequences_ratio.pickle'