# Running document-level MIAs against LLMs

We here provide the instructions, code and data used to run the document-level MIA results 
as reported in ["SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It)"](https://arxiv.org/pdf/2406.17975).

## Datasets

In the paper, we consider full ArXiv documents as members and non-members to conduct document-level MIAs against LLMs. 
We consider two setups: 

**1. ArXiv papers from the train and test split from the Pile**. 
The full documents (n=2000) with the corresponding label for membership can be downloaded [here](https://huggingface.co/datasets/imperial-cpg/pile_arxiv_doc_mia) on Hugging Face. 
We also provide this dataset containing the first 25 sequences of 200 words [here](https://huggingface.co/datasets/imperial-cpg/pile_arxiv_doc_mia_sequences). 

The dataset can be used to develop and evaluate document-level MIAs against LLMs trained on The Pile. 
Target models include the suite of Pythia and GPTNeo models, to be found [here](https://huggingface.co/EleutherAI).

**2. Setup inspired by Regression Discontinuity Design (RDD)**. 
We here sample ArXiv papers release shortly before (members) and after (non-members) the training data cutoff date for the [OpenLLaMA models](https://huggingface.co/openlm-research/open_llama_7b). 

The OpenLLaMA models (V1) have been trained on [RedPajama data](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T).
The last batch of ArXiv papers included in this dataset are papers published in February 2023. 
To get the members close to the cutoff data, we collect the 13,155 papers published in "2302" as part of the training dataset. 
We process the raw LateX files using this [script](https://github.com/togethercomputer/RedPajama-Data/blob/rp_v1/data_prep/arxiv/run_clean.py) and make the resulting data available [here](https://huggingface.co/datasets/imperial-cpg/arxiv_redpajama_2302).

For the non-members, we collect ArXiv papers originally released in March 2023. 
We use the method as made available by ArXiv (at a small cost) to download papers in bulk from their AWS S3 bucket (see [here](https://info.arxiv.org/help/bulk_data_s3.html)), 
using the script from RedPajama [here](https://github.com/togethercomputer/RedPajama-Data/blob/rp_v1/data_prep/arxiv/run_download.py). 
We collect all 16,213 papers released in "2303" and preprocess them in exactly the same way as the member documents.
Unfortunately, we do not have the license to redistribute this set of non-member data.

If members and non-members are collected in this RDD setup, (document-level) MIAs can be developed and evaluated against the suite of [OpenLLaMA models](https://huggingface.co/openlm-research/open_llama_7b). 

## Running document-level MIAs

As mentioned in [the paper](https://arxiv.org/pdf/2406.17975), one can evaluate document-level MIAs by aggregating sequence-level MIA scores for the same document. 
Proposed by [Shi et al.](https://arxiv.org/pdf/2310.16789v3), one method to do so is computing a threshold for which the sequence-level MIA reaches a maximum accuracy (on a training set of documents), 
after which the average sequence-level binary prediction can be used as the document-level scoring function. 

Assuming the sequence-level results have been computed using the `../main.py`, and saved in `PATH_TO_RESULTS`, 
we provide the code to compute the document-level MIA in `from_seq_to_doc_level.py`, 
with an example script `from_seq_to_doc_level.sh`. 
Note that you need access to the original dataset as well, to access the document indices. 

For the other document-level MIAs from [Meeus et al.](https://arxiv.org/pdf/2310.15007) implemented in the paper, 
we refer to [their repository](https://github.com/computationalprivacy/document-level-membership-inference). 