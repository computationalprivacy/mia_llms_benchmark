# Running document-level MIAs against LLMs

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

## Running document-level MIAs