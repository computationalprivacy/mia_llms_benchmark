# Benchmarking MIAs against LLMs. 

This repository contains the code (and instructions to access the datasets) 
for the paper ["SoK: Membership Inference Attacks on LLMs are Rushing
Nowhere (and How to Fix It)"](https://arxiv.org/pdf/2406.17975). 

We provide an easy-to-use framework to evaluate MIAs against LLMs, 
benchmarking their performance compared to MIAs from the literature 
across **the benchmarks recommended in the paper**. 

## (1) Installation

In your python environment, run `pip install -r requirements.txt` to install the required dependencies. 

To run this code, access to GPUs is recommended. To access some datasets and models, you might also want to 
log into to your Hugging Face account using `huggingface cli login` and providing your access token.

## (2) Accessing datasets

The code provided in this repo can be run on any target LLM and sets of member `(label = 1`) and non-member `(label = 0`) sequences of text. 

We specifically consider the evaluation datasets as recommended in the paper:

#### (2.1) MIMIR

MIMIR has been made available by [Duan et al. (COLM 2024)](https://arxiv.org/pdf/2402.07841) [here](https://huggingface.co/datasets/iamgroot42/mimir). 
MIMIR is a collection of text samples from the train (member) and test (non-member) split from [The Pile](https://huggingface.co/datasets/EleutherAI/pile). 

Potential target models include the suite of Pythia models (e.g. [here](https://huggingface.co/EleutherAI/pythia-6.9b)) or GPT-NEO models (e.g. [here](https://huggingface.co/EleutherAI/gpt-neo-1.3B)). 
Our understanding is that the deduplication executed on the Pile to create the "Pythia-dedup" models has been only done on the training dataset, suggesting this dataset of members/non-members also to be valid for these models (e.g. [here](https://huggingface.co/EleutherAI/pythia-6.9b-deduped))).

For more information on how the MIMIR dataset has been created we refer to [Duan et al.](https://arxiv.org/pdf/2402.07841). 

As stated in our paper, we recommend using the split `13_0.8` (yet not the Github split) where authors deduplicate non-member data, removing non-members with more than 80% overlap in 13-grams with the trainign dataset. 
Further deduplication might cause a distribution shift between members and non-members, making the setup not properly randomized for MIA evaluation.

We provide a config file specifically for MIMIR: `config_template_mimir.yaml`, in which 
one could easily change the target model and data subset(s). 

#### (2.2) Copyright Traps

Copyright traps (see [Meeus et al. (ICML 2024)](https://arxiv.org/pdf/2402.09363)) are unique, synthetically generated sequences 
who have been included into the training dataset of [CroissantLLM](https://huggingface.co/croissantllm/CroissantLLMBase). 
The dataset of traps has been released [here](https://huggingface.co/datasets/imperial-cpg/copyright-traps), 
containing non-members and members (for a different number of repetitions in the training dataset (10, 100, 1000)). 
Traps have been generated using [this code](https://github.com/computationalprivacy/copyright-traps) and by sampling text 
from LLaMA-2 7B while controlling for sequence length and perplexity. 
The dataset contains splits according to `seq_len_{XX}_n_rep_{YY}` where `XX={25,50,100}`and `YY={10, 100, 1000}`.

Also additional non-members generated in exactly the same way are provided [here](https://huggingface.co/datasets/imperial-cpg/copyright-traps-extra-non-members), 
which might be required for some MIA methodologies making additional assumptions for the attacker. 

We provide a config file specifically for the copyright traps: `config_template_traps.yaml`, in which 
one could easily change the data subset(s). 

## (3) Running MIAs against LLMs

The following command allows to run a range of MIA methodologies from the literature on the target model and dataset provided in the config file. 

`python main.py -c config_template_mimir.yaml --run-all --output={PATH_TO_OUTPUT}`

The flag `run-all` corresponds to running all attacks specified in the config file, while subsets of attacks can also be specified using e.g. 

`python main.py -c config_template_traps.yaml --attacks loss zlib ratio --output={PATH_TO_OUTPUT}`

All MIAs we implement can be found in `./attacks/`, where new attacks could easily be added. 
Currently, this repo contains the following attacks, where comments or default parameters are added when meaningful: 

| Attack Name               | Original Paper                                       | Comments on implementation |
|---------------------------|-----------------------------------------------------|------------------------|
| Bag of words (no target model) | [Meeus et al. (2024)]((https://arxiv.org/pdf/2406.17975)) | We use a random forest classifier based on the occurrences of words with a minimum document frequency of 0.05. |
| Loss                      | [Yeom et al. (2021)](https://arxiv.org/pdf/1709.01604) | NA |
| Lowercase                 | [Carlini et al. (2021)](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf) | NA|
| Zlib                      | [Carlini et al. (2021)](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf) | NA|
| Ratio-LLaMA-2             | [Carlini et al. (2021)](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf) | We allow for any reference model to used in the Ratio attack, but do recommend using the `meta-llama/Llama-2-7b-hf` for copyright traps. |
| Neighborhood              | [Mattern et al. (2023)](https://arxiv.org/pdf/2305.18462) | We follow the neighborhood generation procedure outlined in Section 2.2 of Mattern et al. with RoBERTa as an MLM, neighborhood size of 50 samples, dropout probability 0.7, and top-k = 10. |
| Min-K% Prob               | [Shi et al. (2023)](https://arxiv.org/pdf/2310.16789v3)           | Following the best performing setting reported in Shi et al., we use k = 20. |
| Min-K%++                  | [Zhang et al. (2024)](https://arxiv.org/pdf/2404.02936v1)            | In line with Min-K% Prob, we use k = 20. |
| ReCaLL                    | [Xie et al. (2024)](https://arxiv.org/pdf/2406.15968)              | We use a fixed prefix with 10 shots drawn from an auxiliary non-member distribution. |
| Probe (real)              | [Liu et al. (2024)](https://arxiv.org/pdf/2406.01333)       | We use the code provided alongside Liu et al.. For proxy model training, we use 4,000 samples from the auxiliary non-member distribution, randomly split into equally-sized subsets of members and non-members. We fine-tune the target model on the member subset for 10 epochs with AdamW optimizer, a constant learning rate of 1e-3, default weight decay of 1e-2, and a batch-size of 100. We then train a logistic regression classifier for the activation vector, for 1000 epochs with AdamW optimizer, a constant learning rate of 1e-3, and a weight decay of 1e-1. |
| PAC                       | [Ye et al. (2024)](https://arxiv.org/pdf/2405.11930)                | Using the implementation provided by Ye et al., we generate 10 augmentations with alpha parameter of 0.3. We compute polarized distance using k_max=0.05 and k_min=0.3, following the best performing results reported in the original paper. |
| SURP                      | [Zhang et al. (2024)](https://arxiv.org/pdf/2407.21248)       | Following the best performing setup reported in Section 5.3 of Zhang et al., we use k = 40 with an entropy threshold of 2.0. |
| CON-ReCall                | [Wang et al. (2024)](https://arxiv.org/pdf/2409.03363)           | We use 10-shot prefixes for both member- and non-member-conditioned likelihoods, drawn randomly and independently for each target example. Non-member prefixes are drawn from the auxiliary non-member distribution. For member prefixes, we assume partial access by the attacker, and draw 10 samples from the training dataset excluding the target example. |
| SaMIA                     | [Kaneko et al. (2024)](https://arxiv.org/pdf/2404.11262)     | Using the default setup in the implementation provided alongside Kaneko et al., we generate 10 candidates using top-k sampling with top-k=50, temperature of 1.0, and maximum length of 1,024. The prefix is set to 50% of the input sample. |

Finally, for document-level MIAs we refer to `./document_level/README.md`. 

## (4) References

If you found this repository useful for your work, kindly cite:

```
@article{meeus2024inherent,
  title={Inherent challenges of post-hoc membership inference for large language models},
  author={Meeus, Matthieu and Jain, Shubham and Rei, Marek and de Montjoye, Yves-Alexandre},
  journal={arXiv preprint arXiv:2406.17975},
  year={2024}
}
```

That is: until Google Scholar updates the title and author list :)