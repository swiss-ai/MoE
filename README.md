# MoE
Some Mixture of Experts implementations :) 

## Overview
The idea of this repo is just to have simple implementations of MoEs in one place, both as an overview and for easy access. We focus on MoEs for large language models (in medium-sized GPT-2s), where they are used to replace the standard feedforward layers in transformers. Plug and play with it inside our modular [llm-baselines](https://github.com/epfml/llm-baselines) codebase, which extends [nanoGPT](https://github.com/karpathy/nanogpt) with different datasets!

For a broader overview of MoEs in the LLM context, see [our shared doc](https://docs.google.com/document/d/1NuQ5jr7V-Jv1ui7p4KrxO_JTz-7bpYcYMmh49EeJ-QA/edit?usp=sharing).

Currently implemented:
* Classical linear gating with softmax + top-k
* Expert choice routing ([paper](https://arxiv.org/pdf/2202.09368v2.pdf))

We have preliminary results on small model pretraining (~65M-250M params, Mixtral style MoE) on different datasets that show a performance improvement similar to a double-depth (double-param) model; all while keeping the FLOPS close to the base dense model (top-2 routing). 

## Files
The files are the following:
```sh
gpt.py      # contains the standard transformer base architecture (GPT-2 style, similar to nanoGPT)
moe.py      # contains the mixture of experts block
aux_losses.py # the typical load balancing losses used for MoEs
```


## Contact
If you are interested in this effort, please reach out to us on the the Swiss AI slack :)

Alex Hägele (alexander.hagele@epfl.ch), Martin Jaggi (martin.jaggi@epfl.ch).
