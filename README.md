# MoE
Some Mixture of Experts implementations :) 

## Overview
The idea for now is to just have simple implementations in one place, both as an overview and for easy access. We focus on MoEs for large language models where they are used to replace the standard feedforward layers in transformers.

For a broader overview of MoEs in this context, see [our shared doc](https://docs.google.com/document/d/1NuQ5jr7V-Jv1ui7p4KrxO_JTz-7bpYcYMmh49EeJ-QA/edit?usp=sharing).

Currently implemented:
* Classical linear gating with softmax + top-k
* Expert choice routing ([paper](https://arxiv.org/pdf/2202.09368v2.pdf))

We have preliminary results on small model pretraining (~65M-250M params, Mixtral style MoE) on different datasets that show a performance (perplexity) improvement similar to a double-depth (double-param) model, while keeping the FLOPS close to the base dense model (top-2 routing). 

Moreover, we already got [MegaBlocks](https://github.com/stanford-futuredata/megablocks) running on the EPFL cluster. It's a MoE library built on top of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), thereby supporting data, expert and pipeline parallel training of MoEs. So far, it seems we are able to reproduce the results from single-GPU training.

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
