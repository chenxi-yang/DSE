# Safe Neurosymbolic Learning with Differentiable Symbolic Execution

This repo contains the implementation for the paper [Safe Neurosymbolic Learning with Differentiable Symbolic Exectuion](https://openreview.net/forum?id=NYBmJN4MyZ) 

by Chenxi Yang, Swarat Chaudhuri. Published in ICLR 2022.


## How to run the code
### Pre-requisites
This repository requires to run on Python 3.8.12. and install PyTorch 1.10.2.

### Usage
Train and evaluate with the `run.py` with the common options:

> run.py
>   --mode
>   --benchmark_name
>   --num_epoch
>   --AI_verifier_num_components
>   --bound_start
>   --bound_end
>   --nn_mode
>   --l
>   --num_components
>   --bs
>   --train_size


## Add new Benchmarks

