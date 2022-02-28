title={{Safe Neurosymbolic Learning with Differentiable Symbolic Execution}}

This repo (master branch) contains the implementation for the paper [Safe Neurosymbolic Learning with Differentiable Symbolic Exectuion](https://openreview.net/forum?id=NYBmJN4MyZ) 

by [Chenxi Yang](https://cxyang1997.github.io/), [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/). Published in ICLR 2022.

--------------------

We study the problem of learning worst-case-safe parameters for programs that use neural networks as well as symbolic, human-written code. Such neurosymbolic programs arise in many safety-critical domains. However, because they can use nondifferentiable operations, it is hard to learn their parameters using existing gradient-based approaches to safe learning. Our approach to this problem, Differentiable Symbolic Execution (DSE), samples control flow paths in a program, symbolically constructs worst-case **safety losses** along these paths, and backpropagates the gradients of these losses through program operations using a generalization of the reinforce estimator. We evaluate the method on a mix of synthetic tasks and real-world benchmarks. Our experiments show that DSE significantly outperforms the state-of-the-art DiffAI method on these tasks. 


## How to run the code

### Pre-requisites
This repository requires to run on Python 3.8.12. and install PyTorch 1.10.2.

## Datasets
The trajectory datasets are available in our [Google drive](https://drive.google.com/drive/folders/1Icj5gYvRMdpm5_Ys_vE2T1W5HKTMnVul?usp=sharing). Please unzip the `Datasets.zip` and put the dataset under the directory of `dataset/` of this repository.

### Usage
Train and evaluate with the `run.py` with the common options:

```sh
run.py
   --mode: Three methods (Ablation, DiffAI+, DSE) described in the paper. 
   --benchmark_name: Benchmark Names.
   --AI_verifier_num_components: Configurations for verification.
   --num_components: Number of components used in training (For safety loss).
   --train_size: Number of training trajectories used for getting the data loss.
   --bound_start: The starting bound for safety constraint. 
     (default: 0)
   --bound_end: The ending bound for safety constraint.
     (default: 1)
   --nn_mode: Select the neural nerwork structure for a benchmark if applied. 
     (default: complex)
   --l: Neural network parameters. 
     (default: 64)
   --num_epoch: The number of epochs to run the training.
```

* `mode` defines the method to train. Our paper gives three methods in evaluation: Ablation, DiffAI+, DSE, which map to [`only_data`, `DiffAI`, `DSE`].

   * `only_data` method trains with data loss only.
   * `DiffAI` method trains with data loss and safety loss extracted from an extended version of [DiffAI](https://files.sri.inf.ethz.ch/website/papers/icml18-diffai.pdf).
   * `DSE` is the method of this work.

*  `benchmark_name` contains the benchmark names:

   * `thermostat_new` gives the thermostat benchmark with a 20-length loop, two neural networks modeling the cooler and heater, and safety constraint about avoiding extreme temperature.
   * `aircraft_collision_new` gives the aircraft collision benchmark with a 20-length loop to model the aircraft controller. Collision avoidance is the safety constraint.
   * `racetrack_relaxed_multi` gives the racetrack benchmark. This benchmark models two vehicles driving in a given map, trying to follow the path planner and not crash into each other or the walls.
   * `cartpole_v2` gives the cartpole benchmark. This benchmark models the linear approximation of [Cartpole](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) benchmark and constrains the position of a cartpole.

* `AI_verifier_num_components` gives the number of input components allowed in the verification part. Given the `AI_verifier_num_components` as `N`, 

   * if there is only one input symbolic variable, the number of input components into the system is `N`.
   * if there are `k` (`k` > 1) input symbolic variables, the number of input components into the system is `N^k`. 

* `num_components` gives the number of input components allowed for safety loss in training.
* `train_size` indicates the number of trajectory examples used for training data loss. Trajectory examples are in the form of a sequence of input-output pairs of the neural network module.
* `bound_start`, `bound_end` is the index of the safety constraint list. For each benchmark, we allow multiple safety constraints. By default, the number of safety constraint is 1.
* `nn_mode`, `l` are the neural network parameters. `nn_mode` is the neural network structure. `l` is the number of parameters. One benchmark has multiple neural network structures. 

## Add new benchmarks
When new datasets are in the form of the ones in the `Datasets.zip`, adding new benchmarks is available. You can add a new `.py` file under `benchmarks/` with the model version of the new benchmark, register the new benchmark in `import_hub.py`, and add the configurations in `constants.py`. More detailed instructions for the format are in `import_hub.py` and `constants.py`.

## References

If you find this work useful for your research, please consider citing
```bib
@inproceedings{
yang2022safe,
title={Safe Neurosymbolic Learning with Differentiable Symbolic Execution},
author={Chenxi Yang and Swarat Chaudhuri},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=NYBmJN4MyZ}
}
```
