# Safe Neurosymbolic Learning with Differentiable Symbolic Execution

This repo contains the implementation for the paper [Safe Neurosymbolic Learning with Differentiable Symbolic Exectuion](https://openreview.net/forum?id=NYBmJN4MyZ) 

by Chenxi Yang, Swarat Chaudhuri. Published in ICLR 2022.


## How to run the code
### Pre-requisites
This repository requires to run on Python 3.8.12. and install PyTorch 1.10.2.

### Usage
Train and evaluate with the `run.py` with the common options:

```sh
run.py
   --mode: Three methods (Ablation, DiffAI+, DSE) described in the paper. 
   --benchmark_name: Benchmark Names.
   --num_epoch: The number of epochs to run the training.
   --AI_verifier_num_components: Configurations for verification.
   --nn_mode: Select the neural nerwork structure for a benchmark if applied. 
     (default: complex)
   --l: Neural network parameters.
   --num_components: Number of components used in training (For safety loss).
   --train_size: Number of training trajectories used for getting the data loss.
   --bound_start: The starting bound for safety constraint. 
     (default: 0)
   --bound_end: The ending bound for safety constraint.
     (default: 1)
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

* `AI_verifier_num_components` gives the number of components allowed in the verification part. 




## Add new Benchmarks

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
