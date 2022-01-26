import os

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'y', 't'):
        return True
    elif v.lower() in ('false', 'n', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    p = argparse.ArgumentParser()

    # the outer loop
    p.add_argument("--quick_mode", default=False, type=str2bool, help="whether increase the learning speed")

    # generate dataset
    p.add_argument("--dataset", default="thermostat", help="define the dataset")
    p.add_argument("--dataset_distribution", default="normal", help="define the distribution the dataset should follow")

    # other parameters
    p.add_argument("--lr", default=1e-03, type=float, help="learning rate")
    p.add_argument("--stop_val", default=0.05, type=float, help="error for stoping")
    p.add_argument("--t_epoch", default=1, type=int, help="epoch for lambda")

    p.add_argument("--w", default=0.5, type=float, help="the measure between two lagarangian iteration")
    p.add_argument("--gamma", default=1.0, type=float, help="The threshold of two learning")

    p.add_argument("--benchmark_name", default=None, help="represent the benchmark")
    p.add_argument("--data_size", default=10000, type=int, help="size of dataset, both for training and testing")
    
    p.add_argument("--num_epoch", default=100, type=int, help="number of epochs for training")
    p.add_argument("--width", default=0.1, type=float, help="width of perturbation") # for DiffAI
    
    p.add_argument("--nn_mode", default='all', help="how many NN used in model, 'single' means only used in the first one")
    p.add_argument("--l", default=10, type=int, help="size of hidden states in NN")
    p.add_argument("--b", default=100, type=int, help="range of lambda")
    p.add_argument("--module", default="linearrelu", help="module in model")

    # dataset
    p.add_argument("--data_attr", default="normal_52.0_59.0", help="dataset_attr")
    p.add_argument("--train_size", default=200, type=int, help="training size")
    p.add_argument("--test_size", default=20000, type=int, help="test size")
    p.add_argument("--generate_dataset", default=False, type=str2bool, help="generate the data set")
    p.add_argument("--fixed_dataset", default=False, type=str2bool, help="whether to use the same dataset")

    # constraint
    p.add_argument("--ini_unsafe_probability", default=0.0, type=float, help="the ini-unsafe_probability to handle")

    # perturbation
    p.add_argument("--num_components", default=10, type=int, help="number of components to split")
    p.add_argument("--bs", default=10, type=int, help="batch size by number of component")
    p.add_argument("--perturbation_width", default=0.3, type=float, help="the perturbation width in extracting input distribution")

    # training
    p.add_argument(
        "--score_f", 
        default="volume", 
        choices=['volume', 'hybrid', 'distance'], # volume: volume based, hybrid: volume + distance based
        help="define the score function used to calculate the sampling probability"
    )
    p.add_argument("--use_smooth_kernel", default=False, type=str2bool, help="decide whether to use smooth kernel")
    p.add_argument("--save", default=True, type=str2bool, help="decide whether to save the model or not")
    p.add_argument("--mode", help="which method used for training")
    p.add_argument("--adaptive_weight", default=False, type=str2bool, help="whether use another weight")
    p.add_argument(
        "--outside_trajectory_loss", 
        default=True, 
        type=str2bool, 
        help="whether use the new loss function"
    )
    p.add_argument("--verify_outside_trajectory_loss", default=False, type=str2bool,  help="define the verification method")
    p.add_argument(
        "--only_data_loss", 
        default=False, 
        type=str2bool, 
        help="only use data loss"
    )
    p.add_argument(
        "--early_stop", 
        default=True, 
        type=str2bool, 
        help="early stop"
    )
    p.add_argument("--data_bs", default=2, type=int, help="number of trajectories to use for data loss")
    p.add_argument("--use_data_loss", default=True, type=str2bool, help="use data loss")
    p.add_argument(
        "--data_safe_consistent", 
        default=True, 
        type=str2bool, 
        help="use data loss and safe loss simultanuously"
    )
    p.add_argument("--use_hoang", default=False, type=str2bool, help="whether use the outest optimization")
    p.add_argument("--bound_start", default=0, type=int, help=f"the index to start with in safe bound list")
    p.add_argument("--bound_end", default=20, type=int, help=f"the index to end within safe bound list")
    p.add_argument(
        "--use_abstract_components", 
        default=True, 
        type=str2bool, 
        help=f"use abstract components in diffAI"
        )
    p.add_argument(
        "--expr_i_number",
        default=3,
        type=int,
        help=f"the number"
    )
    p.add_argument(
        "--optimizer_method",
        default="Adam",
        help="use SGD/Adam"
    )
    p.add_argument(
        "--train_sample_size",
        type=int,
        default=30,
        help="how many paths to sample in DSE during training"
    )

    # smooth kernel in training
    p.add_argument("--sample_std", default=1.0, type=float, help=f"std to sample theta")
    p.add_argument("--sample_width", default=None, type=float, help=f"width to sample")

    # evaluation
    p.add_argument("--test_mode", default=False, type=str2bool, help="decide whether check load model and then test")
    p.add_argument("--extract_one_trajectory", default=False, type=str2bool, help="extract trajectory starting from one point")
    p.add_argument("--AI_verifier_num_components", default=500, type=int, help="components allowed when using AI as a verifier")
    p.add_argument("--SE_verifier_run_times", default=100, type=int, help="Times to run when using SE as a verifier")
    p.add_argument("--SE_verifier_num_components", default=1, type=int, help="components allowed when using SE as a verifier")

    # debug
    p.add_argument("--debug", default=False, type=str2bool, help="decide whether debug")
    p.add_argument("--debug_verifier", default=False, type=str2bool, help="debug verifier")
    p.add_argument("--cuda_debug", default=False, type=str2bool,  help="decide whether de cuda memory bug")
    p.add_argument("--simple_debug", default=False, type=str2bool, help="change max iteration")
    p.add_argument("--run_time_debug", default=False, type=str2bool, help="whether print sub time")
    p.add_argument("--profile", default=False, type=str2bool, help="print the running time of per part")
    
    # plot
    p.add_argument("--plot", default=False, type=str2bool, help="plot or not")
    return p


def get_args():
    return get_parser().parse_args()