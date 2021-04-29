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

    # generate dataset
    p.add_argument("--dataset", default="thermostat", help="define the dataset")
    p.add_argument("--dataset_distribution", default="normal", help="define the distribution the dataset should follow")

    # other parameters
    p.add_argument("--lr", default=0.000001, type=float, help="learning rate")
    p.add_argument("--stop_val", default=0.05, type=float, help="error for stoping")
    p.add_argument("--t_epoch", default=10, type=int, help="epoch for lambda")
    p.add_argument("--optimizer", default="direct", type=str, help="select the optimizer")
    p.add_argument("--w", default=0.5, type=float, help="the measure between two lagarangian iteration")

    p.add_argument("--benchmark_name", default="benchmark", help="represent the benchmark")
    p.add_argument("--data_size", default=10000, type=int, help="size of dataset, both for training and testing")
    p.add_argument("--test_portion", default=0.99, type=float, help="portion of test set of the entire dataset")
    p.add_argument("--num_epoch", default=10, type=int, help="number of epochs for training")
    p.add_argument("--width", default=0.1, type=float, help="width of perturbation") # for DiffAI
    
    p.add_argument("--n", default=5, type=int, help="number of theta sampled around mean")
    p.add_argument("--nn_mode", default='all', help="how many NN used in model, 'single' means only used in the first one")
    p.add_argument("--l", default=10, type=int, help="size of hidden states in NN")
    p.add_argument("--b", default=1000, type=int, help="range of lambda")
    p.add_argument("--module", default="linearrelu", help="module in model")

    # dataset
    p.add_argument("--data_attr", default="normal_52.0_59.0", help="dataset_attr")
    p.add_argument("--train_size", default=200, type=int, help="training size")
    p.add_argument("--test_size", default=20000, type=int, help="test size")
    p.add_argument("--generate_all_dataset", default=False, type=str2bool, help="generate the data set")
    p.add_argument("--fixed_dataset", default=False, type=str2bool, help="whether to use the same dataset")

    # perturbation
    p.add_argument("--num_components", default=10, type=int, help="number of components to split")
    p.add_argument("--bs", default=10, type=int, help="batch size by number of component")
    p.add_argument("--perturbation_width", default=0.3, type=float, help="the perturbation width in extracting input distribution")

    # training
    p.add_argument("--use_smooth_kernel", default=False, type=str2bool, help="decide whether to use smooth kernel")
    p.add_argument("--save", default=True, help="decide whether to save the model or not")
    p.add_argument("--mode", help="which method used for training")
    p.add_argument("--adaptive_weight", default=False, type=str2bool, help="whether use another weight")
    p.add_argument("--outside_trajectory_loss", default=False, type=str2bool, help="whether use the new loss function")
    p.add_argument("--verify_outside_trajectory_loss", default=False, type=str2bool,  help="define the verification method")
    p.add_argument("--only_data_loss", default=False, type=str2bool, help="only use data loss")
    p.add_argument("--data_bs", default=2, type=int, help="number of trajectories to use for data loss")
    p.add_argument("--use_data_loss", default=True, type=str2bool, help="use data loss")
    
    # evaluation
    p.add_argument("--test_mode", default=False, type=str2bool, help="decide whether check load model and then test")
    p.add_argument("--verification_num_components", default=1000, type=int, help="componentts in  verification")
    p.add_argument("--verification_num_abstract_states", default=500, type=int, help="allowed number of abstract states in module sound")
    p.add_argument("--real_unsafe_value", default=True, help="get the real unsafe value")
    p.add_argument("--sound_verify", default=False, type=str2bool, help="use sound verify when test_mode is True")
    p.add_argument("--unsound_verify", default=False, type=str2bool, help="use unsound verify when test_mode is False")
    p.add_argument("--use_probability", default=True, type=str2bool, help="use probability, ow, worst case training")
    
    # debug
    p.add_argument("--debug", default=False, type=str2bool, help="decide whether debug")
    p.add_argument("--cuda_debug", default=False, type=str2bool,  help="decide whether de cuda memory bug")
    return p


def get_args():
    return get_parser().parse_args()