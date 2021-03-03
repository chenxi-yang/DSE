import os

import argparse

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", default=0.000001, type=float, help="learning rate")
    p.add_argument("--stop_val", default=0.05, type=float, help="error for stoping")
    p.add_argument("--t_epoch", default=10, type=int, help="epoch for lambda")
    p.add_argument("--optimizer", default="direct", type=str, help="select the optimizer")
    p.add_argument("--w", default=0.5, type=float, help="the measure between two lagarangian iteration")
    # p.add_argument("--noise", default=0.1, type=float, help="represent the noise range in gradient descent direct noise")
    p.add_argument("--benchmark_id", type=float, help="represent the benchmark")
    p.add_argument("--data_size", default=10000, type=int, help="size of dataset, both for training and testing")
    p.add_argument("--test_portion", default=0.99, type=float, help="portion of test set of the entire dataset")
    p.add_argument("--num_epoch", default=10, type=int, help="number of epochs for training")
    p.add_argument("--width", default=0.1, type=float, help="width of perturbation")
    p.add_argument("--noise", default=0.05, type=float, help="add noise to avoid local min")
    p.add_argument("--bs", default=10, type=int, help="batch size")
    return p


def get_args():
    return get_parser().parse_args()