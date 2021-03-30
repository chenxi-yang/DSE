import os

import argparse

def get_parser():
    p = argparse.ArgumentParser()

    # generate dataset
    p.add_argument("--dataset", default="thermostat", help="define the dataset")
    p.add_argument("--dataset_distribution", default="normal", help="define the distribution the dataset should follow")
    
    return p


def get_args():
    return get_parser().parse_args()