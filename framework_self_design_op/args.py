import os

import argparse

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", default=0.000001, type=float, help="learning rate")
    p.add_argument("--stop_val", default=0.05, type=float, help="error for stoping")
    p.add_argument("--t_epoch", default=10, type=int, help="epoch for lambda")
    p.add_argument("--optimizer", default="direct", type=str, help="select the optimizer")

    return p


def get_args():
    return get_parser().parse_args()