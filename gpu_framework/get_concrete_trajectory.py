import numpy as np
from termcolor import colored

from constants import *
import constants
import importlib

from utils import (
    load_model,
)

import import_hub as hub
importlib.reload(hub)
from import_hub import *


def store_trajectory(output_states, trajectory_path, category=None):
    trajectory_path = trajectory_path + f"_{category}"
    trajectory_path += ".txt"
    trajectory_log_file = open(trajectory_path, 'w')
    trajectory_log_file.write(f"{constants.name_list}\n")
    for trajectory_idx, trajectory in enumerate(output_states['trajectories']):
        trajectory_log_file.write(f"trajectory_idx {trajectory_idx}\n")
        for state in trajectory:
            for x in state:
                trajectory_log_file.write(f"{float(x.left)}, {float(x.right)};")
            trajectory_log_file.write(f"\n")
    trajectory_log_file.close()
    return 


def extract_trajectory(
        model_path,
        model_name,
        ini_states,
        trajectory_path,
    ):
    m = Program(l=l, nn_mode=nn_mode)

    _, m = load_model(m, model_path, name=model_name)
    if m is None:
        print(f"model path: {model_path}/{model_name} no model to extract concrete trajectory")
        return 
        # raise ValueError(f"No model to extract concrete trajectory!")
    if torch.cuda.is_available():
        m.cuda()
    m.eval()

    for param in m.parameters():
        param.requires_grad = False
    
    output_states = m(ini_states)
    store_trajectory(output_states, trajectory_path, category="single")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    print(f"#### Extract Trajectory ####")
    for safe_range_bound in safe_range_bound_list:
        print(f"Safe Range Bound: {safe_range_bound}")

        for i in range(5):
            constants.status = 'verify_AI'
            import import_hub as hub
            importlib.reload(hub)
            from import_hub import *

            ini_states = initialization_components_point()

            extract_trajectory(
                model_path=MODEL_PATH,
                model_name=f"{model_name_prefix}_{safe_range_bound}_{i}_{0}",
                ini_states=ini_states,
                trajectory_path=f"{trajectory_log_prefix}_{safe_range_bound}_{i}"
            )
