
from constants import *
import constants

if mode == 'DSE':
    from gpu_DSE.train import *
    from gpu_DSE.data_generator import load_data
if mode == 'DiffAI':
    from gpu_DiffAI.train import *
    from gpu_DiffAI.data_generator import load_data
if mode == 'SPS':
    from gpu_SPS.train import *
    from gpu_SPS.data_generator import load_data
if mode == 'SPS-sound':
    from gpu_SPS_sound.train import *
    from gpu_SPS_sound.data_generator import load_data

from args import *
from evaluation_sound import verification
from evaluation_unsound import verification_unsound
import domain

import random
import time

from utils import (
    extract_abstract_representation,
    show_cuda_memory,
)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    for path_sample_size in path_num_list:
        for safe_range_bound_idx, safe_range_bound in enumerate(safe_range_bound_list):
            if safe_range_bound_idx < bound_end and safe_range_bound_idx >= bound_start:
                pass
            else:
                continue

            for i in range(1):
                for t in range(t_epoch):
                    analyze_trajectory(
                        trajectory_path=f"{trajectory_log_prefix}_{safe_range_bound}_{i}.txt"
                        )







