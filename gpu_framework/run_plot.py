'''
TODO:
# use a way to smooth data loss
'''
from constants import *
import constants
from args import *
import importlib
import domain

import random
import time

from utils import (
)


if __name__ == "__main__":
    constants.status = 'verify_AI'

    for safe_range_bound in safe_range_bound_list:
        print(f"Safe Range Bound: {safe_range_bound}")
        for idx, safe_range in enumerate(safe_range_list): 
            # Run 5 times
            for i in range(5):
                if benchmark_name == "thermostat":
                    import benchmarks.thermostat as tm
                    importlib.reload(tm)
                    from benchmarks.thermostat import *
                elif benchmark_name == "mountain_car":
                    import benchmarks.mountain_car as mc
                    importlib.reload(mc)
                    from benchmarks.mountain_car import *
                elif benchmark_name == "unsmooth_1":
                    from benchmarks.unsmooth import *
                elif benchmark_name == "unsmooth_2_separate":
                    from benchmarks.unsmooth_2_separate import *
                elif benchmark_name == "unsmooth_2_overall":
                    from benchmarks.unsmooth_2_overall import *
                elif benchmark_name == "path_explosion":
                    from benchmarks.path_explosion import *
                elif benchmark_name == "path_explosion_2":
                    from benchmarks.path_explosion_2 import *
                
                target_model_name = f"{model_name_prefix}_{safe_range_bound}_{i}_{0}"

                m = Program(l=l, nn_mode=nn_mode)
                epochs_to_skip, m = load_model(m, MODEL_PATH, name=target_model_name)
                if m is None:
                    print(f"no model.")

                ini_states = initialize_components()









