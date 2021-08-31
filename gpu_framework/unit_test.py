from utils import select_argmax
import torch
import time

profile = True

def extract_result():
    a = [[[1.0, 1.5], [2.0, 2.5]]] * 20
    trajectories = [a] * 500
    p_list = [0.97] * 500
    safe_interval = [0.5, 2.0]
    component_loss = 0.0
    real_safety_loss = 0.0

    for trajectory, p in zip(trajectories, p_list):
        if profile:
            start_trajectory = time.time()
        unsafe_penalty = 0.0

        for state_idx, state in enumerate(trajectory):
            l, r = state[0][0], state[1][0]

            if profile:
                start_unsafe_penalty = time.time()
            intersection_l, intersection_r = max(l, safe_interval[0]), min(r, safe_interval[1])
            if profile:
                end_intersection = time.time()
                print(f"--INTERSECTION: {end_intersection - start_unsafe_penalty}")
            if intersection_r < intersection_l:
                unsafe_value = max(l - (safe_interval[0]), safe_interval[1] - (r))
                unsafe_value = unsafe_value + 1.0
            else:
                safe_portion = (intersection_r - intersection_l + 1e-10) / ((r - l) + 1e-10)
                unsafe_value = 1 - safe_portion
            if profile:
                end_calculation = time.time()
                print(f"--CALCULATION: {end_calculation - end_intersection}")
            unsafe_penalty = max(unsafe_penalty, unsafe_value)
            if profile:
                end_unsafe_penalty = time.time()
                print(f"---UNSAFE PENALTY: {end_unsafe_penalty - end_calculation}")
        
        component_loss += p * unsafe_penalty + unsafe_penalty
        real_safety_loss += unsafe_penalty
        if profile:
            end_trajectory = time.time()
            print(f"--ONE TRAJECTORY: {end_trajectory - start_trajectory}")
            exit(0)
    component_loss /= len(p_list)
    real_safety_loss /= len(p_list)

    return component_loss, real_safety_loss


def test_select_argmax():
    torch.manual_seed(0)
    interval_right = torch.sigmoid(torch.randn(6, 3))
    interval_left = interval_right - torch.sigmoid(torch.randn(6, 3)) * 0.1
    interval_left[interval_left < 0] = 0
    if torch.cuda.is_available():
        interval_right = interval_right.cuda()
        interval_left = interval_left.cuda()

    print(f"interval_right:\n{interval_right}\ninterval_left:\n{interval_left}")
    index_mask = select_argmax(interval_left, interval_right)
    print(f"index_mask:\n{index_mask}")
    return 

def test_safe_loss_calculation():
    start = time.time()
    extract_result()
    end = time.time()
    print(f"--NORMAL TIME: {end - start}")

test_safe_loss_calculation()