'''
TODO:
# use a way to smooth data loss
'''
from constants import *
import constants
from args import *
import importlib

from data_loader import *
import domain

import random
import time

from utils import (
    extract_abstract_representation,
    count_parameters,
    append_log,
)


def best_lambda(q_hat, c_hat):
    if c_hat.data.item() <= 0.0:
        res_lambda = var(0.0)
    else:
        res_lambda = B
    return q_hat.add(res_lambda.mul(c_hat)) #L_max


def best_theta(tmp_m_name,
    components, 
    lambda_, 
    epoch, 
    target, 
    lr, 
    bs, 
    nn_mode, 
    l, 
    save, 
    epochs_to_skip, 
    data_bs):
    m = Program(l=l, nn_mode=nn_mode)
    q, c, time_out = learning(
        m=m, 
        components=components,
        lambda_=new_lambda,
        epoch=num_epoch, 
        target=target,
        lr=lr,
        bs=bs,
        nn_mode=nn_mode,
        l=l,
        save=save,
        epochs_to_skip=epochs_to_skip,
        model_name=target_model_name,
        data_bs=data_bs,
        )

    return q.add(new_lambda.mul(c))


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    for safe_range_bound in safe_range_bound_list:
        if not debug:
            append_log([file_dir, file_dir_evaluation], f"path_sample_size: {SAMPLE_SIZE}, safa_range_bound: {safe_range_bound}\n")
        print(f"Safe Range Bound: {safe_range_bound}")

        # update target, fix the left endpoint, varify the right endpoint
        target = list()
        for idx, safe_range in enumerate(safe_range_list):
            if multi_agent_mode is True:
                # target distance, x1, x2
                # distance
                target.append(
                    {   # constraint is in box domain
                        "condition": domain.Interval(var(safe_range[0]), var(safe_range_bound)) if map_mode is False else None,
                        "method": method_list[0],
                        "name": name_list[0],
                        "map_condition": None,
                        "map_mode": map_mode,
                        "distance": True,
                    }
                )
                # agent1
                target.append(
                    {   # constraint is in box domain
                        "condition": domain.Interval(var(safe_range[0]), var(safe_range_bound)) if map_mode is False else None,
                        "method": method_list[1],
                        "name": name_list[1],
                        "map_condition": None,
                        "map_mode": map_mode,
                        "distance": False,
                    }
                )
                # agent2
                target.append(
                    {   # constraint is in box domain
                        "condition": domain.Interval(var(safe_range[0]), var(safe_range_bound)) if map_mode is False else None,
                        "method": method_list[2],
                        "name": name_list[2],
                        "map_condition": None,
                        "map_mode": map_mode,
                        "distance": False,
                    }
                )
            elif benchmark_name == 'aircraft_collision_new':
                 target.append(
                    {   # constraint is in box domain
                        "condition": None,
                        "method": method_list[0],
                        "name": name_list[0],
                        "map_condition": None,
                        "map_mode": map_mode,
                        "distance": True,
                    }
                )
            else:
                target.append(
                    {   # constraint is in box domain
                        "condition": domain.Interval(var(safe_range[0]), var(safe_range_bound)) if map_mode is False else None,
                        "method": method_list[idx],
                        "name": name_list[idx],
                        "map_condition": None,
                        "map_mode": map_mode,
                        "distance": False,
                    }
                )
            if map_mode is True:
                map_condition = list()
                for constraint_l in map_safe_range:
                    interval_l = list()
                    for constraint in constraint_l:
                        interval_l.append(domain.Interval(var(constraint[0]), var(constraint[1])))
                    map_condition.append(interval_l)
                if multi_agent_mode is True:
                    target[1]['map_condition'] = map_condition
                    target[2]['map_condition'] = map_condition
                    distance_condition = list()
                    for constraint_l in distance_safe_range:
                        interval_l = list()
                        for constraint in constraint_l:
                            interval_l.append(domain.Interval(var(constraint[0]), var(constraint[1])))
                        distance_condition.append(interval_l)
                    target[0]['map_condition'] = distance_condition
                else:
                    target[0]['map_condition'] = map_condition
                
            N = 20
            
            for i in range(N):
                constants.status = 'train'
                import import_hub as hub
                importlib.reload(hub)
                from import_hub import *

                if mode == 'DSE':
                    import gpu_DSE.train as gt
                    importlib.reload(gt)
                    from gpu_DSE.train import *
                elif mode == 'DiffAI':
                    import gpu_DiffAI.train as gt
                    importlib.reload(gt)
                    from gpu_DiffAI.train import *
                elif mode == 'only_data':
                    import gpu_only_data.train as gt
                    importlib.reload(gt)
                    from gpu_only_data.train import *
                
                preprocessing_time = time.time()
                if benchmark_name in ["thermostat"]:
                    dataset_path = f"{dataset_path_prefix}_{86.0}.txt"
                else:
                    dataset_path = f"{dataset_path_prefix}_{safe_range_bound}.txt"
                Trajectory_train, Trajectory_test = load_data(train_size=train_size, test_size=test_size, dataset_path=dataset_path)
                components = extract_abstract_representation(Trajectory_train, x_l, x_r, num_components)
                print(f"Prepare data: {time.time() - preprocessing_time} sec.")

                lambda_list = list()
                model_list = list()
                q_list = list()
                c_list = list()
                q = var(0.0)

                for t in range(t_epoch):
                    target_model_name = f"{model_name_prefix}_{safe_range_bound}_{i}_{t}"
                    new_lambda = B.mul(q.exp().div(var(1.0).add(q.exp())))

                    m = Program(l=l, nn_mode=nn_mode)
                    epochs_to_skip, m = load_model(m, MODEL_PATH, name=target_model_name)
                    if constants.profile:
                        epochs_to_skip = -1
                        m = None
                    if test_mode:
                        if m is None:
                            print(f"No Model to test.")
                            exit(0)
                        else:
                            break
                    else:
                        if m is None:
                            torch.manual_seed(i)
                            m = Program(l=l, nn_mode=nn_mode)
                    
                    print(f"parameters: {count_parameters(m)}")

                    # try: 
                    q, c, time_out = learning(
                        m, 
                        components,
                        lambda_=new_lambda,
                        epoch=num_epoch, 
                        target=target,
                        lr=lr,
                        bs=bs,
                        nn_mode=nn_mode,
                        l=l,
                        save=save,
                        epochs_to_skip=epochs_to_skip,
                        model_name=target_model_name,
                        data_bs=data_bs,
                        )
                    
                    if not quick_mode and mode != 'only_data':
                        lambda_list.append(new_lambda)
                        model_list.append(target_model_name)
                        q_list.append(q)
                        c_list.append(c)
                        selected_idx = random.choice([idx for idx in len(model_list)])
                        lambda_hat = torch.stack(lambda_list).sum() / len(lambda_list)
                        L_max = best_lambda(q_hat=q_list[selected_idx], c_hat=c_list[selected_idx])
                        if t == 0:
                            if c.data.item() <= 0.0:
                                L_min = var(0.0)
                            else:
                                L_min = B
                        else:
                            L_min = best_theta(
                                tmp_m_name=f"{model_name_prefix}_{safe_range_bound}_{i}_{t}_tmp",
                                components=extract_abstract_representation(Trajectory_train, x_l, x_r, num_components),
                                lambda_=lambda_hat,
                                epoch=num_epoch,
                                target=target,
                                lr=lr,
                                bs=bs,
                                nn_mode=nn_mode,
                                l=l,
                                save=save,
                                epochs_to_skip=-1,
                                data_bs=data_bs,
                            )
                        if abs((L_max - L_min).data.item()) <= gamma:
                            target_model_name = target_model_name
                            break
                    else:
                        # one-time training with a fixed lambda
                        break

                    if time_out == True:
                        continue

                # AI verification use many initial components, as more as possible
                AI_components = extract_abstract_representation(Trajectory_test, x_l, x_r, AI_verifier_num_components)
                # SE verification use one initial components
                SE_components = extract_abstract_representation(Trajectory_test, x_l, x_r, SE_verifier_num_components)
                
                # AI verification, SE verification, Test data loss
                print(f"------------start verification------------")
                print(f"to verify safe bound: {safe_range_bound}")

                # print(f"sys.modules.keys: {sys.modules.keys()}")
                constants.status = 'verify_AI'
                import verifier_AI as vA
                importlib.reload(vA)
                from verifier_AI import *

                verification_time = time.time()
                
                verifier_AI(
                    model_path=MODEL_PATH, 
                    model_name=target_model_name, 
                    components=AI_components, 
                    target=target,
                    trajectory_path=f"{trajectory_log_prefix}_{safe_range_bound}_{i}"
                )
                print(f"---verification AI time: {time.time() - verification_time} sec---")

                import tester as t
                importlib.reload(t)
                from tester import test_data_loss
                
                test_time = time.time()
                test_data_loss(
                    model_path=MODEL_PATH, 
                    model_name=target_model_name, 
                    trajectory_test=Trajectory_test, 
                    target=target,
                )
                print(f"---test data loss time: {time.time() - test_time} sec---")







