
from constants import *
import constants
from args import *

if benchmark_name == "thermostat":
    from benchmarks.thermostat import *
elif benchmark_name == "mountain_car":
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

from data_loader import *
import domain

import random
import time

from utils import (
    extract_abstract_representation,
    show_cuda_memory,
    count_parameters,
    import_module,
    append_log,
)


def import_module():
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


#TODO:  change arguments
def best_lambda(X_train, y_train, m, target):
    # TODO: m is only a path
    c = cal_c(X_train, y_train, m, target)
    q = cal_q(X_train, y_train, m)

    if c.data.item() <= 0.0:
        res_lambda = var(0.0)
    else:
        res_lambda = B
    return res_lambda, q.add(res_lambda.mul(c)) # lambda, L_max


# TODO: change arguments
def best_theta(component_list, lambda_, target):
    m, loss, loss_list, q, c, time_out = learning(
        component_list, 
        lambda_=lambda_, 
        stop_val=stop_val, 
        epoch=num_epoch, 
        lr=lr, 
        bs=bs,
        l=l,
        n=n,
        nn_mode=nn_mode,
        module=module,
        target=target,
        use_smooth_kernel=use_smooth_kernel,
        )

    return m, loss, time_out


def outer_loop(lambda_list, model_list, q):
    m_t = random.choice(model_list)
    lambda_t = var(0.0)
    for i in lambda_list:
        lambda_t = lambda_t.add(i)
    lambda_t = lambda_t.div(var(len(lambda_list)))

    _, l_max = best_lambda(X_train, y_train, m_t, target)
    _, l_min, time_out = best_theta(X, Y, abstract_representation, lambda_t, target)

    print('-------------------------------')
    print('l_max, l_min', l_max, l_min)

    if (torch.abs(l_max.sub(l_min))).data.item() < w:
        return None, m_t
    
    q = q.add(var(lr).mul(cal_c(X_train, y_train, m_t, theta)))

    return q, None


if __name__ == "__main__":
    for safe_range_bound in enumerate(safe_range_bound_list):
        if not debug:
            append_log([file_dir, file_dir_evaluation], f"path_sample_size: {path_sample_size}, safa_range_bound: {safe_range_bound}\n")
        print(f"Safe Range Bound: {safe_range_bound}")

        # update target, fix the left endpoint, varify the right endpoint
        target = list()
        for idx, safe_range in enumerate(safe_range_list):
            target.append(
                {   # constraint is in box domain
                    "condition": domain.Interval(var(safe_range[0]), var(safe_range_bound)),
                    "method": method_list[idx],
                }
            )

            # Run 5 times
            for i in range(5):
                import_module()

                preprocessing_time = time.time()
                if benchmark_name in ["thermostat"]:
                    dataset_path = f"{dataset_path_prefix}_{86.0}.txt"
                else:
                    dataset_path = f"{dataset_path_prefix}_{safe_range_bound}.txt"
                Trajectory_train, Trajectory_test = load_data(train_size=train_size, test_size=test_size, dataset_path=)
                # TODO: update component
                component_list = extract_abstract_representation(Trajectory_train, x_l, x_r, num_components)
                print(f"Prepare data: {time.time() - preprocessing_time} sec.")

                lambda_list = list()
                model_list = list()
                q = var(0.0)

                for t in range(t_epoch):
                    target_model_name = f"{model_name_prefix}_{safe_range_bound}_{i}_{t}"
                    new_lambda = B.mul(q.exp().div(var(1.0).add(q.exp())))

                    m = Program(l=l, nn_mode=nn_mode)
                    epochs_to_skip, m = load_model(m, MODEL_PATH, name=target_model_name)
                    if test_mode:
                        if m is None:
                            print(f"No Model to test.")
                            exit(0)
                        else:
                            break
                    else:
                        if m is None:
                            m = Program(l=l, nn_mode=nn_mode)
                    
                    print(f"parameters: {count_parameters(m)}")

                    # try: 
                    _, loss, loss_list, q, c, time_out = learning(
                        m, 
                        component_list,
                        lambda_=new_lambda, 
                        stop_val=stop_val, 
                        epoch=num_epoch, 
                        target=target,
                        lr=lr,
                        bs=bs,
                        nn_mode=nn_mode,
                        l=l,
                        module=module,
                        save=save,
                        epochs_to_skip=epochs_to_skip,
                        model_name=target_model_name,
                        only_data_loss=only_data_loss,
                        data_bs=data_bs,
                        )
                    
                    #TODO: reduce time, because the gap between cal_c and cal_q does not influence a lot on the performance
                    # m_t = m
                    
                    if use_hoang:
                        lambda_list.append(new_lambda)
                        model_list.append(target_model_name)
                        q, target_m = outer_loop(lambda_lost, model_list, q)
                        if q is None:
                            break
                    else:
                        break

                if time_out == True:
                    continue

                AI_component_list = extract_abstract_representation(Trajectory_test, x_l, x_r, AI_verifier_num_components)
                SE_component_list = extract_abstract_representation(Trajectory_test, x_l, x_r, 1)
                # AI verification, SE verification, Test data loss
                verify(
                    model_path=MODEL_PATH,
                    model_name=target_model_name,
                    AI_component_list=AI_component_list,
                    SE_component_list=SE_component_list,
                    trajectory=Trajectory_test,
                    target=target,
                    trajectory_path=f"{trajectory_log_prefix}_{safe_range_bound}_{i}",
                )

                # Verification
                from verifier_AI import verifier_AI
                from modules_AI import *
                # TODO: check replacement?
                print(f"------------start sound verification------------")
                print(f"to verify safe bound: {safe_range_bound}")

                verification_time = time.time()
                # TODO: change extract_abstract_representation
                
                verifier_AI(
                    model_path=MODEL_PATH, 
                    model_name=target_model_name, 
                    component_list=component_list, 
                    target=target,
                    trajectory_path=f"{trajectory_log_prefix}_{safe_range_bound}_{i}")
                print(f"---verification time: {time.time() - verification_time} sec---")
                
                from verifier_SE import verifier_SE
                from modules_SE import *
                print(f"to verify safe bound(test dataset): {safe_range_bound}")
                verification_time = time.time()
                component_list = extract_abstract_representation(Trajectory_test, x_l, x_r, 1)
                verification_SE(
                    model_path=MODEL_PATH, 
                    model_name=target_model_name, 
                    trajectory_test=Trajectory_test, 
                    component_list=component_list
                    target=target
                )
                print(f"---unsound verification(test dataset)time: {time.time() - verification_time} sec---")

                from tester import test_data_loss
                test_data_loss(
                    model_path=MODEL_PATH, 
                    model_name=target_model_name, 
                    trajectory_test=Trajectory_test, 
                    target=target
                )







