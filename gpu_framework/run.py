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
    torch.autograd.set_detect_anomaly(True)

    for safe_range_bound in safe_range_bound_list:
        if not debug:
            append_log([file_dir, file_dir_evaluation], f"path_sample_size: {SAMPLE_SIZE}, safa_range_bound: {safe_range_bound}\n")
        print(f"Safe Range Bound: {safe_range_bound}")

        # update target, fix the left endpoint, varify the right endpoint
        target = list()
        for idx, safe_range in enumerate(safe_range_list):
            target.append(
                {   # constraint is in box domain
                    "condition": domain.Interval(var(safe_range[0]), var(safe_range_bound)) if map_mode is False else None,
                    "method": method_list[idx],
                    "name": name_list[idx],
                    "map_condition": [
                      domain.Interval(var(constraint[0]), var(constraint[1])) for constraint in map_safe_range
                    ] if map_mode is True else None,
                    "map_mode": map_mode,
                }
            )

            # Run 5 times
            for i in range(25):
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
                elif mode == 'symbol_data_loss_DSE':
                    import gpu_symbol_data_loss_DSE.train as gt
                    importlib.reload(gt)
                    from gpu_symbol_data_loss_DSE.train import *
                elif mode == 'DiffAI_sps':
                    import gpu_DiffAI_sps.train as gt
                    importlib.reload(gt)
                    from gpu_DiffAI_sps.train import *
                
                preprocessing_time = time.time()
                if benchmark_name in ["thermostat"]:
                    dataset_path = f"{dataset_path_prefix}_{86.0}.txt"
                else:
                    dataset_path = f"{dataset_path_prefix}_{safe_range_bound}.txt"
                Trajectory_train, Trajectory_test = load_data(train_size=train_size, test_size=test_size, dataset_path=dataset_path)
                # TODO: update component
                components = extract_abstract_representation(Trajectory_train, x_l, x_r, num_components)
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

                # AI verification use many initial components, as more as possible
                AI_components = extract_abstract_representation(Trajectory_test, x_l, x_r, AI_verifier_num_components)
                # SE verification use one initial components
                SE_components = extract_abstract_representation(Trajectory_test, x_l, x_r, SE_verifier_num_components)
                # AI verification, SE verification, Test data loss
                
                # TODO: check replacement?
                print(f"------------start verification------------")
                print(f"to verify safe bound: {safe_range_bound}")

                # print(f"sys.modules.keys: {sys.modules.keys()}")
                constants.status = 'verify_AI'
                import verifier_AI as vA
                importlib.reload(vA)
                from verifier_AI import *

                verification_time = time.time()
                # TODO: change extract_abstract_representation
                
                verifier_AI(
                    model_path=MODEL_PATH, 
                    model_name=target_model_name, 
                    components=AI_components, 
                    target=target,
                    trajectory_path=f"{trajectory_log_prefix}_{safe_range_bound}_{i}"
                )
                print(f"---verification AI time: {time.time() - verification_time} sec---")

                constants.status = 'verify_SE'
                import verifier_SE as vS
                importlib.reload(vS)
                from verifier_SE import *
                
                verification_time = time.time()
                verifier_SE(
                    model_path=MODEL_PATH, 
                    model_name=target_model_name,
                    components=SE_components,
                    target=target,
                    trajectory_path=f"{trajectory_log_prefix}_{safe_range_bound}_{i}"
                )
                print(f"---verification SE time: {time.time() - verification_time} sec---")

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







