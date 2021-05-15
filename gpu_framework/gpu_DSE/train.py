import torch
import time
import random
from torch.autograd import Variable
import nlopt
import numpy as np
import matplotlib.pyplot as plt
import copy

from constants import benchmark_name

if benchmark_name == "thermostat":
    from gpu_DSE.thermostat_nn import * 
if benchmark_name == "mountain_car":
    from gpu_DSE.mountain_car import *
if benchmark_name == "unsound_1":
    from gpu_DSE.unsound_1 import *
if benchmark_name == "unsound_2_separate":
    from gpu_DSE.unsound_2_separate import *
if benchmark_name == "unsound_2_overall":
    from gpu_DSE.unsound_2_overall import *

from gpu_DSE.data_generator import *

from utils import (
    generate_distribution,
    ini_trajectory,
    batch_pair, 
    show_cuda_memory,
    show_component,
    show_trajectory,
    sample_parameters,
)

random.seed(1)

def distance_f_point(pred_y, y):
    return torch.abs(pred_y.sub(y)) # l1-distance
    # return torch.square(pred_y.sub(y)) # l2-distance


def get_intersection(interval_1, interval_2):
    res_interval = domain.Interval()
    res_interval.left = torch.max(interval_1.left, interval_2.left)
    res_interval.right = torch.min(interval_1.right, interval_2.right)
    return res_interval


def distance_f_interval(symbol_table_list, target):
    if len(symbol_table_list) == 0:
        return var(1.0)
    res = var(0.0)

    for symbol_table in symbol_table_list:
        X = symbol_table['safe_range']
        p = symbol_table['probability']

        # print(f"X: {X.left.data.item(), X.right.data.item()}, p: {p}")
        # print(f"target: {target.left, target.right}")
        intersection_interval = get_intersection(X, target)
        
        #  calculate the reward of each partition
        if intersection_interval.isEmpty():
            # print(f"in empty")
            reward = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
        else:
            # print(f"not empty")
            # print(f"intersection interval get length: {intersection_interval.getLength()}")
            # print(f"X get length: {X.getLength()}")
            reward = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
        # print(f"reward for one partition: {reward}")
    
        tmp_res = reward.mul(p)
        res = res.add(tmp_res)
    res = res.div(var(len(symbol_table_list)).add(EPSILON))

    # print(f"safe loss, res: {res}")
    # res = res_up.div(res_basse)
    return res


def extract_abstract_state_safe_loss(abstract_state, target_component, target_idx):
    # weighted sum of symbol_table loss in one abstract_state
    unsafe_probability_condition = target_component["phi"]
    safe_interval = target_component["condition"]
    method = target_component["method"]
    abstract_loss = var_list([0.0])
    symbol_table_wise_loss_list = list()
    for symbol_table in abstract_state:
        if method == "last":
            trajectory = [symbol_table['trajectory'][-1]]
        elif method == "all":
            trajectory = symbol_table['trajectory'][:]
        else:
            raise NotImplementedError("Error: No trajectory method detected!")
        symbol_table['trajectory_loss'] = list()
        tmp_symbol_table_tra_loss = list()

        trajectory_loss = var_list([0.0])
        # print(f"symbol table p: {float(symbol_table['probability'])}")
        # print(f"start trajectory: ")
        for state in trajectory:
            # print(f"state: {state}")
            X = state[target_idx] # select the variable to measure
            intersection_interval = get_intersection(X, safe_interval)
            if intersection_interval.isEmpty():
                # print(f"point: {X.isPoint()}")
                # print(f"empty")
                if X.isPoint():
                    # min point to interval
                    unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right))
                else:
                    unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength() + eps)
                # unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength().add(EPSILON))
            else:
                # print(f"not empty: {intersection_interval.getLength()}, {X.getLength()}")
                safe_portion = (intersection_interval.getLength() + eps).div(X.getLength() + eps)
                unsafe_value = 1 - safe_portion
            # if float(unsafe_value) > 0:
            #     print(f"X: {float(X.left)}, {float(X.right)}")
            #     print(f"point: {X.isPoint()}")
            #     print(f"unsafe value: {float(unsafe_value)}")
            if outside_trajectory_loss:
                tmp_symbol_table_tra_loss.append(unsafe_value * symbol_table['probability'])
            else:
                trajectory_loss = torch.max(trajectory_loss, unsafe_value)

        if outside_trajectory_loss:
            symbol_table_wise_loss_list.append(tmp_symbol_table_tra_loss)

        # print(f"add part: {trajectory_loss, symbol_table['probability']}")
        if not outside_trajectory_loss:
            abstract_loss += trajectory_loss * symbol_table['probability']
        # print(f"abstract_loss: {abstract_loss}")
    if outside_trajectory_loss:
        abstract_state_wise_trajectory_loss = zip(*symbol_table_wise_loss_list)
        abstract_loss = var_list([0.0])
        for l in abstract_state_wise_trajectory_loss:
            # print(l)
            abstract_loss = torch.max(abstract_loss, torch.sum(torch.stack(l)))
        # if debug:
        #     print(f"abstract_loss: {abstract_loss}")

    return abstract_loss
    

def safe_distance(abstract_state_list, target):
    # measure safe distance in DSE
    # I am using sampling, and many samples the eventual average will be the same as the expectation
    # limited number of abstract states, so count sequentially based on target is ok
    
    loss = var_list([0.0])
    for idx, target_component in enumerate(target):
        target_loss = var_list([0.0])
        for abstract_state in abstract_state_list:
            abstract_state_safe_loss = extract_abstract_state_safe_loss(
                abstract_state, target_component, target_idx=idx, 
            )
            target_loss += abstract_state_safe_loss
        target_loss = target_loss / (len(abstract_state_list) + eps)
        # Weighted loss of different state variables
        # print(f"target_loss: {float(target_loss)}")
        target_loss = target_component["w"] * (target_loss - target_component['phi'])
        # print(f"refined target_loss: {float(target_loss)}")
        # TODO: max(loss - target, 0) change or not
        # target_loss = torch.max(target_loss, var(0.0))
        loss += target_loss
    # print(f"loss: {loss}")
    # exit(0)

    return loss


def cal_data_loss(m, trajectory_list, criterion):
    # for the point in the same batch
    # calculate the data loss of each point
    # add the point data loss together
    if len(trajectory_list) == 0:
        return var_list([0.0])
    X, y = batch_pair(trajectory_list)
    # print(f"after batch pair: {X.shape}, {y.shape}")
    X, y = torch.from_numpy(X).float().cuda(), torch.from_numpy(y).float().cuda()
    # print(X.shape, y.shape)s
    yp = m(X, version="single_nn_learning")
    if debug:
        yp_list = yp.squeeze().detach().cpu().numpy().tolist()
        y_list = y.squeeze().detach().cpu().numpy().tolist()
        print(f"yp: {min(yp_list)}, {max(yp_list)}")
        # print(f"yp: {yp_list[:5]}, {min(yp_list)}, {max(yp_list)}")
        # print(f"y: {y_list[:5]}")
    data_loss = criterion(yp, y)
    # print(f"data_loss: {data_loss}")
    return data_loss


def cal_safe_loss(m, abstract_state, target):
    '''
    DSE: sample paths
    abstract_state = list<{
        'center': vector, 
        'width': vector, 
        'p': var
    }>
    '''
    # show_component(abstract_state)
    ini_abstract_state_list = initialization_abstract_state(abstract_state)
    assert(len(ini_abstract_state_list) == 1)
    res_abstract_state_list = list()

    for i in range(constants.SAMPLE_SIZE):
        # sample one path each time
        # sample_time = time.time()
        # print(f"DSE: in M")
        abstract_list = m(ini_abstract_state_list, 'abstract')
        # show_trajectory(abstract_list)
        # print(f"---------------")
        res_abstract_state_list.append(abstract_list[0]) # only one abstract state returned
    # print(f"length: {len(y_abstract_list)}")
        # print(f"result: {abstract_list[0][0]['x'].c, abstract_list[0][0]['x'].delta}")
        # print(f"run one time")
    
    # TODO: the new safe loss function
    safe_loss = safe_distance(res_abstract_state_list, target)
    return safe_loss


def divide_chunks(component_list, data_safe_consistent, bs=1, data_bs=2):
    '''
    component: {
        'center': 
        'width':
        'p':
        'trajectory_list':
    }
    bs: number of components in a batch
    data_bs: number of trajectories to aggregate the training points

    # components, bs, data_bs
    # whenever a components end, return the components, otherwise 

    return: refineed trajectory_list, return abstract states
    '''
    for i in range(0, len(component_list), bs):
        components = component_list[i:i + bs]
        abstract_states = list()
        trajectory_list, y_list = list(), list()
        for component_idx, component in enumerate(components):
            abstract_state = {
                'center': component['center'],
                'width': component['width'],
                'p': component['p'],
            }
            abstract_states.append(abstract_state)
            for trajectory_idx, trajectory in enumerate(component['trajectory_list']):
                # print(f"before: {len(trajectory)}")
                trajectory_list.append(trajectory)
                # print(f"after: {len(trajectory_list)}")
                if data_safe_consistent:
                    continue
                if (trajectory_idx + data_bs > len(component['trajectory_list']) - 1) and component_idx == len(components) - 1:
                    pass
                elif len(trajectory_list) == data_bs:
                    # print(trajectory_list)
                    yield trajectory_list, [], False
                    trajectory_list = list()
            # print(f"component probability: {component['p']}")

        # print(f"out: {trajectory_list}")
        yield trajectory_list, abstract_states, True # use safe loss


def update_model_parameter(m, theta):
    # for a given parameter module: theta
    # update the parameters in m with theta
    # no grad required
    # TODO: use theta to actually update the element in m.parameters
    with torch.no_grad():
        for idx, p in enumerate(list(m.parameters())):
            p.copy_(theta[idx])
    return m


def extract_parameters(m):
    # extract the parameters in m into the Theta
    # this is for future sampling and derivative extraction
    Theta = list()
    for value in enumerate(m.parameters()):
        Theta.append(value[1].clone())
    return Theta


def learning(
        m, 
        component_list,
        lambda_=lambda_,
        stop_val=0.01, 
        epoch=1000,
        target=None, 
        lr=0.00001,
        bs=10, 
        n=5,
        nn_mode='all',
        l=10,
        module='linearrelu',
        use_smooth_kernel=use_smooth_kernel,
        save=save,
        epochs_to_skip=None,
        model_name=None, 
        only_data_loss=only_data_loss,
        data_bs=data_bs,
        use_data_loss=use_data_loss,
        data_safe_consistent=None,
        sample_std=0.01,
        sample_width=None,
        ):
    print("--------------------------------------------------------------")
    print('====Start Training====')

    TIME_OUT = False

    # m = ThermostatNN(l=l, nn_mode=nn_mode, module=module)
    # print(m)
    m.cuda()

    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(m.parameters(), lr=lr)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-05)
    
    if epochs_to_skip is None:
        epochs_to_skip = -1
    
    start_time = time.time()
    for i in range(epoch):
        if i <= epochs_to_skip:
            continue
        q_loss, c_loss = 0.0, 0.0
        count = 0
        tmp_q_idx = 0
        # if i > 0 and i % 50 == 0 and sample_width is not None:
        #     print(f"before divide: {sample_width}")
        #     sample_width /= 10.0
        #     print(f"after divide: {sample_width}")

        for trajectory_list, abstract_states, use_safe_loss in divide_chunks(component_list, data_safe_consistent=data_safe_consistent, bs=bs, data_bs=data_bs):
            # print(f"x length: {len(x)}")
            # if len(trajectory_list) == 0:
            #     continue
            # show_cuda_memory(f"ini batch free")
            if not use_safe_loss and not use_data_loss:
                continue

            batch_time = time.time()
            grad_data_loss, grad_safe_loss = var_list([0.0]), var_list([0.0])
            real_data_loss, real_safe_loss = 0.0, 0.0
            
            Theta = extract_parameters(m) # extract the parameters now, and then sample around it
            # print(f"Theta before: {Theta}")
            if use_smooth_kernel:
                if use_safe_loss:
                    data_loss_list, safe_loss_list = list(), list()
                    for (sample_theta, sample_theta_p) in sample_parameters(Theta, n=n, sample_std=sample_std, sample_width=sample_width):
                        # show_cuda_memory(f"ini update model(sampled theta) ")
                        
                        m = update_model_parameter(m, sample_theta)

                        # show_cuda_memory(f"end update model(sampled theta) ")
                        
                        sample_time = time.time()

                        if use_data_loss:
                            data_loss = cal_data_loss(m, trajectory_list, criterion)
                            grad_data_loss += float(data_loss) * sample_theta_p #  torch.log(sample_theta_p) # real_q = \expec_{\theta ~ \theta_0}[data_loss]
                            real_data_loss += float(data_loss)
                            data_loss_list.append(float(data_loss))

                        # show_cuda_memory(f"end sampled data loss")
                        
                        # if not only_data_loss:
                        safe_loss = cal_safe_loss(m, abstract_states, target)
                        grad_safe_loss += float(safe_loss) * sample_theta_p # torch.log(sample_theta_p) # real_c = \expec_{\theta ~ \theta_0}[safe_loss]
                        real_safe_loss += float(safe_loss)
                        safe_loss_list.append(float(safe_loss))

                        # print(f"data_loss: {float(data_loss)}, safe_loss: {float(safe_loss)}, Loss TIME: {time.time() - sample_time}, grad data loss: {float(float(data_loss) * sample_theta_p)}, grad safe loss: {float(float(safe_loss) * sample_theta_p)}")
                        # print(f"{'#' * 15}")
                        # print(f"grad_data_loss: {grad_data_loss.data.item()}, grad_safe_loss: {grad_safe_loss.data.item()}")

                    # To maintain the real theta
                    # show_cuda_memory(f"ini update model(Theta)")
                    m = update_model_parameter(m, Theta)
                    # show_cuda_memory(f"end update model(Theta)")

                    grad_data_loss /= n
                    grad_safe_loss /= n
                    real_data_loss /= n
                    real_safe_loss /= n

                    # max_data_loss, min_data_loss = max(data_loss_list), min(data_loss_list)
                    # max_safe_loss, min_safe_loss = max(safe_loss_list), min(safe_loss_list)

                else:
                    if len(trajectory_list) == 0:
                        continue
                    tmp_q_idx += 1
                    data_loss = cal_data_loss(m, trajectory_list, criterion)
                    safe_loss = var_list([0.0])
                    real_data_loss, real_safe_loss = float(data_loss), float(safe_loss)
                    grad_data_loss, grad_safe_loss = data_loss, safe_loss
            else:
                if len(trajectory_list) == 0:
                    continue
                tmp_q_idx += 1
                data_loss = cal_data_loss(m, trajectory_list, criterion)
                safe_loss = var_list([0.0])
                if not only_data_loss:
                    safe_loss = cal_safe_loss(m, abstract_states, target)
                real_data_loss, real_safe_loss = float(data_loss), float(safe_loss)
                grad_data_loss, grad_safe_loss = data_loss, safe_loss

            print(f"use safe loss:{use_safe_loss}, real data_loss: {real_data_loss}, real safe_loss: {real_safe_loss}, TIME: {time.time() - batch_time}")
            print(f"grad data loss: {float(grad_data_loss)}, grad safe loss: {float(grad_safe_loss)}")
            
            q_loss += real_data_loss
            c_loss += real_safe_loss

            # show_cuda_memory(f"end batch")

            # if time.time() - batch_time > 3600/(len(component_list)/bs):
            #     TIME_OUT = True
            #     if i <= 2: # a chance for the first three epoches
            #         pass
            #     else:
            #         break
            
            # print(m.parameters())

            loss = (grad_data_loss + lambda_ * grad_safe_loss) / lambda_
            # loss = lambda_ * grad_safe_loss
            # loss = grad_safe_loss
            loss.backward()
            # print(f"grad of data_loss: {float(grad_data_loss.grad)}, grad of safe_loss: {float(grad_safe_loss.grad)}")
            print(f"value before clip, weight: {m.nn.linear.weight.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear.bias.detach().cpu().numpy().tolist()[0]}")
            # print(m.nn.linear1.weight.grad)
            # print(m.nn.linear2.weight.grad)
            # Theta = extract_parameters(m) 
            # for partial_theta in Theta:
            #     torch.nn.utils.clip_grad_norm_(partial_theta, 1)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
            print(f"grad before step, weight: {m.nn.linear.weight.grad.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear.bias.grad.detach().cpu().numpy().tolist()[0]}")
            optimizer.step()
            print(f"value before step, weight: {m.nn.linear.weight.detach().cpu().numpy().tolist()[0][:3]}, bias: {m.nn.linear.bias.detach().cpu().numpy().tolist()[0]}")
            optimizer.zero_grad()
            # new_theta = extract_parameters(m)
            # print(f"Theta after step: {new_theta}")

            count += 1
            # if count >= 10:
            #     exit(0)
        
        if save:
            if not debug:
                save_model(m, MODEL_PATH, name=model_name, epoch=i)
                print(f"save model")
            
        # if i >= 5 and i%2 == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] *= 0.5
        
        # f_loss = q_loss + lambda_ * c_loss
        print(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}")
        print(f"-----finish {i}-th epoch-----, the batch loss: q: {real_data_loss}, c: {real_safe_loss}")
        if use_smooth_kernel:
            print(f"-----finish {i}-th epoch-----, q: {q_loss}, c: {c_loss}")
        else:
            print(f"-----finish {i}-th epoch-----, q: {q_loss/tmp_q_idx}, c: {c_loss}")
        if not debug:
            log_file = open(file_dir, 'a')
            log_file.write(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}\n")
            log_file.write(f"-----finish {i}-th epoch-----, the batch loss: q: {real_data_loss}, c: {real_safe_loss}\n")
            log_file.write(f"-----finish {i}-th epoch-----, q: {q_loss}, c: {c_loss}\n")
            log_file.flush()
        # print(f"------{i}-th epoch------, avg q: {q_loss_wo_p.div(len(X_train))}, avg c: {c_loss_wo_p.div(len(X_train)/bs)}")
        # if torch.abs(f_loss.data) < var(stop_val):
        #     break
        if float(c_loss) <= 0.0:
            if not debug:
                log_file = open(file_dir, 'a')
                log_file.write('c_loss is small enough. End. \n')
                log_file.close()
            break
        
        if (time.time() - start_time)/(i+1) > 6000 or TIME_OUT:
            if i <= 5: # give a chance for the first few epoch
                pass
            else:
                if not debug:
                    log_file = open(file_dir, 'a')
                    log_file.write('TIMEOUT: avg epoch time > 3600sec \n')
                    log_file.close()
                TIME_OUT = True
                break
    
    res = real_data_loss + float(lambda_) * real_safe_loss# loss # f_loss.div(len(X_train))

    if not debug:
        log_file = open(file_dir, 'a')
        spend_time = time.time() - start_time
        log_file.write('One train: Optimization--' + str(spend_time) + ',' + str(i+1) + ',' + str(spend_time/(i+1)) + '\n')
        log_file.close()
    if debug:
        exit(0)
    
    return [], res, [], 0.0, 0.0, TIME_OUT


def cal_c(X_train, y_train, m, target):
    # TODO: to check the cal_c process
    # only for calculating the value instead of the gradient
    print(f"---in cal_c---")
    # print(f"theta, {theta}")
    c_loss = var(0.0)
    for idx, x in enumerate(X_train):
        x, y = x, y_train[idx]
        loss = var(0.0)
        safe_loss = cal_safe_loss(m, x, width, target)
        c_loss += safe_loss
    c = c_loss.div(len(X_train))
    print(f"---cal_c, {c}")

    return c


def cal_q(X_train, y_train, m):
    root_point = construct_syntax_tree_point(theta)
    # q = var(0.0)

    data_loss = cal_data_loss(m, X_train, y_train)

    # for idx, x in enumerate(X_train):
    #     x, y = x, y_train[idx]
    #     symbol_table_point = initialization_point(x)
    #     symbol_table_point = root_point['entry'].execute(symbol_table_point)

    #     # print('x, pred_y, y', x, symbol_table_point['x'].data.item(), y)
    #     q = q.add(distance_f_point(symbol_table_point['res'], var(y)))

    # q = q.div(var(len(X_train)))
    print(f"cal_q, {data_loss}")
    
    return q
