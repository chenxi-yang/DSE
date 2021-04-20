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

from gpu_DSE.data_generator import *

from utils import (
    generate_distribution,
    ini_trajectory,
    batch_pair, 
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
        # print(f"start trajectory: ")
        for state in trajectory:
            # print(f"state: {state}")
            X = state[target_idx] # select the variable to measure
            # print(f"real state: [0]: {state[0].left.data.item(), state[0].right.data.item()}; \
            #     [1]: {state[1].left.data.item(), state[1].right.data.item()}")
            # print(f"target_idx: {target_idx}")
            # print(f"X: {X.left.data.item()}, {X.right.data.item()}")
            # print(f"safe condition: {safe_interval.left.data.item()}, {safe_interval.right.data.item()}")
            intersection_interval = get_intersection(X, safe_interval)
            if intersection_interval.isEmpty():
                # print(f"point: {X.isPoint()}")
                # print(f"empty")
                if X.isPoint():
                    # min point to interval
                    unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right))
                else:
                    unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength().add(EPSILON))
                # unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength().add(EPSILON))
            else:
                # print(f"not empty: {intersection_interval.getLength()}, {X.getLength()}")
                safe_portion = intersection_interval.getLength().div(X.getLength().add(EPSILON))
                unsafe_value = 1 - safe_portion
            # print(f"unsafe value: {unsafe_value}")
            if outside_trajectory_loss:
                tmp_symbol_table_tra_loss.append(unsafe_value * symbol_table['probability'])
            else:
                trajectory_loss = torch.max(trajectory_loss, unsafe_value)
            # symbol_table['']
        # exit(0)
        # if trajectory_loss.data.item() > 100.0:
        #     print(f"add part: {trajectory_loss, symbol_table['probability']}")
            # exit(0)
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
        target_loss = target_loss / var(len(abstract_state_list)).add(EPSILON)
        # Weighted loss of different state variables
        target_loss = target_component["w"] * (target_loss - target_component['phi'])
        # TODO: max(loss - target, 0)
        target_loss = torch.max(target_loss, var(0.0))
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
    X, y = batch_pair(trajectory_list, data_bs=None)
    # print(f"after batch pair: {X.shape}, {y.shape}")
    X, y = torch.from_numpy(X).float().cuda(), torch.from_numpy(y).float().cuda()
    print(X.shape, y.shape)
    yp = m(X, version="single_nn_learning")
    # if debug:
    #     print(f"yp: {yp.squeeze()}")
    #     print(f"y: {y.squeeze()}")
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
    ini_abstract_state_list = initialization_abstract_state(abstract_state)
    assert(len(ini_abstract_state_list) == 1)
    res_abstract_state_list = list()

    for i in range(constants.SAMPLE_SIZE):
        # sample one path each time
        # sample_time = time.time()
        # print(f"DSE: in M")
        abstract_list = m(ini_abstract_state_list, 'abstract')
        res_abstract_state_list.append(abstract_list[0]) # only one abstract state returned
    # print(f"length: {len(y_abstract_list)}")
        # print(f"result: {abstract_list[0][0]['x'].c, abstract_list[0][0]['x'].delta}")
        # print(f"run one time")
    
    # TODO: the new safe loss function
    safe_loss = safe_distance(res_abstract_state_list, target)
    return safe_loss


def divide_chunks(component_list, bs=1):
    '''
    component: {
        'center': 
        'width':
        'p':
        'trajectory_list':
    }
    return the component={
        'center':
        'width':
        'p':
    }, trajectory
    '''
    for i in range(0, len(component_list), bs):
        components = component_list[i:i + bs]
        abstract_states = list()
        trajectory_list, y_list = list(), list()
        for component in components:
            abstract_state = {
                'center': component['center'],
                'width': component['width'],
                'p': component['p'],
            }
            trajectory_list.extend(component['trajectory_list'])
            abstract_states.append(abstract_state)
            # print(f"component probability: {component['p']}")
        yield trajectory_list, abstract_states
    # TODO: return refineed trajectory_list, return abstract states
    # 


def update_model_parameter(m, theta):
    # for a given parameter module: theta
    # update the parameters in m with theta
    # no grad required
    # TODO: use theta to actually update the element in m.parameters
    with torch.no_grad():
        for idx, p in enumerate(list(m.parameters())):
            p.copy_(theta[idx])
    return m


def normal_pdf(x, mean, std):
    # print(f"----normal_pdf-----\n x: {x} \n mean: {mean} \n std: {std} \n -------")
    y = torch.exp((-((x-mean)**2)/(2*std*std)))/ (std* torch.sqrt(2*var(math.pi)))
    # res = torch.prod(y)
    res = torch.sum(torch.log(y))
    # res *= var(1e)

    return res


def sampled(x):
    res = torch.normal(mean=x, std=var(1.0))
    log_p = normal_pdf(res, mean=x, std=var(1.0))
    # print(f"res: {res} \n p: {p}")
    # exit(0)
    return res, log_p


def sample_parameters(Theta, n=5):
    # theta_0 is a parameter method
    # sample n theta based on the normal distribution with mean=Theta std=1.0
    # return a list of <theta, theta_p>
    # each theta, Theta is a list of Tensor
    theta_list = list()
    for i in range(n):
        sampled_theta = list()
        theta_p = var(1.0)
        for array in Theta:
            sampled_array, sampled_p = sampled(array)
            sampled_theta.append(sampled_array)
            # sum the log(p)
            theta_p += sampled_p
            # theta_p *= sampled_p # !incorrect
        # print(f"each sampled theta: {sampled_theta}")
        # print(f"each probability: {theta_p}")
        theta_list.append((sampled_theta, theta_p))

    return theta_list


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
        ):
    print("--------------------------------------------------------------")
    print('====Start Training====')

    TIME_OUT = False

    # m = ThermostatNN(l=l, nn_mode=nn_mode, module=module)
    # print(m)
    m.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(m.parameters(), lr=lr)
    
    if epochs_to_skip is None:
        epochs_to_skip = -1
    
    start_time = time.time()
    for i in range(epoch):
        if i <= epochs_to_skip:
            continue
        q_loss, c_loss = var_list([0.0]), var_list([0.0])
        count = 0
        if not use_smooth_kernel:
            tmp_q_idx = 0
        for trajectory_list, abstract_states in divide_chunks(component_list, bs=bs):
            # print(f"x length: {len(x)}")
            # if len(trajectory_list) == 0:
            #     continue
            batch_time = time.time()
            grad_data_loss, grad_safe_loss = var_list([0.0]), var_list([0.0])
            real_data_loss, real_safe_loss = var_list([0.0]), var_list([0.0])
            
            Theta = extract_parameters(m) # extract the parameters now, and then sample around it
            # print(f"Theta before: {Theta}")
            if use_smooth_kernel:
                for (sample_theta, sample_theta_p) in sample_parameters(Theta, n=n):
                    m = update_model_parameter(m, sample_theta)
                    sample_time = time.time()

                    data_loss = cal_data_loss(m, trajectory_list, criterion)
                    grad_data_loss += var(data_loss.data.item()) * sample_theta_p #  torch.log(sample_theta_p) # real_q = \expec_{\theta ~ \theta_0}[data_loss]
                    real_data_loss += var(data_loss.data.item())
                    safe_loss = var(0.0)
                    
                    if not only_data_loss:
                        safe_loss = cal_safe_loss(m, abstract_states, target)
                        grad_safe_loss += var(safe_loss.data.item()) * sample_theta_p # torch.log(sample_theta_p) # real_c = \expec_{\theta ~ \theta_0}[safe_loss]
                        real_safe_loss += var(safe_loss.data.item())

                        print(f"data_loss: {data_loss.data.item()}, safe_loss: {safe_loss.data.item()}, Loss TIME: {time.time() - sample_time}")
                    # print(f"{'#' * 15}")
                    # print(f"grad_data_loss: {grad_data_loss.data.item()}, grad_safe_loss: {grad_safe_loss.data.item()}")

                # To maintain the real theta
                m = update_model_parameter(m, Theta)

                real_data_loss /= n
                real_safe_loss /= n
            else:
                if len(trajectory_list) == 0:
                    continue
                tmp_q_idx += 1
                data_loss = cal_data_loss(m, trajectory_list, criterion)
                safe_loss = var_list([0.0])
                if not only_data_loss:
                    safe_loss = cal_safe_loss(m, abstract_states, target)
                real_data_loss, real_safe_loss = data_loss, safe_loss

            print(f"real data_loss: {real_data_loss.data.item()}, real safe_loss: {real_safe_loss.data.item()}, data and safe TIME: {time.time() - batch_time}")
            q_loss += real_data_loss
            c_loss += real_safe_loss

            if time.time() - batch_time > 3600/(len(component_list)/bs):
                TIME_OUT = True
                if i == 0: # a chance for the first epoch
                    pass
                else:
                    break

            loss = real_data_loss + lambda_.mul(real_safe_loss)
            loss.backward()
            for partial_theta in Theta:
                torch.nn.utils.clip_grad_norm_(partial_theta, 1)
            # print(m.nn.linear1.weight.grad)
            # print(m.nn.linear2.weight.grad)
            optimizer.step()
            optimizer.zero_grad()
            # new_theta = extract_parameters(m)
            # print(f"Theta after step: {new_theta}")

            count += 1
            # if count >= 10:
            #     exit(0)
        
        if save:
            if not debug:
                save_model(m, MODEL_PATH, name=model_name, epoch=i)
            
        if i >= 5 and i%2 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5
        
        # f_loss = q_loss + lambda_ * c_loss
        print(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}")
        print(f"-----finish {i}-th epoch-----, the batch loss: q: {real_data_loss.data.item()}, c: {real_safe_loss.data.item()}")
        if use_smooth_kernel:
            print(f"-----finish {i}-th epoch-----, q: {q_loss.data.item()}, c: {c_loss.data.item()}")
        else:
            print(f"-----finish {i}-th epoch-----, q: {q_loss.data.item()/tmp_q_idx}, c: {c_loss.data.item()}")
        if not debug:
            log_file = open(file_dir, 'a')
            log_file.write(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}\n")
            log_file.write(f"-----finish {i}-th epoch-----, the batch loss: q: {real_data_loss.data.item()}, c: {real_safe_loss.data.item()}\n")
            log_file.write(f"-----finish {i}-th epoch-----, q: {q_loss.data.item()}, c: {c_loss.data.item()}\n")

        # print(f"------{i}-th epoch------, avg q: {q_loss_wo_p.div(len(X_train))}, avg c: {c_loss_wo_p.div(len(X_train)/bs)}")
        # if torch.abs(f_loss.data) < var(stop_val):
        #     break
        # if c_loss.data.item() < EPSILON.data.item():
        #     break
        
        if (time.time() - start_time)/(i+1) > 3500 or TIME_OUT:
            if i == 0: # give a chance for the first epoch
                pass
            else:
                log_file = open(file_dir, 'a')
                log_file.write('TIMEOUT: avg epoch time > 2000s \n')
                log_file.close()
                TIME_OUT = True
                break
    
    res = real_data_loss + lambda_ * real_safe_loss# loss # f_loss.div(len(X_train))

    if not debug:
        log_file = open(file_dir, 'a')
        spend_time = time.time() - start_time
        log_file.write('One train: Optimization--' + str(spend_time) + ',' + str(i+1) + ',' + str(spend_time/(i+1)) + '\n')
        log_file.close()
    if debug:
        exit(0)
    
    return m, res, [], data_loss, safe_loss, TIME_OUT


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
