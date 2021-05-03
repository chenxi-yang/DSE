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
    from gpu_DiffAI.thermostat_nn import * 
if benchmark_name == "mountain_car":
    from gpu_DiffAI.mountain_car import *

from utils import (
    generate_distribution,
    ini_trajectory,
    batch_pair,
    batch_points,
    show_cuda_memory,
    sample_parameters,
)

random.seed(1)

def distance_f_point(pred_y, y):
    return torch.abs(pred_y.sub(y)) # l1-distance
    # return torch.square(pred_y.sub(y)) # l2-distance


def get_intersection(interval_1, interval_2):
    res_interval = domain.Interval(interval_1.left, interval_2.right)
    return res_interval


def generate_theta_sample_set(Theta):
    # sample_theta_list = list()
    # sample_theta_probability_list = list()
    # print(f"----------generating Theta --------------")
    # print(f"Original theta: {Theta}")
    sample_theta_list = list()
    for i in range(THETA_SAMPLE_SIZE):
        sample_theta = torch.normal(mean=Theta, std=var(1.0))
        # print(f"Sampled theta: {Theta}")
        # sample_theta = torch.distributions.multivariate_normal.MultivariateNormal(loc=Theta, torch.eye(1.0))
        sample_theta_probability = normal_pdf(sample_theta, Theta, var(1.0))
        sample_theta_list.append((sample_theta, sample_theta_probability))
    # print(f"---------finish generating Theta----------")
    return sample_theta_list


def update_symbol_table_with_sample_theta(sample_theta_list, sample_theta_probability_list, symbol_table_list):
    for symbol_table in symbol_table_list:
        symbol_table['sample_theta'] = copy.deepcopy(sample_theta_list)
        symbol_table['sample_theta_probability'] = copy.deepcopy(sample_theta_probability_list)
    return symbol_table_list


def create_ball(x, width):
    res_l = list()
    res_r = list()
    for value in x:
        res_l.append(value-width)
        res_r.append(value+width)
    return res_l, res_r


def create_point_cloud(res_l, res_r, n=50):
    assert(len(res_l) == len(res_r))
    point_cloud = list()
    for i in range(n):
        point = list()
        for idx, v in enumerate(res_l):
            l = v
            r = res_r[idx]
            x = random.uniform(l, r)
            point.append(x)
        point_cloud.append(point)
    return point_cloud


def cal_data_loss(m, trajectory_list, criterion):
    # for the point in the same batch
    # calculate the data loss of each point
    # add the point data loss together
    X, y = batch_pair(trajectory_list, data_bs=512)
    # print(f"after batch pair: {X.shape}, {y.shape}")
    X, y = torch.from_numpy(X).float().cuda(), torch.from_numpy(y).float().cuda()
    # print(X.shape, y.shape)
    yp = m(X, version="single_nn_learning")
    data_loss = criterion(yp, y)
    # print(f"data_loss: {data_loss}")s
    return data_loss


def generate_small_ball_point(center, width, distribution="Gaussian", unit=10):
    point = list()
    tp_list = list()
    bw_list = list()
    for c in center:
        tp = c + width
        bw = c - width
        x = random.uniform(bw, tp)
        point.append(x)
        tp_list.append(tp)
        bw_list.append(bw)
    ball = (tp_list, bw_list)
    return ball, point


# x: list of points
# width: a number 
# distribution: over small balls
# n: number allowed in one point cloud
def create_ball_cloud(x, width, distribution="Gaussian", n=50):
    point_cloud = list()
    ball_list = list() # list of pair <tp, bw>, tp is a list, bw is a list
    unit = int(n/len(x))
    for idx in range(len(x)):
        center = x[idx]
        ball, points = generate_small_ball_point(center, width, distribution, unit)
        point_cloud.append(points)
        ball_list.append(ball)

    return point_cloud, ball_list


def extract_large_ball(ball_list):
    max_tp, min_bw = ball_list[0][0], ball_list[0][1]
    for ball in ball_list:
        tp, bw = ball[0], ball[1]
        for i in range(len(tp)):
            max_tp[i] = max(tp[i], max_tp[i])
            min_bw[i] = min(bw[i], min_bw[i])
    
    center, new_width = list(), list()
    for i in range(len(max_tp)):
        center.append((max_tp[i] + min_bw[i])/2.0)
        new_width.append((max_tp[i] - min_bw[i])/2.0)
    return center, new_width


def create_small_ball(x_list, width):
    center_list, width_list = list(), list()
    for X in x_list:
        center, w = list(), list()
        for x in X:
            center.append(x)
            w.append(width)
        center_list.append(center)
        width_list.append(w)
    return center_list, width_list


def safe_distance(symbol_tables, target):
    # measure DiffAI safe distance
    # for one trajectory <s1, s2, ..., sn>
    # all_unsafe_value = max_i(unsafe_value(s_i, target))
    # TODO: or avg?
    # loss violate the safe constraint: 
    print(f"[safe_distance] {symbol_tables['x'].c.shape}")
    loss = var_list([0.0])
    for idx, target_component in enumerate(target):
        target_loss = var_list([0.0])
        unsafe_probability_condition = target_component["phi"]
        safe_interval = target_component["condition"]
        for trajectory in symbol_tables['trajectory_list']:
            # show_cuda_memory(f"[update trajectory, {len(trajectory)}]")
            trajectory_loss = var_list([0.0])
            i = 0
            # print(len(trajectory))
            for state in trajectory:
                # show_cuda_memory(f"[update state, state {i}]")
                X = state[idx]
                # print(f"X: {X.left}, {X.right}")
                intersection_interval = get_intersection(X, safe_interval)
                if intersection_interval.isEmpty():
                    if X.isPoint():
                        # min point to interval
                        unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right))
                    else:
                        unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength().add(float(EPSILON)))
                        # unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength())
                else:
                    safe_portion = (intersection_interval.getLength() + eps) / (X.getLength() + eps)
                    # safe_probability = torch.index_select(safe_portion, 0, index0)
                    unsafe_value = 1 - safe_portion
                    # if safe_probability.data.item() > 1 - unsafe_probability_condition.data.item():
                    #     unsafe_value = var_list([0.0])
                    # else:
                    #     # unsafe_value = ((1 - unsafe_probability_condition) - safe_probability) / safe_probability
                    #     unsafe_value = 1 - safe_probability
                trajectory_loss = torch.max(trajectory_loss, unsafe_value)
                i += 1
            target_loss += trajectory_loss
        target_loss = target_loss / (var(len(symbol_tables)).add(float(EPSILON)))
        target_loss = target_component["w"] * (target_loss - unsafe_probability_condition)
        target_loss = torch.max(target_loss, var(0.0))
        loss +=  target_loss

    return loss


def cal_safe_loss(m, trajectory_list, width, target):
    '''
    DiffAI 
    each x is surrounded by a small ball: (x-width, x+width)
    calculate the loss in a batch-wise view
    '''
    # TODO: for now, we only keep 
    # TODO: use batch version, batch the initial states together
    # to reduce cost
    shuffle(trajectory_list)
    trajectory_list = trajectory_list[:int(len(trajectory_list)/2)]
    x = [[ini_trajectory(trajectory)[0][0]] for trajectory in trajectory_list]
    center_list, width_list = create_small_ball(x, width)
    batched_center, batched_width = batch_points(center_list), batch_points(width_list)
    # print(f"[safe loss] center, width: {batched_center.shape}, {batched_width.shape}")
    abstract_data = initialization_nn(batched_center, batched_width)
    # if debug:
    #     exit(0)
    output = m(abstract_data)
    # show_cuda_memory(f"[cal_safe_loss] before safe distance")
    safe_loss = safe_distance(output, target)
    # show_cuda_memory(f"[cal_safe_loss] after safe distance")
    
    return safe_loss

    # safe_loss = var_list([0.0])
    # # batch center_list together
    # for idx, center in enumerate(center_list):
    #     width = width_list[idx]
    #     abstract_data = initialization_nn([center], [width])
    #     # TODO: change the split way
    #     abstract_list = m(abstract_data, 'abstract')
    #     safe_loss += safe_distance(abstract_list, target)
    # safe_loss /= var(len(x)).add(EPSILON)
    # return safe_loss


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
        real_trajectory_list = list()
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
                real_trajectory_list.append(trajectory)
                # print(f"after: {len(trajectory_list)}")
                if data_safe_consistent:
                    continue
                if (trajectory_idx + data_bs > len(component['trajectory_list']) - 1) and component_idx == len(components) - 1:
                    pass
                elif len(trajectory_list) == data_bs:
                    # print(trajectory_list)
                    yield trajectory_list, real_trajectory_list, False
                    trajectory_list = list()
            # print(f"component probability: {component['p']}")

        # print(f"out: {trajectory_list}")
        yield trajectory_list, real_trajectory_list, True # use safe loss


def update_model_parameter(m, theta):
    # for a given parameter module: theta
    # update the parameters in m with theta
    # no grad required
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

    # TODO change all.....
    TIME_OUT = False

    # print(m)
    m.cuda()

    if epochs_to_skip is None:
        epochs_to_skip = -1
    
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(m.parameters(), lr=lr)
    optimizer = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-05)
    
    start_time = time.time()
    for i in range(epoch):
        if i <= epochs_to_skip:
            continue
        q_loss, c_loss = var_list([0.0]), var_list([0.0])
        count = 0
        tmp_q_idx = 0
        for trajectory_list, real_trajectory_list, use_safe_loss in divide_chunks(component_list, data_safe_consistent, bs=bs, data_bs=data_bs):
            # print(f"x length: {len(x)}")
            # print(f"batch size, x: {len(x)}, y: {len(y)}, abstract_states: {len(abstract_states)}")
            # if len(x) == 0: continue  # because DiffAI only makes use of x, y
            batch_time = time.time()
            # if not use_safe_loss and not use_data_loss:
            #     continue

            if use_smooth_kernel:
                if use_safe_loss:
                    tmp_q_idx += 1
                    Theta = extract_parameters(m)
                    grad_data_loss, grad_safe_loss = var_list([0.0]), var_list([0.0])
                    real_data_loss, real_safe_loss = 0.0, 0.0
                    for (sample_theta, sample_theta_p) in sample_parameters(Theta, n=n, sample_std=sample_std, sample_width=sample_width):
                        # sample_theta_p is actually log(theta_p)

                        sample_time = time.time()
                        m = update_model_parameter(m, sample_theta)

                        # show_cuda_memory(f"ini sample")
                        if use_data_loss:
                            data_loss = cal_data_loss(m, trajectory_list, criterion)
                            grad_data_loss += float(data_loss) * sample_theta_p
                            real_data_loss += float(data_loss)

                        # show_cuda_memory(f"after data loss")
                        safe_loss = cal_safe_loss(m, real_trajectory_list, width, target)

                        # show_cuda_memory(f"after safe loss")
                        print(f"data loss: {float(data_loss)}, safe_loss: {float(safe_loss)}, time: {time.time() - sample_time}")

                        # gradient = \exp_{\theta' \sim N(\theta)}[loss * \grad_{\theta}(log(p(\theta', \theta)))]
                        grad_safe_loss += float(safe_loss) * sample_theta_p
                        real_safe_loss += float(safe_loss)

                        # print(f"sample time: {time.time() - sample_time}")
                        # if time.time() - sample_time > 2000/(n*(len(component_list)/bs)):
                        #     TIME_OUT = True
                        #     break
                    # if TIME_OUT:
                    #     break

                    m = update_model_parameter(m, Theta)

                    real_data_loss /= n
                    real_safe_loss /= n
                elif use_data_loss:
                    if len(trajectory_list) == 0:
                        continue
                    tmp_q_idx += 1
                    data_loss = cal_data_loss(m, trajectory_list, criterion)
                    safe_loss = var_list([0.0])
                    real_data_loss, real_safe_loss = float(data_loss), float(safe_loss)
                    grad_data_loss, grad_safe_loss = data_loss, safe_loss

                # print(f"grad data_loss: {grad_data_loss.data.item()}, grad safe_loss: {grad_safe_loss.data.item()}, loss TIME: {time.time() - batch_time}")

            else:
                if use_data_loss:
                    data_loss = cal_data_loss(m, trajectory_list, criterion)
                else:
                    data_loss = var(0.0)
                # print(f"in safe loss: {len(trajectory_list)}")
                safe_loss = cal_safe_loss(m, real_trajectory_list, width, target)

                real_data_loss, real_safe_loss = float(data_loss), float(safe_loss)
                grad_data_loss, grad_safe_loss = data_loss, safe_loss
                tmp_q_idx += 1
                
            print(f"use safe loss:{use_safe_loss}, real data_loss: {real_data_loss}, real safe_loss: {real_safe_loss}, data and safe TIME: {time.time() - batch_time}")
            q_loss += real_data_loss
            c_loss += real_safe_loss

            loss = grad_data_loss + lambda_ * grad_safe_loss
            # loss.backward(retain_graph=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1)
            # print(f"Linear1 grad: [{torch.min(m.nn.linear1.weight.grad)}, {torch.max(m.nn.linear1.weight.grad)}]")
            # print(f"Linear2 grad: [{torch.min(m.nn.linear2.weight.grad)}, {torch.max(m.nn.linear2.weight.grad)}]")

            optimizer.step()
            optimizer.zero_grad()

        if save:
            save_model(m, MODEL_PATH, name=model_name, epoch=i)
        
        # if i >= 5 and i%2 == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] *= 0.5
        
        # f_loss = q_loss + lambda_ * c_loss
        print(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}")
        print(f"-----finish {i}-th epoch-----, the batch loss: q: {real_data_loss}, c: {real_safe_loss}")
        print(f"-----finish {i}-th epoch-----, the epoch loss: q: {q_loss/tmp_q_idx}, c: {c_loss}")
        log_file = open(file_dir, 'a')
        log_file.write(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}\n")
        log_file.write(f"-----finish {i}-th epoch-----, the batch loss: q: {real_data_loss}, c: {real_safe_loss}\n")
        log_file.write(f"-----finish {i}-th epoch-----, the epoch loss: q: {q_loss}, c: {c_loss}\n")
        log_file.flush()

        # print(f"------{i}-th epoch------, avg q: {q_loss_wo_p.div(len(X_train))}, avg c: {c_loss_wo_p.div(len(X_train)/bs)}")
        # if torch.abs(f_loss.data) < var(stop_val):
        #     break

        if float(c_loss) < float(EPSILON):
            if not debug:
                log_file = open(file_dir, 'a')
                log_file.write('c_loss is small enough. End. \n')
                log_file.close()
            break
        
        if (time.time() - start_time)/(i+1) > 3600 or TIME_OUT:
            if not debug:
                log_file = open(file_dir, 'a')
                log_file.write('TIMEOUT: avg epoch time > 3000s \n')
                log_file.close()
            TIME_OUT = True
            if i <= 2:
                pass
            else:
                break
    
    res = loss # loss # f_loss.div(len(X_train))

    log_file = open(file_dir, 'a')
    spend_time = time.time() - start_time
    log_file.write('One train: Optimization--' + str(spend_time) + ',' + str(i+1) + ',' + str(spend_time/(i+1)) + '\n')
    log_file.close()
    
    return [], float(res), [], float(data_loss), float(safe_loss), TIME_OUT


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
    
    return qs


##### create symbolic approximation of perturbation set of input distribution

# def create_ball_perturbation(Trajectory_train, distribution_list, w):
#     perturbation_x_dict = {
#         distribution: list() for distribution in distribution_list
#     }
#     for trajectory in Trajectory_train:
#         state, _ = ini_trajectory(trajectory)
#         # TODO: for now, only use the first input variable
#         x = state[0]
#         l, r = x - w, x + w
#         # print(f"l, r: {l, r}")
#         for distribution in distribution_list:
#             x_list = generate_distribution(x, l, r, distribution, unit=6)
#             # print(f"x_list of {distribution}: {x_list}")
#             perturbation_x_dict[distribution].extend(x_list)
#         # exit(0)
#     return perturbation_x_dict


# def split_component(perturbation_x_dict, x_l, x_r, num_components):
#     # TODO: add vector-wise component split
#     x_min, x_max = x_l[0], x_r[0]
#     for distribution in perturbation_x_dict:
#         x_min, x_max = min(min(perturbation_x_dict[distribution]), x_min), max(max(perturbation_x_dict[distribution]), x_max)
#     component_length = (x_max - x_min) / num_components
#     component_list = list()
#     for i in range(num_components):
#         l = x_min + i * component_length
#         r = x_min + (i + 1) * component_length
#         center = [(l + r) /  2.0]
#         width = [(r - l) / 2.0]
#         component_group = {
#             'center': center,
#             'width': width,
#         }
#         component_list.append(component_group)
#     return component_list


# def extract_upper_probability_per_component(component, perturbation_x_dict):

#     p_list = list()
#     for distribution in perturbation_x_dict:
#         x_list = perturbation_x_dict[distribution]
#         cnt = 0
#         for X in x_list:
#             x = [X] #TODO: X is a value
#             if in_component(x, component):
#                 cnt += 1
#         p = cnt * 1.0 / len(x_list) + eps + random.uniform(0, 0.1)
#         p_list.append(p)
#     return max(p_list)


# def assign_probability(perturbation_x_dict, component_list):
#     '''
#     perturbation_x_dict = {
#         distribution: x_list, # x in x_list are in the form of value
#     }
#     component_list:
#     component: {
#         'center': center # center is vector
#         'width': width # width is vector
#     }
#     keep track of under each distribution, what portiton of x_list fall 
#     in to this component
#     '''
#     for idx, component in enumerate(component_list):
#         p = extract_upper_probability_per_component(component, perturbation_x_dict)
#         component['p'] = p
#         component_list[idx] = component
#     # print(f"sum of upper bound: {sum([component['p'] for component in component_list])}")
#     return component_list


# def in_component(X, component):
#     # TODO: 
#     center = component['center']
#     width = component['width']
#     for i, x in enumerate(X):
#         if x >= center[i] - width[i] and x < center[i] + width[i]:
#             pass
#         else:
#             return False
#     return True


# def assign_data_point(Trajectory_train, component_list):
#     for idx, component in enumerate(component_list):
#         component.update(
#             {
#             'trajectory_list': list(),
#             }
#         )
#         for i, trajectory in enumerate(Trajectory_train):
#             state, action = ini_trajectory(trajectory) # get the initial <state, action> pair in trajectory
#             # when test, only test the first value in state
#             if in_component([state[0]], component): # if the initial state in component
#                 component['trajectory_list'].append(trajectory)
#         component_list[idx] = component
#     return component_list
        

# def extract_abstract_representation(
#     Trajectory_train, 
#     x_l, 
#     x_r, 
#     num_components, 
#     w=0.3):
#     # bs < num_components, w is half of the ball width
#     '''
#     Steps:
#     # 1. generate perturbation, small ball covering following normal, uniform, poission
#     # 2. measure probability 
#     # 3. slice X_train, y_train into component-wise
#     '''
#     # TODO: generate small ball based on init(trajectory), others remain
#     start_t = time.time()
#     # print(f"w: {w}")

#     perturbation_x_dict = create_ball_perturbation(Trajectory_train, 
#         # distribution_list=["normal", "uniform", "beta", "original"], 
#         distribution_list=["normal", "uniform", "original"],  
#         #TODO:  beta distribution does not account for range
#         w=w)
#     component_list = split_component(perturbation_x_dict, x_l, x_r, num_components)
#     # print(f"after split components: {component_list}")

#     # create data for batching, each containing component and cooresponding x, y
#     component_list = assign_probability(perturbation_x_dict, component_list)
#     component_list = assign_data_point(Trajectory_train, component_list)
#     random.shuffle(component_list)

#     print(f"component-wise x length: {[len(component['trajectory_list']) for component in component_list]}")

#     # print(component_list)
#     print(f"-- Generate Perturbation Set --")
#     print(f"--- {time.time() - start_t} sec ---")

#     return component_list