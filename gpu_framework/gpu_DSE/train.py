import torch
import time
import random
from torch.autograd import Variable
import nlopt
import numpy as np
import matplotlib.pyplot as plt
import copy

from helper import *
from data_generator import *
from constants import *
import constants

from thermostat_nn import * 

from utils import generate_distribution

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


def normal_pdf(x, mean, std):
    # print(f"----normal_pdf-----\n x: {x} \n mean: {mean} \n std: {std} \n -------")
    y = torch.exp((-((x-mean)**2)/(2*std*std)))/ (std* torch.sqrt(2*var(math.pi)))
    # res = torch.prod(y)
    res = torch.sum(torch.log(y))
    # res *= var(1e)

    return res


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


def cal_data_loss(m, x, y):
    # for the point in the same batch
    # calculate the data loss of each point
    # add the point data loss together
    data_loss = var(0.0)
    for idx in range(len(x)):
        point, label = x[idx], y[idx]
        point_data = initialization_point_nn(point)
        y_point_list = m(point_data, 'concrete')
        # should be only one partition in y['x']
        # the return value in thermostat is x, index: 2
        data_loss += distance_f_point(y_point_list[0]['x'].c[2], var(label))
    data_loss /= len(x)
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


def cal_safe_loss(m, x, width, target):
    # for each x, generate a small ball:(x-e, x+e)
    # in each ball, follow the distribution predefined to generate point cloud
    # 1). take the union of all point cloud and the larget point cloud
    # 2). take the boundry of all the small ball as the large ball: abstract data
    point_cloud, ball_list = create_ball_cloud(x, width, distribution="Gaussian", n=50)
    center, new_width = extract_large_ball(ball_list) # new width is a list, after generating the new distribution around the point, the width of each element might be different
    # print(f"center: {center}, new_width: {new_width}")
    abstract_data = initialization_nn(center, new_width, point_cloud)
    y_abstract_list = list()
    # TODO: partition in one batch, split strategy
    for i in range(constants.SAMPLE_SIZE):
        # sample one path each time
        # sample_time = time.time()
        abstract_list = m(abstract_data, 'abstract')
        # print(f"sample {i+1}-th path: {time.time() - sample_time}")
        y_abstract_list.append(abstract_list[0])
    # print(f"length: {len(y_abstract_list)}")
    safe_loss = distance_f_interval(y_abstract_list, target)
    return safe_loss


def divide_chunks(X, y, bs=10):
    # return the chunk of size bs from X and 
    # print(f"bs: {bs}")
    for i in range(0, len(X), bs):
        yield X[i:i + bs], y[i:i + bs]


def update_model_parameter(m, theta):
    # for a given parameter module: theta
    # update the parameters in m with theta
    # no grad required
    # TODO: use theta to actually update the element in m.parameters
    with torch.no_grad():
        for idx, p in enumerate(list(m.parameters())):
            p.copy_(theta[idx])
    return m


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
        X_train, 
        y_train,
        lambda_=lambda_,
        stop_val=0.01, 
        epoch=1000, 
        lr=0.00001, 
        theta=None, 
        bs=10, 
        n=5,
        nn_mode='all',
        l=10,
        module='linearrelu'
        ):
    print("--------------------------------------------------------------")
    print('====Start Training====')

    # TODO change all.....
    TIME_OUT = False

    x_min = var(10000.0)
    x_max = var(0.0)

    loop_list = list()
    loss_list = list()

    m = ThermostatNN(l=l, nn_mode=nn_mode, module=module)
    print(m)
    m.cuda()

    optimizer = torch.optim.SGD(m.parameters(), lr=lr)
    
    start_time = time.time()
    for i in range(epoch):
        q_loss, c_loss = var(0.0), var(0.0)
        count = 0
        for x, y in divide_chunks(X_train, y_train, bs=bs):
            # print(f"x length: {len(x)}")
            batch_time = time.time()
            grad_data_loss, grad_safe_loss = var(0.0), var(0.0)
            real_data_loss, real_safe_loss = var(0.0), var(0.0)
            
            Theta = extract_parameters(m) # extract the parameters now, and then sample around it
            # print(f"Theta before: {Theta}")
            for (sample_theta, sample_theta_p) in sample_parameters(Theta, n=n):
                m = update_model_parameter(m, sample_theta)
                data_time = time.time()
                data_loss = cal_data_loss(m, x, y)
                # print(f"{'#' * 15}")
                # print(f"data_loss: {data_loss}, TIME: {time.time() - data_time}")
                # print(f"p: {sample_theta_p}, log_p: {torch.log(sample_theta_p)}")
                grad_data_loss += var(data_loss.data.item()) * sample_theta_p #  torch.log(sample_theta_p) # real_q = \expec_{\theta ~ \theta_0}[data_loss]
                real_data_loss += var(data_loss.data.item())

                safe_time = time.time()
                safe_loss = cal_safe_loss(m, x, width, target)
                # print(f"safe_loss: {safe_loss.data.item()}, Loss TIME: {time.time() - safe_time}")
                # print(f"{'#' * 15}")
                grad_safe_loss += var(safe_loss.data.item()) * sample_theta_p # torch.log(sample_theta_p) # real_c = \expec_{\theta ~ \theta_0}[safe_loss]
                real_safe_loss += var(safe_loss.data.item())

                # exit(0)

            # To maintain the real theta
            m = update_model_parameter(m, Theta)

            real_data_loss /= n
            real_safe_loss /= n

            print(f"real data_loss: {real_data_loss}")
            print(f"real safe_loss: {real_safe_loss}, data and safe TIME: {time.time() - batch_time}")
            q_loss += real_data_loss
            c_loss += real_safe_loss
            loss = grad_data_loss + lambda_.mul(grad_safe_loss)
            loss.backward()
            for partial_theta in Theta:
                torch.nn.utils.clip_grad_norm_(partial_theta, 1)
            # print(m.nn.linear1.weight.grad)
            # print(m.nn.linear2.weight.grad)
            # check: remove all the theta_p, only leave loss.data.item(), check the grad check guola
            optimizer.step()
            optimizer.zero_grad()
            # new_theta = extract_parameters(m)
            # print(f"Theta after step: {new_theta}")

            count += 1
            # if count >= 10:
            #     exit(0)
            
        if i >= 7 and i%2 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5
        
        # f_loss = q_loss + lambda_ * c_loss
        print(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}")
        print(f"-----finish {i}-th epoch-----, the batch loss: q: {real_data_loss.data.item()}, c: {real_safe_loss.data.item()}")
        print(f"-----finish {i}-th epoch-----, q: {q_loss.data.item()}, c: {c_loss.data.item()}")
        log_file = open(file_dir, 'a')
        log_file.write(f"{i}-th Epochs Time: {(time.time() - start_time)/(i+1)}\n")
        log_file.write(f"-----finish {i}-th epoch-----, the batch loss: q: {real_data_loss.data.item()}, c: {real_safe_loss.data.item()}\n")
        log_file.write(f"-----finish {i}-th epoch-----, q: {q_loss.data.item()}, c: {c_loss.data.item()}\n")

        # print(f"------{i}-th epoch------, avg q: {q_loss_wo_p.div(len(X_train))}, avg c: {c_loss_wo_p.div(len(X_train)/bs)}")
        # if torch.abs(f_loss.data) < var(stop_val):
        #     break
        if c_loss.data.item() < EPSILON.data.item():
            break
        
        if (time.time() - start_time)/(i+1) > 2000:
            log_file = open(file_dir, 'a')
            log_file.write('TIMEOUT: avg epoch time > 2000s \n')
            log_file.close()
            TIME_OUT = True
            break
    
    res = real_data_loss + lambda_ * real_safe_loss# loss # f_loss.div(len(X_train))

    log_file = open(file_dir, 'a')
    spend_time = time.time() - start_time
    log_file.write('One train: Optimization--' + str(spend_time) + ',' + str(i+1) + ',' + str(spend_time/(i+1)) + '\n')
    log_file.close()
    
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


def distance_f_interval_REINFORCE(X_list, target, Theta):
    alpha_smooth_max_var = var(alpha_smooth_max)
    res = var(0.0)
    # print('X_list', len(X_list))
    reward_list = list()
    log_p_list = list()
    p_list = list()
    #! Smooth Max
    res_up = var(0.0)
    res_base = var(0.0)
    if len(X_list) == 0:
        res = var(1.0)
        return res
    for X_table in X_list:
        X_min = X_table['x_min'].getInterval()
        X_max = X_table['x_max'].getInterval()
        pi = X_table['probability']
        p = X_table['explore_probability']
        # print('pi, p', pi.data.item(), p.data.item())

        X = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
        X.left = torch.min(X_min.left, X_max.left)
        X.right = torch.max(X_min.right, X_max.right)

        reward = var(0.0)
        intersection_interval = get_intersection(X, target)
        if intersection_interval.isEmpty():
            reward = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
        else:
            reward = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))

        tmp_res = reward.mul(pi.div(p))
        # tmp_res is the reward
        tmp_p = torch.log(pi)

        log_p_list.append(tmp_p)
        reward_list.append(reward)
        p_list.append(p)

        res = res.add(tmp_res)
    res = res.div(var(len(X_list)).add(EPSILON))
    
    return res, p_list, log_p_list, reward_list


def distance_f_interval_new(X_list, target):
    for X_table in X_list:
        X_min = X_table['x_min'].getInterval()
        X_max = X_table['x_max'].getInterval()
        res = var(0.0)

        X = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
        X.left = torch.min(X_min.left, X_max.left)
        X.right = torch.max(X_min.right, X_max.right)

        reward = var(0.0)
        intersection_interval = get_intersection(X, target)
        if intersection_interval.isEmpty():
            # print('isempty')
            reward = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
        else:
            # print('not empty')
            reward = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
        
        res = torch.max(res, reward)
    # print(f"result length, {len(X_list)}, reward: {res}")
    # res is the worst case cost of X_list
    return res


def distance_f_interval_center(X_list, target):
    res = var(0.0)
    for X_table in X_list:
        c = X_table.c
        delta = X_table.delta

        X = domain.Interval(P_INFINITY.data.item(), N_INFINITY.data.item())
        X.left = c.sub(delta)
        X.right = c.add(delta)

        reward = var(0.0)
        intersection_interval = get_intersection(X, target)
        if intersection_interval.isEmpty():
            # print('isempty')
            reward = torch.max(target.left.sub(X.left), X.right.sub(target.right)).div(X.getLength())
        else:
            # print('not empty')
            reward = var(1.0).sub(intersection_interval.getLength().div(X.getLength()))
        
        res = torch.max(res, reward)
    return res


def extract_result_safty(symbol_table_list):
    res_l, res_r = P_INFINITY, N_INFINITY
    for symbol_table in symbol_table_list:
        res_l = torch.min(res_l, symbol_table['x_min'].getInterval().left)
        res_r = torch.max(res_r, symbol_table['x_max'].getInterval().right)
    
    return res_l.data.item(), res_r.data.item()


def create_ball_perturbation(X_train, distribution_list, w):
    perturbation_x_dict = {
        distribution: list() for distribution in distribution_list
    }
    for X in X_train:
        # TODO: for now, only one input variable
        x = X[0]
        l, r = x - w, x + w
        for distribution in distribution_list:
            x_list = generate_distribution(x, l, r, distribution, unit=6)
            perturbation_x_dict[distribution].extend(x_list)
    return perturbation_x_dict


def split_component(perturbation_x_dict, x_l, x_r, num_components):
    # TODO: add vector-wise component split
    x_min, x_max = x_l[0], x_r[0]
    for distribution in perturbation_x_dict:
        x_min, x_max = min(min(perturbation_x_dict[distribution]), x_min), max(max(perturbation_x_dict[distribution]), x_max)
    component_length = (x_max - x_min) / num_components
    component_list = list()
    for i in range(num_components):
        l = x_min + i * component_length
        r = x_min + (i + 1) * component_length
        center = [(l + r) /  2.0]
        width = [(r - l) / 2.0]
        component_group = {
            'center': center,
            'width': width,
        }
        component_list.append(component_group)
    return component_list


def extract_upper_probability_per_component(component, perturbation_x_dict):
    p_list = list()
    for distribution in perturbation_x_dict:
        x_list = perturbation_x_dict[distribution]
        cnt = 0
        for X in x_list:
            x = [X] #TODO: X is a value
            if in_component(x, component):
                cnt += 1
        p = cnt * 1.0 / len(x_list) + eps + random.uniform(0, 0.1)
        p_list.append(p)
    return max(p_list)


def assign_probability(perturbation_x_dict, component_list):
    '''
    perturbation_x_dict = {
        distribution: x_list, # x in x_list are in the form of value
    }
    component_list:
    component: {
        'center': center # center is vector
        'width': width # width is vector
    }
    keep track of under each distribution, what portiton of x_list fall 
    in to this component
    '''
    for idx, component in enumerate(component_list):
        p = extract_upper_probability_per_component(component, perturbation_x_dict)
        component['p'] = p
        component_list[idx] = component
    # print(f"sum of upper bound: {sum([component['p'] for component in component_list])}")
    return component_list


def in_component(X, component):
    # TODO: 
    center = component['center']
    width = component['width']
    for i, x in enumerate(X):
        if x >= center[i] - width[i] and x < center[i] + width[i]:
            pass
        else:
            return False
    return True


def assign_data_point(X_train, y_train, component_list):
    for idx, component in enumerate(component_list):
        component.update(
            {
            'x': list(),
            'y': list(),
            }
        )
        for idx, X in enumerate(X_train):
            if in_component(X, component):
                component['x'].append(X)
                component['y'].append(y_train[idx])
        component_list[idx] = component
    return component_list
        

def extract_abstract_representation(
    X_train, 
    y_train, 
    x_l, 
    x_r, 
    num_components, 
    w=0.3):
    # bs < num_components, w is half of the ball width
    '''
    Steps:
    # 1. generate perturbation, small ball covering following normal, uniform, poission
    # 2. measure probability 
    # 3. slice X_train, y_train into component-wise
    '''
    start_t = time.time()

    perturbation_x_dict = create_ball_perturbation(X_train, 
        distribution_list=["normal", "uniform", "beta", "original"], 
        w=w)
    component_list = split_component(perturbation_x_dict, x_l, x_r, num_components)

    # create data for batching, each containing component and cooresponding x, y
    component_list = assign_probability(perturbation_x_dict, component_list)
    component_list = assign_data_point(X_train, y_train, component_list)
    random.shuffle(component_list)

    print(component_list)
    print(f"-- Generate Perturbation Set --")
    print(f"--- {time.time() - start_t} sec ---")

    return component_list