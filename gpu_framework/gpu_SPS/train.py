import torch
import time
import random
from torch.autograd import Variable
import nlopt
import numpy as np
import matplotlib.pyplot as plt
import copy

from helper import *
from constants import *
import constants

from gpu_DSE.thermostat_nn import * 
from gpu_DSE.data_generator import *

from utils import generate_distribution

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


def extract_abstract_state_safe_loss(abstract_state, target):
    # weighted sum of symbol_table loss in one abstract_state
    abstract_loss = var_list([0.0])
    unsafe_probability_condition = target["phi"]
    safe_interval = target["condition"]
    for symbol_table in abstract_state:
        trajectory_loss = var_list([0.0])
        for X in symbol_table['trajectory']:
            intersection_interval = get_intersection(X, safe_interval)
            if intersection_interval.isEmpty():
                unsafe_value = torch.max(safe_interval.left.sub(X.left), X.right.sub(safe_interval.right)).div(X.getLength())
            else:
                safe_portion = intersection_interval.getLength().div(X.getLength())
                unsafe_value = 1 - safe_portion
            trajectory_loss = torch.max(trajectory_loss, unsafe_value)
        abstract_loss += trajectory_loss * symbol_table['probability']
    return abstract_loss
    

def safe_distance(abstract_state_list, target):
    # measure safe distance in DSE
    # I am using sampling, and many samples the eventual average will be the same as the expectation
    
    loss = var_list([0.0])
    for abstract_state in abstract_state_list:
        abstract_state_safe_loss = extract_abstract_state_safe_loss(
            abstract_state, target
        )
        loss += abstract_state_safe_loss
    loss = loss / var(len(abstract_state_list)).add(EPSILON)
    loss = loss - target['phi']

    return loss


def cal_data_loss(m, x, y):
    # for the point in the same batch
    # calculate the data loss of each point
    # add the point data loss together
    data_loss = var_list([0.0])
    for idx in range(len(x)):
        point, label = x[idx], y[idx]
        point_data = initialization_point_nn(point)
        y_point_list = m(point_data, 'concrete')
        # should be only one partition in y['x']
        # the return value in thermostat is x, index: 2
        # be the first point symbol_table in the first point_list
        data_loss += distance_f_point(y_point_list[0][0]['x'].c[2], var(label))
    data_loss /= var(len(x)).add(EPSILON)
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
        abstract_list = m(ini_abstract_state_list, 'abstract')
        res_abstract_state_list.append(abstract_list[0]) # only one abstract state returned
    # print(f"length: {len(y_abstract_list)}")
    
    # TODO: the new safe loss function
    safe_loss = safe_distance(res_abstract_state_list, target)
    return safe_loss


def divide_chunks(component_list, bs=1):
    '''
    component: {
        'center': 
        'width':
        'p':
        'x':
        'y':
    }
    return the component={
        'center':
        'width':
        'p':
    },X, Y
    '''
    for i in range(0, len(component_list), bs):
        components = component_list[i:i + bs]
        abstract_states = list()
        x_list, y_list = list(), list()
        for component in components:
            abstract_state = {
                'center': component['center'],
                'width': component['width'],
                'p': component['p'],
            }
            x_list.extend(component['x'])
            y_list.extend(component['y'])
            abstract_states.append(abstract_state)
            # print(f"component probability: {component['p']}")

        yield x_list, y_list, abstract_states


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
        model_name=None
        ):
    print("--------------------------------------------------------------")
    print('====Start Training====')

    TIME_OUT = False

    x_min = var(10000.0)
    x_max = var(0.0)

    loop_list = list()
    loss_list = list()

    m = ThermostatNN(l=l, nn_mode=nn_mode, module=module)
    print(m)
    m.cuda()

    optimizer = torch.optim.SGD(m.parameters(), lr=lr)

    if epochs_to_skip is None:
        epochs_to_skip = -1
    
    start_time = time.time()
    for i in range(epoch):
        if i <= epochs_to_skip:
            continue
        q_loss, c_loss = var_list([0.0]), var_list([0.0])
        count = 0
        for x, y, abstract_states in divide_chunks(component_list, bs=bs):
            # print(f"x length: {len(x)}")
            batch_time = time.time()
            grad_data_loss, grad_safe_loss = var_list([0.0]), var_list([0.0])
            real_data_loss, real_safe_loss = var_list([0.0]), var_list([0.0])
            
            Theta = extract_parameters(m) # extract the parameters now, and then sample around it
            # print(f"Theta before: {Theta}")
            for (sample_theta, sample_theta_p) in sample_parameters(Theta, n=n):
                m = update_model_parameter(m, sample_theta)
                sample_time = time.time()

                data_loss = cal_data_loss(m, x, y)
                safe_loss = cal_safe_loss(m, abstract_states, target)

                # print(f"data_loss: {data_loss.data.item()}, safe_loss: {safe_loss.data.item()}, Loss TIME: {time.time() - sample_time}")
                # print(f"{'#' * 15}")
                grad_data_loss += var(data_loss.data.item()) * sample_theta_p #  torch.log(sample_theta_p) # real_q = \expec_{\theta ~ \theta_0}[data_loss]
                real_data_loss += var(data_loss.data.item())
                grad_safe_loss += var(safe_loss.data.item()) * sample_theta_p # torch.log(sample_theta_p) # real_c = \expec_{\theta ~ \theta_0}[safe_loss]
                real_safe_loss += var(safe_loss.data.item())

            # To maintain the real theta
            m = update_model_parameter(m, Theta)

            real_data_loss /= n
            real_safe_loss /= n

            print(f"real data_loss: {real_data_loss.data.item()}, real safe_loss: {real_safe_loss.data.item()}, data and safe TIME: {time.time() - batch_time}")
            q_loss += real_data_loss
            c_loss += real_safe_loss

            loss = grad_data_loss + lambda_.mul(grad_safe_loss)
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
            save_model(m, MODEL_PATH, name=model_name, epoch=i)
            
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
        # if c_loss.data.item() < EPSILON.data.item():
        #     break
        
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


##### create symbolic approximation of perturbation set of input distribution


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
        for i, X in enumerate(X_train):
            if in_component(X, component):
                component['x'].append(X)
                component['y'].append(y_train[i])
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

    print(f"component-wise x length: {[len(component['x']) for component in component_list]}")

    # print(component_list)
    print(f"-- Generate Perturbation Set --")
    print(f"--- {time.time() - start_t} sec ---")

    return component_list