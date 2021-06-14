import numpy as np
from termcolor import colored

from constants import *
# from optimization import *

if benchmark_name == "thermostat":
    from thermostat_nn_batch import (
        ThermostatNN,
        load_model,
        save_model,
        initialization_abstract_state,
        initialization_point_nn,
    )
if benchmark_name == "mountain_car":
    from mountain_car_batch import (
        MountainCar,
        load_model,
        save_model,
        initialization_abstract_state,
        initialization_point_nn,
    )
if benchmark_name == "unsound_1":
    from unsound_1_batch import (
        Unsound_1,
        load_model,
        save_model,
        initialization_abstract_state,
        initialization_point_nn,
    )
if benchmark_name == "sampling_1":
    from sampling_1_batch import (
        Sampling_1,
        load_model,
        save_model,
        initialization_abstract_state,
        # initialization_point_nn,
    )
if benchmark_name == "sampling_2":
    from sampling_2_batch import (
        Sampling_2,
        load_model,
        save_model,
        initialization_abstract_state,
        # initialization_point_nn,
    )
if benchmark_name == "path_explosion":
    from path_explosion_sound import (
        PathExplosion,
        load_model,
        save_model,
        initialization_abstract_state,
    )
if benchmark_name == "path_explosion_2":
    from path_explosion_2_sound import (
        PathExplosion2,
        load_model,
        save_model,
        initialization_abstract_state,
    )
if benchmark_name == "fairness_1":
    from fairness_1_batch import (
        Fairness_1,
        load_model,
        save_model,
        initialization_abstract_state,
    )

import domain


def check_safety(res, target):
    intersection_interval = get_intersection(res, target)
    if intersection_interval.isEmpty():
        # print('isempty')
        score = torch.max(target.left.sub(res.left), res.right.sub(target.right)).div(res.getLength())
    else:
        # print('not empty')
        score = var(1.0).sub(intersection_interval.getLength().div(res.getLength()))
    
    return score


def get_intersection(interval_1, interval_2):
    res_interval = domain.Interval()
    res_interval.left = torch.max(interval_1.left, interval_2.left)
    res_interval.right = torch.min(interval_1.right, interval_2.right)
    return res_interval


def get_symbol_table_trajectory_unsafe_value(symbol_table, target_component, target_idx):
    trajectory_loss = var_list([0.0])
    # print(f"trajectory len: {len(symbol_table['trajectory'])}")
    tmp_symbol_table_tra_loss = list()
    safe_interval = target_component["condition"]
    unsafe_probability_condition = target_component["phi"]
    method = target_component["method"]
    if method == "last":
        trajectory = [symbol_table['trajectory'][-1]]
    elif method == "all":
        trajectory = symbol_table['trajectory'][:]

    for state in trajectory:
        X = state[target_idx]
        # print(f"X:{X.left.data.item(), X.right.data.item()}")
        intersection_interval = get_intersection(X, safe_interval)
        if intersection_interval.isEmpty():
            unsafe_value = var_list([1.0])
        else:
            safe_probability = intersection_interval.getLength().div(X.getLength())
            # TODO: remove this part
            # if safe_probability.data.item() > 1 - unsafe_probability_condition.data.item():
            #     unsafe_value = var_list([0.0])
            # else:
            #     unsafe_value = 1 - safe_probability
            if real_unsafe_value:
                unsafe_value = 1 - safe_probability
            else:
                if safe_probability.data.item() > 1 - unsafe_probability_condition.data.item():
                    unsafe_value = var_list([0.0])
                else:
                    unsafe_value = 1 - safe_probability
        if verify_outside_trajectory_loss:
            # print("loss of one symbol table", unsafe_value, symbol_table["probability"])
            tmp_symbol_table_tra_loss.append(unsafe_value * symbol_table["probability"])
        else:
            trajectory_loss = torch.max(trajectory_loss, unsafe_value)
            # print(f"trajectory_loss: {trajectory_loss.data.item()}")
    if verify_outside_trajectory_loss:
        return tmp_symbol_table_tra_loss
    else:
        return trajectory_loss


def extract_unsafe(abstract_state, target_component, target_idx):
    aggregation_p = var_list([0.0])
    abstract_state_unsafe_value = var_list([0.0])
    if verify_outside_trajectory_loss:
        symbol_table_wise_loss_list = list()
        # print(f"Abstract_state")
        for symbol_table in abstract_state:
            # print('Symbol_Table')
            tmp_symbol_table_tra_loss = get_symbol_table_trajectory_unsafe_value(symbol_table, target_component, target_idx=target_idx)
            symbol_table_wise_loss_list.append(tmp_symbol_table_tra_loss)
            aggregation_p += symbol_table['probability']
        abstract_state_wise_trajectory_loss = zip(*symbol_table_wise_loss_list)
        for l in abstract_state_wise_trajectory_loss:
            # print(l) 
            abstract_state_unsafe_value = torch.max(abstract_state_unsafe_value, torch.sum(torch.stack(l)))
    else:
        for symbol_table in abstract_state:
            trajectory_unsafe_value = get_symbol_table_trajectory_unsafe_value(symbol_table, target_component, target_idx=target_idx)
            # print(f"component p: {symbol_table['probability'].data.item()}, trajectory_unsafe_value: {trajectory_unsafe_value}")
            abstract_state_unsafe_value += symbol_table['probability'] * trajectory_unsafe_value
            aggregation_p += symbol_table['probability']
        # print(f"temporary aggragation p: {aggregation_p}")
    return aggregation_p, abstract_state_unsafe_value


def verify(abstract_state_list, target):
    for idx, target_component in enumerate(target):
        target_name = target_component["name"]
        all_unsafe_probability = var_list([0.0])
        print(f"# of abstract state: {len(abstract_state_list)}")
        for abstract_state in abstract_state_list:
            aggregation_p, unsafe_probability = extract_unsafe(abstract_state, target_component, target_idx=idx)
            print(f"aggregation_p: {aggregation_p.data.item()}, unsafe_probability: {unsafe_probability.data.item()}")
            #! make the aggregation_p make more sense
            aggregation_p = torch.min(var(1.0), aggregation_p)
            all_unsafe_probability += aggregation_p * unsafe_probability
        
        if not debug:
            log_file_evaluation = open(file_dir_evaluation, 'a')
        if all_unsafe_probability.data.item() <= target_component['phi'].data.item():
            print(colored(f"#{target_name}: Verified Safe!", "green"))
            if not debug:
                log_file_evaluation.write(f"Verification of #{target_name}#: Verified Safe!\n")
        else:
            print(colored(f"#{target_name}: Not Verified Safe!", "red"))
            if not debug:
                log_file_evaluation.write(f"Verification of #{target_name}#: Not Verified Safe!\n")
        
        print(f"learnt unsafe_probability: {all_unsafe_probability.data.item()}, target unsafe_probability: {target_component['phi'].data.item()}")
        if not debug:
            log_file_evaluation.write(f"Details#learnt unsafe_probability: {all_unsafe_probability.data.item()}, target unsafe_probability: {target_component['phi'].data.item()}\n")
    if debug:
        exit(0)


def show_component_p(component_list):
    component_p_list = list()
    for component in component_list:
        component_p_list.append(component['p'])
        # print(f"component p: {component['p']}, center: {component['center']}, width: {component['width']}")
    # print(f"component p list: {component_p_list}")
    print(f"sum of component p: {sum(component_p_list)}")
    return 


def verification(model_path, model_name, component_list, target):
    if benchmark_name == "thermostat":
        m = ThermostatNN(l=l, nn_mode=nn_mode, module=module)
    if benchmark_name == "mountain_car":
        m = MountainCar(l=l, nn_mode=nn_mode, module=module)
    _, m = load_model(m, MODEL_PATH, name=model_name)
    if m is None:
        print(f"No model to Verify!!")
        exit(0)
    # m.cuda()
    if torch.cuda.is_available():
        m.cuda()
    m.eval()

    for param in m.parameters():
        param.requires_grad = False
    # print(m.nn.linear1.weight)
    
    abstract_state_list = initialization_abstract_state(component_list)
    print(f"Ini # of abstract state: {len(abstract_state_list)}")
    show_component_p(component_list)
    # print(abstract_state_list[0][0]["x"].c)
    abstract_state_list = m(abstract_state_list)
    verify(abstract_state_list, target)


def trajectory2points(trajectory_list, bs):
    states, actions = list(), list()
    for trajectory in trajectory_list:
        for (state, action) in trajectory:
            states.append(state)
            actions.append([action])
    c = list(zip(states, actions))
    random.shuffle(c)
    states, actions = zip(*c)
    states, actions = np.array(states), np.array(actions)

    for i in range(0, len(states), bs):
        if torch.cuda.is_available():
            yield torch.from_numpy(states[i:i+bs]).float().cuda(), torch.from_numpy(actions[i:i+bs]).float().cuda()
        else:
            yield torch.from_numpy(states[i:i+bs]).float(), torch.from_numpy(actions[i:i+bs]).float()


def test_objective(m, trajectory_test, criterion, test_bs):
    data_loss = 0.0
    count = 0
    test_bs = 8
    for x, y in trajectory2points(trajectory_test, test_bs):
        yp = m(x, version="single_nn_learning")
        batch_data_loss = criterion(yp, y)
        if debug:
            print(f"yp: {yp.squeeze()}, y: {y.squeeze()}")
            print(f"batch data loss: {batch_data_loss}")

        count += 1
        data_loss += batch_data_loss
        # update data_loss
    # print/f.write()
    test_data_loss = data_loss / count

    if not debug:
        log_file_evaluation = open(file_dir_evaluation, 'a')
        log_file_evaluation.write(f"test data loss: {test_data_loss.data.item()}\n")
    print(f"test data loss: {test_data_loss.data.item()}")
    if debug:
        exit(0)
        pass


def ini_state_batch(trajectory_test, test_abstract_bs):
    # extract the initial state from trajectory_test
    # batch initial state
    # initial domain of each batch
    # return domain
    return True


def ini_state_batch_tmp(trajectory_test):
    for trajectory in trajectory_test:
        x = trajectory[0][0] # state, (p, v)
        yield x


def point_test_tmp(res_list, target):
    #! no batch
    symbol_table = res_list[0][0]
    target_safety = list()
    for idx, target_component in enumerate(target):
        safe_interval = target_component["condition"]
        unsafe_probability_condition = target_component["phi"]
        method = target_component["method"]
        if method == "last":
            trajectory = [symbol_table['trajectory'][-1]]
        elif method == "all":
            trajectory = symbol_table['trajectory'][:]
        
        if torch.cuda.is_available():
            safe = torch.tensor([True], dtype=torch.bool).cuda()
        else:
            safe = torch.tensor([True], dtype=torch.bool)
        for state in trajectory:
            X = state[idx]
            if debug:
                print(f"X: {X.left.data.item(), X.right.data.item()}; safe interval: {safe_interval.left.data.item(), safe_interval.right.data.item()}")
                print(f"In: {X.in_other(safe_interval)}")
            safe = torch.cat((safe, X.in_other(safe_interval)), 0)
        if debug:
            print(f"safe: {safe}")
        
        # print(safe)
        if torch.all(safe):
            target_safety.append(True)
        else:
            target_safety.append(False)
    
    return target_safety


def measure_test_safety(all_safe_res, target):
    if not debug:
        log_file_evaluation = open(file_dir_evaluation, 'a')
    for target_idx, target_component in enumerate(target):
        target_name = target_component["name"]
        unsafe_probability = target_component['phi']
        unsafe_count, all_count = 0.0, 0.0
        for safe_res in all_safe_res:
            if safe_res[target_idx]:
                all_count += 1
            else:
                unsafe_count += 1
                all_count += 1
        test_unsafe_probability = unsafe_count * 1.0 / all_count
        
        if test_unsafe_probability <= unsafe_probability:
            print(colored(f"#{target_name}: Unsound Verified Safe!", "green"))
            if not debug:
                log_file_evaluation.write(f"Unsound Verification of #{target_name}#: Verified Safe!\n")
        else:
            print(colored(f"#{target_name}: Unsound Verified Unafe!", "red"))
            if not debug:
                log_file_evaluation.write(f"Unsound Verification of #{target_name}#: Verified Unsafe!\n")
        
        print(f"test_unsafe_probability: {test_unsafe_probability}")
        if not debug:
            log_file_evaluation.write(f"test_unsafe_probability: {test_unsafe_probability}\n")

        
def verify_unsound(m, trajectory_test, target, test_abstract_bs):
    # extract the initial state in trajectory
    # batch initial state, points
    # m(points)
    # test trajectory out of m
    # TODO: batch
    all_safe_res = list()
    for x in ini_state_batch_tmp(trajectory_test):
        input_point = initialization_point_nn(x)
        # print(input_point)
        res_list = m(input_point)
        batch_safe_res = point_test_tmp(res_list, target)
        all_safe_res.append(batch_safe_res)
    measure_test_safety(all_safe_res, target)


def test_data_loss(
    model_path, 
    model_name, 
    trajectory_test, 
    target, 
    test_bs=512,
    test_abstract_bs=32):
    if benchmark_name == "thermostat":
        m = ThermostatNN(l=l, nn_mode=nn_mode, module=module)
    if benchmark_name == "mountain_car":
        m = MountainCar(l=l, nn_mode=nn_mode, module=module)
    if benchmark_name == "unsound_1":
        m = Unsound_1(l=l, nn_mode=nn_mode)
    if benchmark_name == "sampling_1":
        m = Sampling_1(l=l, nn_mode=nn_mode)
    if benchmark_name == "sampling_2":
        m = Sampling_2(l=l, nn_mode=nn_mode)
    if benchmark_name == "path_explosion":
        m = PathExplosion(l=l, nn_mode=nn_mode)
    if benchmark_name == "path_explosion_2":
        m = PathExplosion2(l=l, nn_mode=nn_mode)
    if benchmark_name == "fairness_1":
        m = Fairness_1(l=l, nn_mode=nn_mode)

    _, m = load_model(m, MODEL_PATH, name=model_name)
    if m is None:
        print(f"No model to Unsound Verify!!")
        return 
    # m.cuda()
    if torch.cuda.is_available():
        m.cuda()
    m.eval()

    for param in m.parameters():
        param.requires_grad = False

    criterion = torch.nn.MSELoss()

    # split batch
    test_objective(m, trajectory_test, criterion, test_bs)
    # verify_unsound(m, trajectory_test, target, test_abstract_bs)

    




