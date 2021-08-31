
import torch
import domain


def concatenate_states(states1, states2):
    if len(states1) == 0:
        return states2
    if len(states2) == 0:
        return states1
    
    # x, trajectories, idx_list, p_list
    res_c, res_delta = torch.cat((states1['x'].c, states2['x'].c), 0), torch.cat((states1['x'].delta, states2['x'].delta), 0)
    res_states = {
        'x': domain.Box(res_c, res_delta),
        'trajectories_l': states1['trajectories_l'] + states2['trajectories_l'],
        'trajectories_r': states1['trajectories_r'] + states2['trajectories_r'],
        'idx_list': states1['idx_list'] + states2['idx_list'],
        'p_list': states1['p_list'] + states2['p_list'],
    }
    return res_states


def concatenate_states_list(states_list):
    # check needed, cause pre-selected as a states
    if len(states_list) == 0:
        return states_list[0]
    c_list, delta_list, trajectories_l_list, trajectories_r_list, idx_list, p_list = list(), list(), list(), list(), list(), list()
    for states in states_list:
        c_list.append(states['x'].c)
        delta_list.append(states['x'].delta)
        trajectories_l_list.append(states['trajectories_l'])
        trajectories_r_list.append(states['trajectories_r'])
        idx_list.append(states['idx_list'])
        p_list.append(states['p_list'])
    res_c, res_delta = torch.cat(c_list, 0), torch.cat(delta_list)
    # res_c, res_delta = torch.cat([states['x'].c for states in states_list], 0), \
    #     torch.cat([states['x'].delta for states in states_list])
    # trajectories_list = [states['trajectories'] for states in states_list]
    # idx_list = [states['idx_list'] for states in states_list]
    # p_list = [states['p_list'] for states in states_list]
    res_states = {
        'x': domain.Box(res_c, res_delta),
        'trajectories_l': sum(trajectories_l_list, []),
        'trajectories_r': sum(trajectories_r_list, []),
        'idx_list': sum(idx_list, []),
        'p_list': sum(p_list, []),
    }
    return res_states
