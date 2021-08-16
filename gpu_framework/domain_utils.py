
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
        'trajectories': states1['trajectories'] + states2['trajectories'],
        'idx_list': states1['idx_list'] + states2['idx_list'],
        'p_list': states1['p_list'] + states2['p_list'],
    }
    return res_states


def concatenate_states_list(states_list):
    # check needed, cause pre-selected as a states
    if len(states_list) == 0:
        return states_list[0]

    res_c, res_delta = torch.cat([states['x'].c for states in states_list], 0), \
        torch.cat([states['x'].delta for states in states_list])
    trajectories_list = [states['trajectories'] for states in states_list]
    idx_list = [states['idx_list'] for states in states_list]
    p_list = [states['p_list'] for states in states_list]
    res_states = {
        'x': domain.Box(res_c, res_delta),
        'trajectories': sum(trajectories_list, []),
        'idx_list': sum(idx_list, []),
        'p_list': sum(p_list, []),
    }
    return res_states
