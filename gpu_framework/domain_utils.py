
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