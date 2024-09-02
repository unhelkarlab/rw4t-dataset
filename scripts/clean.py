"""Auxiliary code for processing Unity Data"""

import numpy as np
from copy import deepcopy as copy


def clean_trajectory(states, actions, rewards):
    states, actions = transition_robot_pause(states, actions)
    states, actions = transition_robot_success(states, actions)
    states, actions, rewards = correct_simultaneous_moves(states, actions, rewards)
    states, actions, rewards = remove_waits(states, actions, rewards)
    return states, actions, rewards

def correct_simultaneous_moves(states, actions, rewards):
    '''Separates simultaneous movements
    
    Simmultaneous movements happen when the player moved himself
    and sent a robot at the same time.
    Break down into two moves is necessary as in MDP environment,
    actions are taken one at a time. 
    Correction will be done by first sending robot and then moving
    '''
    # Find where 2d movements and robot usage are happening at same time
    x = states[:,0]
    y = states[:,1]
    r = states[:, -1]

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    # Get indexes only for robot being sent, not deliveries
    dr = (r[1:] - r[:-1]) > 0

    idxs_x = np.where(dx!=0)[0]
    idxs_y = np.where(dy!=0)[0]
    idxs_r = np.where(dr!=0)[0]
    
    idxs = np.concatenate(
        [np.intersect1d(idxs_x, idxs_r),
         np.intersect1d(idxs_y, idxs_r)]
        )

    # For each simultaneous movements at index, insert into actions
    # an additional action of moving robot, and the action of moving.
    for idx in idxs:
        r_status = int(states[idx+1][-1])

        insert_state = copy(states[idx])
        insert_state[-1] = r_status
        states = np.insert(states, idx + 1, insert_state, axis=0)
        actions = np.insert(actions, idx, 'toObj' + str(r_status))
        rewards = np.insert(rewards, idx, -1)
    return states, actions, rewards


def transition_robot_pause(states, actions):
    """If robot is paused, change robot status to idle (=-1)
    
    RW2T data keeps same robot status after pausing it.
    To distinguish states, we enforce pause by changing
    robot status to idle.
    """
    idxs = np.where(actions == 'stop_robot')[0]
    states[idxs + 1][:, -1] = -1
    return states, actions


def transition_robot_success(states, actions):
    """If robot picks an object, immediately change its status to idle"""
    idxs = _robot_picks(states, actions)

    
    for idx in idxs:
        for tmp in range(idx, len(states)):
            states[tmp][-1] = -1
            if 'toObj' in str(actions[tmp]):
                break

    return states, actions


def remove_waits(states, actions: np.array, rewards):
    " Only for discretized"
    # import pdb; pdb.set_trace()
    active_idx = np.where(actions != 'wait')[0]
    wait_idx = _consecutive_robot_usage(actions)
    robot_picks_idx = _robot_picks(states, actions)
    positive_rewards = np.where(rewards > 0)[0]
    keep_idxs = np.concatenate([
        active_idx,
        wait_idx,
        robot_picks_idx,
        positive_rewards])
    keep_idxs = sorted(keep_idxs)
    return states[keep_idxs], actions[keep_idxs], rewards[keep_idxs]


def _consecutive_robot_usage(actions):
    
    wait_idx = np.where(actions != 'wait')[0]
    active_actions = actions[wait_idx]
    tmp = []
    for idx in range(len(active_actions) - 1):
        if 'toObj' in str(active_actions[idx]) and 'toObj' in str(active_actions[idx + 1]):
            tmp.append(wait_idx[idx])
    return np.array(tmp, dtype=int) + 1

def _robot_picks(states, actions):
    """Get indexes when robot picks an object"""
    rescue_status = states[:, 2:-1]
    # Get where rescue status changes, and it's not due participant's
    # collection.
    idxs = np.where(np.any(rescue_status [1:] != rescue_status[:-1], axis=1))[0]
    idxs = idxs[actions[idxs] != 'collect']
    return idxs
