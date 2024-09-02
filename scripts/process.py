"""Auxiliary code to get data in the (inverse) reinforcement learning paradigm"""

import os
import numpy as np
import pandas as pd
from copy import deepcopy as copy


# Rescue World for Teams (RW4T) Configurations
SIDE_LEN = 10
NUM_MEDICAL_KITS = 6
MEDICAL_KIT_IDX = {
    (0, 5): 0,
    (1, 1): 1,
    (1, 7): 2,
    (3, 7): 3,
    (6, 4): 4,
    (8, 6): 5
    }
NUM_TRIALS = 5


###### Get MDPs STATE #######

def get_state(df: pd.DataFrame, num_bins:int=None) -> np.array:
    """Get continuous (or discrete) state sequence"""
    rescue_status = get_rescue_status(df)
    robot_status = get_robot_state(df).reshape( [-1, 1] )
    if num_bins is None:
        position = get_user_pos(df)
        robot_position = get_user_pos(df, "RobotUnityPos")
    else:
        position = get_discrete_user_pos(df, num_bins)
        robot_position = get_discrete_user_pos(df, num_bins, "RobotUnityPos")
    return np.concatenate([
        position,
        rescue_status, 
        robot_position,
        robot_status], axis=1)


def get_rescue_status(df: pd.DataFrame) -> np.ndarray:
    """Get medical kit status at each frame
    
    Args:
        df (pd.DataFrame): dataframe of a RW4T trial. Must have
                           `GridRep` column, which contains 
                           SIDE_LEN x SIDE_LEN digits separated
                           by underscores. The digits are
                           0 - walkable grids
                           1 - unwalkable grids
                           9 - remaining drop off location
    Returns:
        np.ndarray of shape N X NUM_MEDICAL_KITS, where `N = len(df)`
        Each array entry is either 0 or 1, representing whether medical
        kit was delivered (see `LOCATION_KITS`)
    """
    # Placeholder
    medical_kits = np.zeros([len(df), NUM_MEDICAL_KITS])

    # Function to convert a grid from R4WT to numeric np.ndarray
    def grid2numpy_(grid):
        grid = np.array(grid.split("_"), dtype=int)
        grid.resize(SIDE_LEN, SIDE_LEN)
        return grid
        
    grid_seq = df["GridRep"]
    for frame, grid in grid_seq.items():
        grid = grid2numpy_(grid)
        # Check where medical kits are located
        x, y = np.where(grid == 9)
        for coordinate in zip(x, y):
            # Medical kit present at coordinate;
            # Annotate that in its corresponding index
            kit_idx = MEDICAL_KIT_IDX[coordinate]
            medical_kits[frame][kit_idx] = 1

    return medical_kits


def get_robot_state(df: pd.DataFrame) -> np.ndarray:
    """Get robot status at each frame
    
    Args:
        df (pd.DataFrame): dataframe of a RW4T trial. Must have the
                           `RobotState` column, which contains a string
                           with value "Stopped" or 2 integers separated
                           by an underscore, representing the robot's
                           goal location in the grid representation.
                           This location should be one of the locations
                           defined in the `MEDICAL_KIT_IDX` variable.
    Returns:
        np.ndarray of shape N, where `N = len(df)`, with values from
        -1 to 5. -1 represents a Stopped robot, the rest represents
        locations of medical kits as defined in `MEDICAL_KIT_IDX`.
    """
    def distance(x, y):
        dx = np.abs(x[0] - y[0])
        dy = np.abs(x[1] - y[1])
        return dx + dy

    # Update robot status in-place
    robot_status = copy(df["RobotState"].values)

    for idx, status in enumerate(robot_status):
        if status == "Stopped":
            robot_status[idx] = -1
        else:
            x, y = status.split("_")
            coordinate = int(x), int(y)
            nearest = min(MEDICAL_KIT_IDX, key=lambda x: distance(x, coordinate))
            robot_status[idx] = MEDICAL_KIT_IDX[nearest]
    return robot_status.astype(int)


def get_user_pos(df: pd.DataFrame, col_name="PlayerUnityPos") -> np.array:
    """Get user position at each frame
    
    Args:
        df (pd.DataFrame): dataframe of a RW4T trial. Must have the
                           `PlayerUnityPos` column, 3 floating numbers
                           separated by underscores, representing Unity
                           coordinate form.
    Returns:
        np.array of shape N x 2, where `N = len(df)`, with values from
        0 to 80.
    """
    # Process user location in-place
    tmp = df[col_name].values
    user_loc = np.zeros([len(df), 2])
    for idx, loc in enumerate(tmp):
        y, _, x = loc.split("_")
        x = 12 - float(x)
        y = float(y) + 4
        user_loc[idx] = x, y

    return user_loc


def get_discrete_user_pos(df, num_bins, col_name="PlayerUnityPos"):
    positions = get_user_pos(df, col_name)
    bins = np.linspace(0, 80, num_bins + 1)
    x_coord = positions[:, 0]
    y_coord = positions[:, 1]
    x = np.digitize(x_coord, bins) - 1
    y = np.digitize(y_coord, bins) - 1

    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])
    return np.concatenate([x, y], axis=1)


#### MDPs ACTIONS


def get_actions(df, num_bins = None):
    states = get_state(df, num_bins)
    if num_bins is None:
        actions = get_2dcontinuous_actions(states)
    else:
        actions = get_2ddiscrete_actions(states)
        
    actions = get_rescue_actions(df, states, actions)
    actions = add_robot_moves(df, states, actions)

    return actions


def get_2dcontinuous_actions(states):
    positions = states[:, :2]
    prev_pos = positions[:-1]
    next_pos = positions[1:]
    res = next_pos - prev_pos
    dx, dy = res[:,0], res[:, 1]
    angle = np.arctan2(dy, dx)
    wait_idx = np.where(np.abs(dx) + np.abs(dy) == 0)
    angle = angle.astype(object)
    angle[wait_idx] = "wait"
    angle = np.insert(angle, -1, "wait")
    
    return angle

def get_2ddiscrete_actions(states):
    """"""

    positions = states[:, :2]
    prev_pos = positions[:-1]
    next_pos = positions[1:]

    res = next_pos - prev_pos
    steps = {(0,-1): 'up',
             (1,0): 'right',
             (-1,0): 'left',
             (0,1): 'down',
             (0,0): 'wait',
             (-1,1): 'diagonal',
             (1,-1): 'diagonal'
            }
    
    actions = np.zeros(len(positions)).astype(str)
    for idx, dx in enumerate(res):
        actions[idx] = steps[tuple(dx)]

    # Annotate last action
    actions[-1] = 'wait'

    return actions

def get_rescue_actions(df, state, actions):
    picks = np.where(df['ButtonsClicked'] == 'CollectButton')[0] - 1

    rescue_status = state[:, 2:2+6]
    idxs = np.where(np.any(rescue_status [1:] != rescue_status[:-1], axis=1)) 
    
    idxs = np.intersect1d(idxs, picks)
    actions[idxs] = 'collect'
    return actions


def add_robot_moves(df, state, actions):
    """Check change of robot's state:
    If it was idle, it can only go to an active state,
    which is going to a medical kit that a user prompt
    it to go.
    If it was already going to an object, it can either:
        Collect the object and become idle
        fail to collect the object and become idle
        Be Paused by user to send to another object.
    """
    robot_state = state[:, -1].astype(int)
    robot_state_dx = np.where(robot_state[1:] != robot_state[:-1]) 

    button_clicks = df['ButtonsClicked']
    move_click = np.where(button_clicks == 'MoveButton')[0] - 1
    
    idxs = np.intersect1d(robot_state_dx, move_click)
    for idx in idxs:
        actions[idx] = 'toObj' + str(robot_state[idx+1])

    pause_click = np.where(button_clicks == 'PauseButton')[0] - 1
    idxs = np.intersect1d(robot_state_dx, pause_click)
    actions[idxs] = 'stop_robot'
    
    return actions   


###### GET MDPs REWARDS #######
def get_rewards(df):
    """Get rewards from trajectories.

    This function does not include rewards from secondary tasks,
    as currently secondary tasks are not included into the 
    action space.

    """
    human_distributed = df["PlayerNum"].values
    robot_distributed = df["RobotNum"].values
    in_danger = df["DangerView"].values

    rewards = - np.ones_like(in_danger, dtype=float)
    rewards -= 10 * (in_danger == "active").astype(int)
    kit_distr = ((human_distributed[1:] - human_distributed[:-1]) == 1).astype(int)
    kit_distr = np.insert(kit_distr, 0, 0)
    
    robot_distr = ((robot_distributed[1:] - robot_distributed[:-1]) == 1).astype(int)
    robot_distr = np.insert(robot_distr, 0, 0)
    rewards += 25 * (kit_distr)
    rewards += 25 * (robot_distr)

    return rewards

