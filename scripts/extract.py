"""Code to get data in the (inverse) reinforcement learning paradigm"""
import os
import numpy as np
import pandas as pd

from scripts.process import get_state, get_actions, get_rewards
from scripts.clean import clean_trajectory

DATA_FOLDER = "dataset/raw"
TRAJ_DIR = "dataset/trajectories"


def get_user_path(user, trial):
    print("Processing", user, ", trial", trial)
    file = os.path.join(DATA_FOLDER, user, user + '-' + str(trial) + '.csv')
    return file

def get_trial_behavior(user, trial, num_bins=None):
    # Num bins determines if it is continuous or not
    file = get_user_path(user, trial)
    
    df = pd.read_csv(file)
    states = get_state(df, num_bins=num_bins)
    actions = get_actions(df, num_bins=num_bins)
    rewards = get_rewards(df)
    # if (states[:, 2:8]==0).all(1)
    states, actions, rewards = clean_trajectory(states, actions, rewards)
    dones = np.zeros_like(actions, dtype=int)
    dones[-1] = 1
    return states, actions, rewards, dones

def get_trajs(num_bins=None):
    mdp_states, mdp_actions, mdp_dones, mdp_rewards = [], [], [], []
    for user in os.listdir(DATA_FOLDER):
        for trial in range(3, 8):
            st, acts, rews, dones = get_trial_behavior(user, trial, num_bins)
            mdp_states.append(st)
            mdp_actions.append(acts)
            mdp_rewards.append(rews)
            mdp_dones.append(dones)
    mdp_states = np.concatenate(mdp_states)
    mdp_actions = np.concatenate(mdp_actions)
    mdp_rewards = np.concatenate(mdp_rewards)
    mdp_dones = np.concatenate(mdp_dones)
    return mdp_states, mdp_actions, mdp_rewards, mdp_dones

# mdp_states_cont, mdp_actions_cont, mdp_rewards_cont, mdp_dones_cont = get_trajs()
# np.save(os.path.join(TRAJ_DIR,"continuous\\states"), mdp_states_cont)
# np.save(os.path.join(TRAJ_DIR,"continuous\\actions"), mdp_actions_cont, allow_pickle=True)
# np.save(os.path.join(TRAJ_DIR,"continuous\\rewards"), mdp_rewards_cont)
# np.save(os.path.join(TRAJ_DIR,"continuous\\dones"), mdp_dones_cont)
# Discrete processing
mdp_states, mdp_actions, mdp_rewards, mdp_dones = get_trajs(num_bins=10)


np.save(os.path.join(TRAJ_DIR,"discrete\\states"), mdp_states, allow_pickle=False)
np.save(os.path.join(TRAJ_DIR,"discrete\\actions"), mdp_actions)
np.save(os.path.join(TRAJ_DIR,"discrete\\rewards"), mdp_rewards, allow_pickle=False)
np.save(os.path.join(TRAJ_DIR,"discrete\\dones"), mdp_dones, allow_pickle=False)
