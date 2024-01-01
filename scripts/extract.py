"""Code to get data in the (inverse) reinforcement learning paradigm"""

import os
import numpy as np
import pandas as pd

from process import get_state, get_actions, get_rewards
from clean import clean_trajectory


# First continuous processing
mdp_states = []
mdp_actions = []
mdp_rewards = []
mdp_dones = []
parent = "..\\dataset\\raw"
traj_dir = "..\\dataset\\trajectories"

for user_folder in os.listdir(parent):
    print("Processing", user_folder)
    path = os.path.join(parent, user_folder)
    for trial in range(3, 8):
        print("Trial number", trial)
        file = os.path.join(path, user_folder + '-' + str(trial) + '.csv')
        df = pd.read_csv(file)
        states = get_state(df)
        actions = get_actions(df)
        rewards = get_rewards(df)
        states, actions, rewards = clean_trajectory(states, actions, rewards)

        dones = np.zeros_like(actions, dtype=int)
        dones[-1] = 1
        mdp_states.append(states)
        mdp_actions.append(actions)
        mdp_rewards.append(rewards)
        mdp_dones.append(dones)

mdp_states = np.concatenate(mdp_states)
mdp_actions = np.concatenate(mdp_actions)
mdp_rewards = np.concatenate(mdp_rewards)
mdp_dones = np.concatenate(mdp_dones)


np.save(os.path.join(traj_dir,"continuous\\states"), mdp_states, allow_pickle=False)
np.save(os.path.join(traj_dir,"continuous\\actions"), mdp_actions)
np.save(os.path.join(traj_dir,"continuous\\rewards"), mdp_rewards, allow_pickle=False)
np.save(os.path.join(traj_dir,"continuous\\dones"), mdp_dones, allow_pickle=False)

# Discrete processing

mdp_states = []
mdp_actions = []
mdp_rewards = []
mdp_dones = []

for user_folder in os.listdir(parent):
    print("Processing", user_folder)
    path = os.path.join(parent, user_folder)
    for trial in range(3, 8):
        print("Trial number", trial)
        file = os.path.join(path, user_folder + '-' + str(trial) + '.csv')
        df = pd.read_csv(file)
        states = get_state(df, num_bins=10)
        actions = get_actions(df, num_bins=10)
        rewards = get_rewards(df)
        states, actions, rewards = clean_trajectory(states, actions, rewards)

        dones = np.zeros_like(actions, dtype=int)
        dones[-1] = 1
        mdp_states.append(states)
        mdp_actions.append(actions)
        mdp_rewards.append(rewards)
        mdp_dones.append(dones)


mdp_states = np.concatenate(mdp_states)
mdp_actions = np.concatenate(mdp_actions)
mdp_rewards = np.concatenate(mdp_rewards)
mdp_dones = np.concatenate(mdp_dones)

np.save(os.path.join(traj_dir,"discrete\\states"), mdp_states, allow_pickle=False)
np.save(os.path.join(traj_dir,"discrete\\actions"), mdp_actions)
np.save(os.path.join(traj_dir,"discrete\\rewards"), mdp_rewards, allow_pickle=False)
np.save(os.path.join(traj_dir,"discrete\\dones"), mdp_dones, allow_pickle=False)

