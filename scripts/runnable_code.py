
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import os

from scripts.data_types import get_trajectory, get_trajectorywrewards
from latent_active_learning.oracle import Oracle, Random, QueryCapLimit
MEDICAL_KIT_IDX = np.array([
    [5, 0],
    [1, 1],
    [7, 1],
    [7, 3],
    [4, 6],
    [6, 8],
])

def create_video(traj):
    fig = plt.figure()
    im = plt.imshow(traj[0].state_snapshot())

    def animate(t):
        im.set_array(traj[t].state_snapshot())
        return im,
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames = len(traj),
        interval = ANIMATION_SPEED,
        blit = True,
        repeat=False
    )

    plt.show()
    return anim

# TRAJ_DIR = "dataset/trajectories"
# mdp_states = np.load(os.path.join(TRAJ_DIR,"continuous\\states.npy"))
# mdp_actions = np.load(os.path.join(TRAJ_DIR,"continuous\\actions.npy"), allow_pickle=True)
# mdp_rewards = np.load(os.path.join(TRAJ_DIR,"continuous\\rewards.npy"))
# mdp_dones = np.load(os.path.join(TRAJ_DIR,"continuous\\dones.npy"))

# # Get one single trajectory
# traj = get_trajectory(
#     mdp_states, mdp_actions, mdp_rewards, mdp_dones, traj_idx=10, continuous=True
# )

# ANIMATION_SPEED=1
# anim = create_video(traj)
"""
TODO:
1. Append the data into the dataclass, which holds actions, rewards, and so on. I think it's done?
2. Compare to Franka Kitchen, and imitation style things.
3. Add data gym style or something that can be downloadable and fed into the algorithm
"""
TRAJ_DIR = "dataset/trajectories"
mdp_states = np.load(os.path.join(TRAJ_DIR,"discrete\\states.npy"))
mdp_actions = np.load(os.path.join(TRAJ_DIR,"discrete\\actions.npy"), allow_pickle=True)
mdp_rewards = np.load(os.path.join(TRAJ_DIR,"discrete\\rewards.npy"))
mdp_dones = np.load(os.path.join(TRAJ_DIR,"discrete\\dones.npy"))

# Get one single trajectory
traj2 = get_trajectory(
    mdp_states, mdp_actions, mdp_rewards, mdp_dones, traj_idx=5, continuous=False
)

ANIMATION_SPEED=100
# anim = create_video(traj2)
# For HBC training, creatining a gini
trajectories = []
options = []

def get_options(traj):
    last_status = traj.obs[-1][2:8]
    if last_status.sum()>0:
        last_pos2kits = traj.obs[-1][:2] - MEDICAL_KIT_IDX[last_status.astype(bool)]
        last_pos2kits = (last_pos2kits**2).sum(1)
        closest = last_pos2kits.argmin()
        closest = (MEDICAL_KIT_IDX[last_status.astype(bool)][closest] == MEDICAL_KIT_IDX).all(1)
        closest = np.where(closest)[0].item()
    else:
        closest = -1
    med_kits = traj.obs[:, 2:8]
    options = closest * np.ones(len(med_kits))

    prev_med_left = med_kits[-1]
    k = closest
    for idx, status in enumerate(reversed(med_kits)):
        if not np.all(status == prev_med_left):
            if traj.acts[-idx]==7:
                k = np.where(prev_med_left!= status)[0].item()
            prev_med_left = status
        options[-idx-1] = k
    if options[-1]==-1:
        options[-1] = options[-2]
    return options



for traj_idx in range(100):
    traj = get_trajectorywrewards(
        mdp_states,
        mdp_actions,
        mdp_rewards,
        mdp_dones,
        traj_idx=traj_idx
        )
    trajectories.append(traj)
    options.append(get_options(traj))

N_TRAIN_TRAJ = 90
gini = Oracle(
    expert_trajectories=trajectories[:N_TRAIN_TRAJ],
    true_options=options[:N_TRAIN_TRAJ],
    expert_trajectories_test=trajectories[N_TRAIN_TRAJ:],
    true_options_test=options[N_TRAIN_TRAJ:],
)
gini.save("/home/liubove/Documents/my-packages/rw4t-dataset/dataset/trajectories/"
          "discrete/gini_n18")
