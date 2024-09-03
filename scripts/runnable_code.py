
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import os

from scripts.data_types import get_trajectory


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

TRAJ_DIR = "dataset/trajectories"
mdp_states = np.load(os.path.join(TRAJ_DIR,"continuous\\states.npy"))
mdp_actions = np.load(os.path.join(TRAJ_DIR,"continuous\\actions.npy"), allow_pickle=True)
mdp_rewards = np.load(os.path.join(TRAJ_DIR,"continuous\\rewards.npy"))
mdp_dones = np.load(os.path.join(TRAJ_DIR,"continuous\\dones.npy"))

# Get one single trajectory
traj = get_trajectory(
    mdp_states, mdp_actions, mdp_rewards, mdp_dones, traj_idx=0, continuous=True
)

ANIMATION_SPEED=0.00001
anim = create_video(traj)
"""
TODO:
1. Append the data into the dataclass, which holds actions, rewards, and so on. I think it's done?
2. Compare to Franka Kitchen, and imitation style things.
3. Add data gym style or something that can be downloadable and fed into the algorithm
"""
