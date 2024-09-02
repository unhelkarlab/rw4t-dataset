
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

traj_dir = "dataset/trajectories"
mdp_states = np.load(os.path.join(traj_dir,"discrete\\states.npy"))
mdp_actions = np.load(os.path.join(traj_dir,"discrete\\actions.npy"))
mdp_rewards = np.load(os.path.join(traj_dir,"discrete\\rewards.npy"))
mdp_dones = np.load(os.path.join(traj_dir,"discrete\\dones.npy"))

# Get one single trajectory
traj = get_trajectory(
    mdp_states, mdp_actions, mdp_rewards, mdp_dones, traj_idx=1, continuous=False
)

ANIMATION_SPEED=100
anim = create_video(traj)
"""
TODO:
2. Append the data into the dataclass, which holds actions, rewards, and so on. I think it's done?
3. Check and verify that trajectories are well done
    [Done] Why is it being repeated?
    Why is last state not added?
4. Compare to Franka Kitchen, and imitation style things.
"""
