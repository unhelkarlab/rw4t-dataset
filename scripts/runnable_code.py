## HERE IS WHERE DATASET IS CREATED
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
mdp_states = np.load(os.path.join(TRAJ_DIR,"discrete\\states2.npy"))
mdp_actions = np.load(os.path.join(TRAJ_DIR,"discrete\\actions2.npy"), allow_pickle=True)
mdp_rewards = np.load(os.path.join(TRAJ_DIR,"discrete\\rewards2.npy"))
mdp_dones = np.load(os.path.join(TRAJ_DIR,"discrete\\dones2.npy"))
ids = np.load(os.path.join(TRAJ_DIR,"discrete\\ids2.npy")).reshape(-1, 2)

# Get one single trajectory
traj2 = get_trajectory(
    mdp_states, mdp_actions, mdp_rewards, mdp_dones, traj_idx=5, continuous=False
)

# Having what I have, redo to have trajectories like (s,o_r, o_h, a_r, a_h)


ANIMATION_SPEED=100
# anim = create_video(traj2)
# For HBC training, creatining a gini
trajectories = []
options_h = []
options_r = []
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



#  Add robot actions?
positions = mdp_states[:, -3:-1]
prev_pos = positions[:-1]
next_pos = positions[1:]
res = next_pos - prev_pos
steps = {(0,-1): 'up',
        (1,0): 'right',
        (-1,0): 'left',
        (0,1): 'down',
        (0,0): 'wait'
        }

actions_r = np.zeros(len(positions)).astype(str)
for idx, dx in enumerate(res):
    if tuple(dx) in steps:
        actions_r[idx] = steps[tuple(dx)]
    else:
        actions_r[idx] = None

# Annotate last action
actions_r[-1] = 'wait'
from scripts.clean import _robot_picks
picks = _robot_picks(mdp_states, mdp_actions)
actions_r[picks] = 'collect'



for traj_idx in range(100):
    try:
        traj = get_trajectorywrewards(
            mdp_states,
            mdp_actions,
            mdp_rewards,
            mdp_dones,
            traj_idx=traj_idx,
            mdp_r_actions=None #actions_r
            )
    except KeyError as e:
        print("A bug in the code in mdp_actions containing an action `toObj2` that isn't interpretable by the code", e)
        continue
    trajectories.append(traj)
    opth = get_options(traj)
    options_h.append(opth)


    idxs = np.where(mdp_dones==1)[0]
    if traj_idx==0:
        stidx=0
    else:
        stidx=idxs[traj_idx-1] + 1
    endidx=idxs[traj_idx]
    optr = mdp_states[stidx:endidx,-1]
    options_r.append(optr)
    options.append(np.stack((opth, optr), axis=1))

N_TRAIN_TRAJ = 90
gini = Oracle(
    expert_trajectories=trajectories[:N_TRAIN_TRAJ],
    true_options=options[:N_TRAIN_TRAJ],
    expert_trajectories_test=trajectories[N_TRAIN_TRAJ:],
    true_options_test=options[N_TRAIN_TRAJ:],
    # ids = ids[:N_TRAIN_TRAJ]
)
gini.save("/home/liubove/Documents/my-packages/rw4t-dataset/dataset/trajectories/"
          "discrete/gini_n18-1090")

def process_trajectory(traj):
    team_delivery = sum(traj.obs[-1][2:8]==0)
    human_delivery = sum(traj.acts==7)
    robot_delivery = team_delivery - human_delivery
    return team_delivery, robot_delivery

robot_utility = []
team_performance = []
for traj in trajectories:
    t, r = process_trajectory(traj)
    team_performance.append(t)
    robot_utility.append(r)
import time
for participant in range(20):
    for task in range(5):
        idx = participant * 5 + task
        print(idx)
        print("Participant: ", participant, "Task: ", task, "Robot utility: ", robot_utility[idx])
        if not (idx+1)%5:
            print('-------------------------')
            time.sleep(5)
            