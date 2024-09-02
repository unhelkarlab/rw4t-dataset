from dataclasses import dataclass
from copy import deepcopy as copy
import numpy as np
import torch
from scripts.create_rw4t_visuals import Rw4TImage
NUM_MEDICAL_KITS = 6
MEDICAL_KIT_IDX = np.array([
    [5, 0],
    [1, 1],
    [7, 1],
    [7, 3],
    [4, 6],
    [6, 8],
])
HAZARD_LOC = np.array([
    [ 0, 2 ],
    [ 1, 1 ],
    [ 2, 8 ],
    [ 4, 7 ],
    [ 7, 3 ],
    [ 7, 4 ],
    [ 7, 6 ]
])


@dataclass(frozen=True)
class SingleStep:
    state: torch.Tensor
    next_state: torch.Tensor
    action: torch.Tensor
    reward: float
    latents: torch.Tensor
    continuous: bool

    def visualize(self, image=True):
        if image:
            pass
        else:
            print(self.state, '--->', self.action, '--->', self.next_state)
            print('Latent state value is: ', self.latents)
            print('Reward obstained is: ', self.reward)

    def state_snapshot(self):
        viz = Rw4TImage(continuous=self.continuous)
        background = copy(viz.background)
        state = self.state
        xy_coord = state[:2]
        med_kit = state[2:2+6]
        robot_pos = state[2+6:-1]
        robot = state[-1]
        for idx, loc in enumerate(HAZARD_LOC):
            viz.paste_hazard(background, loc)
        for idx, loc in enumerate(MEDICAL_KIT_IDX):
            if med_kit[idx]:
                viz.paste_medical_kit(background, loc)
        
        viz.paste_human(background, xy_coord)
        viz.paste_robot(background, robot_pos)
        return background

@dataclass(frozen=True)
class Steps:
    states: torch.Tensor
    next_states: torch.Tensor
    actions: torch.Tensor
    rewards: float
    latents: torch.Tensor
    continuous: bool

    def append(self, step: SingleStep):
        self.states.append(step.state)
        self.next_states.append(step.next_state)
        self.actions.append(step.action)
        self.rewards.append(step.reward)
        self.latents.append(step.latents)

    def __getitem__(self, idx):
        
        st = self.states[idx]
        next_st = self.next_states[idx]
        act = self.actions[idx]
        rew = self.rewards[idx]
        lats = self.latents[idx]

        if isinstance(idx, int):
            return SingleStep(
                state=st,
                next_state=next_st,
                action=act,
                reward=rew,
                latents=lats,
                continuous=self.continuous
            )
        else:
            return Steps(
                states=st,
                next_states=next_st,
                actions=act,
                rewards=rew,
                latents=lats,
                continuous=self.continuous
            )

    def __len__(self):
        return len(self.states)


def get_trajectory(
        mdp_states,
        mdp_actions,
        mdp_rewards,
        mdp_dones,
        traj_idx,
        continuous
):
    idxs = np.where(mdp_dones==1)[0]
    if traj_idx==0:
        stidx=0
    else:
        stidx=idxs[traj_idx-1] + 1
    endidx=idxs[traj_idx]

    return Steps(
        mdp_states[stidx:endidx-1],
        mdp_states[stidx+1:endidx],
        mdp_actions[stidx:endidx],
        mdp_rewards[stidx:endidx],
        mdp_rewards[stidx:endidx],
        continuous
        )
