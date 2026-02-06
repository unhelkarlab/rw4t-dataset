"""
Dataset creation and Oracle training for RW4T.

This script loads MDP trajectory data from disk, extracts human and robot
options, trains an Oracle (HBC model) for option inference, and computes/prints
performance metrics per participant and task.

TODO:
1. Append the data into the dataclass, which holds actions, rewards, and so on.
2. Compare to Franka Kitchen, and imitation style things.
3. Add data gym style or something that can be downloadable and fed into the
   algorithm
"""
import os
import time

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

from scripts.clean import _robot_picks
from scripts.data_types import get_trajectory, get_trajectorywrewards
# from latent_active_learning.oracle import Oracle

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Grid coordinates of the 6 medical kit locations in the RW4T environment.
# Used for inferring human options (which kit the human is targeting).
MEDICAL_KIT_IDX = np.array([
    [5, 0],
    [1, 1],
    [7, 1],
    [7, 3],
    [4, 6],
    [6, 8],
])

# Mapping from (dx, dy) position deltas to action labels for the robot.
# Used when inferring robot actions from state transitions.
POSITION_DELTA_TO_ACTION = {
    (0, -1): 'up',
    (1, 0): 'right',
    (-1, 0): 'left',
    (0, 1): 'down',
    (0, 0): 'wait',
}

# -----------------------------------------------------------------------------
# Video / Visualization
# -----------------------------------------------------------------------------


def create_video(traj, animation_speed_ms=100):
    """
    Create and display an animated video of a single trajectory.

    Args:
        traj: A Steps object containing the trajectory (from get_trajectory).
        animation_speed_ms: Milliseconds between frames (default 100).

    Returns:
        The FuncAnimation object (kept for reference; plt.show() blocks).
    """
    fig = plt.figure()
    im = plt.imshow(traj[0].state_snapshot())

    def animate(t):
        im.set_array(traj[t].state_snapshot())
        return im,

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(traj),
        interval=animation_speed_ms,
        blit=True,
        repeat=False,
    )
    plt.show()
    return anim


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------


def load_mdp_data(traj_dir, subdir='discrete', suffix='2'):
    """
    Load MDP arrays (states, actions, rewards, dones, ids) from disk.

    Args:
        traj_dir: Base directory for trajectories (e.g.
        "dataset/trajectories").
        subdir: Subdirectory name ('discrete' or 'continuous').
        suffix: Filename suffix (e.g., '2' for states2.npy, actions2.npy, etc.).

    Returns:
        Tuple of (mdp_states, mdp_actions, mdp_rewards, mdp_dones, ids).
    """
    base = os.path.join(traj_dir, subdir)
    mdp_states = np.load(os.path.join(base, f"states{suffix}.npy"))
    mdp_actions = np.load(
        os.path.join(base, f"actions{suffix}.npy"),
        allow_pickle=True,
    )
    mdp_rewards = np.load(os.path.join(base, f"rewards{suffix}.npy"))
    mdp_dones = np.load(os.path.join(base, f"dones{suffix}.npy"))
    ids = None
    # ids = np.load(os.path.join(base, f"ids{suffix}.npy")).reshape(-1, 2)
    return mdp_states, mdp_actions, mdp_rewards, mdp_dones, ids


# -----------------------------------------------------------------------------
# Robot Action Inference
# -----------------------------------------------------------------------------


def compute_robot_actions_from_positions(mdp_states, mdp_actions):
    """
    Infer discrete robot actions from state position transitions.

    Robot actions are derived from (dx, dy) between consecutive positions.
    'collect' actions are identified via _robot_picks (when robot picks a kit).

    Args:
        mdp_states: Full MDP state array (positions in columns -3:-1).
        mdp_actions: Full MDP action array (used to detect robot picks).

    Returns:
        1D array of string action labels: 'up', 'down', 'left', 'right', 'wait',
        'collect', or None for unknown transitions.
    """
    # Positions are stored in the last columns before robot status
    positions = mdp_states[:, -3:-1]
    prev_pos = positions[:-1]
    next_pos = positions[1:]
    deltas = next_pos - prev_pos

    # Map (dx, dy) to action labels
    actions_r = np.zeros(len(positions), dtype=object)
    for idx, dx in enumerate(deltas):
        key = tuple(dx)
        if key in POSITION_DELTA_TO_ACTION:
            actions_r[idx] = POSITION_DELTA_TO_ACTION[key]
        else:
            actions_r[idx] = None

    # Last timestep has no next position; assume 'wait'
    actions_r[-1] = 'wait'

    # Override indices where robot picks a medical kit
    picks = _robot_picks(mdp_states, mdp_actions)
    actions_r[picks] = 'collect'

    return actions_r


# -----------------------------------------------------------------------------
# Human Option Inference
# -----------------------------------------------------------------------------


def get_options(traj):
    """
    Infer which medical kit the human is targeting at each timestep.

    Options are inferred by tracing which kit status changed (delivery) and
    which kit is closest to the human's final position.

    Args:
        traj: TrajectoryWithRew (or similar) with obs and acts.

    Returns:
        1D array of option indices (0-5 for kit index, -1 for none/unclear).
    """
    last_status = traj.obs[-1][2:8]
    if last_status.sum() > 0:
        # Find closest collected kit to final human position
        last_pos2kits = traj.obs[-1][:2] - MEDICAL_KIT_IDX[last_status.astype(
            bool)]
        last_pos2kits = (last_pos2kits**2).sum(1)
        closest = last_pos2kits.argmin()
        closest = (MEDICAL_KIT_IDX[last_status.astype(bool)][closest] ==
                   MEDICAL_KIT_IDX).all(1)
        closest = np.where(closest)[0].item()
    else:
        closest = -1

    med_kits = traj.obs[:, 2:8]
    options = closest * np.ones(len(med_kits))

    # Trace backwards to assign option when kit status changes (delivery)
    prev_med_left = med_kits[-1]
    k = closest
    for idx, status in enumerate(reversed(med_kits)):
        if not np.all(status == prev_med_left):
            if traj.acts[-idx] == 7:  # 7 = 'collect' (human delivery)
                k = np.where(prev_med_left != status)[0].item()
            prev_med_left = status
        options[-idx - 1] = k

    if options[-1] == -1:
        options[-1] = options[-2]

    return options


def get_robot_options_for_trajectory(mdp_states, mdp_dones, traj_idx):
    """
    Extract robot option (target object) for each timestep in a trajectory.

    Robot options are stored in the last column of mdp_states.

    Args:
        mdp_states: Full MDP state array.
        mdp_dones: Full MDP done flags.
        traj_idx: Trajectory index (0-based).

    Returns:
        1D array of robot options for this trajectory.
    """
    idxs = np.where(mdp_dones == 1)[0]
    if traj_idx == 0:
        stidx = 0
    else:
        stidx = idxs[traj_idx - 1] + 1
    endidx = idxs[traj_idx]
    return mdp_states[stidx:endidx, -1]


# -----------------------------------------------------------------------------
# Trajectory Loading and Option Assembly
# -----------------------------------------------------------------------------


def load_trajectories_with_options(
    mdp_states,
    mdp_actions,
    mdp_rewards,
    mdp_dones,
    actions_r=None,
    n_trajectories=100,
):
    """
    Load trajectories and compute human/robot options for each.

    Args:
        mdp_states: Full MDP state array.
        mdp_actions: Full MDP action array.
        mdp_rewards: Full MDP reward array.
        mdp_dones: Full MDP done flags.
        actions_r: Optional precomputed robot actions; if None, not passed to
            get_trajectorywrewards.
        n_trajectories: Number of trajectories to load (default 100).

    Returns:
        Tuple of (trajectories, options_h, options_r, options).
        - trajectories: List of TrajectoryWithRew.
        - options_h: List of human option arrays per trajectory.
        - options_r: List of robot option arrays per trajectory.
        - options: List of (opth, optr) stacked arrays per trajectory.
    """
    trajectories = []
    options_h = []
    options_r = []
    options = []

    for traj_idx in range(n_trajectories):
        try:
            traj = get_trajectorywrewards(
                mdp_states,
                mdp_actions,
                mdp_rewards,
                mdp_dones,
                traj_idx=traj_idx,
                mdp_r_actions=actions_r,
            )
        except KeyError as e:
            print(
                "Bug in mdp_actions: action not interpretable (e.g. `toObj2`):",
                e,
            )
            continue

        trajectories.append(traj)
        opth = get_options(traj)
        options_h.append(opth)
        optr = get_robot_options_for_trajectory(mdp_states, mdp_dones, traj_idx)
        options_r.append(optr)
        options.append(np.stack((opth, optr), axis=1))

    return trajectories, options_h, options_r, options


# -----------------------------------------------------------------------------
# Performance Metrics
# -----------------------------------------------------------------------------


def process_trajectory(traj):
    """
    Compute team and robot delivery counts for a single trajectory.

    Args:
        traj: TrajectoryWithRew with obs and acts.

    Returns:
        Tuple of (team_delivery, robot_delivery).
        - team_delivery: Total kits delivered (human + robot).
        - robot_delivery: Kits delivered by the robot only.
    """
    team_delivery = sum(traj.obs[-1][2:8] == 0)
    human_delivery = sum(traj.acts == 7)  # 7 = 'collect'
    robot_delivery = team_delivery - human_delivery
    return team_delivery, robot_delivery


def compute_trajectory_metrics(trajectories):
    """
    Compute robot utility and team performance for all trajectories.

    Args:
        trajectories: List of TrajectoryWithRew.

    Returns:
        Tuple of (robot_utility, team_performance), each a list of per-trajectory values.
    """
    robot_utility = []
    team_performance = []
    for traj in trajectories:
        t, r = process_trajectory(traj)
        team_performance.append(t)
        robot_utility.append(r)
    return robot_utility, team_performance


# -----------------------------------------------------------------------------
# Oracle Training
# -----------------------------------------------------------------------------


def train_and_save_oracle(
    trajectories,
    options,
    n_train,
    save_path,
    expert_trajectories_test=None,
    true_options_test=None,
):
    """
    Train the Oracle (HBC) model and save to disk.

    Args:
        trajectories: List of expert trajectories.
        options: List of (opth, optr) option arrays per trajectory.
        n_train: Number of trajectories to use for training (rest for test).
        save_path: Path to save the trained Oracle.
        expert_trajectories_test: Optional test trajectories; if None, uses
            trajectories[n_train:].
        true_options_test: Optional test options; if None, uses options[n_train:].
    """
    train_traj = trajectories[:n_train]
    train_opts = options[:n_train]
    test_traj = expert_trajectories_test or trajectories[n_train:]
    test_opts = true_options_test or options[n_train:]

    gini = Oracle(
        expert_trajectories=train_traj,
        true_options=train_opts,
        expert_trajectories_test=test_traj,
        true_options_test=test_opts,
    )
    gini.save(save_path)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def print_participant_results(
    robot_utility,
    team_performance,
    n_participants=20,
    n_tasks_per_participant=5,
    pause_seconds=5,
):
    """
    Print robot utility per participant and task, with optional pause between
    participants.

    Args:
        robot_utility: List of robot delivery counts per trajectory.
        team_performance: List of team delivery counts per trajectory.
        n_participants: Number of participants (default 20).
        n_tasks_per_participant: Tasks per participant (default 5).
        pause_seconds: Seconds to pause after each participant block (default 5).
    """
    for participant in range(n_participants):
        for task in range(n_tasks_per_participant):
            idx = participant * n_tasks_per_participant + task
            if idx >= len(robot_utility):
                return
            print(idx)
            print(
                "Participant:",
                participant,
                "Task:",
                task,
                "Robot utility:",
                robot_utility[idx],
            )
        print("-------------------------")
        time.sleep(pause_seconds)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(
    traj_dir="dataset/trajectories",
    subdir="discrete",
    suffix="2",
    n_trajectories=100,
    n_train=90,
    oracle_save_path=None,
    n_participants=20,
    n_tasks=5,
    pause_seconds=5,
):
    """
    Load data, train Oracle, compute metrics, and print results.

    Args:
        traj_dir: Base directory for trajectory files.
        subdir: Subdirectory ('discrete' or 'continuous').
        suffix: Filename suffix for numpy files.
        n_trajectories: Number of trajectories to load.
        n_train: Number of trajectories for Oracle training.
        oracle_save_path: Path to save Oracle; default uses traj_dir/subdir.
        n_participants: Number of participants for reporting.
        n_tasks: Tasks per participant for reporting.
        pause_seconds: Pause between participant blocks when printing.
    """
    # Load MDP data
    mdp_states, mdp_actions, mdp_rewards, mdp_dones, ids = load_mdp_data(
        traj_dir, subdir=subdir, suffix=suffix)
    print(mdp_rewards[:10])
    print(np.unique(mdp_rewards))
    return

    # Infer robot actions from position transitions (optional; pass None to
    # skip)
    actions_r = compute_robot_actions_from_positions(mdp_states, mdp_actions)

    # Load trajectories and human/robot options
    trajectories, options_h, options_r, options = \
        load_trajectories_with_options(
            mdp_states,
            mdp_actions,
            mdp_rewards,
            mdp_dones,
            actions_r=
            None,  # Set to actions_r to include robot actions in trajectories
            n_trajectories=n_trajectories,
        )

    # Train and save Oracle
    save_path = oracle_save_path or os.path.join(traj_dir, subdir,
                                                 "gini_n18-1090")
    train_and_save_oracle(trajectories, options, n_train, save_path)

    # Compute and print performance metrics
    robot_utility, team_performance = compute_trajectory_metrics(trajectories)
    print_participant_results(
        robot_utility,
        team_performance,
        n_participants=n_participants,
        n_tasks_per_participant=n_tasks,
        pause_seconds=pause_seconds,
    )


if __name__ == "__main__":
    main(
        traj_dir="dataset/trajectories",
        subdir="discrete",
        suffix="2",
        n_trajectories=100,
        n_train=90,
        oracle_save_path=None,  # Uses dataset/trajectories/discrete/gini_n18-1090
        n_participants=20,
        n_tasks=5,
        pause_seconds=5,
    )
