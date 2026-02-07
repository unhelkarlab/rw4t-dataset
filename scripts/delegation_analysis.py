import os

import pandas as pd
import numpy as np

from scripts.clean import _robot_picks
from scripts.extract import (get_user_path, _get_sectasks_mask, DATA_FOLDER,
                             TRIAL_START, TRIAL_END)
from scripts.process import get_state, get_actions

NUM_OBJECTS = 6  # number of objects (medical kits) in the rw4t environment


def get_trial_data(
    user,
    trial,
    data_folder=None,
    num_bins=None,
):
    """
    Load per-step rewards for a single trial.

    Args:
        user: User identifier.
        trial: Trial number.
        data_folder: Base folder for raw CSV data; uses DATA_FOLDER if None.
        step_penalty: Penalty per environment step.
        danger_penalty: Penalty for entering danger zones.

    Returns:
        Array of per-step rewards.
    """
    filepath = get_user_path(user, trial, data_folder)
    df = pd.read_csv(filepath)

    states = get_state(df, num_bins=num_bins)
    actions = get_actions(df, num_bins)
    sectasks = _get_sectasks_mask(df, trial)

    return states, actions, sectasks


def get_all_acts(
    data_folder=None,
    trial_start=TRIAL_START,
    trial_end=TRIAL_END,
    num_bins=None,
    verbose=True,
):
    """
    Compute total returns for all users and trials.

    Args:
        data_folder: Base folder for raw CSV data; uses DATA_FOLDER if None.
        trial_start: First trial index (inclusive).
        trial_end: Last trial index (exclusive).
        step_penalty: Penalty per environment step.
        danger_penalty: Penalty for entering danger zones.
        verbose: If True, print progress.

    Returns:
        Tuple of (returns_array, ids). returns_array is shape (num_users,
        num_trials); ids is a list of (user, trial) tuples.
    """
    base = data_folder or DATA_FOLDER
    all_states, all_actions, all_sectasks = [], [], []
    ids = []

    for user in sorted(os.listdir(base)):
        user_path = os.path.join(base, user)
        if not os.path.isdir(user_path):
            continue

        if verbose:
            print("Processing", user, "...")

        user_states, user_actions, user_sectasks = [], [], []
        for trial in range(trial_start, trial_end):
            if verbose:
                print("  Trial", trial)

            states, acts, sectasks = get_trial_data(
                user,
                trial,
                data_folder=base,
                num_bins=num_bins,
            )
            user_states.append(states)
            user_actions.append(acts)
            user_sectasks.append(sectasks)
            ids.append((user, trial))
            # print("    Unique acts:", np.unique(np.array(acts)))

        all_states.append(user_states)
        all_actions.append(user_actions)
        all_sectasks.append(user_sectasks)
        # break
        if verbose:
            print("================================================")

    return all_states, all_actions, all_sectasks, ids


def count_delegations_in_trajectory(actions):
    """
    Count actions that begin with 'toObj' in a single trajectory.
    """
    return sum(1 for a in actions
               if isinstance(a, str) and a.startswith('toObj'))


def compute_total_delegations_per_task(all_actions):
    # Step 1: Delegations per trajectory per participant
    delegations_per_trajectory = [[
        count_delegations_in_trajectory(traj) for traj in participant_actions
    ] for participant_actions in all_actions]

    # Step 2: Stack so each row = one participant, each col = one task
    arr = np.array(
        delegations_per_trajectory)  # shape: (n_participants, n_tasks)
    avg_dele_per_task = np.mean(arr, axis=0)  # mean over participants
    std_dele_per_task = np.std(arr, axis=0)  # std over participants

    # print('avg_dele_per_task:', avg_dele_per_task)
    # print('std_dele_per_task:', std_dele_per_task)
    print('Total delegations per task:')
    for task_idx, (m, s) in enumerate(zip(avg_dele_per_task,
                                          std_dele_per_task)):
        print(f"Task {task_idx}: {m:.2f} ± {s:.2f}")


def compute_delegation_rates_by_secondary_task(all_actions, all_sectasks):
    """
    For each participant, compute average delegation rate with and without
    secondary tasks. Then return the mean across participants.

    Args:
        all_actions: list[participant][trajectory] of action arrays (per
                     timestep)
        all_sectasks: same shape, boolean arrays - True where secondary task
                      active

    Returns:
        Tuple of (mean_with_secondary, mean_without_secondary). Each is the mean
        across participants of that participant's delegation rate in that
        condition.
    """
    participants_with = []
    participants_without = []

    for p_actions, p_mask in zip(all_actions, all_sectasks):
        # Concatenate all trajectories for this participant
        acts = np.concatenate(
            [np.atleast_1d(np.asarray(t).ravel()) for t in p_actions])
        mask = np.concatenate(
            [np.atleast_1d(np.asarray(m).ravel()) for m in p_mask])

        # Delegation mask: actions starting with 'toObj'
        is_del = np.char.startswith(acts.astype(str), 'toObj')

        # Count delegations and timesteps in each condition (vectorized)
        with_sec = mask
        n_del_with = np.sum(is_del & with_sec)
        n_del_without = np.sum(is_del & ~with_sec)

        rate_with = n_del_with / 7.5  # 7.5 min with sec tasks
        rate_without = n_del_without / 5.0  # 5.0 min with sec tasks

        participants_with.append(rate_with)
        participants_without.append(rate_without)

    print('Average delegation rates:')
    mean_with = np.nanmean(participants_with)
    std_with = np.std(participants_with)
    print(f'With secondary tasks: {mean_with:.2f} ± {std_with:.2f}')

    mean_without = np.nanmean(participants_without)
    std_without = np.std(participants_without)
    print(f'Without secondary tasks: {mean_without:.2f} ± {std_without:.2f}')

    return mean_with, mean_without


def compute_robot_picks_per_task(all_states, all_actions):
    """
    Compute the number of robot picks per task per participant, then average
    over participants for each task.

    Args:
        all_states: list[participant][task] of state arrays (n_timesteps,
                    n_features).
        all_actions: same shape, action arrays (n_timesteps,).

    Returns:
        Tuple of (mean_per_task, std_per_task), each shape (n_tasks,).
        mean_per_task: mean number of robot picks per task per participant.
        std_per_task: std number of robot picks per task per participant.
    """
    robot_picks_per_trajectory = [[
        len(_robot_picks(np.asarray(states), np.asarray(actions)))
        for states, actions in zip(p_states, p_actions)
    ] for p_states, p_actions in zip(all_states, all_actions)]

    arr = np.array(robot_picks_per_trajectory)  # (n_participants, n_tasks)
    mean_per_task = np.mean(arr, axis=0)
    std_per_task = np.std(arr, axis=0)

    print("Robot picks per task:")
    for task_idx, (m, s) in enumerate(zip(mean_per_task, std_per_task)):
        print(f"Task {task_idx}: {m:.2f} ± {s:.2f}")

    return mean_per_task, std_per_task


def count_delegations_per_object_in_trajectory(actions):
    """
    Count how many times the robot is delegated to each object
    (0..NUM_OBJECTS-1) in a single trajectory. Delegation actions are of the
    form 'toObj0', 'toObj1', etc.

    Returns:
        np.ndarray of shape (NUM_OBJECTS,) with counts per object.
    """
    arr = np.atleast_1d(np.asarray(actions).ravel()).astype(str)
    mask = np.char.startswith(arr, 'toObj')
    delegated = arr[mask]
    if delegated.size == 0:
        return np.zeros(NUM_OBJECTS, dtype=np.intp)

    obj_indices = np.array(
        [int(s[5:]) for s in delegated if len(s) > 5 and s[5:].isdigit()],
        dtype=np.intp,
    )
    valid = (obj_indices >= 0) & (obj_indices < NUM_OBJECTS)
    return np.bincount(obj_indices[valid], minlength=NUM_OBJECTS)


def delegations_to_objects_per_task(all_actions, task_idx):
    """
    For a given task index, compute per participant the number of times they
    delegate the robot to each object, then the mean (and std) across
    participants or each object.

    Args:
        all_actions: list[participant][task] of action arrays (as returned by
                     get_all_acts).
        task_idx: int, index of the task (trial) to analyze.

    Returns:
        Tuple of (mean_per_object, std_per_object), each shape (NUM_OBJECTS,).
        mean_per_object[i] is the average across participants of how many times
        they delegated to object i in that task.
    """
    # Per-participant counts for this task: (n_participants, NUM_OBJECTS)
    counts_per_participant = np.array([
        count_delegations_per_object_in_trajectory(
            np.atleast_1d(np.asarray(participant_actions[task_idx]).ravel()))
        for participant_actions in all_actions
    ])

    mean_per_object = np.mean(counts_per_participant, axis=0)
    std_per_object = np.std(counts_per_participant, axis=0)
    print(f"Delegation data for task {task_idx}:")
    for obj_idx, (m, s) in enumerate(zip(mean_per_object, std_per_object)):
        print(f"Object {obj_idx}: {m:.2f} ± {s:.2f}")

    return mean_per_object, std_per_object


def main(
    data_folder=None,
    trial_start=TRIAL_START,
    trial_end=TRIAL_END,
    num_bins=None,
    verbose=True,
):
    """
    Compute returns and print per-trial mean ± std statistics.

    Args:
        data_folder: Base folder for raw CSV data; uses DATA_FOLDER if None.
        trial_start: First trial index (inclusive).
        trial_end: Last trial index (exclusive).
        step_penalty: Penalty per environment step.
        danger_penalty: Penalty for entering danger zones.
        verbose: If True, print progress.
    """
    states, actions, sectasks, ids = get_all_acts(
        data_folder=data_folder,
        trial_start=trial_start,
        trial_end=trial_end,
        num_bins=num_bins,
        verbose=verbose,
    )
    # compute_total_delegations_per_task(actions)
    # print('--------------------------------')
    # compute_delegation_rates_by_secondary_task(actions, sectasks)
    # print('--------------------------------')
    # compute_robot_picks_per_task(states, actions)
    # print('--------------------------------')
    for task_idx in range(TRIAL_END - TRIAL_START):
        delegations_to_objects_per_task(actions, task_idx)
        print('--------------------------------')

    return states, actions, sectasks, ids


if __name__ == "__main__":
    main(
        data_folder=DATA_FOLDER,
        trial_start=TRIAL_START,
        trial_end=TRIAL_END,
        num_bins=10,
        verbose=True,
    )
