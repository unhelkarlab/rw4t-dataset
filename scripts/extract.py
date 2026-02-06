"""
Extract MDP trajectory data from raw CSV files for the RW4T dataset.

This script loads raw trial data from per-user CSV files, processes states,
actions, and rewards through the cleaning pipeline, and optionally saves
the concatenated MDP arrays to disk for use by downstream scripts.
"""
import os

import numpy as np
import pandas as pd

from scripts.clean import clean_trajectory
from scripts.process import get_state, get_actions, get_rewards

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DATA_FOLDER = "dataset/raw"
TRAJ_DIR = "dataset/trajectories"

# Trial range to process (3-7 inclusive = 5 trials per user)
TRIAL_START = 3
TRIAL_END = 8  # exclusive, so range(3, 8) = 3,4,5,6,7

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------


def get_user_path(user, trial, data_folder=None):
    """
    Build the path to a user's trial CSV file.

    Args:
        user: User identifier (directory name under data_folder).
        trial: Trial number.
        data_folder: Base folder for raw data; uses DATA_FOLDER if None.

    Returns:
        Path string (e.g., "dataset/raw/user1/user1-4.csv").
    """
    base = data_folder or DATA_FOLDER
    filename = f"{user}-{trial}.csv"
    return os.path.join(base, user, filename)


# -----------------------------------------------------------------------------
# Trial Processing
# -----------------------------------------------------------------------------


def _get_sectasks_mask(df, trial):
    """
    Build a boolean mask for secondary task segments based on trial and timer.

    Trials 4 and 5 split the task by timer; trial 6 marks the entire run.

    Args:
        df: DataFrame with 'TimerText' column.
        trial: Trial number (4, 5, or 6 have special handling).

    Returns:
        1D boolean array of length len(df).
    """
    sectasks = np.zeros(len(df), dtype=bool)
    s = df["TimerText"]
    s_times = pd.to_timedelta("00:" + s, errors="coerce")
    threshold = s_times.max() / 2

    if trial == 4:
        sectasks[s_times <= threshold] = True
    elif trial == 5:
        comparison = s_times > threshold
        comparison = comparison.where(s != "Timer", True)
        sectasks[comparison] = True
    elif trial == 6:
        sectasks[:] = True

    return sectasks


def get_trial_behavior(user,
                       trial,
                       num_bins=None,
                       data_folder=None,
                       step_penalty=0.0):
    """
    Load and process a single trial into MDP format.

    Reads the CSV, extracts states/actions/rewards, builds sectasks mask,
    and runs the cleaning pipeline.

    Args:
        user: User identifier.
        trial: Trial number (typically 3-7).
        num_bins: If set, discretize positions into this many bins; if None,
            use continuous positions.
        data_folder: Base folder for raw data; uses DATA_FOLDER if None.
        step_penalty: penalty for each environment step.

    Returns:
        Tuple of (states, actions, rewards, dones, sectasks).
    """
    filepath = get_user_path(user, trial, data_folder)
    df = pd.read_csv(filepath)

    states = get_state(df, num_bins=num_bins)
    actions = get_actions(df, num_bins=num_bins)
    rewards = get_rewards(df, step_penalty)
    sectasks = _get_sectasks_mask(df, trial)

    states, actions, rewards, sectasks = clean_trajectory(
        states, actions, rewards, sectasks, step_penalty=step_penalty)

    dones = np.zeros_like(actions, dtype=int)
    dones[-1] = 1

    return states, actions, rewards, dones, sectasks


# -----------------------------------------------------------------------------
# Batch Extraction
# -----------------------------------------------------------------------------


def get_trajs(
    num_bins=None,
    data_folder=None,
    traj_dir=None,
    trial_start=TRIAL_START,
    trial_end=TRIAL_END,
    step_penalty=0.0,
    verbose=True,
):
    """
    Extract trajectories for all users and trials in the raw data folder.

    Iterates over each user subdirectory and trials in [trial_start, trial_end),
    processing each CSV and appending to MDP arrays.

    Args:
        num_bins: If set, discretize positions; if None, continuous.
        data_folder: Base folder for raw CSV data; uses DATA_FOLDER if None.
        traj_dir: Unused; kept for API compatibility.
        trial_start: First trial index (inclusive).
        trial_end: Last trial index (exclusive).
        step_penalty: penalty for each environment step.
        verbose: If True, print progress.

    Returns:
        Tuple of (mdp_states, mdp_actions, mdp_rewards, mdp_dones, ids,
                  sec_task_present). Each MDP array is a list of per-trial
                  arrays; ids is list of (user, trial) tuples.
    """
    base = data_folder or DATA_FOLDER
    mdp_states, mdp_actions, mdp_rewards, mdp_dones = [], [], [], []
    ids = []
    sec_task_present = []

    for user in sorted(os.listdir(base)):
        user_path = os.path.join(base, user)
        if not os.path.isdir(user_path):
            continue

        if verbose:
            print("Processing", user, "...")

        for trial in range(trial_start, trial_end):
            if verbose:
                print("  Trial", trial)

            st, acts, rews, dones, sectaskbool = get_trial_behavior(
                user,
                trial,
                num_bins=num_bins,
                data_folder=base,
                step_penalty=step_penalty,
            )

            mdp_states.append(st)
            mdp_actions.append(acts)
            mdp_rewards.append(rews)
            mdp_dones.append(dones)
            sec_task_present.append(sectaskbool)
            ids.append((user, trial))

        if verbose:
            print("================================================")
        # print(mdp_states[0][0])
        # print(mdp_actions[0][0])
        # print(mdp_rewards[0][0])
        # print(mdp_dones[0][0])
        # print(ids[0])
        # print(sec_task_present[0])

    return (
        mdp_states,
        mdp_actions,
        mdp_rewards,
        mdp_dones,
        ids,
        sec_task_present,
    )


# -----------------------------------------------------------------------------
# Concatenation (for downstream that expects single arrays)
# -----------------------------------------------------------------------------


def concatenate_trajectories(mdp_states, mdp_actions, mdp_rewards, mdp_dones):
    """
    Concatenate per-trial MDP arrays into single arrays.

    Downstream scripts (e.g., runnable_code, data_types) expect concatenated
    arrays where trajectory boundaries are marked by mdp_dones == 1.

    Args:
        mdp_states: List of state arrays (each shape T x D).
        mdp_actions: List of action arrays (each length T).
        mdp_rewards: List of reward arrays (each length T).
        mdp_dones: List of done arrays (each length T, with 1 at trajectory
                   end).

    Returns:
        Tuple of (states, actions, rewards, dones) as concatenated arrays.
    """
    states = np.concatenate(mdp_states, axis=0)
    actions = np.concatenate(mdp_actions, axis=0)
    rewards = np.concatenate(mdp_rewards, axis=0)
    dones = np.concatenate(mdp_dones, axis=0)
    return states, actions, rewards, dones


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------


def save_trajectories(
    mdp_states,
    mdp_actions,
    mdp_rewards,
    mdp_dones,
    ids,
    sectasks=None,
    subdir="discrete",
    suffix="2",
    traj_dir=None,
    concatenate=True,
):
    """
    Save extracted MDP arrays to disk as .npy files.

    Args:
        mdp_states: List of state arrays per trajectory.
        mdp_actions: List of action arrays per trajectory.
        mdp_rewards: List of reward arrays per trajectory.
        mdp_dones: List of done arrays per trajectory.
        ids: List of (user, trial) identifier tuples.
        sectasks: Optional list of sectask boolean arrays per trajectory.
            When concatenate=True, concatenated before saving (one bool per
            timestep). When concatenate=False, saved as list.
        subdir: Subdirectory under traj_dir ('discrete' or 'continuous').
        suffix: Filename suffix (e.g., '2' -> states2.npy).
        traj_dir: Base output directory; uses TRAJ_DIR if None.
        concatenate: If True (default), concatenate MDP arrays before saving so
            downstream (runnable_code, data_types) can index by timestep.
    """
    base = os.path.join(traj_dir or TRAJ_DIR, subdir)
    os.makedirs(base, exist_ok=True)

    if concatenate:
        states, actions, rewards, dones = concatenate_trajectories(
            mdp_states, mdp_actions, mdp_rewards, mdp_dones)
    else:
        states, actions, rewards, dones = (
            mdp_states,
            mdp_actions,
            mdp_rewards,
            mdp_dones,
        )

    np.save(os.path.join(base, f"states{suffix}"), states, allow_pickle=False)
    np.save(
        os.path.join(base, f"actions{suffix}"),
        actions,
        allow_pickle=True,
    )
    np.save(os.path.join(base, f"rewards{suffix}"), rewards, allow_pickle=False)
    np.save(os.path.join(base, f"dones{suffix}"), dones, allow_pickle=False)
    np.save(os.path.join(base, f"ids{suffix}"),
            np.array(ids),
            allow_pickle=False)

    if sectasks is not None:
        if concatenate:
            sectasks_to_save = np.concatenate(sectasks, axis=0)
        else:
            sectasks_to_save = sectasks
        np.save(
            os.path.join(base, f"sectasks{suffix}"),
            sectasks_to_save,
            allow_pickle=False,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(
    data_folder=None,
    traj_dir=None,
    num_bins=10,
    subdir="discrete",
    suffix="2",
    trial_start=TRIAL_START,
    trial_end=TRIAL_END,
    step_penalty=0.0,
    save=True,
    verbose=True,
):
    """
    Extract trajectories from raw data and optionally save to disk.

    Args:
        data_folder: Base folder for raw CSV data; uses DATA_FOLDER if None.
        traj_dir: Output directory for trajectories; uses TRAJ_DIR if None.
        num_bins: Discretization bins for positions (None = continuous).
        subdir: Subdirectory for output ('discrete' or 'continuous').
        suffix: Filename suffix for saved .npy files.
        trial_start: First trial index (inclusive).
        trial_end: Last trial index (exclusive).
        step_penalty: penalty for each environment step.
        save: If True, save arrays to disk.
        verbose: If True, print progress.
    """
    mdp_states, mdp_actions, mdp_rewards, mdp_dones, ids, sectasks = get_trajs(
        num_bins=num_bins,
        data_folder=data_folder,
        traj_dir=traj_dir,
        trial_start=trial_start,
        trial_end=trial_end,
        step_penalty=step_penalty,
        verbose=verbose,
    )

    if save:
        save_trajectories(
            mdp_states,
            mdp_actions,
            mdp_rewards,
            mdp_dones,
            ids,
            sectasks=sectasks,
            subdir=subdir,
            suffix=suffix,
            traj_dir=traj_dir,
        )
        if verbose:
            out_base = os.path.join(traj_dir or TRAJ_DIR, subdir)
            print(f"Saved trajectories to {out_base}")

    return mdp_states, mdp_actions, mdp_rewards, mdp_dones, ids, sectasks


if __name__ == "__main__":
    main(
        data_folder=DATA_FOLDER,
        traj_dir=TRAJ_DIR,
        num_bins=10,  # Discrete processing
        subdir="discrete",
        suffix="2",
        trial_start=TRIAL_START,
        trial_end=TRIAL_END,
        step_penalty=0.0,
        save=False,
        verbose=True,
    )
