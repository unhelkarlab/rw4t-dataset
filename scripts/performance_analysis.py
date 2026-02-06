import os

import numpy as np
import pandas as pd

from scripts.extract import get_user_path, DATA_FOLDER, TRIAL_START, TRIAL_END
from scripts.process import get_rewards


def get_trial_rewards(
    user,
    trial,
    data_folder=None,
    step_penalty=0.0,
    danger_penalty=-0.1666666,
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
    rewards = get_rewards(df, step_penalty, danger_penalty)

    return rewards


def get_rews(
    data_folder=None,
    trial_start=TRIAL_START,
    trial_end=TRIAL_END,
    step_penalty=0.0,
    danger_penalty=-0.1666666,
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
    ids = []
    all_returns = []

    for user in sorted(os.listdir(base)):
        user_path = os.path.join(base, user)
        if not os.path.isdir(user_path):
            continue

        if verbose:
            print("Processing", user, "...")

        user_returns = []
        for trial in range(trial_start, trial_end):
            if verbose:
                print("  Trial", trial)

            rews = get_trial_rewards(
                user,
                trial,
                data_folder=base,
                step_penalty=step_penalty,
                danger_penalty=danger_penalty,
            )
            print('    Return:', np.sum(rews))
            user_returns.append(np.sum(rews))

            ids.append((user, trial))
        all_returns.append(user_returns)

        if verbose:
            print("================================================")

    return np.array(all_returns), ids


def main(
    data_folder=None,
    trial_start=TRIAL_START,
    trial_end=TRIAL_END,
    step_penalty=0.0,
    danger_penalty=-0.1666666,
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
    all_returns, ids = get_rews(
        data_folder=data_folder,
        trial_start=trial_start,
        trial_end=trial_end,
        step_penalty=step_penalty,
        danger_penalty=danger_penalty,
        verbose=verbose,
    )
    all_returns_avg = np.mean(all_returns, axis=0)
    all_returns_stds = np.std(all_returns, axis=0)
    for m, s in zip(all_returns_avg, all_returns_stds):
        print(f"{m:.3f} ± {s:.3f}")

    return all_returns, ids


if __name__ == "__main__":
    main(
        data_folder=DATA_FOLDER,
        trial_start=TRIAL_START,
        trial_end=TRIAL_END,
        step_penalty=0.0,
        danger_penalty=-0.1666666,
        verbose=True,
    )
