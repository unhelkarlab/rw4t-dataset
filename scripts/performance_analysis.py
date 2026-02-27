import os

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from scripts.extract import (get_user_path, _get_sectasks_mask, DATA_FOLDER,
                             TRIAL_START, TRIAL_END)
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
    sectasks = _get_sectasks_mask(df, trial)

    return rewards, sectasks


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
    all_returns, all_rewards, all_sectasks = [], [], []

    for user in sorted(os.listdir(base)):
        user_path = os.path.join(base, user)
        if not os.path.isdir(user_path):
            continue

        if verbose:
            print("Processing", user, "...")

        user_returns, user_rewards, user_sectasks = [], [], []
        for trial in range(trial_start, trial_end):
            if verbose:
                print("  Trial", trial)

            rews, sectasks = get_trial_rewards(
                user,
                trial,
                data_folder=base,
                step_penalty=step_penalty,
                danger_penalty=danger_penalty,
            )
            # print('    Return:', np.sum(rews))
            user_returns.append(np.sum(rews))
            user_rewards.append(rews)
            user_sectasks.append(sectasks)

            ids.append((user, trial))
        all_returns.append(user_returns)
        all_rewards.append(user_rewards)
        all_sectasks.append(user_sectasks)

        if verbose:
            print("================================================")

    return np.array(all_returns), all_rewards, all_sectasks, ids


def compute_rew_by_secondary_task(all_rewards,
                                  all_sectasks,
                                  participant_indices=None,
                                  duration_with_sec=5.0,
                                  duration_without_sec=7.5):
    """
    For each participant, compute returns accumulated during secondary tasks
    and without secondary tasks, then return the mean across participants.

    Args:
        all_rewards: list[participant][trajectory] of reward arrays (per
                     timestep)
        all_sectasks: same shape, boolean arrays - True where secondary task
                      active
        participant_indices: optional 1D array of participant indices to
                            include; if None, use all.
        duration_with_sec: total minutes in "with secondary task" condition.
        duration_without_sec: total minutes in "without secondary task"
                             condition.

    Returns:
        Tuple of (mean_with_secondary, mean_without_secondary). Each is the mean
        across participants of that participant's action rate in that
        condition (per minute).
    """
    if participant_indices is not None:
        all_rewards = [all_rewards[i] for i in participant_indices]
        all_sectasks = [all_sectasks[i] for i in participant_indices]

    participants_with = []
    participants_without = []

    for p_rewards, p_mask in zip(all_rewards, all_sectasks):
        # Concatenate all trajectories for this participant
        rewards = np.concatenate(
            [np.atleast_1d(np.asarray(t).ravel()) for t in p_rewards])
        mask = np.concatenate(
            [np.atleast_1d(np.asarray(m).ravel()) for m in p_mask])

        # Count actions in each condition (vectorized)
        with_sec = mask
        reward_with = rewards[with_sec].sum()
        reward_without = rewards[~with_sec].sum()

        reward_with = reward_with / duration_with_sec
        reward_without = reward_without / duration_without_sec

        participants_with.append(reward_with)
        participants_without.append(reward_without)

    print('Average returns:')
    mean_with = np.nanmean(participants_with)
    std_with = np.std(participants_with)
    print(f'With secondary tasks: {mean_with:.2f} ± {std_with:.2f}')

    mean_without = np.nanmean(participants_without)
    std_without = np.std(participants_without)
    print(f'Without secondary tasks: {mean_without:.2f} ± {std_without:.2f}')

    res = wilcoxon(participants_with, participants_without)
    print('Wilcoxon test:', res.statistic, res.pvalue)

    # res_ttest = ttest_rel(participants_with, participants_without)
    # print('T-test:', res_ttest.statistic, res_ttest.pvalue)

    return mean_with, mean_without


def compute_rew_by_secondary_task_by_task(all_rewards,
                                          all_returns,
                                          all_sectasks,
                                          duration_with_sec=5.0,
                                          duration_without_sec=7.5):
    """
    Split reward by secondary-task presence for bottom- and top-performing
    participants (per task). Bottom/top 25% is computed separately per task.
    """
    participants_with_bottom_25 = []
    participants_without_bottom_25 = []

    participants_with_top_25 = []
    participants_without_top_25 = []

    # Per task: identify bottom 25% and top 25% by score on that task
    task_index_2_bottom_25_idx = {}
    task_index_2_top_25_idx = {}
    for task_idx in range(TRIAL_END - TRIAL_START):
        n_users = all_returns.shape[0]
        n_quarter = max(1, n_users // 4)
        task_scores = all_returns[:, task_idx]
        sorted_idx = np.argsort(task_scores)
        bottom_25_idx = sorted_idx[:n_quarter]
        top_25_idx = sorted_idx[-n_quarter:]
        task_index_2_bottom_25_idx[task_idx] = bottom_25_idx
        task_index_2_top_25_idx[task_idx] = top_25_idx

    # For each (participant, task): split reward into with- and without-sec
    # segments using sectasks mask; keep only bottom/top 25% for that task
    for idx, (p_rewards, p_mask) in enumerate(zip(all_rewards, all_sectasks)):
        for task_idx in range(TRIAL_END - TRIAL_START):

            task_rewards = p_rewards[task_idx]
            task_mask = p_mask[task_idx]

            reward_with = task_rewards[task_mask].sum()
            reward_without = task_rewards[~task_mask].sum()

            if idx in task_index_2_bottom_25_idx[task_idx]:
                participants_with_bottom_25.append(reward_with)
                participants_without_bottom_25.append(reward_without)
            elif idx in task_index_2_top_25_idx[task_idx]:
                participants_with_top_25.append(reward_with)
                participants_without_top_25.append(reward_without)

    # Mean reward per minute: total / (n_quarter * duration)
    mean_with_bottom_25 = np.sum(
        participants_with_bottom_25) / n_quarter / duration_with_sec
    mean_without_bottom_25 = np.sum(
        participants_without_bottom_25) / n_quarter / duration_without_sec

    mean_with_top_25 = np.sum(
        participants_with_top_25) / n_quarter / duration_with_sec
    mean_without_top_25 = np.sum(
        participants_without_top_25) / n_quarter / duration_without_sec

    print('Bottom 25%:')
    print(f'With secondary tasks: {mean_with_bottom_25:.2f}')
    print(f'Without secondary tasks: {mean_without_bottom_25:.2f}')
    print('Top 25%:')
    print(f'With secondary tasks: {mean_with_top_25:.2f}')
    print(f'Without secondary tasks: {mean_without_top_25:.2f}')


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
    all_returns, all_rewards, all_sectasks, ids = get_rews(
        data_folder=data_folder,
        trial_start=trial_start,
        trial_end=trial_end,
        step_penalty=step_penalty,
        danger_penalty=danger_penalty,
        verbose=verbose,
    )

    # mean ± std over all users for each task
    all_returns_avg = np.mean(all_returns, axis=0)
    all_returns_stds = np.std(all_returns, axis=0)

    # Rank users by mean return across trials; take top/bottom 25%
    user_means = np.mean(all_returns, axis=1)  # (num_users,)
    print('user_means:', user_means)
    n_users = len(user_means)
    n_quarter = max(1, n_users // 4)
    sorted_idx = np.argsort(user_means)  # ascending: [0] = worst
    print('sorted_idx:', sorted_idx)
    bottom_25_idx = sorted_idx[:n_quarter]
    top_25_idx = sorted_idx[-n_quarter:]

    bottom_25_avg = np.mean(all_returns[bottom_25_idx], axis=0)
    bottom_25_stds = np.std(all_returns[bottom_25_idx], axis=0)
    top_25_avg = np.mean(all_returns[top_25_idx], axis=0)
    top_25_stds = np.std(all_returns[top_25_idx], axis=0)

    n_half = max(1, n_users // 2)
    bottom_50_idx = sorted_idx[:n_half]
    top_50_idx = sorted_idx[-n_half:]
    res_learning = wilcoxon(all_returns[:, 0], all_returns[:, 4])
    print('Learning effect (Wilcoxon test):', res_learning.statistic,
          res_learning.pvalue)
    res_sectask = wilcoxon(all_returns[:, 3], all_returns[:, 4])
    print('Secondary task effect (Wilcoxon test):', res_sectask.statistic,
          res_sectask.pvalue)
    res_sectask = wilcoxon(all_returns[bottom_50_idx, 3],
                           all_returns[bottom_50_idx, 4])
    print('Secondary task effect bottom 50 (Wilcoxon test):',
          res_sectask.statistic, res_sectask.pvalue)
    res_sectask = wilcoxon(all_returns[top_50_idx, 3], all_returns[top_50_idx,
                                                                   4])
    print('Secondary task effect top 50 (Wilcoxon test):',
          res_sectask.statistic, res_sectask.pvalue)

    # Per-trial top/bottom 25%: for each trial, take the lowest/highest 25% of
    # scores and average
    sorted_returns = np.sort(all_returns, axis=0)
    n_quarter_trial = max(1, all_returns.shape[0] // 4)
    per_trial_bottom_25_avg = np.mean(sorted_returns[:n_quarter_trial], axis=0)
    per_trial_bottom_25_stds = np.std(sorted_returns[:n_quarter_trial], axis=0)
    per_trial_top_25_avg = np.mean(sorted_returns[-n_quarter_trial:], axis=0)
    per_trial_top_25_stds = np.std(sorted_returns[-n_quarter_trial:], axis=0)

    print("All users (mean ± std):")
    for m, s in zip(all_returns_avg, all_returns_stds):
        print(f"  {m:.2f} ± {s:.2f}")

    print("Bottom 25% users (mean ± std):")
    for m, s in zip(bottom_25_avg, bottom_25_stds):
        print(f"  {m:.2f} ± {s:.2f}")

    print("Top 25% users (mean ± std):")
    for m, s in zip(top_25_avg, top_25_stds):
        print(f"  {m:.2f} ± {s:.2f}")

    print("Per-trial bottom 25% scores (mean ± std):")
    for m, s in zip(per_trial_bottom_25_avg, per_trial_bottom_25_stds):
        print(f"  {m:.2f} ± {s:.2f}")

    print("Per-trial top 25% scores (mean ± std):")
    for m, s in zip(per_trial_top_25_avg, per_trial_top_25_stds):
        print(f"  {m:.2f} ± {s:.2f}")

    # print('--------------------------------')
    # print('All users:')
    # compute_rew_by_secondary_task(all_rewards, all_sectasks)
    # print('Bottom 25%:')
    # compute_rew_by_secondary_task(all_rewards,
    #                               all_sectasks,
    #                               participant_indices=bottom_25_idx)
    # print('Top 25%:')
    # compute_rew_by_secondary_task(all_rewards,
    #                               all_sectasks,
    #                               participant_indices=top_25_idx)

    # print('By task:')
    # compute_rew_by_secondary_task_by_task(all_rewards, all_returns,
    #                                       all_sectasks)

    return all_returns, ids


if __name__ == "__main__":
    all_returns, _ids = main(
        data_folder=DATA_FOLDER,
        trial_start=TRIAL_START,
        trial_end=TRIAL_END,
        step_penalty=0.0,
        danger_penalty=-0.1666666,
        verbose=True,
    )

    # np.save('all_returns.npy', all_returns)
