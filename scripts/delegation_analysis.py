import os

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

from scripts.clean import _robot_picks
from scripts.extract import (get_user_path, _get_sectasks_mask, DATA_FOLDER,
                             TRIAL_START, TRIAL_END)
from scripts.performance_analysis import get_rews
from scripts.process import get_state, get_actions

NUM_OBJECTS = 6  # number of objects (medical kits) in the rw4t environment

ROBOT_PICKS = [[
    0.0,
    0.98,
    0.0,
    0.0,
    1.0,
    1.0,
], [
    0.0,
    0.99,
    0.0,
    0.0,
    0.98,
    1.0,
], [
    0.0,
    0.98,
    0.0,
    0.0,
    0.98,
    1.0,
], [
    0.0,
    0.98,
    0.0,
    0.0,
    1.0,
    1.0,
], [
    0.0,
    0.98,
    0.0,
    0.0,
    1.0,
    1.0,
]]


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

    s = df["TimerText"]
    timer_times = pd.to_timedelta("00:" + s, errors="coerce")

    return states, actions, sectasks, timer_times


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
    all_states, all_actions, all_sectasks, all_timer_times = [], [], [], []
    ids = []

    for user in sorted(os.listdir(base)):
        user_path = os.path.join(base, user)
        if not os.path.isdir(user_path):
            continue

        print("Processing", user, "...")

        (user_states, user_actions, user_sectasks,
         user_timer_times) = [], [], [], []
        for trial in range(trial_start, trial_end):
            if verbose:
                print("  Trial", trial)

            states, acts, sectasks, timer_times = get_trial_data(
                user,
                trial,
                data_folder=base,
                num_bins=num_bins,
            )
            # print('states[0]:', states[0])
            user_states.append(states)
            user_actions.append(acts)
            user_sectasks.append(sectasks)
            user_timer_times.append(timer_times)
            ids.append((user, trial))
            # print("    Unique acts:", np.unique(np.array(acts)))
            # break

        all_states.append(user_states)
        all_actions.append(user_actions)
        all_sectasks.append(user_sectasks)
        all_timer_times.append(user_timer_times)
        # break
        if verbose:
            print("================================================")

    return all_states, all_actions, all_sectasks, all_timer_times, ids


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


def _first_half_mask(timer_times):
    """
    Boolean mask True where timer_times > timer_times.max() / 2 (handles NaT).
    """
    t = pd.Series(timer_times)
    threshold = t.max() / 2
    return (t > threshold).values


def filter_trajectories_by_timer_half(all_actions, all_sectasks,
                                      all_timer_times):
    """
    Return (actions, sectasks) keeping only steps where
    timer_times > timer_times.max() / 2 per trajectory.

    Use this to get actions and sectasks for the first half of each episode
    (by actual timer) for feeding into compute_action_rates_by_secondary_task
    or other analyses. All three inputs must be list[participant][trajectory]
    with matching lengths per trajectory.

    Args:
        all_actions: list[participant][trajectory] of action arrays.
        all_sectasks: same shape, boolean arrays.
        all_timer_times: same shape, timedelta-like arrays (e.g. from
                        get_trial_data / get_all_acts).

    Returns:
        (filtered_actions, filtered_sectasks): same structure, each trajectory
        sliced to indices where timer_times > max(timer_times)/2.
    """
    filtered_actions = []
    filtered_sectasks = []
    for p_actions, p_sectasks, p_timers in zip(all_actions, all_sectasks,
                                               all_timer_times):
        pa_list = []
        ps_list = []
        for acts, sec, timers in zip(p_actions, p_sectasks, p_timers):
            first_half = _first_half_mask(timers)
            assert len(acts) == len(sec) == len(first_half)
            pa_list.append(acts[first_half])
            ps_list.append(sec[first_half])
        filtered_actions.append(pa_list)
        filtered_sectasks.append(ps_list)
    return filtered_actions, filtered_sectasks


def compute_action_rates_by_secondary_task(all_actions,
                                           all_sectasks,
                                           timer_times,
                                           action_type='delegation',
                                           participant_indices=None,
                                           use_first_half_only=False,
                                           duration_with_sec=5.0,
                                           duration_without_sec=7.5):
    """
    For each participant, compute average action rate (delegations or collects)
    with and without secondary tasks, then return the mean across participants.

    Action type can be 'delegation' (sending robot to object, 'toObj*') or
    'collect' (human picking up a kit). Rates are normalized by trial duration.
    When use_first_half_only is True, only steps where
    timer_times > timer_times.max()/2 are used (first half of each episode).

    Args:
        all_actions: list[participant][trajectory] of action arrays (per
                     timestep)
        all_sectasks: same shape, boolean arrays - True where secondary task
                      active
        timer_times: same shape, timedelta arrays per timestep (from
                     get_trial_data / get_all_acts). Used only if
                     use_first_half_only is True.
        action_type: 'delegation' to rate delegations (toObj*), 'collect' to
                     rate human collects.
        participant_indices: optional 1D array of participant indices to
                            include; if None, use all.
        use_first_half_only: if True, restrict to steps where
                             timer_times > timer_times.max()/2 per trajectory.
        duration_with_sec: total minutes in "with secondary task" condition
                           (use half, e.g. 2.5, when use_first_half_only).
        duration_without_sec: total minutes in "without secondary task"
                             condition (use half, e.g. 3.75, when
                             use_first_half_only).

    Returns:
        Tuple of (mean_with_secondary, mean_without_secondary). Each is the mean
        across participants of that participant's action rate in that
        condition (per minute).
    """
    if participant_indices is not None:
        all_actions = [all_actions[i] for i in participant_indices]
        all_sectasks = [all_sectasks[i] for i in participant_indices]
        timer_times = [timer_times[i] for i in participant_indices]

    if use_first_half_only:
        all_actions, all_sectasks = filter_trajectories_by_timer_half(
            all_actions, all_sectasks, timer_times)
        duration_with_sec = duration_with_sec / 2
        duration_without_sec = duration_without_sec / 2

    participants_with = []
    participants_without = []

    for p_actions, p_mask in zip(all_actions, all_sectasks):
        # Concatenate all trajectories for this participant
        acts = np.concatenate(
            [np.atleast_1d(np.asarray(t).ravel()) for t in p_actions])
        mask = np.concatenate(
            [np.atleast_1d(np.asarray(m).ravel()) for m in p_mask])

        # Action mask: 'toObj*' for delegation, 'collect' for human collect
        if action_type == 'delegation':
            action_str = 'toObj'
        elif action_type == 'collect':
            action_str = 'collect'
        else:
            raise ValueError(f'Invalid action type: {action_type}')
        is_del = np.char.startswith(acts.astype(str), action_str)

        # Count actions in each condition (vectorized)
        with_sec = mask
        n_del_with = np.sum(is_del & with_sec)
        n_del_without = np.sum(is_del & ~with_sec)

        rate_with = n_del_with / duration_with_sec
        rate_without = n_del_without / duration_without_sec

        participants_with.append(rate_with)
        participants_without.append(rate_without)

    print(f'Average {action_type} rates:')
    mean_with = np.nanmean(participants_with)
    std_with = np.std(participants_with)
    print(f'With secondary tasks: {mean_with:.2f} ± {std_with:.2f}')

    mean_without = np.nanmean(participants_without)
    std_without = np.std(participants_without)
    print(f'Without secondary tasks: {mean_without:.2f} ± {std_without:.2f}')

    res_wil = wilcoxon(participants_with, participants_without)
    print('Wilcoxon test:', res_wil.statistic, res_wil.pvalue)

    # res_ttest = ttest_rel(participants_with, participants_without)
    # print('T-test:', res_ttest.statistic, res_ttest.pvalue)

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


def count_robot_picks_per_object_in_trajectory(states, actions):
    """
    Count how many times the robot picks up each object (0..NUM_OBJECTS-1)
    in a single trajectory, using _robot_picks to find pick timesteps and
    state changes to identify which object was picked.

    Returns:
        np.ndarray of shape (NUM_OBJECTS,) with counts per object.
    """
    states = np.asarray(states)
    actions = np.atleast_1d(np.asarray(actions).ravel())
    pick_idxs = _robot_picks(states, actions)
    if len(pick_idxs) == 0:
        return np.zeros(NUM_OBJECTS, dtype=np.intp)
    rescue_status = states[:, 2:2 + NUM_OBJECTS]
    # At each pick index, the object picked is where status goes 1 -> 0
    obj_indices = []
    for idx in pick_idxs:
        if idx + 1 >= len(rescue_status):
            continue
        diff = rescue_status[idx] - rescue_status[idx + 1]
        assert np.any(diff == 1)
        obj_indices.append(np.argmax(diff))
    return np.bincount(obj_indices, minlength=NUM_OBJECTS)


def l1_distance(p, q):
    p, q = np.asarray(p).ravel(), np.asarray(q).ravel()
    return np.sum(np.abs(p - q))


def l2_distance(p, q):
    p, q = np.asarray(p).ravel(), np.asarray(q).ravel()
    return np.sqrt(np.sum((p - q)**2))


def robot_picks_per_object_per_task(
    all_states,
    all_actions,
    task_idx,
    participant_indices=None,
    label="All participants",
):
    """
    For a given task index, compute per participant the number of times the
    robot picks up each object, then the mean (and std) across participants
    for each object.

    Args:
        all_states: list[participant][task] of state arrays (as returned by
                    get_all_acts).
        all_actions: list[participant][task] of action arrays (as returned by
                     get_all_acts).
        task_idx: int, index of the task (trial) to analyze.
        participant_indices: optional 1d array or list of participant indices
                             to include; if None, use all participants.
        label: str, label for this group in printed output.

    Returns:
        Tuple of (mean_per_object, std_per_object), each shape (NUM_OBJECTS,).
        mean_per_object[i] is the average across participants of how many times
        the robot picked up object i in that task.
    """
    if participant_indices is not None:
        all_states = [all_states[i] for i in participant_indices]
        all_actions = [all_actions[i] for i in participant_indices]
    counts_per_participant = np.array([
        count_robot_picks_per_object_in_trajectory(
            np.asarray(participant_states[task_idx]),
            np.atleast_1d(np.asarray(participant_actions[task_idx]).ravel()),
        ) for participant_states, participant_actions in zip(
            all_states, all_actions)
    ])
    mean_per_object = np.mean(counts_per_participant, axis=0)
    std_per_object = np.std(counts_per_participant, axis=0)
    print(f"Robot picks per object for task {task_idx} ({label}):")
    for obj_idx, (m, s) in enumerate(zip(mean_per_object, std_per_object)):
        print(f"Object {obj_idx}: {m:.2f} ± {s:.2f}")
    return mean_per_object, std_per_object


def main(
    data_folder=None,
    trial_start=TRIAL_START,
    trial_end=TRIAL_END,
    num_bins=None,
    verbose=True,
    top_bottom_by_mean_across_tasks=False,
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
        top_bottom_by_mean_across_tasks: If False, top/bottom 25% are computed
            per task (by score on that task only). If True, top/bottom 25% are
            computed once by ranking participants by mean return across all
            tasks, and the same participant set is used for every task.
    """
    states, actions, sectasks, timer_times, ids = get_all_acts(
        data_folder=data_folder,
        trial_start=trial_start,
        trial_end=trial_end,
        num_bins=num_bins,
        verbose=verbose,
    )

    # Participant order matches get_rews (same sorted user loop and trial range)
    all_returns, _, _, _ = get_rews(
        data_folder=data_folder,
        trial_start=trial_start,
        trial_end=trial_end,
        verbose=False,
    )
    n_users = all_returns.shape[0]
    n_quarter = max(1, n_users // 4)

    # If using mean across tasks, compute top/bottom 25% participant indices
    # once (same set for all tasks)
    if top_bottom_by_mean_across_tasks:
        user_means = np.mean(all_returns, axis=1)
        print('Sorted means:', np.sort(user_means))
        sorted_idx = np.argsort(user_means)
        bottom_25_idx_global = sorted_idx[:n_quarter]
        top_25_idx_global = sorted_idx[-n_quarter:]
        bottom_label = "Bottom 25% (by mean return across tasks)"
        top_label = "Top 25% (by mean return across tasks)"
    else:
        bottom_label = "Bottom 25% by score on this task"
        top_label = "Top 25% by score on this task"

    # compute_total_delegations_per_task(actions)
    print('--------------------------------')
    user_means = np.mean(all_returns, axis=1)
    sorted_idx = np.argsort(user_means)
    bottom_25_idx_global = sorted_idx[:n_quarter]
    top_25_idx_global = sorted_idx[-n_quarter:]

    print('\nAll participants:')
    compute_action_rates_by_secondary_task(actions,
                                           sectasks,
                                           timer_times,
                                           action_type='delegation',
                                           use_first_half_only=False)
    compute_action_rates_by_secondary_task(actions,
                                           sectasks,
                                           timer_times,
                                           action_type='collect',
                                           use_first_half_only=False)

    print('\nBottom 25%:')
    compute_action_rates_by_secondary_task(
        actions,
        sectasks,
        timer_times,
        action_type='delegation',
        participant_indices=bottom_25_idx_global,
        use_first_half_only=False)

    compute_action_rates_by_secondary_task(
        actions,
        sectasks,
        timer_times,
        action_type='collect',
        participant_indices=bottom_25_idx_global,
        use_first_half_only=False)

    print('\nTop 25%:')
    compute_action_rates_by_secondary_task(
        actions,
        sectasks,
        timer_times,
        action_type='delegation',
        participant_indices=top_25_idx_global,
        use_first_half_only=False)

    compute_action_rates_by_secondary_task(
        actions,
        sectasks,
        timer_times,
        action_type='collect',
        participant_indices=top_25_idx_global,
        use_first_half_only=False)
    print('--------------------------------')

    # compute_robot_picks_per_task(states, actions)
    # print('--------------------------------')
    for task_idx in range(TRIAL_END - TRIAL_START):
        delegations_to_objects_per_task(actions, task_idx)
        print('--------------------------------')
        if top_bottom_by_mean_across_tasks:
            bottom_25_idx = bottom_25_idx_global
            top_25_idx = top_25_idx_global
        else:
            task_scores = all_returns[:, task_idx]
            print('Sorted task scores:', np.sort(task_scores))
            sorted_idx = np.argsort(task_scores)
            bottom_25_idx = sorted_idx[:n_quarter]
            top_25_idx = sorted_idx[-n_quarter:]
        robot_picks_per_object_per_task(states, actions, task_idx)
        (bottom_25_robot_picks_avg,
         _bottom_25_robot_picks_std) = robot_picks_per_object_per_task(
             states,
             actions,
             task_idx,
             participant_indices=bottom_25_idx,
             label=bottom_label,
         )
        (top_25_robot_picks_avg,
         _top_25_robot_picks_std) = robot_picks_per_object_per_task(
             states,
             actions,
             task_idx,
             participant_indices=top_25_idx,
             label=top_label,
         )
        # Distance from robot pick distributions to reference ROBOT_PICKS
        ref = np.array(ROBOT_PICKS[task_idx])
        print(f"Task {task_idx} distance to ROBOT_PICKS:")
        print(
            f"  Bot 25%: L1={l1_distance(bottom_25_robot_picks_avg, ref):.2f}, "
            f"L2={l2_distance(bottom_25_robot_picks_avg, ref):.2f}")
        print(f"  Top 25%: L1={l1_distance(top_25_robot_picks_avg, ref):.2f}, "
              f"L2={l2_distance(top_25_robot_picks_avg, ref):.2f}")
        print('--------------------------------')

    return states, actions, sectasks, ids


if __name__ == "__main__":
    main(
        data_folder=DATA_FOLDER,
        trial_start=TRIAL_START,
        trial_end=TRIAL_END,
        num_bins=10,
        verbose=False,
        top_bottom_by_mean_across_tasks=False,
    )
