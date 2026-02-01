import numpy as np
import os
import matplotlib.pyplot as plt
from extract import get_trajs

mdp_states, mdp_actions, mdp_rewards, mdp_dones, ids, sectasks = get_trajs(num_bins=10)

high_workload_picks = []
low_workload_picks = []

for user in range(20):
    for trial in range(5):
        if trial == 0:
            continue
        idx = user*5 + trial
        secs = sectasks[idx]
        if trial==3:
            # only high workloads
            picks = (mdp_rewards[idx][secs]==24).sum()
            high_workload_picks.append(picks//2)
            high_workload_picks.append(picks-picks//2)
        elif trial==4:
            picks = (mdp_rewards[idx][~secs]==24).sum()
            low_workload_picks.append(picks//2)
            low_workload_picks.append(picks-picks//2)
        else:
            high_workload_picks.append((mdp_rewards[idx][secs]==24).sum())
            low_workload_picks.append((mdp_rewards[idx][~secs]==24).sum())
        # secs.append(sectasks[idx])

mean_low, std_low = np.mean(low_workload_picks), np.std(low_workload_picks)
mean_high, std_high = np.mean(high_workload_picks), np.std(high_workload_picks)


labels = ['Low Workload', 'High Workload']
means = [mean_low, mean_high]
stds = [std_low, std_high]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, means, yerr=stds, capsize=6,  edgecolor='black', linewidth=2)

# Add grid and labels
ax.set_ylabel('Objects Picked Up by Team', fontsize=12, fontweight='bold')
ax.set_title('Effect of Workload on Performance', fontsize=14, fontweight='bold')
ax.grid(axis='y', linestyle='--', linewidth=1.5, alpha=0.6)
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')
# Format axes
ax.tick_params(axis='both', width=2, length=5)
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("article_images/effect_of_workload_on_performance.png")
plt.show()
