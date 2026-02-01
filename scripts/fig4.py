import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from extract import get_trajs
mdp_states, mdp_actions, mdp_rewards, mdp_dones, ids, sectasks = get_trajs(num_bins=10)

# Data
data_scores = [
    [75, 20, -84, -5, 24],
    [109, 105, 83, 131, 142],
    [147, 122, 106, 150, 150],
    [9, 55, 15, 75, 100],
    [100, 65, -8, 100, 65],
    [125, 120, 125, 120, 150],
    [125, 75, 150, 125, 150],
    [109, 43, 32, 47, 107],
    [100, 75, 125, 90, 150],
    [100, 62, 75, 95, 125],
    [150, 142, 110, 100, 150],
    [150, 125, 108, 150, 150],
    [71, 30, 115, 95, 150],
    [63, 70, 115, 68, 150],
    [142, 106, 76, 133, 143],
    [50, 14, 90, 90, 125],
    [64, 16, 123, 110, 122],
    [45, 112, 137, 125, 150],
    [109, 121, 140, 150, 121],
    [125, 85, 82, 150, 150]
]

categories = [
    ("beginner", "yearly"),
    ("intermediate", "weekly"),
    ("advanced", "daily"),
    ("beginner", "yearly"),
    ("no experience", "never"),
    ("beginner", "yearly"),
    ("no experience", "never"),
    ("intermediate", "monthly"),
    ("expert", "weekly"),
    ("beginner", "monthly"),
    ("advanced", "daily"),
    ("intermediate", "yearly"),
    ("beginner", "yearly"),
    ("expert", "yearly"),
    ("intermediate", "weekly"),
    ("intermediate", "monthly"),
    ("beginner", "yearly"),
    ("advanced", "monthly"),
    ("advanced", "monthly"),
    ("intermediate", "weekly"),
]

# Merge function
def merge_experience(exp):
    exp = exp.lower()
    if exp in ['beginner', 'no experience']:
        return 'Beginner/No Exp'
    elif exp in ['expert', 'advanced']:
        return 'Expert/Advanced'
    else:
        return 'Intermediate'

# Create DataFrame
df = pd.DataFrame(data_scores)
df["experience"] = [merge_experience(exp) for exp, freq in categories]
df["mean_score"] = df.iloc[:, :5].mean(axis=1)

# Group and aggregate
grouped = df.groupby("experience")["mean_score"].agg(["mean", "std"]).sort_values("mean", ascending=True)

# Plotting in requested style
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ['orange', 'orange', 'orange']
bars = ax.bar(grouped.index, grouped["mean"], yerr=grouped["std"],
              capsize=5, color=bar_colors, edgecolor='black', linewidth=2,
              error_kw=dict(ecolor='dimgrey', elinewidth=1.5, capsize=5, capthick=1.5))

# Add reference line and band
mean_ref = 142.04
std_ref = 0.28
ax.axhline(mean_ref, color='black', linestyle='--', linewidth=1.5)
ax.fill_between([-0.5, len(grouped) - 0.5], mean_ref - std_ref, mean_ref + std_ref,
                color='black', alpha=0.1)
ax.text(-0.6, 143.1, 'RL-Derived Optimal Score', fontsize=10, fontweight='bold')

# Annotate bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold')

# Style and labels
ax.set_ylabel('Average Score', fontsize=14, fontweight='bold')
ax.set_title('Average Score by Experience Level', fontsize=16, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.tick_params(axis='both', labelsize=12)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig("article_images/score_by_experience.png")
plt.show()
