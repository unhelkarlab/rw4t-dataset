import matplotlib.pyplot as plt
import numpy as np

# X-axis categories
x_labels = ['RW4T Dataset', 'RL-Derived Optimal Policy']
x = np.arange(len(x_labels))

# Data: Robot-only and total team object deliveries
robot_only = np.array([1.98, 2.78])
team_total = np.array([4.91, 6.00])
human_part = team_total - robot_only  # Human contribution (top of bar)

# Standard deviations
robot_only_std = [0.76, 0.70]
team_total_std = [1.28, 0.00]  # Total SD (for reference if needed)

# Plot settings
bar_width = 0.5
fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Error bar style
err_style = dict(
    ecolor='dimgrey',
    elinewidth=1.5,
    capsize=5,
    capthick=1.5,
)

# Plot bottom (robot) and stacked (human) bars
bars_robot = ax.bar(x, robot_only,
                    width=bar_width,
                    label='Robot (via Delegation)',
                    color='salmon',
                    edgecolor='black',
                    linewidth=2,
                    yerr=robot_only_std,
                    error_kw=err_style)

bars_human = ax.bar(x, human_part,
                    bottom=robot_only,
                    width=bar_width,
                    label='Human',
                    color='skyblue',
                    edgecolor='black',
                    yerr=team_total_std,
                    error_kw=err_style,
                    linewidth=2)

# # Annotate total values on top of bars
# for i in range(len(x)):
#     total = team_total[i]
#     ax.annotate(f'{total:.2f}',
#                 xy=(x[i], total),
#                 xytext=(0, 3),
#                 textcoords="offset points",
#                 ha='center',
#                 va='bottom',
#                 fontsize=12,
#                 fontweight='bold')

# Labels and styling
ax.set_title("Team's Performance", fontsize=16, fontweight='bold')
ax.set_ylabel("Objects Delivered", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=13, fontweight='bold')
ax.tick_params(axis='y', labelsize=12)
ax.legend(loc='upper left', prop={'weight': 'bold', 'size': 12})
ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=1.5)

# Bold axis lines
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("article_images/team_performance_stacked_bar.png")
plt.show()
