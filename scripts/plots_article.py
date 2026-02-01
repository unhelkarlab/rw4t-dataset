import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# TO PLOT THIS
# number objects delivered as a team 4.91+/-1.28  (vs 6)
# number of objects delegated to robot 1.98+/-0.76 (2.78 +/- 0.70)
# score: 99.19+/-45.11 vs 142.04+/-0.28


import numpy as np
from scipy import stats

# Sample data
group_human = np.concatenate(data)
group_rl = np.ones(10) * 6

# Perform independent two-sample t-test (assuming equal variances)
t_statistic, p_value = stats.ttest_ind(group_human, group_rl)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Data
data2 = [
    [2, 2, 0, 1, 2],
    [1, 2, 2, 1, 2],
    [3, 2, 2, 2, 2],
    [2, 2, 3, 2, 3],
    [1, 1, 2, 2, 3],
    [3, 3, 2, 2, 3],
    [2, 0, 3, 2, 3],
    [2, 2, 2, 2, 2],
    [2, 1, 2, 1, 2],
    [2, 2, 2, 1, 1],
    [3, 2, 1, 2, 2],
    [3, 2, 3, 2, 2],
    [2, 0, 2, 3, 2],
    [2, 2, 3, 2, 3],
    [3, 3, 2, 2, 2],
    [1, 2, 1, 1, 1],
    [1, 0, 2, 1, 1],
    [2, 3, 3, 3, 3],
    [3, 2, 3, 1, 3],
    [2, 2, 2, 2, 2]
]
data = [
    [3, 4, 2, 1, 3],
    [5, 6, 6, 6, 6],
    [6, 5, 6, 6, 6],
    [2, 3, 3, 3, 4],
    [4, 3, 4, 4, 3],
    [5, 6, 5, 6, 6],
    [5, 3, 6, 5, 6],
    [5, 6, 6, 6, 6],
    [4, 3, 5, 4, 6],
    [4, 4, 3, 5, 5],
    [6, 6, 5, 6, 6],
    [6, 5, 6, 6, 6],
    [3, 2, 5, 5, 6],
    [4, 4, 5, 5, 6],
    [6, 6, 6, 6, 6],
    [2, 3, 4, 4, 5],
    [5, 4, 6, 6, 6],
    [3, 6, 6, 5, 6],
    [6, 6, 6, 6, 6],
    [5, 5, 6, 6, 6]
]

# Create DataFrame
df2 = pd.DataFrame(data, columns=['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5'])
df2_long = df2.melt(var_name='Task', value_name='Score')

# Set style
sns.set(style="whitegrid", font_scale=1.2)

# Create figure
fig, ax = plt.subplots(figsize=(5, 5))
palette = ['#88bde6', '#88bde6', '#88bde6', '#88bde6', '#88bde6']

# Boxplot with bold median
sns.boxplot(
    x='Task', y='Score', data=df2_long,
    linewidth=1.2, width=0.6, fliersize=3,
    palette=palette, ax=ax,
    medianprops=dict(color='black', linewidth=2.5)
)

# Add horizontal line at y=4
ax.axhline(y=6, linestyle='--', color='black', linewidth=1)
ax.text(-0.4, 6.05, 'Optimal Team Performance', fontsize=9, fontweight='bold')

# Axis labels and limits
ax.set_ylim(0, 6.4)
ax.set_ylabel("Medical Kits Delivered")
ax.set_xlabel("")
ax.set_title("Team Performance")

plt.tight_layout()
plt.show()





# Create DataFrame
df2 = pd.DataFrame(data2, columns=['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5'])
df2_long = df2.melt(var_name='Task', value_name='Score')

# Set style
sns.set(style="whitegrid", font_scale=1.2)

# Create figure
fig, ax = plt.subplots(figsize=(5, 5))
palette = ['#88bde6'] * 5

# Boxplot with bold median
sns.boxplot(
    x='Task', y='Score', data=df2_long,
    linewidth=1.2, width=0.6, fliersize=3,
    palette=palette, ax=ax,
    medianprops=dict(color='black', linewidth=2.5)
)

# Add horizontal line at y=4
ax.axhline(y=4, linestyle='--', color='black', linewidth=1)
ax.text(-0.4, 4.05, 'Optimal Robot Utility', fontsize=9, fontweight='bold')

# Set integer y-ticks
ax.set_yticks(range(0, 5, 1))

# Axis labels and limits
ax.set_ylim(0, 4.4)
ax.set_ylabel("Medical Kits Delivered")
ax.set_xlabel("")
ax.set_title("Robot's Utility")

plt.tight_layout()
plt.show()



