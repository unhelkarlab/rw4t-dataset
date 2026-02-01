import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
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

# Convert to DataFrame
df = pd.DataFrame(data, columns=['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5'])

# Remove outliers column-wise using IQR
def remove_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return series[(series >= Q1 - 1.5 * IQR) & (series <= Q3 + 1.5 * IQR)]

df_cleaned = df.apply(remove_outliers)

# Calculate statistics
means = df_cleaned.mean().to_numpy()
mins = df_cleaned.min().to_numpy()
maxs = df_cleaned.max().to_numpy()

# Plot
tasks = df.columns
x = np.arange(len(tasks))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, means, marker='o', color='#88bde6', linewidth=3)
ax.fill_between(x, mins, maxs, color='#88bde6', alpha=0.3)
ax.set_ylim(0, 6.5)
# Optimal performance line
ax.axhline(y=6, linestyle='--', color='black', linewidth=1.5)
ax.text(0, 6.1, 'Optimal Team Performance', fontsize=10, fontweight='bold')
# Style
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=12, fontweight='bold')
ax.set_yticks(range(0, 7))
ax.set_ylabel("Medical Kits Delivered", fontsize=12, fontweight='bold')
ax.set_title("Team Performance by Task", fontsize=14, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=1.5)

# Thicker axes
ax.tick_params(axis='both', width=2, length=5)
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("article_images/teamperformance_by_task_iqr_band.png")
plt.show()
