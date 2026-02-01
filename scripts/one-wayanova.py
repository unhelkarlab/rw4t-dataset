import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Your data
beginner = np.concatenate([[75, 20, -84, -5, 24],[9, 55, 15, 75, 100],[100, 65, -8, 100, 65],
            [125, 120, 125, 120, 150], [125, 75, 150, 125, 150],[64, 16, 123, 110, 122],
            [100, 62, 75, 95, 125],[71, 30, 115, 95, 150],])
intermediate = np.concatenate([[109, 105, 83, 131, 142], [125, 85, 82, 150, 150], [142, 106, 76, 133, 143],
    [50, 14, 90, 90, 125],[109, 43, 32, 47, 107],[150, 125, 108, 150, 150],])
advanced = np.concatenate([[147, 122, 106, 150, 150],[45, 112, 137, 125, 150],[109, 121, 140, 150, 121],
            [100, 75, 125, 90, 150],[63, 70, 115, 68, 150],[150, 142, 110, 100, 150],])

# Combine data for Tukey HSD
data = np.concatenate([beginner, intermediate, advanced])
groups = (['beginner'] * len(beginner)) + (['intermediate'] * len(intermediate)) + (['advanced'] * len(advanced))

# Perform ANOVA
f_stat, p_val = stats.f_oneway(beginner, intermediate, advanced)

# Calculate omega squared
k = 3
N = len(data)
df_between = k - 1
df_within = N - k
ss_between = sum([len(g) * (np.mean(g) - np.mean(data))**2 for g in [beginner, intermediate, advanced]])
ss_within = sum([sum((g - np.mean(g))**2) for g in [beginner, intermediate, advanced]])
ms_within = ss_within / df_within
omega_sq = (ss_between - (df_between * ms_within)) / (ss_between + ss_within + ms_within)

# Perform Tukey HSD
tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

# Print results
print(f"Beginner n: {len(beginner)}")
print(f"Intermediate n: {len(intermediate)}")
print(f"Advanced n: {len(advanced)}")
print(f"Total N: {N}")
print(f"df_between: {df_between}")
print(f"df_within: {df_within}")
print(f"F-statistic: {f_stat:.3f}")
print(f"p-value: {p_val:.4f}")
print(f"Omega squared: {omega_sq:.3f}")
print("\nTukey HSD results:")
print(tukey_df)



import numpy as np
from scipy import stats

# Example data (replace with your data arrays)
low_workload_picks
high_workload_picks
# t-test
t_stat, p_value = stats.ttest_ind(low_workload_picks, high_workload_picks, equal_var=False)  # Welch's t-test

# Effect size (Cohen's d)
mean_diff = np.mean(low_workload_picks) - np.mean(high_workload_picks)
pooled_std = np.sqrt(((len(low_workload_picks) - 1)*np.var(low_workload_picks, ddof=1) + (len(high_workload_picks) - 1)*np.var(high_workload_picks, ddof=1)) / (len(low_workload_picks) + len(high_workload_picks) - 2))
cohens_d = mean_diff / pooled_std

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Cohen's d: {cohens_d:.3f}")
