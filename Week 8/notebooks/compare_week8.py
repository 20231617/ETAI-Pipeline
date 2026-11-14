
import numpy as np
from scipy.stats import t

# Nested CV ROC-AUC scores from your reports
with_fs = np.array([0.726, 0.703, 0.731, 0.734, 0.734])
without_fs = np.array([0.729, 0.702, 0.737, 0.739, 0.734])

# Paired differences
differences = without_fs - with_fs

# Nadeau-Bengio corrected paired t-test
n = len(differences)
test_train_ratio = 1 / 4

corrected_se = np.sqrt(
    (1 / n + test_train_ratio)
    * np.var(differences, ddof=1)
)

t_stat = np.mean(differences) / corrected_se
p_value = 2 * t.sf(abs(t_stat), df=n - 1)

print(f"Mean ROC-AUC with FS: {with_fs.mean():.3f}")
print(f"Mean ROC-AUC without FS: {without_fs.mean():.3f}")
print(f"Corrected t-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Statistically significant difference.")
else:
    print("No statistically significant difference.")
