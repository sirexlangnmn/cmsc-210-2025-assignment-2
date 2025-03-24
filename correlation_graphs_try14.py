import numpy as np
from scipy.stats import pearsonr

# Define the data
x = np.arange(10, 20)
y = np.array([2, 1, 4, 5, 8, 12, 18, 25, 96, 48])

# Compute Pearson correlation using NumPy
r_matrix = np.corrcoef(x, y)
print("Correlation matrix:")
print(r_matrix)
print(f"Pearson correlation coefficient (NumPy): {r_matrix[0, 1]:.2f}")

# Compute Pearson correlation using SciPy
r_value, _ = pearsonr(x, y)
print(f"Pearson correlation coefficient (SciPy): {r_value:.2f}")