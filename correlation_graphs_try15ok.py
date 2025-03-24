import numpy as np
import matplotlib.pyplot as plt
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

# Plot the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label=f'Pearson r = {r_value:.2f}')
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Scatter Plot with Pearson Correlation')
plt.legend()
plt.grid()

# Save the plot as an image
plt.savefig("scatter_plot.png")
# plt.show()
