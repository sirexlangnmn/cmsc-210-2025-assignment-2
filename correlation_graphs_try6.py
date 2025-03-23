import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with specific correlations
size = 100  # Number of data points

# Perfect negative correlation (-1)
x1 = np.linspace(0, 10, size)
y1 = -x1

# Strong negative correlations (-0.9 to -0.4)
y2 = -0.9 * x1 + np.random.normal(0, 1, size)
y3 = -0.8 * x1 + np.random.normal(0, 1, size)
y4 = -0.7 * x1 + np.random.normal(0, 1, size)
y5 = -0.6 * x1 + np.random.normal(0, 1, size)
y6 = -0.5 * x1 + np.random.normal(0, 1, size)
y7 = -0.4 * x1 + np.random.normal(0, 1, size)

# Weak/no correlation (-0.3 to 0.3)
y8 = -0.3 * x1 + np.random.normal(0, 1, size)
y9 = -0.2 * x1 + np.random.normal(0, 1, size)
y10 = np.random.normal(0, 1, size)  # Near zero correlation
y11 = 0.2 * x1 + np.random.normal(0, 1, size)
y12 = 0.3 * x1 + np.random.normal(0, 1, size)

# Strong positive correlations (0.4 to 1)
y13 = 0.4 * x1 + np.random.normal(0, 1, size)
y14 = 0.5 * x1 + np.random.normal(0, 1, size)
y15 = 0.6 * x1 + np.random.normal(0, 1, size)
y16 = 0.7 * x1 + np.random.normal(0, 1, size)
y17 = 0.8 * x1 + np.random.normal(0, 1, size)
y18 = 0.9 * x1 + np.random.normal(0, 1, size)

# Perfect positive correlation (+1)
y19 = x1

# Create a DataFrame
df = pd.DataFrame({
    "-1.00": y1, "-0.90": y2, "-0.80": y3, "-0.70": y4, "-0.60": y5, "-0.50": y6, "-0.40": y7,
    "-0.30": y8, "-0.20": y9, "-0.10": y10, "0.00": y10, "0.10": y11, "0.20": y12, "0.30": y13,
    "0.40": y14, "0.50": y15, "0.60": y16, "0.70": y17, "0.80": y18, "0.90": y19, "1.00": y19
})

# Create the pairplot
g = sns.PairGrid(df)

# Plot scatter plots in blue
g.map_upper(sns.scatterplot, color="blue", alpha=0.6)

# Add correlation coefficient labels
def corrfunc(x, y, **kwargs):
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    ax.set_title(f"{r:.2f}", fontsize=12, fontweight='bold')

g.map_lower(corrfunc)

# Hide diagonal (optional, for cleaner look)
for ax in g.diag_axes:
    ax.set_visible(False)

# Adjust layout
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Pearson Correlation Coefficient", fontsize=16, fontweight="bold")

# Show plot
plt.show()
