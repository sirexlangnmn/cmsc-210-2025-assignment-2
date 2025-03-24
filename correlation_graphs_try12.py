import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Example DataFrame
data = {
    'Variable_1': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'Variable_2': [2, 1, 4, 5, 8, 12, 18, 25, 96, 48]
}
df = pd.DataFrame(data)

# Compute the correlation matrix
correlation_matrix = df.corr(method='pearson')

# Create a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Define the directory and filename
save_dir = "C:/Users/YourUsername/Documents"  # Change this to your desired path
filename = "correlation_heatmap.png"
save_path = os.path.join(save_dir, filename)

# Save the figure
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Show the plot (optional)
plt.show()

print(f"Heatmap saved at: {save_path}")
