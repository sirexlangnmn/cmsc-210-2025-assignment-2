import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr

# Load data from JSON file
with open("cleanData.json", "r") as file:
    data = json.load(file)

# Extract and convert age and wake-up time to integers
x = []
y = []
for entry in data:
    try:
        age = int(entry.get("How would you describe your sleeping environment?", 0))
        wake_up_time = int(entry.get("How comfortable is your usual sleeping position?", 0))
        x.append(age)
        y.append(wake_up_time)
    except ValueError:
        continue  # Skip entries with invalid data

x = np.array(x)
y = np.array(y)

# Compute Pearson correlation using NumPy
if len(x) > 1 and len(y) > 1:
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
    plt.xlabel('sleeping environment')
    plt.ylabel('sleeping position')
    plt.title('Scatter Plot of sleeping environment vs. sleeping position')
    plt.legend()
    plt.grid()

    # Save the plot as an image
    plt.savefig("Scatter Plot of sleeping environment vs. sleeping position.png")
    plt.show()
else:
    print("Insufficient valid data for correlation calculation.")