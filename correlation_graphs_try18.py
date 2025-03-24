import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr

# Load data from JSON file
with open("cleanData.json", "r") as file:
    data = json.load(file)

# Extract and convert sleeping environment and sleeping position to integers
sleeping_environment = []
sleeping_position = []

for entry in data:
    try:
        env_score = entry.get("How would you describe your daily water intake?")
        pos_score = entry.get("How consistent is your sleep schedule?")
        
        if env_score is not None and pos_score is not None:
            env_score = int(env_score)
            pos_score = int(pos_score)
            sleeping_environment.append(env_score)
            sleeping_position.append(pos_score)
    except ValueError:
        continue  # Skip entries with invalid data

# Convert lists to NumPy arrays
x = np.array(sleeping_environment)
y = np.array(sleeping_position)

print('x (How would you describe your daily water intake?  ) ===>> ', x)
print('y (How consistent is your sleep schedule?  ) ===>> ', y)

# Compute Pearson correlation only if sufficient data exists
if len(x) > 1 and len(y) > 1:
    # Compute correlation using NumPy
    r_matrix = np.corrcoef(x, y)
    print("Correlation matrix:")
    print(r_matrix)
    print(f"Pearson correlation coefficient (NumPy): {r_matrix[0, 1]:.2f}")

    # Compute correlation using SciPy
    r_value, _ = pearsonr(x, y)
    print(f"Pearson correlation coefficient (SciPy): {r_value:.2f}")

    # Plot the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label=f'Pearson r = {r_value:.2f}')

    xlabel = "water intake"
    ylabel = "sleeping schedule"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = f"{xlabel} vs {ylabel}"
    plt.title(title)
    plt.legend()
    plt.grid()

    # Save the plot as an image
    plt.savefig(f"Scatter Plot of {title}.png")
    plt.show()
else:
    print("Insufficient valid data for correlation calculation.")
