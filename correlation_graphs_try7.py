import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re  # For filename cleaning

# Load the JSON file
file_path = "csvjson_part1_v3.json"  # Update with your actual file path
df = pd.read_json(file_path)

# Define the numeric columns
numeric_columns = [
    "What is your age? (Input in years,  e.g. 19 for 19y/o)",
    "What time do you usually go to bed? (Military time, e.g., 22 for 10 PM)"
]

# Convert columns to numeric values (handling errors)
df_numeric = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
df_numeric = df_numeric.dropna()

# Compute Pearson Correlation Coefficient
pearson_corr = df_numeric.corr(method='pearson')
correlation_value = pearson_corr.iloc[0, 1]

# Print the correlation result
print(f"Pearson Correlation Coefficient (Age vs. Bedtime): {correlation_value:.3f}")

# Create output directory for graphs
output_dir = "correlation_graphs"
os.makedirs(output_dir, exist_ok=True)

# Function to clean filenames for saving images
def clean_filename(text):
    return re.sub(r'[^\w\-_]', '_', text)  # Replace non-alphanumeric characters with '_'


# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df_numeric[numeric_columns[0]], df_numeric[numeric_columns[1]], alpha=0.6, color='blue')

# Add labels and title
plt.xlabel("Age (years)")
plt.ylabel("Bedtime (Military Time)")
plt.title("Scatter Plot: Age vs. Bedtime")
plt.grid(True)

# Save the figure
plt.savefig("age_vs_bedtime_correlation.png")  # Saves the file locally
# plt.show()

# Sanitize filename
filename = f"{output_dir}/{clean_filename(1)}_vs_{clean_filename(2)}.png"

# Save the plot
plt.savefig(filename)
