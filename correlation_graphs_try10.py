import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load JSON data
file_path = 'csvjson_part1_v3.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame(data)

# Select relevant numerical columns
columns_to_keep = [
    "What is your age? (Input in years,  e.g. 19 for 19y/o)",
    "On average, how many hours do you sleep per night? (input number only, e.g., 8 for 8 hours )",
    "What time do you usually wake up? (Military time, e.g., 6 for 6 AM)",
    "How would you rate your stress level? (Stressed have felt in past week)"
]
df = df[columns_to_keep]

# Convert columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Compute Pearson Correlation
correlation_matrix = df.corr(method='pearson')

# Display correlation matrix
print("Pearson Correlation Matrix:")
print(correlation_matrix)

# Save correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Pearson Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# Generate scatter plots for key relationships
for col in columns_to_keep[1:]:  # Compare each variable with age
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[columns_to_keep[0]], y=df[col])
    plt.xlabel("Age")
    plt.ylabel(col)
    plt.title(f"Scatter Plot: Age vs {col}")
    plt.savefig(f"scatter_{col.replace(' ', '_')}.png")
    plt.close()

# Example: Compute specific correlation between Age and Hours of Sleep
corr, p_value = pearsonr(df[columns_to_keep[0]], df[columns_to_keep[1]])
print(f"Pearson Correlation between Age and Hours of Sleep: {corr} (p-value: {p_value})")
