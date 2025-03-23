import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re  # For filename cleaning

# Load JSON file
file_path = "csvjson_part1_v3.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame(data)

# Define numeric attributes
numeric_columns = [
    "What is your age? (Input in years,  e.g. 19 for 19y/o)",
    "Do you consume caffeine (coffee, tea, energy drinks)?",
    # "Do you smoke?",  # Removed as in the previous correlation script
    "Do you consume alcohol?",
    "How would you describe your sleeping environment?",
    "On average, how many hours do you sleep per night? (input number only, e.g., 8 for 8 hours )",
    "What time do you usually go to bed? (Military time, e.g., 22 for 10 PM)",
    "What time do you usually wake up? (Military time, e.g., 6 for 6 AM)",
    "How many times do you wake up during the night?  (e.g., number from 0-5)",
    "How often do you exercise per week? (e.g., 0-7)",
    "How would you rate your stress level? (Stressed have felt in past week)",
    "How often do you use electronic devices (phone, TV, computer) before sleep?",
    "How would you describe your daily water intake?",
    "How comfortable is your usual sleeping position?",
    "How consistent is your sleep schedule?"
]

# Convert columns to numeric values
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Keep only selected numeric attributes
df_numeric = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop non-numeric columns (if any columns are entirely non-numeric, drop them)
df_numeric = df_numeric.dropna(axis=1, how='all')

# Compute correlation matrix
corr_matrix = df_numeric.corr()

# Convert to long format and remove self-correlations
corr_pairs = corr_matrix.unstack().reset_index()
corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']
corr_pairs = corr_pairs[corr_pairs['Variable 1'] != corr_pairs['Variable 2']]

# Remove duplicate pairs (keep unique pairs only)
corr_pairs['Sorted Pair'] = corr_pairs.apply(lambda row: tuple(sorted([row['Variable 1'], row['Variable 2']])), axis=1)
corr_pairs = corr_pairs.drop_duplicates(subset=['Sorted Pair']).drop(columns=['Sorted Pair'])

# Sort by absolute correlation value (strongest to weakest)
corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False, key=abs).reset_index(drop=True)

# Debugging: Print the top 10 correlations
print("\n📊 **Top 10 Correlations After Processing**")
print(corr_pairs.head(10))  # Print top 10 correlations to verify if they match previous results

# Ensure scatter plot uses the exact dataset from which correlations were computed
df_numeric = df_numeric.dropna()

# Create output directory for graphs
output_dir = "correlation_graphs"
os.makedirs(output_dir, exist_ok=True)

# Function to clean filenames for saving images
def clean_filename(text):
    return re.sub(r'[^\w\-_]', '_', text)  # Replace non-alphanumeric characters with '_'

# Generate scatter plots for all correlation pairs
for index, row in corr_pairs.iterrows():
    var1 = row["Variable 1"]
    var2 = row["Variable 2"]
    correlation = row["Correlation"]

    # print('var1 ==>> ', var1)
    # print('var2 ==>> ', var2)
    # print('correlation ==>> ', correlation)

    # Check if variables exist in the dataset
    if var1 in df_numeric.columns and var2 in df_numeric.columns:
        print('var1 ==>> ', var1)
        # print('df_numeric.columns ==>> ', df_numeric.columns)
        print('df_numeric[var1] ==>> ', df_numeric[var1])
        # print('df_numeric[var2] ==>> ', df_numeric[var2])
        # Create scatter plot
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=df_numeric[var1], y=df_numeric[var2], color="blue")
        plt.title(f"{var1} vs. {var2}\n(r = {correlation:.3f})")
        plt.xlabel(var1)
        plt.ylabel(var2)

        # Sanitize filename
        filename = f"{output_dir}/{clean_filename(var1)}_vs_{clean_filename(var2)}.png"
        
        # Save the plot
        plt.savefig(filename)
        plt.close()  # Close the figure to free memory

print(f"\n✅ Correlation graphs saved in '{output_dir}' folder.")
