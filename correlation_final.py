import pandas as pd
import json

# Load JSON file
file_path = "cleanData.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Define the numeric attributes we want to keep
numeric_columns = [
    "What is your age? (Input in years,  e.g. 19 for 19y/o)",
    # "Do you consume caffeine (coffee, tea, energy drinks)?",
    # "Do you smoke?", # If a column has the same value for every row, its standard deviation is zero, making correlation undefined.
    # "Do you consume alcohol?",
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

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Convert all columns that contain numeric values stored as strings to real numbers
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numbers, set errors to NaN


# Keep only the selected numeric attributes
df_numeric = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop non-numeric columns (if any columns are entirely non-numeric, drop them)
df_numeric = df_numeric.dropna(axis=1, how='all')

# Compute correlation matrix
corr_matrix = df_numeric.corr()

# Convert matrix to long format and remove self-correlations
corr_pairs = corr_matrix.unstack().reset_index()
corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']
corr_pairs = corr_pairs[corr_pairs['Variable 1'] != corr_pairs['Variable 2']]

# Remove duplicate pairs (keep unique pairs only)
corr_pairs['Sorted Pair'] = corr_pairs.apply(lambda row: tuple(sorted([row['Variable 1'], row['Variable 2']])), axis=1)
corr_pairs = corr_pairs.drop_duplicates(subset=['Sorted Pair']).drop(columns=['Sorted Pair'])

# Sort by absolute correlation value (strongest to weakest)
corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False, key=abs).reset_index(drop=True)


pd.set_option('display.max_rows', None)  # Show all rows
# Display result
print(corr_pairs)
