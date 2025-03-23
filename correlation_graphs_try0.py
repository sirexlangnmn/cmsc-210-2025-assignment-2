import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load JSON file
file_path = "csvjson_part1.json"  # Update with correct file path
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame(data)

# Define numeric attributes
numeric_columns = [
    "What is your age? (Input in years,  e.g. 19 for 19y/o)",
    "Do you consume caffeine (coffee, tea, energy drinks)?",
    # "Do you smoke?", # If a column has the same value for every row, its standard deviation is zero, making correlation undefined.
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
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Keep only numeric data
df_numeric = df[numeric_columns].dropna()

# Compute correlation matrix
corr_matrix = df_numeric.corr()

# Find strongest and weakest correlations
corr_pairs = corr_matrix.unstack().reset_index()
corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']
corr_pairs = corr_pairs[corr_pairs['Variable 1'] != corr_pairs['Variable 2']]

# Remove duplicate pairs
corr_pairs['Sorted Pair'] = corr_pairs.apply(lambda row: tuple(sorted([row['Variable 1'], row['Variable 2']])), axis=1)
corr_pairs = corr_pairs.drop_duplicates(subset=['Sorted Pair']).drop(columns=['Sorted Pair'])

# Sort by absolute correlation
corr_pairs = corr_pairs.sort_values(by="Correlation", ascending=False, key=abs)

# Select strongest and weakest correlations
strongest_correlation = corr_pairs.iloc[0]
weakest_correlation = corr_pairs.iloc[-1]

# Extract column names for visualization
strongest_pair = (strongest_correlation["Variable 1"], strongest_correlation["Variable 2"])
weakest_pair = (weakest_correlation["Variable 1"], weakest_correlation["Variable 2"])

# Generate scatter plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Strongest Correlation Plot
sns.scatterplot(x=df_numeric[strongest_pair[0]], y=df_numeric[strongest_pair[1]], ax=axes[0], color="blue")
axes[0].set_title(f"Strongest Correlation: {strongest_pair[0]} vs. {strongest_pair[1]}\n(r = {strongest_correlation['Correlation']:.3f})")
axes[0].set_xlabel(strongest_pair[0])
axes[0].set_ylabel(strongest_pair[1])

# Weakest Correlation Plot
sns.scatterplot(x=df_numeric[weakest_pair[0]], y=df_numeric[weakest_pair[1]], ax=axes[1], color="red")
axes[1].set_title(f"Weakest Correlation: {weakest_pair[0]} vs. {weakest_pair[1]}\n(r = {weakest_correlation['Correlation']:.3f})")
axes[1].set_xlabel(weakest_pair[0])
axes[1].set_ylabel(weakest_pair[1])

# Show plots
plt.tight_layout()
# plt.show()
plt.savefig("correlation_graphs.png")  # Saves the figure as an image
print("📊 Correlation graphs saved as 'correlation_graphs.png'")


# Print Results
print("\n📊 **Correlation Analysis**")
print(f"✅ **Strongest Correlation**: {strongest_pair[0]} vs. {strongest_pair[1]} (r = {strongest_correlation['Correlation']:.3f})")
print(f"❌ **Weakest Correlation**: {weakest_pair[0]} vs. {weakest_pair[1]} (r = {weakest_correlation['Correlation']:.3f})")

# Explanation
print("\n📌 **Analysis and Discussion**")
print(f"1️⃣ **Strongest Correlation** ({strongest_correlation['Correlation']:.3f}):")
print(f"   - {strongest_pair[0]} and {strongest_pair[1]} show a strong relationship.")
print(f"   - This means that as one increases, the other also tends to increase.")
print(f"   - Real-life example: If someone sleeps later, they also wake up later.")

print(f"\n2️⃣ **Weakest Correlation** ({weakest_correlation['Correlation']:.3f}):")
print(f"   - {weakest_pair[0]} and {weakest_pair[1]} show almost no correlation.")
print(f"   - This suggests that they do not influence each other directly.")
print(f"   - Real-life example: Smoking may not necessarily affect how consistent one's sleep schedule is.")

