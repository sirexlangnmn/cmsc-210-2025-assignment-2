import pandas as pd 

# Load the CSV file (Assuming the file is named 'data.csv')
# df = pd.read_csv('SolisPotolin_SleepQualityAndHabits.csv')


df = pd.read_json('csvjson_part1.json')


print('df =>> ', df)
# print('df.head() =>> ', df.head())  # Show first few rows of the dataset
# print('df.dtypes =>> ', df.dtypes)  # Check the data types of each column


# Select only numeric columns
df_numeric = df.select_dtypes(include=['number'])

# Compute the correlation matrix
corr_matrix = df_numeric.corr()

# Convert matrix to long format and remove self-correlations
corr_pairs = corr_matrix.unstack().reset_index()
corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']
corr_pairs = corr_pairs[corr_pairs['Variable 1'] != corr_pairs['Variable 2']]

# Remove duplicate pairs (e.g., keep only 'Height & Weight', remove 'Weight & Height')
corr_pairs['Sorted Pair'] = corr_pairs.apply(lambda row: tuple(sorted([row['Variable 1'], row['Variable 2']])), axis=1)
corr_pairs = corr_pairs.drop_duplicates(subset=['Sorted Pair']).drop(columns=['Sorted Pair'])

# Sort from strongest to weakest correlation
corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False, key=abs).reset_index(drop=True)

# Display result
print(corr_pairs)
