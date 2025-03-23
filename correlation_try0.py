import pandas as pd

# Sample dataset
data = {
    'Height': [165, 170, 175, 180, 185],
    'Weight': [55, 65, 75, 85, 95],
    'Age': [22, 25, 30, 35, 40],
    'Salary': [30000, 40000, 50000, 60000, 70000]
}

df = pd.DataFrame(data)

# Compute correlation matrix
corr_matrix = df.corr()

# Convert matrix to long format and remove self-correlations
corr_pairs = corr_matrix.unstack().reset_index()
corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']
corr_pairs = corr_pairs[corr_pairs['Variable 1'] != corr_pairs['Variable 2']]


# Remove duplicate pairs
# corr_pairs = corr_pairs.drop_duplicates(subset=['Correlation'], keep='first')


# Remove duplicate pairs correctly (considering variable names)
corr_pairs['Sorted Pair'] = corr_pairs.apply(lambda row: tuple(sorted([row['Variable 1'], row['Variable 2']])), axis=1)
corr_pairs = corr_pairs.drop_duplicates(subset=['Sorted Pair']).drop(columns=['Sorted Pair'])


# Sort by absolute correlation value (strongest to weakest)
# corr_pairs = corr_pairs.reindex(corr_pairs['Correlation'].abs().sort_values(ascending=False).index)
# corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False, key=abs)
corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False, key=abs).reset_index(drop=True)



# Display result
print(corr_pairs)
