import pandas as pd

# Sample dataset
data = {
    'Height': [165, 170, 175, 180, 185],
    'Weight': [55, 65, 75, 85, 95],
    'Age': [22, 25, 30, 35, 40],
    'Salary': [30000, 40000, 50000, 60000, 70000]
}

df = pd.DataFrame(data)

# Compute the Correlation Matrix
# 💡 Correlation tells us how things are related!
#
# Do taller people also weigh more?
#
# Do older people have higher salaries?
#
#  This line checks all possible relationships between height, weight, age, and salary.

corr_matrix = df.corr()



# Convert matrix to long format and remove self-correlations
#💡 We are organizing the relationships into a list!
#
# Before: Correlation is stored in a matrix (like a square table).
#
# After: We unstack it to turn it into a simple list of relationships.
#
# We also remove self-correlations (Height with Height is always 1.0, which is useless).

corr_pairs = corr_matrix.unstack().reset_index()
corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']
corr_pairs = corr_pairs[corr_pairs['Variable 1'] != corr_pairs['Variable 2']]




# Remove Duplicate Pairs
#
# If we compare Height & Weight, we don’t need to compare Weight & Height again.
#
# This step keeps only one version of each pair.

corr_pairs['Sorted Pair'] = corr_pairs.apply(lambda row: tuple(sorted([row['Variable 1'], row['Variable 2']])), axis=1)
corr_pairs = corr_pairs.drop_duplicates(subset=['Sorted Pair']).drop(columns=['Sorted Pair'])



# Sort the Relationships from Strongest to Weakest
#
# 💡 Sorting from most important to least important!
#
# The strongest relationships (close to 1.0 or -1.0) come first.
#
# The weakest relationships (close to 0.0) come last.
#
# .reset_index(drop=True) makes sure the numbering is clean and ordered.

corr_pairs = corr_pairs.sort_values(by='Correlation', ascending=False, key=abs).reset_index(drop=True)



# Display result
print(corr_pairs)
