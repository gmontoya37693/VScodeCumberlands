# -----------------------------------
# Step 1: Introduction
# This script analyzes the Adult Income dataset using Python.
# It covers correlation analysis, frequency distribution, chi-square test, and multiple regression.
# -----------------------------------

# Step 2: Data Preparation
import pandas as pd

# Load the dataset
df = pd.read_csv('adult.csv')

# Display basic info
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nNumber of unique values for each variable:")
print(df.nunique())

# -----------------------------------
# Step 3: Correlation Analysis
# We will select numeric columns and compute their correlation matrix.
# -----------------------------------

# Select numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_cols].corr()
print("\nCorrelation matrix for numeric variables:")
print(corr_matrix)

# Interpretation:
# Values close to +1 or -1 indicate strong relationships.
# Values near 0 indicate weak or no relationship.

# -----------------------------------
# Step 4: Frequency Distribution Analysis
# We will analyze the frequency distribution of the 'education' variable.
# -----------------------------------

education_counts = df['education'].value_counts()
print("\nFrequency distribution for 'education':")
print(education_counts)

# Interpretation:
# The most common education level will have the highest count.
# The least common will have the lowest.

# -----------------------------------
# Step 5: Chi-Square Test for Independence
# We will test the relationship between 'workclass' and 'education'.
# -----------------------------------

from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(df['workclass'], df['education'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("\nChi-square test between 'workclass' and 'education':")
print(f"Chi2 Statistic: {chi2}, p-value: {p}")

# Interpretation:
# If p-value < 0.05, 'workclass' and 'education' are related.
# If p-value > 0.05, there is no evidence of a relationship.

# -----------------------------------
# Step 6: Multiple Regression Model to Predict Income
# We will use logistic regression to predict if income is >50K.
# -----------------------------------

import statsmodels.api as sm

# Convert income to binary
df['income_binary'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Select predictors
X = df[['age', 'education-num', 'hours-per-week']]
X = sm.add_constant(X)
y = df['income_binary']

model = sm.Logit(y, X).fit()
print("\nMultiple regression model summary:")
print(model.summary())

# Interpretation:
# Significant predictors have p-values < 0.05.
# Positive coefficients increase the odds of earning >50K.

# -----------------------------------
# Step 7: Conclusions and Insights
# Summarize findings from all analyses here (add your own interpretation).