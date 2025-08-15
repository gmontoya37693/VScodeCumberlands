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
print("-" * 60)
print("First 5 rows of the dataset:")
print(df.head())
print("-" * 60)
print("\nDataset info:")
print(df.info())
print("-" * 60)
print("\nMissing values per column:")
print(df.isnull().sum())
print("-" * 60)
print("\nNumber of unique values for each variable:")
print(df.nunique())
print("-" * 60)
print("\nDescriptive statistics for numeric variables:")
print(df.describe())
print("-" * 60)

# -----------------------------------
# Step 3: Correlation Analysis
# We will select numeric columns and compute their correlation matrix.
# -----------------------------------

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_cols].corr()
print("\nCorrelation matrix for numeric variables:")
print(corr_matrix)
print("-" * 60)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# -----------------------------------
# Step 4: Frequency Distribution Analysis
# We will analyze the frequency distribution of the 'education' variable.
# -----------------------------------

print("-" * 60)
education_counts = df['education'].value_counts()
print("\nFrequency distribution for 'education':")
print(education_counts)
print("-" * 60)

# -----------------------------------
# Step 5: Chi-Square Test for Independence
# We will test the relationship between 'workclass' and 'education'.
# -----------------------------------

from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(df['workclass'], df['education'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("\nChi-square test between 'workclass' and 'education':")
print(f"Chi2 Statistic: {chi2}, p-value: {p}")
print("-" * 60)

# -----------------------------------
# Step 6: Multiple Regression Model to Predict Income
# We will use logistic regression to predict if income is >50K.
# -----------------------------------

import statsmodels.api as sm

df['income_binary'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
X = df[['age', 'education-num', 'hours-per-week']]
X = sm.add_constant(X)
y = df['income_binary']

model = sm.Logit(y, X).fit()
print("\nMultiple regression model summary:")
print(model.summary())
print("-" * 60)

# -----------------------------------
# Step 7: Conclusions and Insights
# Summarize findings from all analyses here (add your own interpretation).
# -----------------------------------