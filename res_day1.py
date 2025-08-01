"""
Residency Day 1: Linear Regression
Quick analysis of a dataset using linear regression
August 1, 2025
Germán Montoya
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('multiple_linear_regression_dataset.csv')

# --- Data Exploration and Visualization ---

# Show the first few rows
print("First five rows of the dataset:")
print(df.head())

# Show the column headers
print("\nColumn headers:")
print(df.columns.tolist())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Show data types
print("\nData types:")
print(df.dtypes)

# Show summary statistics
print("\nSummary statistics:")
print(df.describe())

# Pairplot for quick visualization of relationships
sns.pairplot(df)
plt.suptitle("Pairplot of Age, Experience, and Income", y=1.02)
plt.show()

# --- Regression Analysis ---

# Define independent variables (X) and dependent variable (y)
X = df[['age', 'experience']]
y = df['income']

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 score:", model.score(X, y))

# Add a constant (intercept) to the model
X_sm = sm.add_constant(X)

# Fit the model using statsmodels
model_sm = sm.OLS(y, X_sm).fit()

# Print the full summary
print(model_sm.summary())