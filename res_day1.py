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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

# --- 3D Scatter Plot with Regression Plane ---

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['age'], df['experience'], df['income'], color='blue', label='Data')

# Create grid to plot regression plane
age_range = np.linspace(df['age'].min(), df['age'].max(), 10)
exp_range = np.linspace(df['experience'].min(), df['experience'].max(), 10)
age_grid, exp_grid = np.meshgrid(age_range, exp_range)
income_pred = (model.intercept_ +
               model.coef_[0] * age_grid +
               model.coef_[1] * exp_grid)
ax.plot_surface(age_grid, exp_grid, income_pred, color='orange', alpha=0.5)

ax.set_xlabel('Age')
ax.set_ylabel('Experience')
ax.set_zlabel('Income')
ax.set_title('3D Scatter Plot of Age, Experience, and Income with Regression Plane')
plt.legend()
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

# --- Residual Plot ---
y_pred = model.predict(X)
residuals = y - y_pred

plt.figure(figsize=(8, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Predicted Income')
plt.ylabel('Residuals')
plt.title('Residual Plot: Predicted Income vs. Residuals')
plt.axhline(0, color='black', linestyle='--')
plt.show()