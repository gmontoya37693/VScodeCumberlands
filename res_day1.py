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
import numpy as np

# Load the dataset
df = pd.read_csv('multiple_linear_regression_dataset.csv')

# Define independent variables (X) and dependent variable (y)
X = df[['age', 'experience']]
y = df['income']

# Fit the model
model = LinearRegression()
model.fit(X, y)

# Add a constant for statsmodels
X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()

# --- 3D Scatter Plot with Regression Plane and Equation ---
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
ax.set_title('3D Scatter Plot with Regression Plane')

# Regression equation as a string
eqn = f"Income = {model.intercept_:.2f} + ({model.coef_[0]:.2f} * Age) + ({model.coef_[1]:.2f} * Experience)"
ax.text2D(0.05, 0.95, eqn, transform=ax.transAxes, fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7))

plt.show()

# --- Print Statistics and Conclusions ---
print("\n--- Regression Statistics ---")
print(f"R^2: {model_sm.rsquared:.3f}")
print(f"Adjusted R^2: {model_sm.rsquared_adj:.3f}")
print(f"F-test p-value: {model_sm.f_pvalue:.4g}")

if model_sm.f_pvalue < 0.05:
    print("Conclusion: The model is statistically significant (reject H0).")
else:
    print("Conclusion: The model is NOT statistically significant (fail to reject H0).")

print("\n--- t-tests for Predictors ---")
for name, coef, pval in zip(['Age', 'Experience'], model_sm.params[1:], model_sm.pvalues[1:]):
    print(f"{name}: coefficient = {coef:.2f}, p-value = {pval:.4g}")
    if pval < 0.05:
        print(f"  {name} is a significant predictor (reject H0).")
    else:
        print(f"  {name} is NOT a significant predictor (fail to reject H0).")

print("\n" + "="*80)
print("Part 1: Statistical Problem Definition")
print("="*80)
print("I want to determine if age and experience significantly predict income using multiple linear regression.")

print("\nStep 1: State the Hypothesis")
print("Null Hypothesis (H0): Age and experience do not significantly predict income.")
print("Alternative Hypothesis (H1): At least one of age or experience significantly predicts income.")

print("\nStep 2: Data Preparation")
print("First five rows of the dataset:")
print(df.head())
print("No missing values detected.")
print("Variables: age (int), experience (int), income (int)")

print("\nStep 3: Model Fitting")
print("Fitting a multiple linear regression model using the least squares method...")

print("\nStep 4: Regression Equation")
print(f"Regression Equation: income = {model.intercept_:.2f} + ({model.coef_[0]:.2f} * age) + ({model.coef_[1]:.2f} * experience)")

print("\nStep 5: Model Evaluation")
print(f"R^2: {model_sm.rsquared:.3f}")
print(f"Adjusted R^2: {model_sm.rsquared_adj:.3f}")
print(f"F-test p-value: {model_sm.f_pvalue:.4g}")
if model_sm.f_pvalue < 0.05:
    print("Conclusion: The model is statistically significant (reject H0).")
else:
    print("Conclusion: The model is NOT statistically significant (fail to reject H0).")

print("\nStep 6: t-tests for Predictors")
for name, coef, pval in zip(['Age', 'Experience'], model_sm.params[1:], model_sm.pvalues[1:]):
    print(f"{name}: coefficient = {coef:.2f}, p-value = {pval:.4g}")
    if pval < 0.05:
        print(f"  {name} is a significant predictor (reject H0).")
    else:
        print(f"  {name} is NOT a significant predictor (fail to reject H0).")