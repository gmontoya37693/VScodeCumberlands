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
print("Part 1: Introduction and Statistical Problem Definition")
print("="*80)
print("""
In the context of business decision-making, understanding the relationships between variables is essential for forecasting, optimization, and strategic planning. One of the most widely used techniques for this purpose is linear regression analysis, which enables researchers to model the behavior of a dependent quantitative variable based on one or more explanatory variables (Favero & Belfiore, 2019, p. 436). These explanatory variables may be metric or dummy variables, and the goal is to estimate how they influence the dependent variable while satisfying certain assumptions and conditions.

For example, researchers may be interested in understanding how a company’s expenses vary with changes in production capacity or work hours, or how real estate prices are influenced by the number of bedrooms and floor space. In all these cases, the dependent variable is quantitative, making linear regression an appropriate modeling technique (Favero & Belfiore, 2019, p. 436).

The statistical problem addressed in this assignment involves estimating and interpreting a multiple linear regression model using the least squares method. While the model aims to explain how a set of independent variables affects a dependent variable, what stood out in the reading was the insight that a high R² value alone is not sufficient to validate the model or its predictive power. As Favero and Belfiore (2019) emphasize, additional statistical tests must be introduced to ensure the model is well-grounded. These include the F-test to assess the overall significance of the regression by testing the correlation between the dependent variable and its predictors, and t-tests for individual coefficients to evaluate the significance of each explanatory variable independently (pp. 438–439).

To better understand these concepts in practice, this assignment will briefly test them using a fresh dataset. The goal is to apply the least squares method and evaluate the model not only by its R² value but also by examining the statistical significance of the model and its predictors. Following the structured approach outlined in regression modeling literature (Emmert-Streib & Dehmer, 2019), the analysis will proceed by describing the statistical problem and then framing a strategy to solve it. This includes stating the hypothesis, preparing the data, selecting and fitting the model, evaluating its performance, interpreting the results, and drawing conclusions.
""")

print("="*80)
print("Part 2: Strategy to Frame the Problem")
print("="*80)
print("""
To explore the regression modeling process, I’m using a publicly available dataset from Kaggle titled Salary Dataset - Simple Linear Regression (Abhishek, 2020). The dataset includes three variables: age, years of experience, and income. The statistical problem I’m addressing is to determine how well age and experience predict income among individuals. Specifically, I want to quantify the relationship between the independent variables (age and experience) and the dependent variable (income) using a multiple linear regression model. This will help me understand the impact of each predictor on income and assess the overall fit of the model.

To do this, I’ll begin by stating the hypothesis and preparing the dataset. Then I’ll select and fit a multiple linear regression model using the least squares method. After fitting the model, I’ll evaluate its performance using R², adjusted R², the F-test for overall model significance, and t-tests for individual coefficients. Finally, I’ll interpret the results and draw conclusions based on the statistical evidence.
""")

print("="*80)
print("Step 1: State the Hypothesis")
print("="*80)
print("""
Null Hypothesis (H0): Age and experience do not significantly predict income.
Alternative Hypothesis (H1): At least one of age or experience significantly predicts income.
""")

print("="*80)
print("Step 2: Data Preparation")
print("="*80)
print("First five rows of the dataset:")
print(df.head())
print("\nColumn headers:", df.columns.tolist())
print("\nMissing values in each column:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe())

print("="*80)
print("Step 3: Model Fitting")
print("="*80)
print("Fitting a multiple linear regression model using the least squares method...")