# -----------------------------------
# Step 2: Data Preparation
# -----------------------------------
import pandas as pd

# Load the dataset
df = pd.read_csv('adult.csv')

print("-" * 60)
print("First 5 rows of the dataset:")
print(df.head())
print("-" * 60)

print("Dataset info:")
print(df.info())
print("-" * 60)

print("Missing values per column:")
print(df.isnull().sum())
print("-" * 60)

print("Number of unique values for each variable:")
print(df.nunique())
print("-" * 60)

print("Descriptive statistics for numeric variables:")
print(df.describe())
print("-" * 60)

# -----------------------------------
# Step 3: Correlation Analysis
# -----------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

df['income'] = df['income'].str.strip()
# Add income_binary for correlation analysis
df['income_binary'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Convert income_binary to numeric
df['income_binary'] = pd.to_numeric(df['income_binary'], errors='coerce')

# Select numeric columns (including income_binary)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_cols].corr()
print(corr_matrix)
print("-" * 60)

# Correlation Coefficient Strength Guide
print("Correlation Coefficient Strength Guide")
print("Absolute Value of r\tStrength\t\tMeaning")
print("0.00 – 0.10\t\tNegligible\t\tNo linear relationship.")
print("0.10 – 0.30\t\tWeak\t\t\tSmall tendency for variables to move together.")
print("0.30 – 0.50\t\tModerate\t\tNoticeable, but not strong, linear relationship.")
print("0.50 – 0.70\t\tStrong\t\t\tVariables tend to move together significantly.")
print("0.70 – 0.90\t\tVery Strong\t\tHigh predictability between variables.")
print("0.90 – 1.00\t\tExtremely Strong\tAlmost perfect linear relationship.")
print("-" * 60)

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.xlabel("Variables")
plt.ylabel("Variables")
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=10)               # Keep y-axis labels
plt.show()

print(df['income_binary'].dtype)
print(df['income_binary'].unique())
print(df['income_binary'].value_counts())
print("-" * 60)

print("-" * 60)
print("Summary Table: Correlation with income_binary")
print("{:<15} {:<30} {:<12} {}".format("Variable", "Correlation with income_binary", "Strength", "Interpretation"))
print("{:<15} {:<30} {:<12} {}".format("education-num", "0.34", "Moderate", "Higher education → higher income"))
print("{:<15} {:<30} {:<12} {}".format("hours-per-week", "0.23", "Weak/Moderate", "More hours → higher income"))
print("{:<15} {:<30} {:<12} {}".format("age", "0.23", "Weak/Moderate", "Older age → higher income"))
print("{:<15} {:<30} {:<12} {}".format("capital-gain", "0.22", "Weak", "Capital gains → higher income"))
print("{:<15} {:<30} {:<12} {}".format("capital-loss", "0.15", "Weak", "Capital losses → higher income"))
print("{:<15} {:<30} {:<12} {}".format("fnlwgt", "-0.01", "Negligible", "Not useful for income prediction"))
print("-" * 60)

# -----------------------------------
# Step 4: Frequency Distribution Analysis
# -----------------------------------

# Frequency distribution for 'education' (16 categories)
education_counts = df['education'].value_counts()
print("Frequency distribution for 'education':")
print(education_counts)
print("-" * 60)

# Bar plot for 'education' with Pareto line
education_counts = df['education'].value_counts().sort_values(ascending=False)
cumulative = education_counts.cumsum() / education_counts.sum() * 100

plt.figure(figsize=(10, 6))
ax = education_counts.plot(kind='bar', color='skyblue')
plt.title("Pareto Chart of Education Levels")
plt.xlabel("Education Level")
plt.ylabel("Count")
plt.xticks(rotation=45)

# Add Pareto line
ax2 = ax.twinx()
ax2.plot(cumulative.values, color='red', marker='o', linestyle='-')
ax2.set_ylabel('Cumulative %')

# Annotate Pareto values
for i, value in enumerate(cumulative.values):
    ax2.text(i, value, f"{value:.1f}%", color='red', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
print("-" * 60)

# -----------------------------------
# Step 5: Chi-Square Test for Independence
# -----------------------------------
from scipy.stats import chi2_contingency

# State hypotheses
print("Hypotheses for Chi-Square Test:")
print("H0: 'workclass' and 'education' are independent (no association).")
print("H1: 'workclass' and 'education' are associated (not independent).")
print("-" * 60)

# Create contingency table
contingency_table = pd.crosstab(df['workclass'], df['education'])

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Chi-square test between 'workclass' and 'education':")
print(f"Chi2 Statistic: {chi2:.2e}, p-value: {p:.2e}, Degrees of Freedom: {dof}")
print("-" * 60)

# Interpretation with hypothesis reference
if p < 0.05:
    print("Result: p-value < 0.05. Reject H0. There is a statistically significant association between 'workclass' and 'education'.")
else:
    print("Result: p-value >= 0.05. Fail to reject H0. No statistically significant association between 'workclass' and 'education'.")
print("-" * 60)

print("Contingency table for 'workclass' and 'education':")
print(contingency_table)
print("-" * 60)

plt.figure(figsize=(12, 6))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues")
plt.title("Contingency Table Heatmap: Workclass vs Education")
plt.xlabel("Education")
plt.ylabel("Workclass")
plt.tight_layout()
plt.show()
print("-" * 60)

# Contingency table as percentages
contingency_table_pct = contingency_table / contingency_table.values.sum() * 100
print("Contingency table for 'workclass' and 'education' (percentages):")
print(contingency_table_pct.round(2))
print("-" * 60)

# Heatmap of percentages
plt.figure(figsize=(12, 6))
sns.heatmap(contingency_table_pct, annot=True, fmt=".2f", cmap="Blues")
plt.title("Contingency Table Heatmap (%): Workclass vs Education")
plt.xlabel("Education")
plt.ylabel("Workclass")
plt.tight_layout()
plt.show()
print("-" * 60)

# -----------------------------------
# Step 6: Multiple Regression Model to Predict Income
# -----------------------------------
import statsmodels.api as sm

# Use only predictors with highest correlation to income_binary
X = df[['education-num', 'hours-per-week', 'age']]
X = sm.add_constant(X)
y = df['income_binary']

# Build logistic regression model
model = sm.Logit(y, X).fit()
print(model.summary())
print("-" * 60)

# Interpretation:
print("Interpretation:")
print("Significant predictors have p-values < 0.05. Positive coefficients indicate higher odds of earning >50K.")
print("-" * 60)

predictors = ['education-num', 'hours-per-week', 'age']
for col in predictors:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[col], df['income_binary'], alpha=0.3)
    plt.title(f"{col} vs. Income Binary")
    plt.xlabel(col)
    plt.ylabel("Income Binary (1 = >50K, 0 = <=50K)")
    plt.tight_layout()
    plt.show()

# Get predicted probabilities from the model
df['pred_prob'] = model.predict(X)

for col in predictors:
    plt.figure(figsize=(8, 5))
    plt.scatter(df[col], df['pred_prob'], alpha=0.3, color='green')
    plt.title(f"Predicted Probability vs. {col}")
    plt.xlabel(col)
    plt.ylabel("Predicted Probability of >50K Income")
    plt.tight_layout()
    plt.show()