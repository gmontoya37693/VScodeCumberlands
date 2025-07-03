import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns

# Correct file path based on the output
file_path = "london_houses.csv"

# Load the dataset 
df = pd.read_csv(file_path)
print("Dataset loaded successfully.")
print(df.head())

# Inspect column names to confirm the correct one
print(df.columns)
print()  # Print a blank line for spacing
space_size = len(df)
print(f"Total number of rows in the file: {space_size}")

# Use the correct column name 'Price (£)'
prices = df['Price (£)'].dropna()
print()  # Print a blank line for spacing

# Set your sample size
Conf_level = 0.95
z = 1.96  # For a 95% confidence level
std_price = df['Price (£)'].std()
E = 100000  # Margin of error in pounds
print(f"Standard Deviation of Price (£): {std_price}")
print(f"Margin of Error: £{E}")

# Calculate required sample size
n = int((z * std_price / E) ** 2)
print(f"Required sample size for ±£{E} margin of error: {n}")
k = int(round(space_size / n))
print(f"Sampling factor (k, rounded): {k}")
print()  # Print a blank line for spacing

# Systematic sampling
systematic_sample = prices.iloc[::k][:n]
print(systematic_sample.head())
print(f"Sample size: {len(systematic_sample)}")
# To export:
systematic_sample.to_frame().to_excel("systematic_sample_prices.xlsx", index=False)
print()  # Print a blank line for spacing

# Shapiro-Wilk test
stat, p_value = shapiro(systematic_sample)
print(f"Shapiro-Wilk Test Statistic: {stat}")
print(f"p-value: {p_value}")
print()  # Print a blank line for spacing
