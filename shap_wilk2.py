import pandas as pd
from scipy.stats import shapiro, levene
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

# Systematic sampling for the first sample
systematic_sample1 = prices.iloc[::k][:n].to_frame(name='Price (£)')
print(systematic_sample1.head())
print(f"Sample 1 size: {len(systematic_sample1)}")
# To export:
systematic_sample1.to_excel("systematic_sample1_prices.xlsx", index=False)
print()  # Print a blank line for spacing

# Systematic sampling for the second sample (offset by 1 to avoid overlap)
systematic_sample2 = prices.iloc[1::k][:n].to_frame(name='Price (£)')
print(systematic_sample2.head())
print(f"Sample 2 size: {len(systematic_sample2)}")
# To export:
systematic_sample2.to_excel("systematic_sample2_prices.xlsx", index=False)
print()  # Print a blank line for spacing

# Shapiro-Wilk test on the full dataset
stat_full, p_value_full = shapiro(prices)
print(f"Shapiro-Wilk Test (Full Data) Statistic: {stat_full}")
print(f"Shapiro-Wilk Test (Full Data) p-value: {p_value_full}")
print()  # Print a blank line for spacing

# Shapiro-Wilk test on the systematic samples
stat1, p_value1 = shapiro(systematic_sample1)
print(f"Shapiro-Wilk Test (Systematic Sample 1) Statistic: {stat1}")
print(f"Shapiro-Wilk Test (Systematic Sample 1) p-value: {p_value1}")
print()  # Print a blank line for spacing

stat2, p_value2 = shapiro(systematic_sample2)
print(f"Shapiro-Wilk Test (Systematic Sample 2) Statistic: {stat2}")
print(f"Shapiro-Wilk Test (Systematic Sample 2) p-value: {p_value2}")
print()  # Print a blank line for spacing

# Levene's test for homogeneity of variance between the two systematic samples
stat_levene, p_value_levene = levene(systematic_sample1.iloc[:, 0], systematic_sample2.iloc[:, 0])
print(f"Levene's Test Statistic (Sample 1 vs Sample 2): {stat_levene}")
print(f"Levene's Test p-value (Sample 1 vs Sample 2): {p_value_levene}")
print()  # Print a blank line for spacing
