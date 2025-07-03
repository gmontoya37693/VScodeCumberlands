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

# Use the correct column name 'Price (£)'
prices = df['Price (£)'].dropna()

# Set your sample size
Conf_level = 0.95
z = 1.96  # For a 95% confidence level
std_price = df['Price (£)'].std()
print(f"Standard Deviation of Price (£): {std_price}")

# Shapiro-Wilk test
stat, p_value = shapiro(prices.sample(n=500, random_state=1)) # sample to avoid memory issues
print(f"Shapiro-Wilk Test Statistic: {stat}")
print(f"p-value: {p_value}")
