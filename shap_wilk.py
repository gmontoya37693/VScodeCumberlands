import pandas as pd
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns

# Correct file path based on the output
file_path = "/kaggle/input/houses-in-london/london_houses.csv"

# Load the dataset 
df = pd.read_csv(file_path)
print("Dataset loaded successfully.")
print(df.head())

# Inspect column names to confirm the correct one
print(df.columns)

# Use the correct column name 'Price (£)'
prices = df['Price (£)'].dropna()

# Shapiro-Wilk test
stat, p_value = shapiro(prices.sample(n=500, random_state=1)) # sample to avoid memory issues
