import pandas as pd
from statsmodels.stats.descriptivestats import sign_test

# Load the systematic sample 1 and the full dataset
df1 = pd.read_excel('systematic_sample1_prices.xlsx')
df_fp = pd.read_csv('london_houses.csv')

# Extract the price column (adjust if needed)
sample1_prices = df1['Price (£)']
full_prices = df_fp['Price (£)']

# Calculate the population median
population_median = full_prices.median()
print(f"Population median: £{population_median:,.2f}")

# Perform the one-sample Sign Test
stat, p_value = sign_test(sample1_prices, mu0=population_median)
print(f"Sign Test statistic: {stat}")
print(f"Sign Test p-value: {p_value}")

if p_value < 0.05:
    print("Result: The sample median is significantly different from the population median (p < 0.05).")
else:
    print("Result: No significant difference between the sample median and the population median (p >= 0.05).")
