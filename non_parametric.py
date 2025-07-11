import pandas as pd
from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import mannwhitneyu

# Hypotheses for the one-sample Sign Test
print("One-sample Sign Test Hypotheses:")
print("H0 (Null Hypothesis): The median of the sample (systematic_sample1) is equal to the population median.")
print("H1 (Alternative Hypothesis): The median of the sample (systematic_sample1) is different from the population median.\n")

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

print("\n" + "-"*60 + "\n")

# Hypotheses for the Mann-Whitney U Test
print("Mann-Whitney U Test Hypotheses:")
print("H0 (Null Hypothesis): The distributions of the two samples (systematic_sample1 and systematic_sample2) are equal.")
print("H1 (Alternative Hypothesis): The distributions of the two samples are different.\n")

# Load the second systematic sample
df2 = pd.read_excel('systematic_sample2_prices.xlsx')
sample2_prices = df2['Price (£)']

# Perform the Mann-Whitney U test
tu_stat, tu_p_value = mannwhitneyu(sample1_prices, sample2_prices, alternative='two-sided')
print(f"Mann-Whitney U statistic: {tu_stat}")
print(f"Mann-Whitney U p-value: {tu_p_value}")

if tu_p_value < 0.05:
    print("Result: The distributions of the two samples are significantly different (p < 0.05).")
else:
    print("Result: No significant difference between the distributions of the two samples (p >= 0.05).")
