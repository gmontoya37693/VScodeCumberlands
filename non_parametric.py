import pandas as pd
from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import mannwhitneyu, wilcoxon

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

print("\n" + "-"*60 + "\n")

# Regenerating systematic samples with all columns from the original dataset
# 1. Load the full dataset
df = pd.read_csv('london_houses.csv')

# 3. Calculate total number of rows
space_size = len(df)
print(f"Total number of rows in the file: {space_size}")

# 4. Calculate required sample size and sampling interval (copied from shap_wilk2.py)
Conf_level = 0.95
z = 1.96  # For a 95% confidence level
std_price = df['Price (£)'].std()
E = 100000  # Margin of error in pounds
print(f"Standard Deviation of Price (£): {std_price}")
print(f"Margin of Error: £{E}")

n = int((z * std_price / E) ** 2)
print(f"Required sample size for ±£{E} margin of error: {n}")
k = int(round(space_size / n))
print(f"Sampling factor (k, rounded): {k}")

# Note: If systematic samples are to be regenerated, the previous samples should be dropped first.

# Simple matching: sort both samples by Bedrooms and pair in order
print("\nSimple matching: sorting both samples by Bedrooms and pairing in order...")

# Load systematic samples with all columns (assuming you have them as Excel or regenerate from df)
df1_full = df.iloc[::k][:n].reset_index(drop=True)
df2_full = df.iloc[1::k][:n].reset_index(drop=True)

# Sort by Bedrooms
sampleA = df1_full.sort_values('Bedrooms').reset_index(drop=True)
sampleB = df2_full.sort_values('Bedrooms').reset_index(drop=True)

# Paired prices for paired test
pairedA_prices = sampleA['Price (£)']
pairedB_prices = sampleB['Price (£)']

print(f"Number of paired samples: {len(pairedA_prices)}")

print("\nWilcoxon signed-rank test for paired samples (Price (£)):")
# Perform the Wilcoxon signed-rank test
wilcoxon_stat, wilcoxon_p = wilcoxon(pairedA_prices, pairedB_prices)
print(f"Wilcoxon statistic: {wilcoxon_stat}")
print(f"Wilcoxon p-value: {wilcoxon_p}")

if wilcoxon_p < 0.05:
    print("Result: The paired samples are significantly different (p < 0.05).")
else:
    print("Result: No significant difference between the paired samples (p >= 0.05).")

# Export the matched/paired samples to Excel for reference
sampleA.to_excel('paired_sampleA_bedrooms.xlsx', index=False)
sampleB.to_excel('paired_sampleB_bedrooms.xlsx', index=False)
print("Paired samples exported to 'paired_sampleA_bedrooms.xlsx' and 'paired_sampleB_bedrooms.xlsx'.")
