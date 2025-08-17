"""
Final assignment: Identify most influential attributes for 
predicting advertised vehicle price by regression.
August 17, 2025
German Montoya
Data Source: Adapted from Huang et al. (2021)
"""

# -----------------------------------
# Step 1: Import Libraries and Data
# -----------------------------------

import pandas as pd
import numpy as np

# Import RData file
import pyreadr

# Load the RData file
result = pyreadr.read_r('car_ads_fp.RData')
df = result[list(result.keys())[0]]

# Inspect the first few rows and columns 
print(df.shape)     # Check the number of rows and columns in the dataset
print(df.columns)   # List all column names
print(df.head())    # Show the first 5 rows for a quick look at the data
print(df.info())    # Display data types and non-null counts for each column
print(df.describe()) # Statistical summary of numeric columns (mean, std, min, max, etc.)

print()

print("Missing values per column:")  # Show count of missing values for
print(df.isnull().sum())

print("Skewness of numeric columns:")  # Assess asymmetry of numeric distributions
print(df.skew(numeric_only=True))

print("Kurtosis of numeric columns:")  # Assess peakedness of numeric distributions
print(df.kurtosis(numeric_only=True))

# -----------------------------------
# Step 1a: Dive deeper into data understanding
# -----------------------------------

# Show number of unique values and top 10 most frequent for Genmodel
print(f"\nNumber of unique vehicle models (Genmodel): {df['Genmodel'].nunique()}")
print("Top 10 most frequent vehicle models:")
print(df['Genmodel'].value_counts().head(10))
print()

# Show number of unique values and top 10 most frequent for Bodytype
print(f"\nNumber of unique body types (Bodytype): {df['Bodytype'].nunique()}")
print("Top 10 most frequent body types:")
print(df['Bodytype'].value_counts().head(10))
print()

# Show number of unique values and top 10 most frequent for Fuel_type
print(f"\nNumber of unique fuel types (Fuel_type): {df['Fuel_type'].nunique()}")
print("Top 10 most frequent fuel types:")
print(df['Fuel_type'].value_counts().head(10))
print()

# Show number of unique values and top 10 most frequent for Color
print(f"\nNumber of unique colors (Color): {df['Color'].nunique()}")
print("Top 10 most frequent colors:")
print(df['Color'].value_counts().head(10))
print()

# List all unique colors in the dataset
print("\nFull list of unique colors:")
print(df['Color'].dropna().unique())
print()

# Check for duplicate rows in the dataset
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# -----------------------------------
# Step 1b: Confirmation of Initial Findings
# -----------------------------------
"""
Confirmation of Initial Findings:
- The dataset contains 268,255 rows and 16 columns.
- Key columns include: Maker, Genmodel, Genmodel_ID, Adv_ID, 
  Adv_year, Adv_month, Color, Reg_year, Bodytype, Runned_Miles, 
  Engin_size, Gearbox, Fuel_type, Price, Seat_num, Door_num.
- Several columns have missing values, notably Color, Bodytype, 
  Engin_size, Fuel_type, Seat_num, and Door_num.
- Numeric columns show varying degrees of skewness and kurtosis.
- There are 896 unique vehicle models, 18 unique body types, 13 unique 
  fuel types, and 22 unique colors.
- The most frequent models, body types, fuel types, and colors have been 
  identified.
- No duplicate rows are present in the dataset.
"""

