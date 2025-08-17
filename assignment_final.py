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
print()  

print("Skewness of numeric columns:")  # Assess asymmetry of numeric distributions
print(df.skew(numeric_only=True))

print("Kurtosis of numeric columns:")  # Assess peakedness of numeric distributions
print(df.kurtosis(numeric_only=True))
print()
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

# Rows with all missing values
print("Rows with all missing values:", df.isnull().all(axis=1).sum())

# Check for duplicated Adv_IDs
print("Duplicated Adv_IDs:", df['Adv_ID'].duplicated().sum())

# Non-numeric values in numeric columns
print("Non-numeric Price values:")
print(df[~df['Price'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]['Price'].unique())
print()

# Non-standard or unexpected values in categorical columns
print("Unexpected Fuel_type values:")
print(df['Fuel_type'].value_counts())

# Tabulate min/max values for key numeric columns for better readability
min_max_data = {
    'Column': ['Adv_year', 'Reg_year', 'Price', 'Seat_num', 'Door_num'],
    'Min': [
        df['Adv_year'].min(),
        df['Reg_year'].min(),
        pd.to_numeric(df['Price'], errors='coerce').min(),
        df['Seat_num'].min(),
        df['Door_num'].min()
    ],
    'Max': [
        df['Adv_year'].max(),
        df['Reg_year'].max(),
        pd.to_numeric(df['Price'], errors='coerce').max(),
        df['Seat_num'].max(),
        df['Door_num'].max()
    ]
}

min_max_df = pd.DataFrame(min_max_data)
print("\nMin/Max values for key numeric columns:")
print(min_max_df.to_string(index=False))
print()

# -----------------------------------
# Step 1c: Data Integrity Confirmation and Next Steps
# -----------------------------------
"""
Data Integrity Confirmation:
- No duplicate rows found in the dataset.
- No duplicated Adv_IDs; each advertisement is unique.
"""

# -----------------------------------
# Step 2: Filtering by Assignment Requirements
# -----------------------------------
"""
Filtering will be performed according to assignment instructions:
- Only include vehicle models: L200, XC90, Sorento, Outlander
- Only include body types: SUV, Pickup
- Only include fuel type: Diesel
- Only include the six most frequently advertised colors among filtered vehicles
"""

# Filter by vehicle models
models_required = ['L200', 'XC90', 'Sorento', 'Outlander']
df_filtered = df[df['Genmodel'].isin(models_required)]
print(f"Observations after model filter: {df_filtered.shape[0]}")

# Filter by body types
bodytypes_required = ['SUV', 'Pickup']
df_filtered = df_filtered[df_filtered['Bodytype'].isin(bodytypes_required)]
print(f"Observations after body type filter: {df_filtered.shape[0]}")

# Filter by fuel type
df_filtered = df_filtered[df_filtered['Fuel_type'] == 'Diesel']
print(f"Observations after fuel type filter: {df_filtered.shape[0]}")

# Filter by top 6 colors among filtered vehicles
top6_colors = df_filtered['Color'].value_counts().nlargest(6).index.tolist()
df_filtered = df_filtered[df_filtered['Color'].isin(top6_colors)]
print(f"Observations after color filter: {df_filtered.shape[0]}")

# Final filtered dataset
print("\nFinal number of observations after all filters:")
print(df_filtered.shape[0])
print("Filtered columns:", df_filtered.columns.tolist())
print("Filtered unique models:", df_filtered['Genmodel'].unique())
print("Filtered unique body types:", df_filtered['Bodytype'].unique())
print("Filtered unique fuel types:", df_filtered['Fuel_type'].unique())
print("Filtered top 6 colors:", top6_colors)
print()

# -----------------------------------
# Step 2 Results Summary
# -----------------------------------
"""
Step 2 Results Summary:
- Observations after model filter (L200, XC90, Sorento, Outlander): 3091
- Observations after body type filter (SUV, Pickup): 3014
- Observations after fuel type filter (Diesel): 2604
- Observations after color filter (top 6 colors): 2380

Final filtered dataset:
- Number of observations: 2380
- Columns retained: Maker, Genmodel, Genmodel_ID, Adv_ID, Adv_year, Adv_month, Color, Reg_year, Bodytype, Runned_Miles, Engin_size, Gearbox, Fuel_type, Price, Seat_num, Door_num
- Unique models: Sorento, L200, Outlander, XC90
- Unique body types: SUV, Pickup
- Unique fuel type: Diesel
- Top 6 colors: Black, Silver, Grey, White, Blue, Red
"""

