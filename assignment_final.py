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

# -----------------------------------
# Step 3: Inspection of Filtered Dataset
# -----------------------------------

# 1. Check for missing values in each column
print("Missing values per column in filtered dataset:")
print(df_filtered.isnull().sum())
print()

# 2. Check for non-numeric Price values
print("Non-numeric Price values in filtered dataset:")
print(df_filtered[~df_filtered['Price'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]['Price'].unique())
print()

# 3. Tabulate min/max values for key numeric columns
min_max_data = {
    'Column': ['Adv_year', 'Reg_year', 'Price', 'Seat_num', 'Door_num', 'Runned_Miles'],
    'Min': [
        df_filtered['Adv_year'].min(),
        df_filtered['Reg_year'].min(),
        pd.to_numeric(df_filtered['Price'], errors='coerce').min(),
        df_filtered['Seat_num'].min(),
        df_filtered['Door_num'].min(),
        pd.to_numeric(df_filtered['Runned_Miles'], errors='coerce').min()
    ],
    'Max': [
        df_filtered['Adv_year'].max(),
        df_filtered['Reg_year'].max(),
        pd.to_numeric(df_filtered['Price'], errors='coerce').max(),
        df_filtered['Seat_num'].max(),
        df_filtered['Door_num'].max(),
        pd.to_numeric(df_filtered['Runned_Miles'], errors='coerce').max()
    ]
}
min_max_df = pd.DataFrame(min_max_data)
print("Min/Max values for key numeric columns in filtered dataset:")
print(min_max_df.to_string(index=False))
print()

# 4. Check for outliers using IQR method
numeric_cols = ['Price', 'Reg_year', 'Seat_num', 'Door_num', 'Runned_Miles', 'Adv_year']
for col in numeric_cols:
    col_data = pd.to_numeric(df_filtered[col], errors='coerce')
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_filtered[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
    print(f"{col}: {outliers.shape[0]} outliers")
print()

# 5. Check for values outside expected ranges (example thresholds)
print("Reg_year < 1980:")
print(df_filtered[df_filtered['Reg_year'] < 1980])
print()
print("Seat_num > 8:")
print(df_filtered[df_filtered['Seat_num'] > 8])
print()
print("Price < 500:")
print(df_filtered[pd.to_numeric(df_filtered['Price'], errors='coerce') < 500])
print()

# 6. Check for duplicates in Adv_ID
print("Duplicated Adv_IDs in filtered dataset:", df_filtered['Adv_ID'].duplicated().sum())
print()

# -----------------------------------
# Step 3: Convert Numeric Columns to Numeric Types
# -----------------------------------
"""
After filtering, convert all columns that will be used as numeric to numeric types.
This ensures all subsequent analysis (imputation, outlier detection, etc.) is accurate.
"""
numeric_columns = ['Adv_year', 'Reg_year', 'Price', 'Seat_num', 'Door_num', 'Runned_Miles', 'Engin_size']
df_filtered['Engin_size'] = df_filtered['Engin_size'].str.replace('L', '', regex=False)
for col in numeric_columns:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

# -----------------------------------
# Step 3b: Handling Missing Values in Numeric Variables
# -----------------------------------
"""
Missing values in numeric variables:
- Engin_size: 2 missing
- Seat_num: 114 missing
- Door_num: 17 missing

Impute missing values in Engin_size, Seat_num, and Door_num using the median value for each variable.
Verified that all missing values in these numeric columns have been filled.
The dataset is now complete for these variables and ready for further outlier analysis and modeling.
"""
for column in ['Engin_size', 'Seat_num', 'Door_num']:
    median_value = df_filtered[column].median()
    df_filtered[column] = df_filtered[column].fillna(median_value)
    print(f"Imputed missing values in {column} with median: {median_value}")
print("Missing values in numeric variables after imputation:")
print(df_filtered[['Engin_size', 'Seat_num', 'Door_num']].isnull().sum())
print()

# -----------------------------------
# Step 3c: Outlier Detection and Visualization for Runned_Miles (Before Capping)
# -----------------------------------
"""
Detect outliers in Runned_Miles using the IQR method and visualize them.
- Outliers above Q3 + 1.5*IQR and below Q1 - 1.5*IQR are highlighted in red.
- This plot documents the reason for capping or correcting outliers.
"""
df_filtered['Runned_Miles'] = pd.to_numeric(df_filtered['Runned_Miles'], errors='coerce')
Q1 = df_filtered['Runned_Miles'].quantile(0.25)
Q3 = df_filtered['Runned_Miles'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

import matplotlib.pyplot as plt
reg_year = df_filtered['Reg_year'].astype(int)
runned_miles = df_filtered['Runned_Miles']
outlier_mask = (runned_miles < lower_bound) | (runned_miles > upper_bound)
normal_mask = ~outlier_mask

plt.figure(figsize=(10,6))
plt.scatter(reg_year[normal_mask], runned_miles[normal_mask], alpha=0.5, label='Normal', color='blue')
plt.scatter(reg_year[outlier_mask], runned_miles[outlier_mask], alpha=0.7, label='Outlier', color='red')
plt.xlabel('Reg_year')
plt.ylabel('Runned_Miles')
plt.title('Scatter Plot of Runned_Miles vs Reg_year (Outliers Highlighted)')
plt.legend()
plt.grid(True)
years_full = list(range(reg_year.min(), reg_year.max() + 1))
plt.xticks(years_full, [str(year) for year in years_full], rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------
# Step 3d: Outlier Capping and Correction for Runned_Miles
# -----------------------------------
"""
Cap outliers in Runned_Miles:
- Outliers above Q3 + 1.5*IQR are capped to the upper bound.
- Outliers below Q1 - 1.5*IQR (including negatives) are capped to the lower bound.
Replace negative Runned_Miles values with the median for the same Reg_year.
"""
df_filtered.loc[df_filtered['Runned_Miles'] > upper_bound, 'Runned_Miles'] = upper_bound
df_filtered.loc[df_filtered['Runned_Miles'] < lower_bound, 'Runned_Miles'] = lower_bound

print("Max Runned_Miles after capping:", df_filtered['Runned_Miles'].max())
print("Min Runned_Miles after capping:", df_filtered['Runned_Miles'].min())
print()

# Replace negative Runned_Miles with median for the same Reg_year
neg_miles_mask = df_filtered['Runned_Miles'] < 0
for reg_year in df_filtered.loc[neg_miles_mask, 'Reg_year'].unique():
    median_miles = df_filtered.loc[
        (df_filtered['Reg_year'] == reg_year) & (df_filtered['Runned_Miles'] >= 0),
        'Runned_Miles'
    ].median()
    df_filtered.loc[
        (df_filtered['Reg_year'] == reg_year) & (df_filtered['Runned_Miles'] < 0),
        'Runned_Miles'
    ] = median_miles

# Verify correction
print("Negative Runned_Miles after replacement:", (df_filtered['Runned_Miles'] < 0).sum())
print()

# -----------------------------------
# Step 3: Outlier Detection for Price (Do Not Cap Yet)
# -----------------------------------
"""
Detect outliers in Price using the IQR method.
- Outliers above Q3 + 1.5*IQR and below Q1 - 1.5*IQR are identified.
- Outliers are not capped yet; decision pending further analysis.
"""

# Calculate IQR bounds for Price
Q1 = df_filtered['Price'].quantile(0.25)
Q3 = df_filtered['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Extract outlier rows
price_outliers = df_filtered[(df_filtered['Price'] < lower_bound) | (df_filtered['Price'] > upper_bound)]

# Inspect the outliers (summary only)
print(f"Total price outliers: {price_outliers.shape[0]}")
print()

# -----------------------------------
# Step 3d: Visualize Price Outliers (Before Any Capping)
# -----------------------------------
"""
Scatter Plot Analysis for Price:
- The scatter plot below shows Price vs Reg_year, with outliers highlighted in red.
- This visualization helps assess whether price outliers are concentrated in specific years or models.
- If outliers are clustered for a single model/year, it may indicate a market anomaly or rare case.
- If outliers are spread across models/years, further investigation is needed.
"""

import matplotlib.pyplot as plt

reg_year = df_filtered['Reg_year'].astype(int)
price = df_filtered['Price']

outlier_mask = (price < lower_bound) | (price > upper_bound)
normal_mask = ~outlier_mask

plt.figure(figsize=(10,6))
plt.scatter(reg_year[normal_mask], price[normal_mask], alpha=0.5, label='Normal', color='blue')
plt.scatter(reg_year[outlier_mask], price[outlier_mask], alpha=0.7, label='Outlier', color='red')
plt.xlabel('Reg_year')
plt.ylabel('Price')
plt.title('Scatter Plot of Price vs Reg_year (Outliers Highlighted)')
plt.legend()
plt.grid(True)
years_full = list(range(reg_year.min(), reg_year.max() + 1))
plt.xticks(years_full, [str(year) for year in years_full], rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------
# Step 3e: Visualize Median Price Trends by Genmodel and Reg_year
# -----------------------------------
"""
Trend Analysis:
- Plot the median Price by Reg_year for each Genmodel.
- This visualization helps determine if price growth is unique to XC90 or seen in other models.
- If only XC90 shows a sharp increase, it supports the market anomaly conclusion.
"""

price_trend = df_filtered.groupby(['Genmodel', 'Reg_year'])['Price'].median().reset_index()

plt.figure(figsize=(10,6))
for model in df_filtered['Genmodel'].unique():
    subset = price_trend[price_trend['Genmodel'] == model]
    plt.plot(subset['Reg_year'], subset['Price'], marker='o', label=model)
plt.xlabel('Reg_year')
plt.ylabel('Median Price')
plt.title('Median Price Growth by Genmodel')
plt.legend()
plt.grid(True)
# Remove decimals from x-axis (Reg_year)
years_full = sorted(df_filtered['Reg_year'].dropna().astype(int).unique())
plt.xticks(years_full, years_full, rotation=45)  # Use integer years directly
plt.tight_layout()
plt.show()

# -----------------------------------
# Step 3f: Visualize Reg_year Outliers and Print All Columns for Outlier Row(s)
# -----------------------------------
"""
Visualize Reg_year outliers to assess if they are data errors or rare cases.
Print all columns for any Reg_year outlier row(s) for further inspection.
"""

reg_year_data = df_filtered['Reg_year']
Q1 = reg_year_data.quantile(0.25)
Q3 = reg_year_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outlier_mask = (reg_year_data < lower_bound) | (reg_year_data > upper_bound)
normal_mask = ~outlier_mask

plt.figure(figsize=(8,4))
plt.scatter(range(len(reg_year_data[normal_mask])), reg_year_data[normal_mask], alpha=0.5, label='Normal', color='blue')
plt.scatter(np.where(outlier_mask)[0], reg_year_data[outlier_mask], alpha=0.7, label='Outlier', color='red')
plt.xlabel('Index')
plt.ylabel('Reg_year')
plt.title('Reg_year Outliers Highlighted')
plt.legend()
plt.grid(True)
plt.yticks(sorted(df_filtered['Reg_year'].dropna().astype(int).unique()), 
           [str(int(year)) for year in sorted(df_filtered['Reg_year'].dropna().astype(int).unique())])
plt.tight_layout()
plt.show()

# Print all columns for Reg_year outlier row(s)
outlier_rows = df_filtered[outlier_mask]
print("Reg_year outlier row(s) (all columns):")
for idx, row in outlier_rows.iterrows():
    print(f"\nReg_year outlier at index {idx}:")
    for col in outlier_rows.columns:
        print(f"{col}: {row[col]}")
# -----------------------------------
# Step 3g: Reg_year Outlier Decision Documentation
# -----------------------------------
"""
Reg_year Outlier Decision:
- The only Reg_year outlier in the filtered dataset is 2001 (Mitsubishi L200).
- This value is valid and plausible for a vehicle registration year.
- It is flagged as an outlier only because most vehicles in the filtered data are newer.
- Decision: KEEP the 2001 Reg_year outlier for modeling and analysis.
- This preserves all available information and avoids unnecessary bias.
"""

# -----------------------------------
# Step 3h: Visualize Door_num Outliers
# -----------------------------------
"""
Visualize outliers for Door_num.
- Outliers are detected using the IQR method.
- Only the graph is shown; no printing of outlier rows.
"""

door_num_data = df_filtered['Door_num']
Q1 = door_num_data.quantile(0.25)
Q3 = door_num_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

door_outlier_mask = (door_num_data < lower_bound) | (door_num_data > upper_bound)
door_normal_mask = ~door_outlier_mask

plt.figure(figsize=(8,4))
plt.scatter(range(len(door_num_data[door_normal_mask])), door_num_data[door_normal_mask], alpha=0.5, label='Normal', color='blue')
plt.scatter(np.where(door_outlier_mask)[0], door_num_data[door_outlier_mask], alpha=0.7, label='Outlier', color='red')
plt.xlabel('Index')
plt.ylabel('Door_num')
plt.title('Door_num Outliers Highlighted')
plt.legend()
plt.grid(True)
plt.yticks(sorted(df_filtered['Door_num'].dropna().astype(int).unique()), 
           [str(int(val)) for val in sorted(df_filtered['Door_num'].dropna().astype(int).unique())])
plt.tight_layout()
plt.show()

# After Door_num outlier graph
print("\nDoor_num outlier Genmodel/Bodytype summary:")
print(df_filtered[door_outlier_mask][['Genmodel', 'Bodytype', 'Door_num']].value_counts())

# Tabulate Door_num outlier summary for better readability
print("\nDoor_num outlier Genmodel/Bodytype summary (tabulated):")
door_outlier_summary = df_filtered[door_outlier_mask][['Genmodel', 'Bodytype', 'Door_num']].value_counts().reset_index(name='Count')
print(door_outlier_summary.to_string(index=False))

# -----------------------------------
# Step 3i: Visualize Adv_year Outliers
# -----------------------------------
"""
Visualize outliers for Adv_year.
- Outliers are detected using the IQR method.
- Only the graph is shown; no printing of outlier rows.
"""

adv_year_data = df_filtered['Adv_year']
Q1 = adv_year_data.quantile(0.25)
Q3 = adv_year_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

adv_outlier_mask = (adv_year_data < lower_bound) | (adv_year_data > upper_bound)
adv_normal_mask = ~adv_outlier_mask

plt.figure(figsize=(8,4))
plt.scatter(range(len(adv_year_data[adv_normal_mask])), adv_year_data[adv_normal_mask], alpha=0.5, label='Normal', color='blue')
plt.scatter(np.where(adv_outlier_mask)[0], adv_year_data[adv_outlier_mask], alpha=0.7, label='Outlier', color='red')
plt.xlabel('Index')
plt.ylabel('Adv_year')
plt.title('Adv_year Outliers Highlighted')
plt.legend()
plt.grid(True)
plt.yticks(sorted(df_filtered['Adv_year'].dropna().astype(int).unique()), 
           [str(int(val)) for val in sorted(df_filtered['Adv_year'].dropna().astype(int).unique())])
plt.tight_layout()
plt.show()

# -----------------------------------
# Count the number of doors for L200 marked as SUV
l200_suv_doors = df_filtered[(df_filtered['Genmodel'] == 'L200') & (df_filtered['Bodytype'] == 'SUV')]['Door_num'].value_counts()
print("Number of doors for L200 marked as SUV:")
print(l200_suv_doors)

# -----------------------------------
# Step 3j: Correct Door_num and Bodytype for L200 SUV Outliers
# -----------------------------------
"""
Data Correction:
- Change the single L200 SUV with 2 doors to Bodytype 'Pickup' (valid configuration).
- Change the single L200 SUV with 4 doors to Door_num 5 (most common for L200 SUV).
"""

# Change L200 SUV with 2 doors to Pickup
mask_2door_suv = (df_filtered['Genmodel'] == 'L200') & (df_filtered['Bodytype'] == 'SUV') & (df_filtered['Door_num'] == 2)
df_filtered.loc[mask_2door_suv, 'Bodytype'] = 'Pickup'

# Change L200 SUV with 4 doors to 5 doors
mask_4door_suv = (df_filtered['Genmodel'] == 'L200') & (df_filtered['Bodytype'] == 'SUV') & (df_filtered['Door_num'] == 4)
df_filtered.loc[mask_4door_suv, 'Door_num'] = 5

# Verify corrections
corrected_outliers = df_filtered[mask_2door_suv | mask_4door_suv]
print("Corrected L200 SUV outlier(s):")
print(corrected_outliers[['Genmodel', 'Bodytype', 'Door_num']])

# -----------------------------------
# Step 3i: Adv_year Outlier Decision Documentation
# -----------------------------------
"""
Adv_year Outlier Decision:
- Although some Adv_year values are flagged as outliers by the IQR method, they 
  are valid and consistent with the DVM-CAR dataset documentation.
- The reference states the advertising year range is approximately 2000 to 2020.
- Therefore, all Adv_year values in the filtered dataset are retained for analysis.
"""

# -----------------------------------
# Step 4: Modeling and Feature Comparison
# -----------------------------------
"""
Step 4: Modeling and Feature Comparison
- Build ExtraTreesRegressor models with different feature sets.
- Compare R² and RMSE for each model.
- Visualize actual vs predicted prices for the 6-variable model.
"""

# Build an Extremely Randomized Trees (ExtraTreesRegressor) model to predict vehicle price.
# Encode categorical variables.
# Split data into train/test sets.
# Fit the model and print feature importances and score.

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# Select features and target
features = ['Genmodel', 'Reg_year', 'Bodytype', 'Runned_Miles', 'Engin_size', 'Gearbox', 'Fuel_type', 'Seat_num', 'Door_num', 'Color', 'Adv_year', 'Adv_month']
target = 'Price'

X = df_filtered[features]
y = df_filtered[target]

# One-hot encode categorical variables
categorical_cols = ['Genmodel', 'Bodytype', 'Gearbox', 'Fuel_type', 'Color']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Build and fit ExtraTreesRegressor
etr = ExtraTreesRegressor(n_estimators=100, random_state=42)
etr.fit(X_train, y_train)

# Print feature importances
importances = pd.Series(etr.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nFeature importances (ExtraTreesRegressor):")
print(importances)

# Print model score
score = etr.score(X_test, y_test)
print(f"\nExtraTreesRegressor R^2 score on test set: {score:.3f}")

# -----------------------------------
# Step 4: Compare ExtraTreesRegressor with 6-Variable Feature Set
# -----------------------------------
"""
Compare model performance using:
- All relevant variables
- Reduced set: Reg_year, Engin_size, Runned_Miles, Genmodel
- 6-variable set: Reg_year, Engin_size, Runned_Miles, Genmodel, Gearbox, Adv_month
Show R² scores and RMSE for all models in a table with lines for easy copy-paste.
Also, plot Actual vs Predicted Price for the 6-variable model.
"""

# 1. Full feature set (already fitted above)
y_pred_full = etr.predict(X_test)
r2_full = etr.score(X_test, y_test)
rmse_full = mean_squared_error(y_test, y_pred_full, squared=False)

# 2. Reduced feature set
features_reduced = ['Reg_year', 'Engin_size', 'Runned_Miles', 'Genmodel']
X_reduced = df_filtered[features_reduced]
y_reduced = df_filtered[target]

X_reduced_encoded = pd.get_dummies(X_reduced, columns=['Genmodel'], drop_first=True)
if X_reduced_encoded['Runned_Miles'].dtype == 'object':
    X_reduced_encoded['Runned_Miles'] = pd.to_numeric(X_reduced_encoded['Runned_Miles'], errors='coerce')

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced_encoded, y_reduced, test_size=0.2, random_state=42)
etr_reduced = ExtraTreesRegressor(n_estimators=100, random_state=42)
etr_reduced.fit(X_train_r, y_train_r)
y_pred_reduced = etr_reduced.predict(X_test_r)
r2_reduced = etr_reduced.score(X_test_r, y_test_r)
rmse_reduced = mean_squared_error(y_test_r, y_pred_reduced, squared=False)

# 3. Six-variable feature set
features_six = ['Reg_year', 'Engin_size', 'Runned_Miles', 'Genmodel', 'Gearbox', 'Adv_month']
X_six = df_filtered[features_six]
y_six = df_filtered[target]

# Encode categorical variables
X_six_encoded = pd.get_dummies(X_six, columns=['Genmodel', 'Gearbox'], drop_first=True)
if X_six_encoded['Runned_Miles'].dtype == 'object':
    X_six_encoded['Runned_Miles'] = pd.to_numeric(X_six_encoded['Runned_Miles'], errors='coerce')

X_train_six, X_test_six, y_train_six, y_test_six = train_test_split(X_six_encoded, y_six, test_size=0.2, random_state=42)
etr_six = ExtraTreesRegressor(n_estimators=100, random_state=42)
etr_six.fit(X_train_six, y_train_six)
y_pred_six = etr_six.predict(X_test_six)
r2_six = etr_six.score(X_test_six, y_test_six)
rmse_six = mean_squared_error(y_test_six, y_pred_six, squared=False)

# 4. Compare results in a table with lines
results = pd.DataFrame({
    'Model': [
        'All Relevant Variables',
        'Reduced Set (Reg_year, Engin_size, Runned_Miles, Genmodel)',
        '6-Variable Set (Reg_year, Engin_size, Runned_Miles, Genmodel, Gearbox, Adv_month)'
    ],
    'R2 Score': [r2_full, r2_reduced, r2_six],
    'RMSE': [rmse_full, rmse_reduced, rmse_six]
})

print("\nModel Comparison Table:")
print("+--------------------------------------------------------------+----------+----------+")
print("| Model                                                        | R2 Score |   RMSE   |")
print("+--------------------------------------------------------------+----------+----------+")
for i, row in results.iterrows():
    print(f"| {row['Model']:<60} | {row['R2 Score']:<8.3f} | {row['RMSE']:<8.2f} |")
print("+--------------------------------------------------------------+----------+----------+")

# 5. Plot Actual vs Predicted Price for 6-variable model
plt.figure(figsize=(8,6))
plt.scatter(y_test_six, y_pred_six, alpha=0.5, label='Predicted')
plt.plot([y_test_six.min(), y_test_six.max()], [y_test_six.min(), y_test_six.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price (6-Variable ExtraTreesRegressor)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()