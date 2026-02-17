import pandas as pd
import numpy as np
from pathlib import Path

def explore_dataset(df, dataset_name):
    """
    Comprehensive exploration of a dataset
    """
    print(f"\n{'='*60}")
    print(f"EXPLORATION REPORT FOR: {dataset_name}")
    print(f"{'='*60}")
    
    # Basic info
    print(f"\n✅ BASIC INFORMATION:")
    print(f"Total observations (rows): {len(df)}")
    print(f"Total variables (columns): {len(df.columns)}")
    print(f"Dataset shape: {df.shape}")
    
    # Create comprehensive variable summary table
    print(f"\n✅ VARIABLES SUMMARY TABLE:")
    variable_summary = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        unique_count = df[col].nunique()
        data_type = str(df[col].dtype)
        
        variable_summary.append({
            'Variable': col,
            'Data_Type': data_type,
            'Missing_Count': missing_count,
            'Missing_%': round(missing_percent, 2),
            'Unique_Values': unique_count
        })
    
    summary_df = pd.DataFrame(variable_summary)
    print(summary_df.to_string(index=False))
    
    # Basic statistics for numeric columns
    print(f"\n✅ STATISTICAL SUMMARY (Numeric columns):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("No numeric columns found")
    
    # First few rows
    print(f"\n✅ FIRST 5 ROWS:")
    print(df.head())
    
    return df

# Load and explore the datasets
try:
    # Load first Excel file
    print("Loading Property Base rent SQFT file...")
    df1 = pd.read_excel('Property Base rent SQFT.xlsx')
    df1_explored = explore_dataset(df1, "Property Base rent SQFT")
    
    # Load second Excel file  
    print("\nLoading acento_apartments_fixed file...")
    df2 = pd.read_excel('acento_apartments_fixed.xlsx')
    df2_explored = explore_dataset(df2, "acento_apartments_fixed")
    
except FileNotFoundError as e:
    print(f"❌ Error: Could not find the Excel file - {e}")
    print("Please make sure the Excel files are in the same directory as this Python script")
except Exception as e:
    print(f"❌ Error loading files: {e}")
