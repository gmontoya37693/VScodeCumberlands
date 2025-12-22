"""
MODIVES Updated Data Analysis Script
===================================

Purpose: Analyze the updated MODIVES Excel file (Modives_updated.xlsx)
Tasks:
- Load Excel file data
- Check variables and observations  
- Add enumeration variable starting from 1
- Display file structure and preview

Author: German Montoya
Date: December 22, 2025
File: modives_updated_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_updated_file(file_path):
    """
    Analyze the Modives_updated.xlsx file
    
    Args:
        file_path (str): Path to the updated Excel file
        
    Returns:
        pd.DataFrame: Loaded data with enumeration
    """
    try:
        print("="*80)
        print("MODIVES UPDATED DATA ANALYSIS")
        print("="*80)
        
        # Load the Excel file
        print(f"Loading file: {file_path}")
        df = pd.read_excel(file_path, keep_default_na=False, na_values=[''])
        print(f"✓ File loaded successfully!")
        
        # Display basic information
        print(f"\nFILE STRUCTURE:")
        print(f"Number of observations (rows): {len(df):,}")
        print(f"Number of variables (columns): {len(df.columns)}")
        
        print(f"\nCOLUMN INFORMATION:")
        print("-" * 80)
        print(f"{'#':<3} {'Column Name':<35} {'Data Type':<15} {'Non-null Count':<15}")
        print("-" * 80)
        
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            print(f"{i:<3} {col:<35} {dtype:<15} {non_null:,}")
        
        # Add enumeration variable starting from 1
        df.insert(0, 'Unit_ID', range(1, len(df) + 1))
        print(f"\n✓ Added 'Unit_ID' column with enumeration starting from 1")
        print(f"  Updated dataset now has {len(df.columns)} columns")
        
        # Display first few rows
        print(f"\nDATA PREVIEW (first 5 rows):")
        print("-" * 120)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.max_colwidth', 20)
        print(df.head())
        
        # Check for missing values
        print(f"\nMISSING VALUES SUMMARY:")
        print("-" * 50)
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        missing_summary = []
        for col in df.columns:
            if missing[col] > 0:
                missing_summary.append({
                    'Column': col,
                    'Missing_Count': missing[col],
                    'Missing_Percentage': round(missing_pct[col], 2)
                })
        
        if missing_summary:
            missing_df = pd.DataFrame(missing_summary)
            print(missing_df.to_string(index=False))
        else:
            print("✓ No missing values found in any column!")
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:  # More than just Unit_ID
            print(f"\nNUMERIC COLUMNS SUMMARY:")
            print("-" * 80)
            print(df[numeric_cols].describe().round(2))
        
        # Sample of unique values for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\nCATEGORICAL COLUMNS - UNIQUE VALUES:")
            print("-" * 80)
            for col in categorical_cols[:5]:  # Show first 5 categorical columns
                unique_vals = df[col].unique()
                unique_count = len(unique_vals)
                sample_vals = unique_vals[:10] if len(unique_vals) > 10 else unique_vals
                print(f"{col}: {unique_count} unique values")
                print(f"  Sample: {', '.join(map(str, sample_vals))}")
                if len(unique_vals) > 10:
                    print(f"  ... and {len(unique_vals) - 10} more")
                print()
        
        print("="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return df
        
    except FileNotFoundError:
        print(f"✗ Error: File '{file_path}' not found")
        print("  Please check the file name and location")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None

def main():
    """Main function to analyze the updated MODIVES data"""
    
    # File path for the updated data
    file_path = 'Modives_updated.xlsx'
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"✗ File '{file_path}' not found in current directory")
        print("Available Excel files:")
        excel_files = list(Path('.').glob('*.xlsx')) + list(Path('.').glob('*.xls'))
        for f in excel_files:
            print(f"  - {f.name}")
        return None
    
    # Analyze the updated file
    df = analyze_updated_file(file_path)
    
    if df is not None:
        # Save the processed data with Unit_ID for future use
        output_file = 'modives_updated_with_unit_id.xlsx'
        df.to_excel(output_file, index=False)
        print(f"✓ Processed data saved to: {output_file}")
        
        return df
    else:
        return None

if __name__ == "__main__":
    result = main()