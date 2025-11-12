"""
MODIVES Data Processing Script
=============================

Purpose: Upload and process Excel file for MODIVES analysis
Tasks:
- Load Excel file data
- Check variables and observations
- Count total observations
- Select random sample of size 360

Author: German Montoya
Date: November 12, 2025
File: modives.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
SAMPLE_SIZE = 360
RANDOM_STATE = 42  # For reproducible results

def load_excel_file(file_path):
    """
    Load Excel file and return DataFrame
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        # Try different parameters to handle potential formatting issues
        df = pd.read_excel(file_path, na_values=['', ' ', 'nan', 'NaN', 'NULL', 'null'])
        print(f"✓ File loaded successfully: {file_path}")
        
        # Check for hidden characters or spaces in Status column
        if 'Status' in df.columns:
            # Strip whitespace and check for empty strings
            df['Status'] = df['Status'].astype(str).str.strip()
            df['Status'] = df['Status'].replace('nan', pd.NA)
            df['Status'] = df['Status'].replace('', pd.NA)
            
        return df
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None

def analyze_variables(df):
    """
    Analyze variables in the dataset
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n" + "="*60)
    print("VARIABLE ANALYSIS")
    print("="*60)
    
    print(f"\nTotal Variables (Columns): {len(df.columns)}")
    print(f"Total Observations (Rows): {len(df)}")
    
    print("\n" + "-"*60)
    print("DETAILED VARIABLE INFORMATION")
    print("-"*60)
    
    for i, col in enumerate(df.columns, 1):
        print(f"\n{i}. Variable: '{col}'")
        print(f"   Data Type: {df[col].dtype}")
        print(f"   Missing Values: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")
        print(f"   Non-null Count: {df[col].count()}")
        
        # Determine variable nature
        if df[col].dtype in ['object', 'string']:
            print(f"   Nature: Categorical/Text")
            unique_values = df[col].dropna().unique()
            print(f"   Unique Values: {len(unique_values)}")
            if len(unique_values) <= 10:
                print(f"   Values: {list(unique_values)}")
            else:
                print(f"   Sample Values: {list(unique_values[:10])}...")
        elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            print(f"   Nature: Numerical")
            print(f"   Min: {df[col].min()}")
            print(f"   Max: {df[col].max()}")
            print(f"   Mean: {df[col].mean():.2f}")
            print(f"   Unique Values: {df[col].nunique()}")
        elif df[col].dtype == 'bool':
            print(f"   Nature: Boolean")
            print(f"   Values: {df[col].value_counts().to_dict()}")
        else:
            print(f"   Nature: Other ({df[col].dtype})")
            print(f"   Unique Values: {df[col].nunique()}")

def get_summary_statistics(df):
    """
    Get summary statistics for the dataset
    
    Args:
        df (pd.DataFrame): Dataset to summarize
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # General info
    print(f"Dataset Shape: {df.shape}")
    print(f"Total Cells: {df.size}")
    print(f"Total Missing Values: {df.isnull().sum().sum()}")
    print(f"Missing Data Percentage: {(df.isnull().sum().sum() / df.size) * 100:.2f}%")
    
    # Data types summary
    print("\nData Types Summary:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} variables")

def check_duplicates(df):
    """
    Check for duplicate values in each variable
    
    Args:
        df (pd.DataFrame): Dataset to check for duplicates
    """
    print("\n" + "="*60)
    print("DUPLICATE VALUES ANALYSIS")
    print("="*60)
    
    for col in df.columns:
        total_values = len(df[col])
        unique_values = df[col].nunique()
        duplicates = total_values - unique_values
        
        if duplicates > 0:
            print(f"\n'{col}' has {duplicates} duplicate values")
            
            # Show the actual duplicate values
            duplicate_values = df[col][df[col].duplicated(keep=False)].value_counts()
            print(f"   Most frequent duplicates:")
            for value, count in duplicate_values.head(10).items():
                print(f"   '{value}': appears {count} times")

def check_unit_duplicates(df):
    """
    Check for duplicate values specifically in the Unit variable
    
    Args:
        df (pd.DataFrame): Dataset to check for Unit duplicates
    """
    print("\n" + "="*60)
    print("UNIT DUPLICATE VALUES ANALYSIS")
    print("="*60)
    
    if 'Unit' not in df.columns:
        print("✗ 'Unit' column not found in dataset")
        return
    
    total_units = len(df['Unit'])
    unique_units = df['Unit'].nunique()
    duplicates = total_units - unique_units
    
    print(f"Total Unit entries: {total_units}")
    print(f"Unique Unit values: {unique_units}")
    print(f"Duplicate Unit entries: {duplicates}")
    
    if duplicates > 0:
        # Show the actual duplicate values
        duplicate_units = df['Unit'][df['Unit'].duplicated(keep=False)].value_counts()
        print(f"\nDuplicate Units (showing all):")
        for unit, count in duplicate_units.items():
            print(f"   '{unit}': appears {count} times")
    else:
        print("✓ No duplicate Unit values found")

def investigate_status_column(df):
    """
    Investigate the Status column to understand missing values
    
    Args:
        df (pd.DataFrame): Dataset to investigate
    """
    print("\n" + "="*60)
    print("STATUS COLUMN INVESTIGATION")
    print("="*60)
    
    if 'Status' not in df.columns:
        print("✗ 'Status' column not found")
        return
    
    # Show value counts including NaN
    print("Status value counts (including missing):")
    status_counts = df['Status'].value_counts(dropna=False)
    for value, count in status_counts.items():
        if pd.isna(value):
            print(f"   Missing/NaN: {count}")
        else:
            print(f"   '{value}': {count}")
    
    # Check for rows where Status is missing but other fields have data
    missing_status = df[df['Status'].isna()]
    print(f"\nRows with missing Status: {len(missing_status)}")
    
    if len(missing_status) > 0:
        print("\nSample of rows with missing Status (first 5):")
        print(missing_status[['Unit', 'Resident', 'Status', 'Carrier']].head())
        
        # Check if missing Status correlates with missing other fields
        print(f"\nOf {len(missing_status)} missing Status entries:")
        print(f"   Also missing Resident: {missing_status['Resident'].isna().sum()}")
        print(f"   Also missing Carrier: {missing_status['Carrier'].isna().sum()}")
        print(f"   Also missing Policy #: {missing_status['Policy #'].isna().sum()}")

def main():
    """Main function to process MODIVES data"""
    
    # Look for Excel files in current directory
    excel_files = list(Path('.').glob('*.xlsx')) + list(Path('.').glob('*.xls'))
    
    if not excel_files:
        print("✗ No Excel files found in current directory")
        return None
    
    if len(excel_files) == 1:
        excel_file_path = str(excel_files[0])
        print(f"Found Excel file: {excel_file_path}")
    else:
        print("Multiple Excel files found:")
        for i, file in enumerate(excel_files, 1):
            print(f"{i}. {file}")
        choice = int(input("Select file number: ")) - 1
        excel_file_path = str(excel_files[choice])
    
    # Load the file
    df = load_excel_file(excel_file_path)
    
    if df is not None:
        # Analyze variables
        analyze_variables(df)
        
        # Investigate Status column
        investigate_status_column(df)
        
        # Check for Unit duplicates only
        check_unit_duplicates(df)
        
        # Get summary statistics
        get_summary_statistics(df)
        
        return df
    else:
        return None

if __name__ == "__main__":
    data = main()
