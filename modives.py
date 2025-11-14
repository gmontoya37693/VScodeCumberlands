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
        # Load the Excel file - don't convert "None" to NaN
        df = pd.read_excel(file_path, keep_default_na=False, na_values=[''])
        print(f"✓ File loaded successfully: {file_path}")
        
        # Create database key by concatenating Property and Unit
        if 'Property' in df.columns and 'Unit' in df.columns:
            df.insert(0, 'Database_Key', df['Property'].astype(str) + '_' + df['Unit'].astype(str))
            print(f"✓ Database key created: Property + Unit concatenation")
            print(f"   Sample keys: {list(df['Database_Key'].head(3))}")
        else:
            print("⚠ Warning: Could not create database key - Property or Unit column missing")
            
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
            if len(unique_values) <= 15:
                print(f"   Values: {list(unique_values)}")
            else:
                print(f"   Sample Values: {list(unique_values[:10])}...")
        elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            print(f"   Nature: Numerical")
            print(f"   Min: {df[col].min()}")
            print(f"   Max: {df[col].max()}")
            print(f"   Mean: {df[col].mean():.2f}")
            print(f"   Unique Values: {df[col].nunique()}")
        elif df[col].dtype in ['datetime64[ns]', 'datetime64']:
            print(f"   Nature: Date/Time")
            print(f"   Earliest: {df[col].min()}")
            print(f"   Latest: {df[col].max()}")
            print(f"   Unique Values: {df[col].nunique()}")
        elif df[col].dtype == 'bool':
            print(f"   Nature: Boolean")
            print(f"   Values: {df[col].value_counts().to_dict()}")
        else:
            print(f"   Nature: Other ({df[col].dtype})")
            print(f"   Unique Values: {df[col].nunique()}")

def calculate_risk_metrics(df):
    """
    Calculate key risk and compliance metrics
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n" + "="*80)
    print("RISK EXPOSURE AND COMPLIANCE METRICS")
    print("="*80)
    
    # Basic counts
    total_units = len(df)
    total_properties = df['Property'].nunique()
    
    print(f"Total Apartment Units: {total_units:,}")
    print(f"Total Properties: {total_properties}")
    
    # Status analysis
    if 'Status' in df.columns:
        print("\nStatus Distribution:")
        status_counts = df['Status'].value_counts()
        for status, count in status_counts.items():
            print(f"   {status}: {count:,} units ({count/total_units*100:.1f}%)")
        
        # Key metrics
        none_policies = status_counts.get('None', 0)
        vacant_units = status_counts.get('Vacant', 0)
        
        print(f"\nKey Metrics:")
        print(f"   Units without policies (None): {none_policies:,}")
        print(f"   Vacant apartments: {vacant_units:,}")
        print(f"   Actual tenant count: {total_units - vacant_units:,}")
        print(f"   Baseline compliant tenants: {total_units - vacant_units - none_policies:,}")
    
    # Liability analysis for risk calculation
    if 'Liability' in df.columns:
        print(f"\nLiability Coverage Analysis:")
        
        # Clean liability data - convert to numeric
        liability_clean = pd.to_numeric(df['Liability'], errors='coerce')
        liability_stats = liability_clean.describe()
        
        print(f"   Minimum coverage: ${liability_stats['min']:,.0f}")
        print(f"   Median coverage: ${liability_stats['50%']:,.0f}")
        print(f"   Maximum coverage: ${liability_stats['max']:,.0f}")
        print(f"   Mean coverage: ${liability_stats['mean']:,.0f}")
        
        # Minimum Risk Exposure calculation
        median_coverage = liability_stats['50%']
        min_risk_exposure = none_policies * median_coverage
        
        print(f"\nMinimum Risk Exposure Calculation:")
        print(f"   Units without policies: {none_policies:,}")
        print(f"   × Median liability coverage: ${median_coverage:,.0f}")
        print(f"   = Minimum Risk Exposure: ${min_risk_exposure:,.0f}")
        
        # Additional risk metrics
        if none_policies > 0:
            risk_per_unit = min_risk_exposure / none_policies
            print(f"   Average risk per uninsured unit: ${risk_per_unit:,.0f}")
            
        risk_percentage = (min_risk_exposure / (total_units * median_coverage)) * 100
        print(f"   Risk as % of total potential exposure: {risk_percentage:.1f}%")

def main():
    """Main function to process MODIVES data"""
    
    # Look for files containing "modives" (case insensitive)
    all_excel_files = list(Path('.').glob('*.xlsx')) + list(Path('.').glob('*.xls'))
    modives_files = [f for f in all_excel_files if 'modives' in f.name.lower()]
    
    if not modives_files:
        print("✗ No MODIVES Excel files found in current directory")
        print("   Looking for files containing 'modives' in filename")
        return None
    
    if len(modives_files) == 1:
        excel_file_path = str(modives_files[0])
        print(f"Found MODIVES file: {excel_file_path}")
    else:
        print("Multiple MODIVES files found:")
        for i, file in enumerate(modives_files, 1):
            print(f"{i}. {file}")
        try:
            choice = int(input("Select MODIVES file number: ")) - 1
            if 0 <= choice < len(modives_files):
                excel_file_path = str(modives_files[choice])
            else:
                print("✗ Invalid selection")
                return None
        except ValueError:
            print("✗ Invalid input. Please enter a number.")
            return None
    
    # Load the file and create database key
    df = load_excel_file(excel_file_path)
    
    if df is not None:
        print(f"\nDataset loaded with {len(df)} rows and {len(df.columns)} columns")
        print(f"Database key column added as first column")
        
        # Analyze variables
        analyze_variables(df)
        
        # Calculate risk metrics
        calculate_risk_metrics(df)
        
        return df
    else:
        return None

if __name__ == "__main__":
    data = main()
