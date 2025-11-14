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
        
        # Risk categorization
        none_policies = status_counts.get('None', 0)
        vacant_units = status_counts.get('Vacant', 0)
        cancelled_policies = status_counts.get('Cancelled', 0)
        active_policies = status_counts.get('Active', 0)
        
        print(f"\nRisk Assessment Categories:")
        print(f"   HIGH RISK - No Policy (None): {none_policies:,} units")
        print(f"   MEDIUM RISK - Expired/Cancelled: {cancelled_policies:,} units")
        print(f"   LOW RISK - Active Policies: {active_policies:,} units")
        print(f"   NO RISK - Vacant Units: {vacant_units:,} units")
        
        print(f"\nKey Metrics:")
        print(f"   Occupied units: {total_units - vacant_units:,}")
        print(f"   Current baseline compliant: {total_units - vacant_units - none_policies:,}")
        print(f"   True compliant (Active only): {active_policies:,}")
        
        # Compliance rates
        occupied_units = total_units - vacant_units
        if occupied_units > 0:
            baseline_rate = ((occupied_units - none_policies) / occupied_units) * 100
            true_compliance_rate = (active_policies / occupied_units) * 100
            
            print(f"\nCompliance Rates:")
            print(f"   Baseline compliance rate: {baseline_rate:.1f}%")
            print(f"   True compliance rate (Active only): {true_compliance_rate:.1f}%")
    
    # Enhanced liability analysis
    if 'Liability' in df.columns:
        print(f"\nLiability Coverage Analysis:")
        
        # Clean liability data - convert to numeric
        liability_clean = pd.to_numeric(df['Liability'], errors='coerce')
        liability_stats = liability_clean.describe()
        
        print(f"   Minimum coverage: ${liability_stats['min']:,.0f}")
        print(f"   Median coverage: ${liability_stats['50%']:,.0f}")
        print(f"   Maximum coverage: ${liability_stats['max']:,.0f}")
        print(f"   Mean coverage: ${liability_stats['mean']:,.0f}")
        
        # Risk exposure scenarios
        median_coverage = liability_stats['50%']
        
        print(f"\nRisk Exposure Scenarios:")
        
        # Scenario 1: Only None policies (immediate risk)
        immediate_risk = none_policies * median_coverage
        print(f"   Immediate Risk (None policies only):")
        print(f"     Units: {none_policies:,} × Coverage: ${median_coverage:,.0f}")
        print(f"     = ${immediate_risk:,.0f}")
        
        # Scenario 2: All non-active policies (potential risk if not renewed)
        potential_risk_units = none_policies + cancelled_policies
        potential_risk = potential_risk_units * median_coverage
        print(f"   Potential Risk (None + Cancelled):")
        print(f"     Units: {potential_risk_units:,} × Coverage: ${median_coverage:,.0f}")
        print(f"     = ${potential_risk:,.0f}")
        
        # Risk comparison
        if immediate_risk > 0:
            additional_risk = potential_risk - immediate_risk
            print(f"\n   Risk Analysis:")
            print(f"     Additional exposure if cancelled not renewed: ${additional_risk:,.0f}")
            print(f"     Potential risk multiplier: {potential_risk/immediate_risk:.1f}x")
            
        # Risk as percentage of total exposure
        total_exposure = total_units * median_coverage
        immediate_pct = (immediate_risk / total_exposure) * 100
        potential_pct = (potential_risk / total_exposure) * 100
        
        print(f"\n   Risk as % of Total Portfolio:")
        print(f"     Immediate risk: {immediate_pct:.1f}%")
        print(f"     Potential risk: {potential_pct:.1f}%")

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
