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
import math

# Configuration
SAMPLE_SIZE = 360
RANDOM_STATE = 42  # For reproducible results

def create_risk_chart(df):
    """
    Create a risk chart showing status distribution, counts, and exposure
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n" + "="*80)
    print("RISK EXPOSURE CHART")
    print("="*80)
    
    if 'Status' not in df.columns or 'Liability' not in df.columns:
        print("✗ Required columns (Status, Liability) not found for risk chart")
        return None
    
    # Get liability median for calculations
    liability_clean = pd.to_numeric(df['Liability'], errors='coerce')
    median_coverage = liability_clean.median()
    
    # Status distribution with exposure calculations
    status_counts = df['Status'].value_counts()
    
    # Create chart
    total_units = 0
    total_exposure = 0
    
    print(f"{'Status':<20} {'Count':<8} {'%':<6} {'Unit Exposure':<15} {'Total Exposure':<20}")
    print("-" * 75)
    
    for status, count in status_counts.items():
        percentage = (count / len(df)) * 100
        
        # Calculate exposure (None and Cancelled have exposure, others don't)
        if status in ['None', 'Cancelled']:
            unit_exposure = median_coverage
            exposure = count * median_coverage
        else:
            unit_exposure = 0
            exposure = 0
        
        total_units += count
        total_exposure += exposure
        
        # Format and print row
        print(f"{status:<20} {count:<8,} {percentage:<5.1f}% ${unit_exposure:<14,.0f} ${exposure:<19,.0f}")
    
    # Print totals
    print("-" * 75)
    print(f"{'TOTAL':<20} {total_units:<8,} {'100.0%':<6} {'':<15} ${total_exposure:<19,.0f}")
    
    print(f"\nTotal Risk Exposure: ${total_exposure:,.0f}")
    print(f"Risk as % of Portfolio: {(total_exposure/(total_units * median_coverage))*100:.1f}%")

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
        
        # Fix case sensitivity issues in Status column
        if 'Status' in df.columns:
            # Standardize "active" to "Active"
            df['Status'] = df['Status'].replace('active', 'Active')
            print("✓ Status column standardized (active → Active)")
        
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
        
        # Key counts
        none_policies = status_counts.get('None', 0)
        vacant_units = status_counts.get('Vacant', 0)
        cancelled_policies = status_counts.get('Cancelled', 0)
        active_policies = status_counts.get('Active', 0)
        
        print(f"\nKey Metrics:")
        print(f"   Occupied units: {total_units - vacant_units:,}")
        print(f"   Baseline compliant: {total_units - vacant_units - none_policies:,}")
        print(f"   Active policies: {active_policies:,}")
        
        # Compliance rates
        occupied_units = total_units - vacant_units
        if occupied_units > 0:
            baseline_rate = ((occupied_units - none_policies) / occupied_units) * 100
            active_rate = (active_policies / occupied_units) * 100
            
            print(f"\nCompliance Rates:")
            print(f"   Baseline compliance: {baseline_rate:.1f}%")
            print(f"   Active compliance: {active_rate:.1f}%")
    
    # Liability analysis
    if 'Liability' in df.columns:
        print(f"\nLiability Coverage Analysis:")
        
        # Clean liability data - convert to numeric
        liability_clean = pd.to_numeric(df['Liability'], errors='coerce')
        liability_stats = liability_clean.describe()
        
        print(f"   Minimum coverage: ${liability_stats['min']:,.0f}")
        print(f"   Median coverage: ${liability_stats['50%']:,.0f}")
        print(f"   Maximum coverage: ${liability_stats['max']:,.0f}")
        print(f"   Mean coverage: ${liability_stats['mean']:,.0f}")
        
        # Risk exposure calculations
        median_coverage = liability_stats['50%']
        
        print(f"\nRisk Exposure:")
        
        # None policies exposure
        none_risk = none_policies * median_coverage
        print(f"   None policies: {none_policies:,} × ${median_coverage:,.0f} = ${none_risk:,.0f}")
        
        # Cancelled policies exposure  
        cancelled_risk = cancelled_policies * median_coverage
        print(f"   Cancelled policies: {cancelled_policies:,} × ${median_coverage:,.0f} = ${cancelled_risk:,.0f}")
        
        # Total potential risk
        total_risk = none_risk + cancelled_risk
        print(f"   Total potential exposure: ${total_risk:,.0f}")
        
        # Portfolio percentage
        total_portfolio = total_units * median_coverage
        risk_percentage = (total_risk / total_portfolio) * 100
        print(f"   Risk as % of portfolio: {risk_percentage:.1f}%")

def property_summary_table(df):
    """
    Create a summary table showing properties and their unit counts
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n" + "="*60)
    print("PROPERTY SUMMARY TABLE")
    print("="*60)
    
    if 'Property' not in df.columns:
        print("✗ 'Property' column not found")
        return None
    
    # Get property counts
    property_counts = df['Property'].value_counts().sort_index()
    
    print(f"{'Property':<10} {'Units':<8}")
    print("-" * 20)
    
    total_units = 0
    for property_code, count in property_counts.items():
        print(f"{property_code:<10} {count:<8,}")
        total_units += count
    
    print("-" * 20)
    print(f"{'TOTAL':<10} {total_units:<8,}")
    
    print(f"\nSummary:")
    print(f"   Total Properties: {len(property_counts)}")
    print(f"   Total Units: {total_units:,}")
    print(f"   Average Units per Property: {total_units/len(property_counts):.1f}")
    
    return property_counts

def calculate_sample_size_binary(df):
    """
    Calculate sample size for binary distribution (compliant/non-compliant)
    Using 95% confidence interval and 5% margin of error
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n" + "="*80)
    print("SAMPLE SIZE CALCULATION FOR BINARY DISTRIBUTION")
    print("="*80)
    
    if 'Status' not in df.columns:
        print("✗ 'Status' column not found")
        return None
    
    # Population parameters
    N = len(df)  # Total population
    vacant_units = (df['Status'] == 'Vacant').sum()
    none_policies = (df['Status'] == 'None').sum()
    
    # Define compliant population (exclude Vacant and None)
    compliant_population = N - vacant_units - none_policies
    occupied_units = N - vacant_units
    
    # Estimated proportion (p̂) - current compliance rate
    p_hat = compliant_population / occupied_units
    
    print(f"Population Parameters:")
    print(f"   Total units (N): {N:,}")
    print(f"   Vacant units: {vacant_units:,}")
    print(f"   None policies: {none_policies:,}")
    print(f"   Occupied units: {occupied_units:,}")
    print(f"   Compliant units: {compliant_population:,}")
    print(f"   Estimated compliance rate (p̂): {p_hat:.3f}")
    
    # Sample size calculation parameters
    confidence_level = 0.95
    z_score = 1.96  # For 95% confidence
    margin_of_error = 0.05  # 5% accuracy
    
    print(f"\nSample Size Parameters:")
    print(f"   Confidence Level: {confidence_level*100:.0f}%")
    print(f"   Z-score: {z_score}")
    print(f"   Margin of Error: ±{margin_of_error*100:.0f}%")
    print(f"   Population proportion (p̂): {p_hat:.3f}")
    print(f"   Complement (1-p̂): {1-p_hat:.3f}")
    
    # Formula for infinite population
    # n = (Z²× p̂ × (1-p̂)) / E²
    n_infinite = (z_score**2 * p_hat * (1 - p_hat)) / (margin_of_error**2)
    
    # Finite population correction
    # n_finite = n_infinite / (1 + (n_infinite - 1) / N)
    n_finite = n_infinite / (1 + (n_infinite - 1) / occupied_units)
    
    # Round up to nearest integer
    n_recommended = math.ceil(n_finite)
    
    print(f"\nSample Size Calculations:")
    print(f"   Infinite population formula: n = (Z²×p̂×(1-p̂))/E²")
    print(f"   n_infinite = ({z_score}² × {p_hat:.3f} × {1-p_hat:.3f}) / {margin_of_error}²")
    print(f"   n_infinite = {n_infinite:.1f}")
    
    print(f"\n   Finite population correction: n_finite = n_infinite / (1 + (n_infinite-1)/N)")
    print(f"   n_finite = {n_infinite:.1f} / (1 + ({n_infinite:.1f}-1)/{occupied_units:,})")
    print(f"   n_finite = {n_finite:.1f}")
    
    print(f"\n" + "="*50)
    print(f"RECOMMENDED SAMPLE SIZE: {n_recommended:,}")
    print(f"="*50)
    
    print(f"\nSample Characteristics:")
    print(f"   Sample size: {n_recommended:,}")
    print(f"   Percentage of occupied units: {(n_recommended/occupied_units)*100:.1f}%")
    print(f"   Percentage of total population: {(n_recommended/N)*100:.1f}%")
    
    # Comparison with your current SAMPLE_SIZE
    print(f"\nComparison with current setting:")
    print(f"   Current SAMPLE_SIZE setting: {SAMPLE_SIZE}")
    print(f"   Statistically required: {n_recommended:,}")
    if SAMPLE_SIZE >= n_recommended:
        print(f"   ✓ Current setting is adequate")
    else:
        print(f"   ⚠ Current setting is below statistical requirement")
        print(f"   Recommendation: Update SAMPLE_SIZE to {n_recommended}")
    
    return n_recommended

def main():
    """Main function to process MODIVES data"""
    
    # Look for files containing "modives" (case insensitive)
    all_excel_files = list(Path('.').glob('*.xlsx')) + list(Path('.').glob('*.xls'))
    
    # Filter for modives files and exclude temporary Excel files (starting with ~$)
    modives_files = [f for f in all_excel_files 
                     if 'modives' in f.name.lower() and not f.name.startswith('~$')]
    
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
        
        # Property summary table
        property_summary_table(df)
        
        # Calculate risk metrics
        calculate_risk_metrics(df)
        
        # Create risk chart
        create_risk_chart(df)
        
        # Calculate required sample size
        recommended_sample = calculate_sample_size_binary(df)
        
        return df
    else:
        return None

if __name__ == "__main__":
    data = main()
