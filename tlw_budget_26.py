#!/usr/bin/env python3
"""
TLW Budget 26 - MODIVES Data Analysis
Created on February 11, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_excel_data(file_path):
    """
    Load Excel file and perform initial variable analysis
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        print("="*80)
        print("TLW BUDGET 26 - MODIVES DATA ANALYSIS")
        print("="*80)
        
        # Load the Excel file
        print(f"Loading file: {file_path}")
        df = pd.read_excel(file_path, keep_default_na=False, na_values=[''])
        print(f"✓ File loaded successfully!")
        
        # Display basic information
        print(f"\nFILE STRUCTURE:")
        print(f"Number of observations (rows): {len(df):,}")
        print(f"Number of variables (columns): {len(df.columns)}")
        
        # Show column information
        print(f"\nCOLUMN INFORMATION:")
        print("-" * 100)
        print(f"{'#':<3} {'Column Name':<25} {'Data Type':<15} {'Non-null Count':<15} {'Unique Values'}")
        print("-" * 100)
        
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            unique_values = df[col].nunique()
            print(f"{i:<3} {col:<25} {dtype:<15} {non_null:<15,} {unique_values:,}")
        
        # Transform None status values to Cancelled
        if 'Status' in df.columns:
            none_count = (df['Status'] == 'None').sum()
            if none_count > 0:
                df['Status'] = df['Status'].replace('None', 'Cancelled')
                print(f"\n✓ Converted {none_count} 'None' status values to 'Cancelled'")
            else:
                print(f"\n✓ No 'None' status values found to convert")
        
        return df
        
    except FileNotFoundError:
        print(f"✗ Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None

def validate_data_integrity(df):
    """
    Validate data integrity by checking for duplicates and analyzing key columns
    
    Args:
        df (pd.DataFrame): Dataset to validate
    """
    print(f"\nDATA VALIDATION & ANALYSIS:")
    print("="*80)
    
    # 1. Check for duplicate Unit_IDs
    if 'Unit_ID' in df.columns:
        duplicate_unit_ids = df['Unit_ID'].duplicated().sum()
        if duplicate_unit_ids > 0:
            print(f"✗ Found {duplicate_unit_ids} duplicate Unit_IDs")
            duplicates = df[df['Unit_ID'].duplicated(keep=False)]['Unit_ID'].unique()
            print(f"  Duplicate Unit_IDs: {list(duplicates)}")
        else:
            print(f"✓ No duplicate Unit_IDs found ({len(df['Unit_ID'].unique())} unique values)")
    
    # 2. Check for duplicate Keys
    if 'Key' in df.columns:
        duplicate_keys = df['Key'].duplicated().sum()
        if duplicate_keys > 0:
            print(f"✗ Found {duplicate_keys} duplicate Keys")
            duplicates = df[df['Key'].duplicated(keep=False)]['Key'].unique()
            print(f"  Duplicate Keys: {list(duplicates[:5])}{'...' if len(duplicates) > 5 else ''}")
        else:
            print(f"✓ No duplicate Keys found ({len(df['Key'].unique())} unique values)")
    
    # 3. Check for missing Property values
    if 'Property' in df.columns:
        missing_properties = df['Property'].isnull().sum() + (df['Property'] == '').sum()
        if missing_properties > 0:
            print(f"✗ Found {missing_properties} missing Property values")
        else:
            print(f"✓ No missing Property values found (all {len(df)} records have property assigned)")
    
    # 4. Check for missing AND duplicate addresses (look for address-like columns)
    address_columns = [col for col in df.columns if any(addr_word in col.lower() 
                      for addr_word in ['address', 'addr', 'street', 'location'])]
    
    if address_columns:
        for addr_col in address_columns:
            # Check for missing addresses
            missing_addresses = df[addr_col].isnull().sum() + (df[addr_col] == '').sum()
            if missing_addresses > 0:
                print(f"✗ Found {missing_addresses} missing {addr_col} values")
            else:
                print(f"✓ No missing {addr_col} values found")
            
            # Check for duplicate addresses (should be unique per unit)
            duplicate_addresses = df[addr_col].duplicated().sum()
            if duplicate_addresses > 0:
                print(f"✗ Found {duplicate_addresses} duplicate {addr_col} values")
                duplicates = df[df[addr_col].duplicated(keep=False)][addr_col].value_counts()
                print(f"  Most common duplicates:")
                for addr, count in duplicates.head(3).items():
                    print(f"    '{addr}': {count} occurrences")
            else:
                print(f"✓ All {addr_col} values are unique (as expected for unit addresses)")
    else:
        print("Note: No address columns identified for validation")
    
    print()  # Add spacing before next section

def analyze_property_status_distribution(df):
    """
    Analyze Property and Status column distributions
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print(f"PROPERTY & STATUS ANALYSIS:")
    print("="*80)
    
    # Property analysis
    if 'Property' in df.columns:
        property_counts = df['Property'].value_counts()
        total_properties = len(property_counts)
        
        print(f"PROPERTY DISTRIBUTION:")
        print(f"Total different properties: {total_properties}")
        print("-" * 70)
        print(f"{'#':<3} {'Property':<20} {'Units':<8} {'Percentage'}")
        print("-" * 70)
        
        for i, (prop, count) in enumerate(property_counts.items(), 1):
            pct = (count / len(df)) * 100
            print(f"{i:<3} {prop:<20} {count:>6,} {pct:>8.1f}%")
        
        print("-" * 70)
        print(f"{'':>3} {'TOTAL':<20} {len(df):>6,} {'100.0%':>8}")
        print()
    
    # Status analysis
    if 'Status' in df.columns:
        status_counts = df['Status'].value_counts()
        total_statuses = len(status_counts)
        
        print(f"STATUS DISTRIBUTION:")
        print(f"Total different status values: {total_statuses}")
        print("-" * 60)
        print(f"{'Status':<20} {'Units':<8} {'Percentage'}")
        print("-" * 60)
        
        for status, count in status_counts.items():
            pct = (count / len(df)) * 100
            print(f"{status:<20} {count:>6,} {pct:>8.1f}%")
        
        print("-" * 60)
        print(f"{'TOTAL':<20} {len(df):>6,} {'100.0%':>8}")
        print()
        
        # Show unique status values
        print(f"Status values found: {list(status_counts.index)}")
        print(f"\nNote: Original data had 7 unique status values, now showing {total_statuses}")
        print(f"      'None' values were converted to 'Cancelled' for analysis consistency")

def main():
    """Main analysis function"""
    
    # File path - adjust as needed
    file_path = 'modives_updated_with_unit_id.xlsx'
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"✗ File '{file_path}' not found")
        print(f"\nAvailable Excel files:")
        excel_files = list(Path('.').glob('*.xlsx')) + list(Path('.').glob('*.xls'))
        if excel_files:
            for f in excel_files:
                print(f"  - {f.name}")
        else:
            print("  No Excel files found")
        return None
    
    # Load data
    df = load_excel_data(file_path)
    
    if df is not None:
        # Perform data validation and analysis
        validate_data_integrity(df)
        analyze_property_status_distribution(df)
        
        print("\n" + "="*80)
        print("INITIAL ANALYSIS COMPLETE - Ready for additional functions")
        print("="*80)
        
        return df
    
    return None

if __name__ == "__main__":
    result = main()
