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
        # Load the Excel file
        df = pd.read_excel(file_path)
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
    
    # Load the file and create database key
    df = load_excel_file(excel_file_path)
    
    if df is not None:
        print(f"\nDataset loaded with {len(df)} rows and {len(df.columns)} columns")
        print(f"Database key column added as first column")
        return df
    else:
        return None

if __name__ == "__main__":
    data = main()
