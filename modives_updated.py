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
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        # Clean Status column - replace NaN/missing values with "None"
        if 'Status' in df.columns:
            missing_status_count = df['Status'].isnull().sum() + (df['Status'] == '').sum()
            if missing_status_count > 0:
                df['Status'] = df['Status'].fillna('None')
                df['Status'] = df['Status'].replace('', 'None')
                print(f"\n✓ Data cleaning: Replaced {missing_status_count} missing Status values with 'None'")
            
            # Also standardize the lowercase "active" to "Active"
            lowercase_active = (df['Status'] == 'active').sum()
            if lowercase_active > 0:
                df['Status'] = df['Status'].replace('active', 'Active')
                print(f"✓ Data cleaning: Standardized {lowercase_active} lowercase 'active' to 'Active'")
            
            # Simplify status names for better readability
            df['Status'] = df['Status'].replace('Pending Cancellation', 'Pending')
        
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
        
        # Status column distribution analysis
        print(f"\nSTATUS COLUMN DISTRIBUTION:")
        print("-" * 80)
        if 'Status' in df.columns:
            status_counts = df['Status'].value_counts(dropna=False)
            status_pct = df['Status'].value_counts(normalize=True, dropna=False) * 100
            
            print(f"{'Status':<25} {'Count':<10} {'Percentage'}")
            print("-" * 50)
            for status, count in status_counts.items():
                pct = status_pct[status]
                print(f'{status:<25}: {count:>6,} units ({pct:>5.1f}%)')
            
            print(f"\nTotal units: {len(df):,}")
        
        # Examine missing status observations (after cleaning)
        print(f"\nMISSING STATUS OBSERVATIONS (after cleaning):")
        print("-" * 80)
        missing_status = df[df['Status'].isnull() | (df['Status'] == '')]
        if len(missing_status) > 0:
            print(f'Found {len(missing_status)} observations still with missing Status')
            print(f'\nDetails of remaining missing Status observations:')
            print(missing_status[['Unit_ID', 'Key', 'Property', 'Unit', 'Resident', 'Carrier', 'Status']].to_string(index=False))
        else:
            print('✓ No missing Status observations found - all cleaned successfully!')

        # Create property vs status heatmap table
        print(f"\nPROPERTY vs STATUS HEATMAP TABLE:")
        print("-" * 80)
        create_property_status_heatmap(df)

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
def create_property_status_heatmap(df):
    """
    Create a heatmap table showing Property vs Status distribution
    
    Args:
        df (pd.DataFrame): Dataset with Property and Status columns
    """
    try:
        # Create crosstab with totals
        crosstab = pd.crosstab(df['Property'], df['Status'], margins=True, margins_name='TOTAL')
        
        print("PROPERTY vs STATUS CROSS-TABULATION TABLE:")
        print("=" * 120)
        print(crosstab)
        
        # Create heatmap focusing on Active status for color coding
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        
        # Heatmap 1: Full crosstab (without totals row/column for better visualization)
        crosstab_no_totals = crosstab.iloc[:-1, :-1]  # Remove totals
        
        # Create heatmap with all status values
        sns.heatmap(crosstab_no_totals, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   ax=ax1,
                   cbar_kws={'label': 'Number of Units'})
        ax1.set_title('Property vs Status Distribution\n(All Status Categories)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Status', fontsize=12)
        ax1.set_ylabel('Property', fontsize=12)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Heatmap 2: Active status only with Red-Green color scheme
        if 'Active' in crosstab_no_totals.columns:
            active_data = crosstab_no_totals[['Active']].copy()
            
            # Create custom colormap from red (low) to green (high)
            sns.heatmap(active_data,
                       annot=True,
                       fmt='d',
                       cmap='RdYlGn',  # Red-Yellow-Green colormap
                       ax=ax2,
                       cbar_kws={'label': 'Number of Active Units'})
            ax2.set_title('Active Units by Property\n(Red=Lowest, Green=Highest)', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Status', fontsize=12)
            ax2.set_ylabel('Property', fontsize=12)
            
            # Add percentage annotations
            total_active = active_data['Active'].sum()
            for i, property_name in enumerate(active_data.index):
                active_count = active_data.loc[property_name, 'Active']
                percentage = (active_count / total_active) * 100
                ax2.text(0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('property_status_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary statistics
        print(f"\nSUMMARY STATISTICS:")
        print("-" * 50)
        if 'Active' in crosstab.columns:
            active_by_property = crosstab['Active'].iloc[:-1].sort_values(ascending=False)
            print("Top 5 Properties by Active Units:")
            for i, (prop, count) in enumerate(active_by_property.head().items(), 1):
                total_units = crosstab.loc[prop, 'TOTAL']
                pct_active = (count / total_units) * 100
                print(f"{i:2d}. {prop}: {count:,} active units ({pct_active:.1f}% of property)")
            
            print(f"\nBottom 5 Properties by Active Units:")
            for i, (prop, count) in enumerate(active_by_property.tail().items(), 1):
                total_units = crosstab.loc[prop, 'TOTAL'] 
                pct_active = (count / total_units) * 100 if total_units > 0 else 0
                print(f"{i:2d}. {prop}: {count:,} active units ({pct_active:.1f}% of property)")
        
        print(f"\n✓ Heatmap saved as 'property_status_heatmap.png'")
        
    except Exception as e:
        print(f"✗ Error creating heatmap: {e}")
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