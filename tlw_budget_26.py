#!/usr/bin/env python3
"""
TLW Budget 26 - MODIVES Data Analysis
Created on February 11, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
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
        print(f"\nNote: Status values of None were replaced with Canceled")
    
    # Carrier analysis
    if 'Carrier' in df.columns:
        carrier_counts = df['Carrier'].value_counts(dropna=False)
        total_carriers = len(carrier_counts)
        
        print(f"\nCARRIER ANALYSIS:")
        print(f"Total different carriers: {total_carriers}")
        
        # Validation: Check Active status records have carriers
        print(f"\nCARRIER VALIDATION FOR ACTIVE STATUS:")
        print("-" * 50)
        if 'Status' in df.columns:
            active_records = df[df['Status'] == 'Active']
            if len(active_records) > 0:
                missing_carriers = active_records['Carrier'].isnull().sum() + (active_records['Carrier'] == '').sum()
                if missing_carriers > 0:
                    print(f"✗ Found {missing_carriers} Active records with missing/blank Carrier")
                    print(f"  This represents {(missing_carriers/len(active_records)*100):.1f}% of Active records")
                else:
                    print(f"✓ All {len(active_records):,} Active records have Carrier information")
            else:
                print("No Active status records found")
        print()

def create_status_pie_chart(df):
    """
    Create a pie chart showing status distribution with heatmap-style colors
    
    Args:
        df (pd.DataFrame): Dataset with Status column
    """
    if 'Status' not in df.columns:
        print("✗ No Status column found for pie chart")
        return
    
    # Get status counts (keep all statuses including Future)
    status_counts = df['Status'].value_counts()
    
    # Define color mapping (heatmap style: vacant->green->red)
    color_map = {
        'Vacant': '#FFFFFF',      # White (no color)
        'Active': '#2E8B57',      # Sea Green  
        'Override': '#DAA520',    # Golden Rod 
        'Pending': '#FF8C00',     # Dark Orange 
        'Future': '#FF6347',      # Tomato 
        'Cancelled': '#DC143C'    # Crimson Red
    }
    
    # Get colors for existing statuses (in order of counts)
    colors = []
    labels = []
    values = []
    
    for status in status_counts.index:
        labels.append(status)
        values.append(status_counts[status])
        colors.append(color_map.get(status, '#808080'))
    
    # Create the pie chart
    plt.figure(figsize=(14, 10))
    
    # Only show labels for slices > 3% AND hide Future label specifically
    labels_display = []
    for i, (status, count) in enumerate(zip(labels, values)):
        pct = (count / sum(values)) * 100
        if pct > 3.0 and status != 'Future':  # Hide Future label but keep others
            labels_display.append(status)
        else:
            labels_display.append('')
    
    # Create pie chart
    wedges, texts, autotexts = plt.pie(values, labels=labels_display, colors=colors, 
                                      autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(values)):,})' if pct > 2.0 else '',
                                      startangle=90, 
                                      labeldistance=1.15,
                                      pctdistance=0.85,
                                      wedgeprops=dict(edgecolor='black', linewidth=1.5),
                                      textprops={'fontsize': 11})
    
    # Enhance text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    
    # Set title with total units and properties
    total_units = len(df)
    total_properties = df['Property'].nunique() if 'Property' in df.columns else 0
    
    plt.title(f'Property Status Distribution\n{total_units:,} Units in {total_properties} Properties', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend with detailed information
    legend_labels = [f'{status}: {count:,} units ({count/len(df)*100:.1f}%)' 
                    for status, count in zip(labels, values)]
    
    plt.legend(wedges, legend_labels, title="Status Details", loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 PIE CHART GENERATED")
    print(f"Statuses shown: {len(status_counts)} (Future wedge visible but unlabeled)")
    print(f"Units displayed: {len(df):,}")

def analyze_states_from_addresses(df):
    """
    Analyze US states from address columns
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print(f"\nUS STATES ANALYSIS:")
    print("="*80)
    
    # Find address columns
    address_columns = [col for col in df.columns if any(addr_word in col.lower() 
                      for addr_word in ['address', 'addr', 'street', 'location'])]
    
    if not address_columns:
        print("✗ No address columns found")
        return
    
    print(f"Address columns found: {address_columns}")
    
    # US state abbreviations and full names
    us_states = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
        'DC': 'District of Columbia'
    }
    
    states_found = set()
    state_counts = {}
    
    # Analyze each address column
    for addr_col in address_columns:
        print(f"\nAnalyzing column: {addr_col}")
        
        # Get non-null addresses
        addresses = df[addr_col].dropna().astype(str)
        
        for address in addresses:
            # Look for state abbreviations (2 capital letters)
            # Common patterns: "City, ST ZIP" or "City ST ZIP" or "ST ZIP"
            state_patterns = [
                r'\b([A-Z]{2})\s+\d{5}',  # ST followed by ZIP
                r',\s*([A-Z]{2})\s*\d{5}', # , ST ZIP
                r',\s*([A-Z]{2})\s*$',    # , ST at end
                r'\b([A-Z]{2})\s*$'       # ST at end
            ]
            
            for pattern in state_patterns:
                matches = re.findall(pattern, address.upper())
                for match in matches:
                    if match in us_states:
                        states_found.add(match)
                        state_counts[match] = state_counts.get(match, 0) + 1
                        break
    
    # Display results
    if states_found:
        print(f"\n🗺️  STATES IDENTIFIED:")
        print(f"Total US states found: {len(states_found)}")
        print("-" * 60)
        print(f"{'State':<15} {'Full Name':<20} {'Units':<8}")
        print("-" * 60)
        
        # Sort by count (descending)
        sorted_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)
        
        for state_abbr, count in sorted_states:
            full_name = us_states[state_abbr]
            print(f"{state_abbr:<15} {full_name:<20} {count:>6,}")
        
        print("-" * 60)
        print(f"{'TOTAL':<36} {sum(state_counts.values()):>6,}")
        
        # Summary
        print(f"\nStates list: {', '.join(sorted(states_found))}")
        
    else:
        print("✗ No US states could be identified from address data")
        print("First few addresses for reference:")
        sample_addresses = df[address_columns[0]].dropna().head(5).tolist()
        for i, addr in enumerate(sample_addresses, 1):
            print(f"  {i}. {addr}")
    
    return states_found, state_counts

def analyze_property_portfolio(df):
    """
    Analyze property sizes for rollout planning (excluding WEA, CCA, CHA, CWA)
    
    Args:
        df (pd.DataFrame): Dataset with Property and current units
    """
    print(f"\nPROPERTY PORTFOLIO ANALYSIS FOR TLW ROLLOUT:")
    print("="*80)
    
    if 'Property' not in df.columns:
        print("✗ No Property column found")
        return None
    
    # Calculate units per property
    property_units = df.groupby('Property').size().reset_index()
    property_units.columns = ['Property', 'Units']
    
    # Filter out excluded properties for TLW rollout
    excluded_properties = ['WEA', 'CCA', 'CHA', 'CWA']
    original_count = len(property_units)
    excluded_units = property_units[property_units['Property'].isin(excluded_properties)]['Units'].sum()
    
    property_units = property_units[~property_units['Property'].isin(excluded_properties)]
    property_units = property_units.sort_values('Units')
    
    print(f"EXCLUSIONS FOR TLW ROLLOUT:")
    print(f"Excluded properties: {', '.join(excluded_properties)}")
    print(f"Excluded units: {excluded_units:,}")
    print(f"Remaining for TLW: {len(property_units)} properties, {property_units['Units'].sum():,} units")
    
    # Categorize properties by size
    property_units['Category'] = pd.cut(property_units['Units'], 
                                      bins=[0, 100, 200, 400, float('inf')],
                                      labels=['Micro (<100)', 'Small (100-200)', 'Medium (200-400)', 'Large (400+)'])
    
    print(f"\nTLW ROLLOUT PROPERTY SIZE DISTRIBUTION:")
    print("-" * 70)
    print(f"{'#':<3} {'Property':<25} {'Units':<8} {'Category':<15}")
    print("-" * 70)
    
    for i, row in property_units.iterrows():
        print(f"{property_units.index.get_loc(i)+1:<3} {row['Property']:<25} {row['Units']:<8} {row['Category']}")
    
    print("-" * 70)
    print(f"{'':>3} {'TLW ROLLOUT TOTAL':<25} {property_units['Units'].sum():<8}")
    
    # Category summary
    print(f"\nTLW ROLLOUT CATEGORY SUMMARY:")
    category_summary = property_units.groupby('Category').agg({
        'Property': 'count',
        'Units': ['sum', 'mean']
    }).round(1)
    
    print(category_summary)
    
    return property_units

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
        
        # Create visualizations and additional analysis
        create_status_pie_chart(df)
        analyze_states_from_addresses(df)
        
        # TLW Rollout Planning Analysis
        print("\n" + "="*80)
        print("TENANT LIABILITY WAIVER (TLW) ROLLOUT PLANNING")
        print("="*80)
        
        property_units = analyze_property_portfolio(df)
        
        print("\n" + "="*80)
        print("COMPLETE ANALYSIS FINISHED")
        print("="*80)
        
        return df
    
    return None

if __name__ == "__main__":
    result = main()
