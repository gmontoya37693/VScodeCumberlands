#!/usr/bin/env python3
"""
TLW Budget 26 - MODIVES Data Analysis & Enrollment Planning
Created on February 11, 2026
Updated on February 12, 2026

DESCRIPTION:
Comprehensive analysis and visualization tool for Tenant Liability Waiver (TLW) 
enrollment planning across Acento Apartments portfolio. Processes MODIVES data 
to generate enrollment schedules, visualizations, and Excel reports for budget planning.

INPUT FILE:
- modives_updated_with_unit_id.xlsx (Property portfolio with unit details)

OUTPUTS:
1. VISUALIZATIONS:
   - Status distribution pie chart (all properties)
   - Weekly cumulative enrollment curve (Week 15-52, April-December 2026)
   - Property Gantt chart (26 TLW properties enrollment timeline)

2. EXCEL REPORTS:
   - acento_apartments.xlsx: All 30 properties (Unit_ID, Key, Property, Units)
   - tlw_rollout.xlsx: 26 TLW properties with rollout weeks (Unit_ID, Key, Property, Units, Week)

3. ANALYSIS REPORTS:
   - Data validation and integrity checks
   - Property status distribution analysis
   - State-based geographical analysis  
   - TLW enrollment timeline (April-December 2026)
   - Property size categorization and scheduling optimization
   - Weekly enrollment statistics and milestone tracking

PHASES:
Phase 1: Data validation and portfolio analysis
Phase 2: Timeline design and enrollment curve planning
Phase 3: Weekly cumulative enrollment visualization
Phase 4: Property-by-property Gantt chart generation
Phase 5: Excel reports export for budget planning

TIMELINE REFERENCE:
- Today: February 12, 2026 (Week 7)
- TLW Rollout: April-December 2026 (Weeks 15-52)
- Target: 6,359 apartment units across 26 properties
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

def prepare_tlw_enrollment_portfolio(df):
    """
    Phase 1: Data Preparation & Property Sizing for TLW Enrollment
    Extract eligible properties, calculate sizes, and validate totals
    
    Args:
        df (pd.DataFrame): Dataset with Property and units
        
    Returns:
        pd.DataFrame: Prepared portfolio for TLW enrollment
    """
    print(f"\nTLW ENROLLMENT - PHASE 1: PORTFOLIO PREPARATION")
    print("="*80)
    
    if 'Property' not in df.columns:
        print("✗ No Property column found")
        return None
    
    # Calculate total units before exclusions
    total_original_units = len(df)
    total_properties = df['Property'].nunique()
    
    print(f"ORIGINAL PORTFOLIO:")
    print(f"Total properties: {total_properties}")
    print(f"Total units: {total_original_units:,}")
    
    # Calculate units per property
    property_units = df.groupby('Property').size().reset_index()
    property_units.columns = ['Property', 'Units']
    property_units = property_units.sort_values('Units', ascending=False)
    
    # Filter out excluded properties for TLW enrollment
    excluded_properties = ['WEA', 'CCA', 'CHA', 'CWA']
    excluded_data = property_units[property_units['Property'].isin(excluded_properties)]
    excluded_units = excluded_data['Units'].sum()
    excluded_count = len(excluded_data)
    
    # Get TLW eligible portfolio
    tlw_portfolio = property_units[~property_units['Property'].isin(excluded_properties)].copy()
    tlw_units = tlw_portfolio['Units'].sum()
    tlw_properties = len(tlw_portfolio)
    
    print(f"\nEXCLUSIONS:")
    if excluded_count > 0:
        print(f"Excluded properties: {excluded_count}")
        for _, row in excluded_data.iterrows():
            print(f"  - {row['Property']}: {row['Units']:,} units")
        print(f"Total excluded units: {excluded_units:,}")
    else:
        print(f"No properties found matching exclusion criteria: {', '.join(excluded_properties)}")
    
    print(f"\nTLW ELIGIBLE PORTFOLIO:")
    print(f"Properties for TLW: {tlw_properties}")
    print(f"Units for TLW: {tlw_units:,}")
    print(f"Average units per property: {tlw_units/tlw_properties:.1f}")
    
    # Categorize properties by size for enrollment planning
    def categorize_property_size(units):
        if units < 100:
            return 'Micro (<100)'
        elif units < 200:
            return 'Small (100-199)'
        elif units < 400:
            return 'Medium (200-399)'
        else:
            return 'Large (400+)'
    
    tlw_portfolio['Size_Category'] = tlw_portfolio['Units'].apply(categorize_property_size)
    tlw_portfolio['Enrollment_Priority'] = range(1, len(tlw_portfolio) + 1)  # Largest first
    
    # Category analysis
    print(f"\nPROPERTY SIZE DISTRIBUTION:")
    print("-" * 80)
    print(f"{'Category':<18} {'Properties':<12} {'Units':<10} {'Avg Size':<10} {'% of Total':<10}")
    print("-" * 80)
    
    category_summary = tlw_portfolio.groupby('Size_Category').agg({
        'Property': 'count',
        'Units': ['sum', 'mean']
    }).round(1)
    
    for category in ['Large (400+)', 'Medium (200-399)', 'Small (100-199)', 'Micro (<100)']:
        if category in category_summary.index:
            prop_count = category_summary.loc[category, ('Property', 'count')]
            total_units = category_summary.loc[category, ('Units', 'sum')]
            avg_size = category_summary.loc[category, ('Units', 'mean')]
            pct_total = (total_units / tlw_units) * 100
            print(f"{category:<18} {prop_count:<12} {total_units:<10,.0f} {avg_size:<10.1f} {pct_total:<10.1f}%")
    
    print("-" * 80)
    print(f"{'TOTAL':<18} {tlw_properties:<12} {tlw_units:<10,} {tlw_units/tlw_properties:<10.1f} {'100.0%':<10}")
    
    print(f"\nReady for Phase 2: Timeline & Enrollment Curve Design")
    
    return tlw_portfolio

def design_enrollment_timeline(tlw_portfolio):
    """
    Phase 2: Timeline & Enrollment Curve Design
    Create April-December enrollment schedule with steady slope
    
    Args:
        tlw_portfolio (pd.DataFrame): Prepared portfolio with properties and units
        
    Returns:
        dict: Timeline data with monthly targets and property schedule
    """
    print(f"\nTLW ENROLLMENT - PHASE 2: TIMELINE & ENROLLMENT CURVE DESIGN")
    print("="*80)
    
    # Timeline parameters
    start_month = 4  # April 2026
    end_month = 12   # December 2026
    total_months = end_month - start_month + 1  # 9 months
    total_units = tlw_portfolio['Units'].sum()
    
    print(f"ENROLLMENT TIMELINE:")
    print(f"Start date: April 2026")
    print(f"End date: December 2026")
    print(f"Duration: {total_months} months")
    print(f"Total units: {total_units:,}")
    print(f"Average monthly target: {total_units/total_months:,.0f} units")
    
    # Create steady slope enrollment curve
    # Linear distribution with slight acceleration toward end
    months = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    # Calculate monthly targets using linear slope
    base_monthly = total_units / total_months
    monthly_targets = {}
    cumulative_target = 0
    
    # Slight S-curve: slower start, steady middle, stronger finish
    curve_factors = [0.8, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 1.1]  # Sums to ~9.0
    factor_sum = sum(curve_factors)
    
    print(f"\nMONTHLY ENROLLMENT TARGETS (Steady Slope):")
    print("-" * 70)
    print(f"{'Month':<12} {'Target Units':<12} {'Cumulative':<12} {'% Complete':<12}")
    print("-" * 70)
    
    for i, (month, factor) in enumerate(zip(months, curve_factors)):
        monthly_target = int((base_monthly * factor * total_months) / factor_sum)
        cumulative_target += monthly_target
        
        # Adjust last month to hit exact total
        if i == len(months) - 1:
            monthly_target += (total_units - cumulative_target)
            cumulative_target = total_units
        
        pct_complete = (cumulative_target / total_units) * 100
        monthly_targets[month] = monthly_target
        
        print(f"{month:<12} {monthly_target:<12,} {cumulative_target:<12,} {pct_complete:<11.1f}%")
    
    print("-" * 70)
    
    # Property scheduling optimization to minimize parallel rollouts
    print(f"\nPROPERTY SCHEDULING OPTIMIZATION:")
    print("="*50)
    
    # Sort properties by size (largest first) for efficiency
    sorted_portfolio = tlw_portfolio.sort_values('Units', ascending=False).copy()
    
    # Estimate enrollment time per property (based on size)
    def estimate_enrollment_months(units):
        if units >= 400:
            return 2  # Large properties: 2 months
        elif units >= 200:
            return 1  # Medium properties: 1 month  
        else:
            return 1  # Small/Micro properties: 1 month
    
    sorted_portfolio['Estimated_Duration'] = sorted_portfolio['Units'].apply(estimate_enrollment_months)
    
    # Schedule properties to minimize overlaps
    property_schedule = {}
    monthly_capacity = {}
    
    # Initialize monthly capacity tracking
    for month in months:
        monthly_capacity[month] = monthly_targets[month]
    
    print(f"PROPERTY ENROLLMENT SCHEDULE:")
    print("-" * 88)
    print(f"{'#':<3} {'Property':<12} {'Units':<8} {'Duration':<10} {'Start Month':<12} {'End Month'}")
    print("-" * 88)
    
    # Assign properties to months
    property_counter = 1
    for _, property_row in sorted_portfolio.iterrows():
        prop = property_row['Property']
        units = property_row['Units']
        duration = property_row['Estimated_Duration']
        
        # Find best start month with available capacity
        best_start_idx = None
        for start_idx in range(len(months)):
            # Check if property can fit in available months
            total_needed = units
            months_needed = duration
            can_fit = True
            
            for month_offset in range(months_needed):
                if start_idx + month_offset >= len(months):
                    can_fit = False
                    break
                    
                month = months[start_idx + month_offset]
                month_allocation = total_needed / months_needed
                
                if monthly_capacity[month] < month_allocation:
                    can_fit = False
                    break
            
            if can_fit:
                best_start_idx = start_idx
                break
        
        # Assign property to best available slot
        if best_start_idx is not None:
            start_month = months[best_start_idx]
            
            if duration == 1:
                end_month = start_month
                monthly_capacity[start_month] -= units
            else:
                end_month_idx = min(best_start_idx + duration - 1, len(months) - 1)
                end_month = months[end_month_idx]
                
                # Distribute units across enrollment months
                units_per_month = units / duration
                for month_offset in range(duration):
                    if best_start_idx + month_offset < len(months):
                        month = months[best_start_idx + month_offset]
                        monthly_capacity[month] -= units_per_month
            
            property_schedule[prop] = {
                'units': units,
                'start_month': start_month,
                'end_month': end_month,
                'duration': duration
            }
            
            print(f"{property_counter:<3} {prop:<12} {units:<8} {duration:<10} {start_month:<12} {end_month}")
            property_counter += 1
        else:
            # Fallback: assign to month with most remaining capacity
            max_capacity_month = max(monthly_capacity.items(), key=lambda x: x[1])[0]
            monthly_capacity[max_capacity_month] -= units
            
            property_schedule[prop] = {
                'units': units,
                'start_month': max_capacity_month,
                'end_month': max_capacity_month,
                'duration': 1
            }
            
            print(f"{property_counter:<3} {prop:<12} {units:<8} {'1':<10} {max_capacity_month:<12} {max_capacity_month}")
            property_counter += 1
    
    print("-" * 88)
    
    # Show monthly property counts for parallel analysis
    print(f"\nPARALLEL PROPERTIES BY MONTH:")
    print("=" * 80)
    print(f"{'Month':<12} {'Count':<8} {'Total Units':<12} {'Active Properties'}")
    print("=" * 80)
    
    for month in months:
        active_props = []
        month_units = 0
        
        for prop, schedule in property_schedule.items():
            # Check if property is active in this month
            start_idx = months.index(schedule['start_month'])
            end_idx = months.index(schedule['end_month'])
            current_idx = months.index(month)
            
            if start_idx <= current_idx <= end_idx:
                active_props.append(prop)
                month_units += schedule['units'] / schedule['duration']
        
        # Format property list for better readability
        if len(active_props) <= 4:
            props_display = ', '.join(active_props)
        else:
            props_display = f"{', '.join(active_props[:3])}, +{len(active_props)-3} more"
            
        print(f"{month:<12} {len(active_props):<8} {month_units:<12,.0f} {props_display}")
    
    print("=" * 80)
    
    timeline_data = {
        'monthly_targets': monthly_targets,
        'property_schedule': property_schedule,
        'months': months,
        'total_units': total_units,
        'total_months': total_months
    }
    
    print(f"\n✓ Timeline design complete - Ready for Phase 3: Visualization")
    return timeline_data

def create_weekly_cumulative_enrollment_graph(timeline_data):
    """
    Create a weekly cumulative enrollment distribution graph
    
    Args:
        timeline_data (dict): Timeline data from design_enrollment_timeline()
    """
    print(f"\nPHASE 3: WEEKLY CUMULATIVE ENROLLMENT VISUALIZATION")
    print("="*80)
    
    # Extract data
    property_schedule = timeline_data['property_schedule']
    months = timeline_data['months']
    target_total = timeline_data['total_units']
    
    # Convert months to weeks (approximately 4.33 weeks per month)
    weeks_per_month = 4.33
    total_weeks = int(len(months) * weeks_per_month)  # April-December = 9 months = ~39 weeks
    
    # Today is Feb 12, 2026 = Week 7, April starts at Week 15
    current_week = 7  # Feb 12, 2026
    april_start_week = 15  # Week 15 is approximate start of April
    
    # Create week labels starting from April (Week 15)
    week_numbers = list(range(april_start_week, april_start_week + total_weeks))
    week_labels = [f"Week {i}" for i in week_numbers]
    
    # Initialize weekly enrollment tracking
    weekly_enrollments = [0] * total_weeks
    
    print(f"Converting monthly schedule to weekly distribution...")
    print(f"Timeline: April-December = {len(months)} months → {total_weeks} weeks")
    
    # Distribute property enrollments across weeks
    for prop, schedule in property_schedule.items():
        units = schedule['units']
        start_month = schedule['start_month']
        end_month = schedule['end_month']
        duration = schedule['duration']
        
        # Convert months to week indices
        start_month_idx = months.index(start_month)
        end_month_idx = months.index(end_month)
        
        start_week = int(start_month_idx * weeks_per_month)
        end_week = min(int((end_month_idx + 1) * weeks_per_month) - 1, total_weeks - 1)
        
        # Distribute units evenly across enrollment weeks
        if duration == 1:
            # Single month enrollment - concentrate in mid-month weeks
            mid_week = start_week + 2 if start_week + 2 < total_weeks else start_week
            weekly_enrollments[mid_week] += units
        else:
            # Multi-month enrollment - distribute evenly
            enrollment_weeks = end_week - start_week + 1
            units_per_week = units / enrollment_weeks
            
            for week_idx in range(start_week, end_week + 1):
                if week_idx < total_weeks:
                    weekly_enrollments[week_idx] += units_per_week
    
    # Calculate cumulative enrollments
    cumulative_enrollments = []
    running_total = 0
    
    for weekly_units in weekly_enrollments:
        running_total += weekly_units
        cumulative_enrollments.append(int(running_total))
    
    # Create the visualization
    plt.figure(figsize=(15, 8))
    
    # Plot cumulative enrollment line using correct week numbers
    plt.plot(week_numbers, cumulative_enrollments, 
             linewidth=3, color='#2E86AB', marker='o', markersize=4, 
             label='Cumulative Apartments Enrolled')
    
    # Add area under curve
    plt.fill_between(week_numbers, cumulative_enrollments, 
                     alpha=0.3, color='#A23B72', label='Enrollment Progress')
    
    # Formatting
    plt.title(f'TLW Enrollment: Weekly Cumulative Distribution\n{target_total:,} Apartment Units Timeline (April-December)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f'Timeline - April to December 2026 (Week {week_numbers[0]} to {week_numbers[-1]})', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Apartments Enrolled', fontsize=12, fontweight='bold')
    
    # Add milestone markers (25%, 50%, 75% only - 100% is shown in final annotation)
    milestones = [0.25, 0.5, 0.75]
    milestone_colors = ['orange', 'gold', 'lightgreen']
    
    for i, (milestone, color) in enumerate(zip(milestones, milestone_colors)):
        milestone_units = int(target_total * milestone)
        plt.axhline(y=milestone_units, color=color, linestyle=':', alpha=0.6,
                   label=f'{int(milestone*100)}% ({milestone_units:,} units)')
        
        # Find intersection week for this milestone
        milestone_week_idx = next((w for w, cum in enumerate(cumulative_enrollments) 
                             if cum >= milestone_units), total_weeks-1)
        milestone_week_num = week_numbers[milestone_week_idx]
        
        # Add annotation at intersection
        plt.annotate(f'Week {milestone_week_num}\n{milestone_units:,} units', 
                    xy=(milestone_week_num, milestone_units),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Format y-axis with comma separators
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set x-axis limits to show full timeline
    plt.xlim(week_numbers[0]-1, week_numbers[-1]+1)
    
    # Add x-axis ticks at key intervals
    tick_step = 5 if len(week_numbers) > 20 else 2
    plt.xticks(range(week_numbers[0], week_numbers[-1]+1, tick_step))
    
    # Add text annotations for initial and final numbers
    plt.text(week_numbers[0], cumulative_enrollments[0] + target_total*0.02, 
             f'Start: Week {week_numbers[0]}\n{cumulative_enrollments[0]} units', 
             ha='left', va='bottom', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    plt.annotate(f'End: Week {week_numbers[-1]}\n{cumulative_enrollments[-1]:,} units', 
                xy=(week_numbers[-1], cumulative_enrollments[-1]),
                xytext=(-10, -10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Legend
    plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    # Tight layout
    plt.tight_layout()
    
    # Display statistics
    print(f"\nWEEKLY ENROLLMENT STATISTICS:")
    print("-" * 50)
    print(f"Total Timeline: {total_weeks} weeks (April-December = {len(months)} months)")
    print(f"Total Units: {target_total:,} apartments")
    print(f"Average per week: {target_total/total_weeks:,.1f} units")
    print(f"Peak week enrollment: {max(weekly_enrollments):,.0f} units")
    print(f"Final cumulative: {cumulative_enrollments[-1]:,} units")
    
    # Show key milestones
    print(f"\nKEY MILESTONES (April-December 2026):")
    print("-" * 50)
    months_names = timeline_data['months']  # Should be ['April', 'May', 'June', etc.]
    
    for milestone in milestones:
        milestone_units = int(target_total * milestone)
        # Find first week reaching this milestone
        milestone_week_idx = next((i for i, cum in enumerate(cumulative_enrollments) 
                             if cum >= milestone_units), total_weeks-1)
        milestone_week_num = week_numbers[milestone_week_idx]
        milestone_month_idx = int(milestone_week_idx / weeks_per_month)
        milestone_month_name = months_names[milestone_month_idx] if milestone_month_idx < len(months_names) else "December"
        
        print(f"{int(milestone*100):>3}%: Week {milestone_week_num:>2} ({milestone_month_name}) - {milestone_units:,} units")
    
    print("-" * 50)
    
    plt.show()
    print(f"\n✓ Weekly cumulative enrollment graph generated successfully!")

def create_property_gantt_chart(timeline_data):
    """
    Create a Gantt chart showing enrollment timeline for each property
    
    Args:
        timeline_data (dict): Timeline data from design_enrollment_timeline()
    """
    print(f"\nPHASE 4: PROPERTY ENROLLMENT GANTT CHART")
    print("="*80)
    
    property_schedule = timeline_data['property_schedule']
    months = timeline_data['months']
    
    # Convert months to week numbers (starting from Week 15 = April)
    april_start_week = 15
    weeks_per_month = 4.33
    
    # Prepare data for Gantt chart
    properties = []
    start_weeks = []
    durations_weeks = []
    units = []
    colors = []
    
    # Color palette for different property sizes
    color_map = {
        'Large (400+ units)': '#FF6B6B',     # Red
        'Medium (200-399 units)': '#4ECDC4',  # Teal  
        'Small (100-199 units)': '#45B7D1',   # Blue
        'Micro (<100 units)': '#96CEB4'       # Green
    }
    
    print(f"PROPERTY ENROLLMENT SCHEDULE:")
    print("-" * 95)
    print(f"{'Property':<12} {'Units':<8} {'Start':<12} {'End':<12} {'Duration':<10} {'Weeks':<10}")
    print("-" * 95)
    
    for prop, schedule in property_schedule.items():
        prop_units = schedule['units']
        start_month = schedule['start_month']
        end_month = schedule['end_month']
        duration_months = schedule['duration']
        
        # Convert to weeks
        start_month_idx = months.index(start_month)
        end_month_idx = months.index(end_month)
        
        start_week = april_start_week + int(start_month_idx * weeks_per_month)
        end_week = april_start_week + int((end_month_idx + 1) * weeks_per_month) - 1
        duration_weeks = end_week - start_week + 1
        
        # Categorize by size for color coding
        if prop_units >= 400:
            category = 'Large (400+ units)'
        elif prop_units >= 200:
            category = 'Medium (200-399 units)'
        elif prop_units >= 100:
            category = 'Small (100-199 units)'
        else:
            category = 'Micro (<100 units)'
        
        properties.append(prop)
        start_weeks.append(start_week)
        durations_weeks.append(duration_weeks)
        units.append(prop_units)
        colors.append(color_map[category])
        
        print(f"{prop:<12} {prop_units:<8} {start_month:<12} {end_month:<12} {duration_months:<10} {duration_weeks:<10}")
    
    print("-" * 95)
    
    # Create Gantt chart
    fig, ax = plt.subplots(figsize=(16, max(8, len(properties) * 0.4)))
    
    # Create horizontal bars
    y_positions = range(len(properties))
    bars = ax.barh(y_positions, durations_weeks, left=start_weeks, 
                   color=colors, alpha=0.8, height=0.6)
    
    # Add unit count labels on bars
    for i, (bar, unit_count) in enumerate(zip(bars, units)):
        width = bar.get_width()
        x_pos = bar.get_x() + width / 2
        y_pos = bar.get_y() + bar.get_height() / 2
        
        # Add unit count in the middle of the bar
        ax.text(x_pos, y_pos, f'{unit_count} units', 
                ha='center', va='center', fontweight='bold', fontsize=9,
                color='white' if width > 3 else 'black')
        
        # Add week range at the end of the bar
        end_week = start_weeks[i] + durations_weeks[i] - 1
        ax.text(end_week + 0.5, y_pos, f'W{start_weeks[i]}-{end_week}', 
                ha='left', va='center', fontsize=8, fontweight='bold')
    
    # Customize chart
    ax.set_yticks(y_positions)
    ax.set_yticklabels(properties)
    ax.set_xlabel('Timeline - April to December 2026 (Week Numbers)', fontweight='bold')
    ax.set_ylabel('Properties', fontweight='bold')
    ax.set_title('TLW Property Enrollment Schedule - Gantt Chart\nProperty-by-Property Timeline', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add vertical lines for month boundaries
    month_boundaries = []
    for i in range(len(months)):
        week_boundary = april_start_week + int(i * weeks_per_month)
        month_boundaries.append(week_boundary)
        ax.axvline(x=week_boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Add month labels at the top
        if i < len(months):
            ax.text(week_boundary + 2, len(properties), months[i], 
                   rotation=45, ha='left', va='bottom', fontsize=9, fontweight='bold')
    
    # Add Today's week marker
    today_week = 7  # Feb 12, 2026
    ax.axvline(x=today_week, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'Today (Week {today_week})')
    
    # Format x-axis
    ax.set_xlim(0, april_start_week + int(len(months) * weeks_per_month) + 2)
    
    # Add grid - both horizontal and vertical for easy tracking
    ax.grid(True, alpha=0.3, axis='both')
    ax.set_axisbelow(True)
    
    # Create legend for property sizes
    legend_elements = []
    for category, color in color_map.items():
        count = sum(1 for c in colors if c == color)
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                           label=f'{category} ({count} properties)'))
    
    # Add today marker to legend
    legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=2, 
                                    label=f'Today: Week {today_week} (Feb 12)'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Tight layout
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    total_properties = len(properties)
    total_units_gantt = sum(units)
    earliest_start = min(start_weeks)
    latest_end = max([start + duration for start, duration in zip(start_weeks, durations_weeks)])
    
    print(f"\nGANTT CHART SUMMARY:")
    print("-" * 50)
    print(f"Total properties: {total_properties}")
    print(f"Total units: {total_units_gantt:,}")
    print(f"Timeline span: Week {earliest_start} to Week {latest_end-1}")
    print(f"Total duration: {latest_end - earliest_start} weeks")
    print(f"Average units per property: {total_units_gantt/total_properties:,.0f}")
    
    # Show property size distribution
    print(f"\nPROPERTY SIZE DISTRIBUTION:")
    print("-" * 50)
    for category, color in color_map.items():
        count = sum(1 for c in colors if c == color)
        category_units = sum(u for u, c in zip(units, colors) if c == color)
        if count > 0:
            print(f"{category}: {count} properties, {category_units:,} units")
    
    print(f"\n✓ Property Gantt chart generated successfully!")

def generate_excel_reports(df, tlw_portfolio, timeline_data):
    """
    Generate Excel reports for Acento Apartments and TLW Rollout
    
    Args:
        df (pd.DataFrame): Original dataset
        tlw_portfolio (pd.DataFrame): TLW properties portfolio  
        timeline_data (dict): Timeline data from design_enrollment_timeline()
    """
    print(f"\nPHASE 5: EXCEL REPORTS GENERATION")
    print("="*80)
    
    # 1. Generate Acento Apartments report (all 30 properties)
    print(f"Generating Acento Apartments report...")
    
    # Select required columns for all properties
    acento_columns = ['Unit_ID', 'Key', 'Property', 'Units']
    acento_report = df[acento_columns].copy()
    
    # Sort by Property and then by Unit_ID 
    acento_report = acento_report.sort_values(['Property', 'Unit_ID'])
    
    # Generate Excel file
    acento_filename = 'acento_apartments.xlsx'
    with pd.ExcelWriter(acento_filename, engine='openpyxl') as writer:
        acento_report.to_excel(writer, sheet_name='All_Properties', index=False)
    
    print(f"✓ Created: {acento_filename}")
    print(f"  - Total properties: {acento_report['Property'].nunique()}")
    print(f"  - Total units: {len(acento_report):,}")
    
    # 2. Generate TLW Rollout report (26 properties with week assignments)
    print(f"\nGenerating TLW Rollout report...")
    
    # Filter for TLW properties only
    tlw_properties = tlw_portfolio['Property'].tolist()
    tlw_data = df[df['Property'].isin(tlw_properties)].copy()
    
    # Calculate week rollout for each unit based on property schedule
    property_schedule = timeline_data['property_schedule']
    months = timeline_data['months']
    weeks_per_month = 4.33
    april_start_week = 15
    
    # Create week assignment mapping
    week_assignments = []
    
    print(f"\nCalculating rollout weeks for each unit...")
    print("-" * 70)
    print(f"{'Property':<12} {'Units':<8} {'Start Week':<12} {'End Week':<12}")
    print("-" * 70)
    
    for prop in tlw_properties:
        if prop in property_schedule:
            schedule = property_schedule[prop]
            start_month = schedule['start_month']
            end_month = schedule['end_month']
            
            # Convert to weeks
            start_month_idx = months.index(start_month)
            end_month_idx = months.index(end_month)
            
            start_week = april_start_week + int(start_month_idx * weeks_per_month)
            end_week = april_start_week + int((end_month_idx + 1) * weeks_per_month) - 1
            
            # Get all units for this property
            property_units = tlw_data[tlw_data['Property'] == prop]
            total_units_prop = len(property_units)
            
            # Distribute units evenly across the rollout period
            rollout_weeks = list(range(start_week, end_week + 1))
            
            # Assign each unit to a specific week
            for idx, (_, unit_row) in enumerate(property_units.iterrows()):
                # Distribute units evenly across available weeks
                week_idx = idx % len(rollout_weeks) if len(rollout_weeks) > 0 else 0
                assigned_week = rollout_weeks[week_idx]
                
                week_assignments.append({
                    'Unit_ID': unit_row['Unit_ID'],
                    'Key': unit_row['Key'],
                    'Property': unit_row['Property'],
                    'Units': unit_row['Units'],
                    'Week': assigned_week
                })
            
            print(f"{prop:<12} {total_units_prop:<8} {start_week:<12} {end_week:<12}")
    
    print("-" * 70)
    
    # Create TLW rollout DataFrame
    tlw_rollout_df = pd.DataFrame(week_assignments)
    
    # Sort by rollout week, then by property, then by unit_id
    tlw_rollout_df = tlw_rollout_df.sort_values(['Week', 'Property', 'Unit_ID'])
    
    # Generate simple Excel file with single sheet
    tlw_filename = 'tlw_rollout.xlsx'
    tlw_rollout_df.to_excel(tlw_filename, index=False)
    
    print(f"✓ Created: {tlw_filename}")
    print(f"  - TLW properties: {tlw_rollout_df['Property'].nunique()}")
    print(f"  - Total units: {len(tlw_rollout_df):,}")
    print(f"  - Week range: {tlw_rollout_df['Week'].min()} to {tlw_rollout_df['Week'].max()}")
    print(f"  - Columns: Unit_ID, Key, Property, Units, Week")
    
    # Display rollout statistics
    print(f"\nTLW ROLLOUT STATISTICS:")
    print("-" * 50)
    week_stats = tlw_rollout_df['Week'].value_counts().sort_index()
    
    print(f"{'Week':<8} {'Units':<8} {'% of Total'}")
    print("-" * 30)
    total_tlw_units = len(tlw_rollout_df)
    
    for week, count in week_stats.head(10).items():  # Show first 10 weeks
        pct = (count / total_tlw_units) * 100
        print(f"Week {week:<3} {count:<8} {pct:>6.1f}%")
    
    if len(week_stats) > 10:
        remaining_weeks = len(week_stats) - 10
        remaining_units = week_stats.tail(remaining_weeks).sum()
        remaining_pct = (remaining_units / total_tlw_units) * 100
        print(f"... and {remaining_weeks} more weeks with {remaining_units} units ({remaining_pct:.1f}%)")
    
    print("-" * 50)
    print(f"Total: {total_tlw_units} units across {len(week_stats)} weeks")
    
    print(f"\n✓ Excel reports generated successfully!")
    print(f"Files created:")
    print(f"  - {acento_filename} (All 30 properties)")
    print(f"  - {tlw_filename} (26 TLW properties with rollout weeks)")

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
        
        # TLW Enrollment Analysis - Phase 1
        print("\n" + "="*80)
        print("TENANT LIABILITY WAIVER (TLW) ENROLLMENT PLANNING")
        print("="*80)
        
        tlw_portfolio = prepare_tlw_enrollment_portfolio(df)
        
        # Phase 2: Timeline & Enrollment Curve Design
        timeline_data = design_enrollment_timeline(tlw_portfolio)
        
        # Phase 3: Visualization - Weekly Cumulative Enrollment Graph
        create_weekly_cumulative_enrollment_graph(timeline_data)
        
        # Phase 4: Property Gantt Chart 
        create_property_gantt_chart(timeline_data)
        
        # Phase 5: Generate Excel Reports
        generate_excel_reports(df, tlw_portfolio, timeline_data)
        
        return df, tlw_portfolio, timeline_data
    
    return None, None, None

if __name__ == "__main__":
    df, tlw_portfolio, timeline_data = main()
    
    # Generate additional visualizations if data is available
    if timeline_data is not None:
        print("\n" + "="*80)
        print("ADDITIONAL PHASE 3 VISUALIZATIONS")
        print("="*80)
        print("\n📊 Generating weekly cumulative enrollment graph...")
