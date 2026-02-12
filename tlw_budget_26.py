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

def optimize_rollout_timeline(property_units, start_date='2026-04-01', target_monthly_units=None):
    """
    Simple, working rollout timeline optimization
    """
    print(f"\nROLLOUT TIMELINE OPTIMIZATION:")
    print("="*80)
    
    from datetime import datetime, timedelta
    import calendar
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime(2026, 12, 31)
    properties = property_units.copy().sort_values('Units')  # Start with smallest
    
    total_units = properties['Units'].sum()
    total_months = 9  # April through December
    calculated_target = total_units / total_months
    
    if target_monthly_units is None:
        target_monthly_units = calculated_target
    
    print(f"Constraint: Complete {total_units:,} units from April-December 2026")
    print(f"Target: {target_monthly_units:.0f} units/month")
    
    # Simple sequential approach with smart overlaps
    timeline = []
    monthly_totals = {i: 0 for i in range(4, 13)}  # April=4 to December=12
    current_start = start_dt
    
    print(f"\nProperty sequence:")
    print("-" * 80)
    print(f"{'Property':<12} {'Units':<8} {'Start':<12} {'End':<12} {'Duration':<10}")
    print("-" * 80)
    
    for _, prop in properties.iterrows():
        # Calculate duration: 1 month minimum, 1.5 months for large properties
        if prop['Units'] >= 400:
            duration_days = 45  # 1.5 months for large properties
        elif prop['Units'] >= 200:
            duration_days = 35  # ~1.2 months for medium
        else:
            duration_days = 30  # 1 month for small
        
        prop_end = current_start + timedelta(days=duration_days)
        
        # Ensure we don't exceed December 31, 2026
        if prop_end > end_dt:
            prop_end = end_dt
            duration_days = (prop_end - current_start).days
        
        # Distribute units across the months this property spans
        units_per_day = prop['Units'] / duration_days if duration_days > 0 else 0
        
        # Calculate monthly contributions
        current_date = current_start
        while current_date < prop_end:
            month_num = current_date.month
            if 4 <= month_num <= 12:  # April to December only
                # Calculate last day of current month
                _, last_day = calendar.monthrange(current_date.year, month_num)
                month_end_date = datetime(current_date.year, month_num, last_day)
                
                # Find the actual end date for this month
                actual_end = min(month_end_date, prop_end - timedelta(days=1))
                days_in_this_month = (actual_end - current_date).days + 1
                
                if days_in_this_month > 0:
                    monthly_totals[month_num] += units_per_day * days_in_this_month
            
            # Move to first day of next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        timeline.append({
            'Property': prop['Property'],
            'Units': prop['Units'],
            'Start_Date': current_start,
            'End_Date': prop_end,
            'Duration_Days': duration_days
        })
        
        print(f"{prop['Property']:<12} {prop['Units']:<8} "
              f"{current_start.strftime('%Y-%m-%d'):<12} "
              f"{prop_end.strftime('%Y-%m-%d'):<12} "
              f"{duration_days} days")
        
        # Aggressive overlapping to fit all properties
        # Calculate remaining properties and time
        remaining_props = len(properties) - len(timeline)
        remaining_days = (end_dt - current_start).days
        
        if remaining_props > 0 and remaining_days > 0:
            # Ensure we can fit remaining properties
            max_gap = min(10, remaining_days // remaining_props)  # Max 10 days between starts
            if prop['Units'] >= 400:
                current_start = current_start + timedelta(days=max(max_gap, 7))  # Week minimum
            elif prop['Units'] >= 200:
                current_start = current_start + timedelta(days=max(max_gap, 5))  # 5 days minimum
            else:
                current_start = current_start + timedelta(days=max(max_gap, 3))  # 3 days minimum
        else:
            current_start = current_start + timedelta(days=7)  # Default weekly
    
    # Calculate results
    month_values = [monthly_totals[i] for i in range(4, 13)]
    actual_avg = np.mean(month_values)
    std_dev = np.std(month_values)
    
    print("-" * 80)
    print(f"\nMONTHLY DISTRIBUTION:")
    print("-" * 60)
    month_names = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    for i, month_name in enumerate(month_names):
        month_num = i + 4
        units = monthly_totals[month_num]
        variance_pct = ((units - target_monthly_units) / target_monthly_units) * 100 if target_monthly_units > 0 else 0
        print(f"{month_name:<12} {units:>7.0f} units ({variance_pct:>+5.1f}%)")
    
    print("-" * 60)
    print(f"Target:       {target_monthly_units:>7.0f} units/month")
    print(f"Actual Avg:   {actual_avg:>7.0f} units/month")
    print(f"Std Dev:      {std_dev:>7.0f} units ({(std_dev/actual_avg)*100:.1f}%)")
    print(f"Total:        {sum(month_values):>7.0f} units")
    print(f"Properties:   {len(timeline)}/{len(properties)}")
    
    return timeline, monthly_totals

def optimize_property_sequence(properties, target_monthly_units):
    """
    Find optimal property sequence to minimize monthly variance
    """
    # Start with smallest properties for early wins
    small_props = properties[properties['Units'] <= 200].copy()
    medium_props = properties[(properties['Units'] > 200) & (properties['Units'] <= 400)].copy()
    large_props = properties[properties['Units'] > 400].copy()
    
    # Strategic sequencing: mix sizes to balance monthly loads
    sequence = []
    
    # Add smallest first (quick wins)
    sequence.extend(small_props.sort_values('Units').to_dict('records'))
    
    # Intersperse medium properties
    sequence.extend(medium_props.sort_values('Units').to_dict('records'))
    
    # Add large properties at strategic points
    sequence.extend(large_props.sort_values('Units').to_dict('records'))
    
    return pd.DataFrame(sequence)

def add_months(date, months):
    """Helper function to add months to a date"""
    import calendar
    month = date.month + months
    year = date.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    day = min(date.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day)

def create_rollout_visualization(timeline, monthly_totals):
    """
    Create rollout table and progress visualization
    """
    
    # Table: Weekly Rollout Schedule
    print(f"\n" + "="*100)
    print("TLW ROLLOUT SCHEDULE - WEEKLY BREAKDOWN")
    print("="*100)
    
    # Create weekly timeline table
    from datetime import datetime, timedelta
    import calendar
    
    # Generate week numbers for 2026
    start_date = datetime(2026, 4, 1)  # April 1, 2026
    week_headers = []
    week_dates = []
    current_week = start_date
    
    for week_num in range(1, 40):  # Generate enough weeks through December
        week_end = current_week + timedelta(days=6)
        if current_week.month > 12:  # Stop at end of year
            break
        week_headers.append(f"W{week_num:2d}")
        week_dates.append(f"{current_week.strftime('%m/%d')}-{week_end.strftime('%m/%d')}")
        current_week += timedelta(days=7)
    
    # Print header
    print(f"{'Property':<12} {'Units':<6} {'Start':<12} {'End':<12} {'Weeks':<6} {'Weekly Schedule'}")
    print("-" * 100)
    
    # Create timeline for each property
    for prop in timeline:
        start_date = prop['Start_Date']
        end_date = prop['End_Date']
        duration_weeks = max(1, int((end_date - start_date).days / 7))
        
        # Find which weeks this property spans
        base_date = datetime(2026, 4, 1)
        start_week = max(1, int((start_date - base_date).days / 7) + 1)
        end_week = min(len(week_headers), start_week + duration_weeks - 1)
        
        # Create visual timeline
        timeline_str = ""
        for i in range(len(week_headers)):
            if start_week <= i+1 <= end_week:
                timeline_str += "█"
            elif i+1 == start_week - 1 or i+1 == end_week + 1:
                timeline_str += "▌"
            else:
                timeline_str += "░"
            
        print(f"{prop['Property']:<12} {prop['Units']:<6} "
              f"{start_date.strftime('%Y-%m-%d'):<12} {end_date.strftime('%Y-%m-%d'):<12} "
              f"{duration_weeks:<6} {timeline_str[:35]}...")
    
    # Print week number reference
    print("-" * 100)
    print(f"{'Week #':<12} {'Units':<6} {'Start':<12} {'End':<12} {'Weeks':<6} " + 
          "".join([f"{i+1:2d}" if (i+1) % 2 == 1 else "  " for i in range(35)]))
    print(f"{'Dates':<48} " + 
          " ".join([week_dates[i][:5] if i < len(week_dates) and (i+1) % 2 == 1 else "     " for i in range(35)]))
    
    print("\nLegend: █ = Active Period  ▌ = Transition  ░ = Not Active")
    
    # Summary table by month
    print(f"\n" + "="*80)
    print("MONTHLY SUMMARY TABLE")
    print("="*80)
    
    months = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    target = 803
    
    print(f"{'Month':<12} {'Units':<8} {'Target':<8} {'Variance':<10} {'Variance %':<12} {'Status'}")
    print("-" * 70)
    
    for i, month in enumerate(months):
        month_num = i + 4
        units = monthly_totals[month_num]
        variance = units - target
        variance_pct = (variance / target) * 100
        
        # Status indicator
        if abs(variance_pct) <= 10:
            status = "✓ Good"
        elif abs(variance_pct) <= 20:
            status = "⚠ Caution"
        else:
            status = "⚠ Review"
            
        print(f"{month:<12} {units:<8.0f} {target:<8} {variance:<10.0f} {variance_pct:<12.1f}% {status}")
    
    print("-" * 70)
    total_units = sum([monthly_totals[i] for i in range(4, 13)])
    avg_variance = np.std([monthly_totals[i] for i in range(4, 13)])
    print(f"{'TOTAL':<12} {total_units:<8.0f} {target*9:<8.0f} {total_units-target*9:<10.0f} "
          f"Std Dev: {avg_variance:.0f}")
    
    # Chart 2: Monthly Progress Distribution (keep the visual chart)
    plt.figure(figsize=(16, 8))
    plt.title('TLW Rollout - Monthly Unit Distribution', fontsize=18, fontweight='bold', pad=30)
    
    months_short = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    values = [monthly_totals[i] for i in range(4, 13)]
    
    # Create bars with gradient colors based on variance
    colors_list = []
    for val in values:
        variance = (val - target) / target
        if variance > 0.2:  # High above target
            colors_list.append('#ff4444')  # Red
        elif variance > 0.1:
            colors_list.append('#ffaa44')  # Orange
        elif variance > -0.1:
            colors_list.append('#44ff44')  # Green (close to target)
        elif variance > -0.2:
            colors_list.append('#44aaff')  # Blue
        else:
            colors_list.append('#4444ff')  # Dark blue (low)
    
    bars = plt.bar(months_short, values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=2)
    plt.axhline(y=target, color='red', linestyle='--', linewidth=3, label=f'Target ({target} units)', alpha=0.8)
    
    # Enhanced value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        variance = ((height - target) / target) * 100
        
        label_y = height + 30 if height > target * 0.8 else height + 50
        
        plt.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{height:.0f}\n({variance:+.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.ylabel('Units Enrolled', fontweight='bold', fontsize=14)
    plt.xlabel('Month (2026)', fontweight='bold', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add summary statistics box  
    mean_val = np.mean(values)
    std_val = np.std(values)
    min_val = min(values)
    max_val = max(values)
    target = 707  # Use correct TLW target
    
    stats_text = f'''TLW Rollout Summary:
Target: {target:.0f} units/month
Actual Avg: {mean_val:.0f} units/month
Std Dev: {std_val:.0f} units ({std_val/mean_val*100:.1f}%)
Range: {min_val:.0f} - {max_val:.0f} units
Variance: {max_val - min_val:.0f} units'''
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Summary Report
    print(f"\n📊 ROLLOUT PLANNING COMPLETE")
    print(f"Properties scheduled: {len(timeline)}")
    print(f"Timeline: April 2026 - December 2026") 
    print(f"Monthly variance: ±{np.std(values):.0f} units ({np.std(values)/np.mean(values)*100:.1f}%)")
    print(f"Peak month: {months_short[values.index(max(values))]} ({max(values):.0f} units)")
    print(f"Lowest month: {months_short[values.index(min(values))]} ({min(values):.0f} units)")

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
        if property_units is not None:
            timeline, monthly_totals = optimize_rollout_timeline(property_units)
            create_rollout_visualization(timeline, monthly_totals)
        
        print("\n" + "="*80)
        print("COMPLETE ANALYSIS FINISHED - Including TLW Rollout Plan")
        print("="*80)
        
        return df
    
    return None

if __name__ == "__main__":
    result = main()
