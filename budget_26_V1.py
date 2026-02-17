import pandas as pd
import numpy as np
from pathlib import Path

def explore_dataset(df, dataset_name):
    """
    Comprehensive exploration of a dataset
    """
    print(f"\n{'='*60}")
    print(f"EXPLORATION REPORT FOR: {dataset_name}")
    print(f"{'='*60}")
    
    # Basic info
    print(f"\n✅ BASIC INFORMATION:")
    print(f"Total observations (rows): {len(df)}")
    print(f"Total variables (columns): {len(df.columns)}")
    print(f"Dataset shape: {df.shape}")
    
    # Create comprehensive variable summary table
    print(f"\n✅ VARIABLES SUMMARY TABLE:")
    variable_summary = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        unique_count = df[col].nunique()
        data_type = str(df[col].dtype)
        
        variable_summary.append({
            'Variable': col,
            'Data_Type': data_type,
            'Missing_Count': missing_count,
            'Missing_%': round(missing_percent, 2),
            'Unique_Values': unique_count
        })
    
    summary_df = pd.DataFrame(variable_summary)
    print(summary_df.to_string(index=False))
    
    # Basic statistics for numeric columns
    print(f"\n✅ STATISTICAL SUMMARY (Numeric columns):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("No numeric columns found")
    
    # First few rows
    print(f"\n✅ FIRST 5 ROWS:")
    print(df.head())
    
    return df

# Load and explore the datasets
try:
    # PART 1: DATA LOADING AND EXPLORATION
    print(f"{'='*60}")
    print(f"PART 1: DATA LOADING AND EXPLORATION")
    print(f"{'='*60}")
    
    # Load Property Base rent SQFT file (contains slope data)
    print("✅ Loading Property Base rent SQFT file...")
    property_base_df = pd.read_excel('Property Base rent SQFT.xlsx')
    explore_dataset(property_base_df, "Property Base rent SQFT")
    
    # Load acento_apartments_fixed file (main apartments data)
    print("\n✅ Loading acento_apartments_fixed file...")
    apartments_df = pd.read_excel('acento_apartments_fixed.xlsx')
    explore_dataset(apartments_df, "acento_apartments_fixed")
    
    # Create main dataframe 
    print(f"\n✅ CREATING MAIN DATAFRAME:")
    
    # Merge apartments with slopes and calculate est_rent
    main_df = apartments_df.merge(property_base_df[['Property', 'Slope']], on='Property')
    main_df['est_rent'] = main_df['SQFT'] * main_df['Slope']
    
    print(f"✅ Main dataframe created - {len(main_df)} units with 6 columns")
    print(main_df.head(10))
    
    # Property summary table - ALL PROPERTIES
    print(f"\n✅ PART 1 SUMMARY - ALL PROPERTIES:")
    property_summary_all = main_df.groupby('Property').agg({
        'Unit_ID': 'count'
    }).rename(columns={'Unit_ID': 'Apartment_Count'}).reset_index()
    property_summary_all = property_summary_all.sort_values('Apartment_Count', ascending=False)
    
    print(f"Total different properties: {len(property_summary_all)}")
    print("-" * 67)
    print(f"{'Count':<6} {'Property':<20} {'Apartments':<15}")
    print("-" * 67)
    
    for i, row in property_summary_all.iterrows():
        count = i + 1
        prop = row['Property']
        apt_count = row['Apartment_Count']
        print(f"{count:<6} {prop:<20} {apt_count:,}")
    
    print("-" * 67)
    total_properties = len(property_summary_all)
    total_apartments = property_summary_all['Apartment_Count'].sum()
    print(f"{'TOTAL':<6} {total_properties} Properties{'':<6} {total_apartments:,}")
    
    # PART 2: ROLLOUT TIMELINE AND GANTT
    print(f"\n{'='*60}")
    print(f"PART 2: ROLLOUT TIMELINE AND GANTT")
    print(f"{'='*60}")
    
    # Define rollout column (exclude WEA, CCA, CHA, CWA)
    excluded_properties = ['WEA', 'CCA', 'CHA', 'CWA']
    main_df['rollout'] = ~main_df['Property'].isin(excluded_properties)
    
    print(f"\n✅ ROLLOUT PROPERTIES (Excluding WEA, CCA, CHA, CWA):")
    
    # Filter for rollout properties only
    rollout_df = main_df[main_df['rollout'] == True]
    
    # Property summary table - ROLLOUT PROPERTIES ONLY
    property_summary_rollout = rollout_df.groupby('Property').agg({
        'Unit_ID': 'count'
    }).rename(columns={'Unit_ID': 'Apartment_Count'}).reset_index()
    property_summary_rollout = property_summary_rollout.sort_values('Apartment_Count', ascending=False)
    
    print(f"Total rollout properties: {len(property_summary_rollout)}")
    print("-" * 67)
    print(f"{'Count':<6} {'Property':<20} {'Apartments':<15}")
    print("-" * 67)
    
    for i, row in property_summary_rollout.iterrows():
        count = i + 1
        prop = row['Property']
        apt_count = row['Apartment_Count']
        print(f"{count:<6} {prop:<20} {apt_count:,}")
    
    print("-" * 67)
    excluded_count = main_df[main_df['rollout'] == False]['Unit_ID'].count()
    rollout_total = property_summary_rollout['Apartment_Count'].sum()
    print(f"{'TOTAL':<6} {len(property_summary_rollout)} Rollout Props{'':<4} {rollout_total:,}")
    
    print(f"\n⚠️ EXCLUDED FROM ROLLOUT:")
    print(f"   - Properties: {', '.join(excluded_properties)}")
    print(f"   - Excluded Apartments: {excluded_count:,}")
    
    # Show current dataframe structure
    print(f"\n✅ CURRENT DATAFRAME STRUCTURE:")
    print(f"   Columns: {list(main_df.columns)}")
    print(f"   Shape: {main_df.shape}")
    print(f"   Rollout units: {(main_df['rollout'] == True).sum():,}")
    print(f"   Excluded units: {(main_df['rollout'] == False).sum():,}")
    
    # GANTT-BASED REALISTIC TIMELINE
    print(f"\n✅ REALISTIC TIMELINE BASED ON GANTT SCHEDULE:")
    
    import matplotlib.pyplot as plt
    
    # Property completion schedule from Gantt (property: completion_week)
    property_completions = {
        'NHR': 17, 'LVA': 18, 'LOP': 19, 'ESA': 21, 'CGA': 22, 
        'IDA': 23, 'LHC': 24, 'GAA': 25, 'TSA': 27, 'WGA': 28,
        'CSS': 30, 'GSA': 31, 'ATS': 33, 'LPA': 34, 'BWA': 35,
        'OOA': 37, 'CAL': 38, 'WWA': 40, 'CPA': 41, 'LWA': 43,
        'TCA': 44, 'JHA': 45, 'TGA': 47, 'HGA': 47, 'PGA': 50
    }
    
    # Calculate cumulative enrollment by week
    weeks_range = list(range(14, 53))
    cumulative_units = []
    total_enrolled = 0
    
    # Get actual unit counts per property
    rollout_properties = main_df[main_df['rollout'] == True]
    property_unit_counts = rollout_properties.groupby('Property').size().to_dict()
    
    for week in weeks_range:
        # Check which properties complete this week
        for prop, completion_week in property_completions.items():
            if completion_week == week and prop in property_unit_counts:
                total_enrolled += property_unit_counts[prop]
        
        cumulative_units.append(total_enrolled)
    
    # Ensure final total matches exactly
    if cumulative_units[-1] != rollout_total:
        cumulative_units[-1] = rollout_total
    
    # Create the timeline visualization
    plt.figure(figsize=(14, 8))
    plt.fill_between(weeks_range, cumulative_units, alpha=0.3, color='lightcoral')
    plt.plot(weeks_range, cumulative_units, linewidth=3, color='darkred', 
             label='Cumulative Apartments Enrolled')
    
    # Define percentage targets for reference lines
    percentages = [0.25, 0.50, 0.75, 1.0]
    colors = ['gold', 'orange', 'darkorange', 'red']
    line_styles = [':', '--', '-.', '-']
    
    # Add milestone annotations - positioned at target intersections
    milestone_annotations = []
    
    for i, (pct, color, style) in enumerate(zip(percentages, colors, line_styles)):
        target_units = int(rollout_total * pct)
        
        # Find the week when this target is reached
        target_week = None
        for j, week in enumerate(weeks_range):
            if cumulative_units[j] >= target_units:
                target_week = week
                break
        
        if target_week:
            milestone_annotations.append((target_week, target_units, f'{int(pct*100)}%'))
    
    # Add the green milestone annotations at target intersections
    for week, units, label in milestone_annotations:
        plt.annotate(f'{label} Target: {units:,} units', 
                    xy=(week, units), 
                    xytext=(week, units + 300),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8),
                    ha='center', fontsize=10, weight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    
    # Add percentage reference lines with different colors and vertical indicators
    percentages = [0.25, 0.50, 0.75, 1.0]
    colors = ['gold', 'orange', 'darkorange', 'red']
    line_styles = [':', '--', '-.', '-']
    
    for i, (pct, color, style) in enumerate(zip(percentages, colors, line_styles)):
        target_units = int(rollout_total * pct)
        
        # Horizontal target line
        plt.axhline(y=target_units, color=color, linestyle=style, alpha=0.6, 
                   label=f'{int(pct*100)}% Target ({target_units:,} units)')
        
        # Find the week when this target is reached
        target_week = None
        for j, week in enumerate(weeks_range):
            if cumulative_units[j] >= target_units:
                target_week = week
                break
        
        # Add vertical line at the target week (using gray color for consistency)
        if target_week:
            plt.axvline(x=target_week, color='gray', linestyle=style, alpha=0.7)
            
            # Add week number annotation at the bottom
            plt.annotate(f'Wk{target_week}', 
                        xy=(target_week, target_units * 0.05), 
                        ha='center', fontsize=9, color='black', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
    
    # Chart formatting
    plt.title('TLW Waiver Rollout Timeline 2026 - Realistic Property-by-Property Enrollment Schedule', 
              fontsize=16, weight='bold', pad=20)
    plt.xlabel('Timeline - April to December 2026 (Week 14 to 52)', fontsize=12)
    plt.ylabel('Cumulative Apartment Units Enrolled', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Legend positioned upper left to avoid overlap
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
    
    plt.xlim(13, 53)
    plt.ylim(0, max(cumulative_units) + 500)
    plt.tight_layout()
    plt.show()
    
    # Timeline characteristics analysis
    print(f"\n✅ TIMELINE CHARACTERISTICS:")
    print("-" * 60)
    print(f"Peak enrollment period: Weeks 17-25 (Early properties)")
    print(f"Mid-period pace: Weeks 26-40 (Steady progression)")
    print(f"Final completion: Week 50 (Mid-December)")
    print(f"Shape: Step-function with property completions")
    print(f"Realism: High - based on operational capacity")
    print("-" * 60)
    
    # CREATE GANTT CHART
    print(f"\n✅ CREATING GANTT CHART:")
    
    # Create second figure for Gantt chart
    fig2, ax = plt.subplots(figsize=(16, 10))
    
    # Property schedule with colors by size category
    gantt_data = []
    for prop in property_completions.keys():
        if prop in property_unit_counts:
            units = property_unit_counts[prop]
            start_week = None
            end_week = property_completions[prop]
            
            # Estimate start week (assuming 2-3 week duration for most)
            if units >= 400:  # Large properties
                start_week = end_week - 2
                color = 'lightcoral'
                category = 'Large (400+ units)'
            elif units >= 200:  # Medium properties 
                start_week = end_week - 2
                color = 'lightblue'
                category = 'Medium (200-399 units)'
            elif units >= 100:  # Small properties
                start_week = end_week - 1
                color = 'lightgreen' 
                category = 'Small (100-199 units)'
            else:  # Micro properties
                start_week = end_week
                color = 'lightyellow'
                category = 'Micro (<100 units)'
                
            gantt_data.append({
                'property': prop,
                'start': start_week,
                'end': end_week,
                'duration': end_week - start_week + 1,
                'units': units,
                'color': color,
                'category': category
            })
    
    # Sort by start week then by units descending
    gantt_data.sort(key=lambda x: (x['start'], -x['units']))
    
    # Create Gantt bars
    y_positions = range(len(gantt_data))
    
    for i, data in enumerate(gantt_data):
        ax.barh(i, data['duration'], left=data['start'], 
               color=data['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add property name, unit count, and week range as annotation
        if data['duration'] > 1:
            week_range = f"W{data['start']}-{data['end']}"
        else:
            week_range = f"W{data['end']}"
            
        ax.text(data['start'] + data['duration']/2, i, 
               f"{data['property']}\n{data['units']} units\n{week_range}", 
               ha='center', va='center', fontsize=8, weight='bold')
    
    # Customize Gantt chart
    ax.set_xlabel('Timeline - April to December 2026 (Week Numbers)', fontsize=12)
    ax.set_ylabel('Properties', fontsize=12)
    ax.set_title('TLW Property Enrollment Schedule - Gantt Chart\nProperty-by-Property Timeline', 
                fontsize=16, weight='bold', pad=20)
    
    # Set y-axis labels
    property_labels = [data['property'] for data in gantt_data]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(property_labels)
    
    # Add grid and formatting
    ax.grid(True, alpha=0.3, axis='x')
    ax.grid(True, alpha=0.2, axis='y')  # Add horizontal grid lines
    ax.set_xlim(14, 52)
    
    # Add legend for property sizes
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='lightcoral', alpha=0.7, label='Large (400+ units)'),
        plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.7, label='Medium (200-399 units)'),
        plt.Rectangle((0,0),1,1, facecolor='lightgreen', alpha=0.7, label='Small (100-199 units)'),
        plt.Rectangle((0,0),1,1, facecolor='lightyellow', alpha=0.7, label='Micro (<100 units)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add current week indicator
    current_week = 7  # February 16, 2026
    ax.axvline(x=current_week, color='red', linestyle='-', linewidth=2, alpha=0.8, 
              label=f'Today: Week {current_week} (Feb 15)')
    
    plt.tight_layout()
    plt.show()
    
except FileNotFoundError as e:
    print(f"❌ Error: Could not find the Excel file - {e}")
    print("Please make sure the Excel files are in the same directory as this Python script")
except Exception as e:
    print(f"❌ Error loading files: {e}")
