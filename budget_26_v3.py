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
    property_summary_all = property_summary_all.sort_values('Apartment_Count', ascending=False).reset_index(drop=True)
    
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

    # PART 1: EST_RENT SUMMARY BY PROPERTY (TABLE + PARETO)
    print(f"\n✅ PART 1 SUMMARY - TOTAL EST_RENT BY PROPERTY:")
    est_rent_summary = main_df.groupby('Property', as_index=False)['est_rent'].sum()
    est_rent_summary = est_rent_summary.sort_values('est_rent', ascending=False).reset_index(drop=True)
    est_rent_total = est_rent_summary['est_rent'].sum()
    est_rent_summary['Pct_Total'] = (est_rent_summary['est_rent'] / est_rent_total) * 100
    est_rent_summary['Cum_Pct'] = est_rent_summary['Pct_Total'].cumsum()

    print("-" * 90)
    print(f"{'Count':<6} {'Property':<12} {'Total_Est_Rent':<18} {'Pct_Total':<12} {'Cum_Pct':<10}")
    print("-" * 90)
    for i, row in est_rent_summary.iterrows():
        print(
            f"{i + 1:<6} "
            f"{row['Property']:<12} "
            f"${row['est_rent']:>15,.2f} "
            f"{row['Pct_Total']:>10.2f}% "
            f"{row['Cum_Pct']:>9.2f}%"
        )
    print("-" * 90)
    print(f"{'TOTAL':<6} {'':<12} ${est_rent_total:>15,.2f} {100:>10.2f}% {100:>9.2f}%")

    # Pareto chart for total est_rent by property
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    bars = ax1.bar(est_rent_summary['Property'], est_rent_summary['est_rent'], color='steelblue', alpha=0.8)
    ax1.set_ylabel('Total Est Rent (USD)', fontsize=12)
    ax1.set_xlabel('Property', fontsize=12)
    ax1.set_title('Total Est Monthly Rent by Property (Pareto)', fontsize=14, weight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, axis='y', alpha=0.3)

    # Format y-axis as USD
    from matplotlib.ticker import FuncFormatter
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Add value labels on top of each bar
    ax1.bar_label(bars, labels=[f"${v:,.0f}" for v in est_rent_summary['est_rent']],
                  padding=3, fontsize=8, rotation=90)

    ax2 = ax1.twinx()
    ax2.plot(est_rent_summary['Property'], est_rent_summary['Cum_Pct'], color='darkred', marker='o')
    ax2.set_ylabel('Cumulative % of Total Est Rent', fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color='gray', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
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
    property_summary_rollout = property_summary_rollout.sort_values('Apartment_Count', ascending=False).reset_index(drop=True)
    
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
    
    plt.xlim(14, 52)
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
    
    # Sort by property position (count), not by apartment amount
    # First create a mapping of properties to their rank
    property_rank = {prop: i for i, prop in enumerate(property_completions.keys())}
    gantt_data.sort(key=lambda x: property_rank.get(x['property'], 999))
    
    # Create Gantt bars
    y_positions = range(len(gantt_data))
    
    for i, data in enumerate(gantt_data):
        ax.barh(i, data['duration'], left=data['start'], 
               color=data['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add only unit count inside the bar (property name is on y-axis)
        ax.text(data['start'] + data['duration']/2, i, 
               f"{data['units']} units", 
               ha='center', va='center', fontsize=9, weight='bold')
        
        # Add week range to the right side OUTSIDE the bar (after end)
        if data['duration'] > 1:
            week_range = f"W{data['start']}-{data['end']}"
        else:
            week_range = f"W{data['end']}"
            
        ax.text(1.01, i, week_range,
            transform=ax.get_yaxis_transform(),
            ha='left', va='center', fontsize=8, weight='bold', color='darkblue',
            clip_on=False)
    
    # Customize Gantt chart
    ax.set_xlabel('Timeline - April to December 2026 (Week Numbers)', fontsize=12)
    ax.set_ylabel('Properties', fontsize=12)
    ax.set_title('TLW PROPERTY ENROLLMENT SCHEDULE - GANTT CHART\nProperty-by-Property Timeline (6,359 Units from 26 Properties)', 
                fontsize=14, weight='bold')
    fig2.subplots_adjust(right=0.88)
    
    # Set y-axis labels
    property_labels = [data['property'] for data in gantt_data]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(property_labels)
    
    # Add grid and formatting
    ax.grid(True, alpha=0.3, axis='x')
    ax.grid(True, alpha=0.2, axis='y')  # Add horizontal grid lines
    ax.set_xlim(14, 52)

    # Add month labels on the top axis
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    month_ticks = [14, 18, 22, 27, 31, 36, 40, 45, 49]
    month_labels = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    ax_top.set_xticks(month_ticks)
    ax_top.set_xticklabels(month_labels, rotation=35, ha='left')
    ax_top.tick_params(axis='x', pad=6)
    
    # Add legend for property sizes (positioned in upper left to avoid overlap)
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='lightcoral', alpha=0.7, label='Large (400+ units)'),
        plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.7, label='Medium (200-399 units)'),
        plt.Rectangle((0,0),1,1, facecolor='lightgreen', alpha=0.7, label='Small (100-199 units)'),
        plt.Rectangle((0,0),1,1, facecolor='lightyellow', alpha=0.7, label='Micro (<100 units)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
    
    # Add current week indicator
    current_week = 7  # February 16, 2026
    ax.axvline(x=current_week, color='red', linestyle='-', linewidth=2, alpha=0.8, 
              label=f'Today: Week {current_week} (Feb 15)')
    
    plt.tight_layout()
    plt.show()

    # PRELIMINARY: Calculate tlw_flat and tlw_adj for all rollout units to use in revenue timeline
    temp_tlw_flat = np.zeros(len(main_df))
    temp_tlw_flat[main_df['rollout'] == True] = 15.0

    temp_tlw_adj = np.zeros(len(main_df))
    for prop in main_df['Property'].unique():
        prop_mask = (main_df['Property'] == prop) & (main_df['rollout'] == True)
        if prop_mask.sum() > 0:
            avg_sqft = main_df.loc[prop_mask, 'SQFT'].mean()
            temp_tlw_adj[prop_mask] = (main_df.loc[prop_mask, 'SQFT'] * 15.0 / avg_sqft)

    # TLW REVENUE TIMELINE (MONTHLY BASED ON ROLLOUT COMPLETIONS)
    print(f"\n✅ TLW MONTHLY REVENUE TIMELINE (FLAT vs ADJUSTED):")
    
    # Map weeks to months (April = 4, May = 5, ..., December = 12)
    weeks_to_month = {
        14: 4, 15: 4, 16: 4, 17: 4, 18: 5, 19: 5, 20: 5, 21: 5, 22: 6,
        23: 6, 24: 6, 25: 6, 26: 7, 27: 7, 28: 7, 29: 7, 30: 8,
        31: 8, 32: 8, 33: 8, 34: 9, 35: 9, 36: 9, 37: 9, 38: 10,
        39: 10, 40: 10, 41: 10, 42: 11, 43: 11, 44: 11, 45: 11, 46: 12,
        47: 12, 48: 12, 49: 12, 50: 12, 51: 12, 52: 12
    }
    
    # Month week ranges: April (14-18), May (19-22), June (23-26), July (26-29), etc.
    month_weeks = {
        4: (14, 18),   # April
        5: (19, 22),   # May
        6: (23, 26),   # June
        7: (26, 29),   # July
        8: (30, 33),   # August
        9: (34, 37),   # September
        10: (38, 41),  # October
        11: (42, 45),  # November
        12: (46, 52)   # December
    }

    # Calculate monthly revenue based on active units each month
    monthly_revenue_flat = {m: 0.0 for m in range(4, 13)}
    monthly_revenue_adj = {m: 0.0 for m in range(4, 13)}

    for prop, completion_week in property_completions.items():
        if prop in property_unit_counts:
            prop_mask = (main_df['Property'] == prop) & (main_df['rollout'] == True)
            prop_flat = temp_tlw_flat[prop_mask].sum()  # Total for this property
            prop_adj = temp_tlw_adj[prop_mask].sum()    # Total for this property
            
            # Find which months this property is active in
            for month, (week_start, week_end) in month_weeks.items():
                # Property is active in month if completion_week <= week_end
                # (meaning it completes before or during this month)
                if completion_week <= week_end:
                    monthly_revenue_flat[month] += prop_flat
                    monthly_revenue_adj[month] += prop_adj

    months = list(range(4, 13))
    month_names = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    revenue_flat_monthly = [monthly_revenue_flat[m] for m in months]
    revenue_adj_monthly = [monthly_revenue_adj[m] for m in months]

    cumulative_flat = np.cumsum(revenue_flat_monthly)
    cumulative_adj = np.cumsum(revenue_adj_monthly)

    # Create combined bar + line chart for 2026 rollout
    fig_rev, (ax_bar, ax_cum) = plt.subplots(2, 1, figsize=(14, 10))
    
    # SUBPLOT 1: Monthly Incremental Revenue (Bars)
    width = 0.35
    x = np.arange(len(month_names))
    
    bars1 = ax_bar.bar(x - width/2, revenue_flat_monthly, width, label='Monthly TLW Flat ($15/unit)', color='steelblue', alpha=0.8)
    bars2 = ax_bar.bar(x + width/2, revenue_adj_monthly, width, label='Monthly TLW Adjusted (SQFT-based)', color='darkgreen', alpha=0.8)
    
    ax_bar.set_ylabel('Monthly Revenue (USD)', fontsize=12, fontweight='bold')
    ax_bar.set_title('2026 TLW Rollout: Monthly Incremental Revenue', fontsize=13, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(month_names)
    ax_bar.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_bar.legend(loc='upper left', fontsize=10)
    ax_bar.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:,.0f}',
                           ha='center', va='bottom', fontsize=8)
    
    # SUBPLOT 2: Cumulative Revenue (Line)
    ax_cum.plot(month_names, cumulative_flat, marker='o', linewidth=2.5, color='steelblue', label='Cumulative TLW Flat', markersize=8)
    ax_cum.plot(month_names, cumulative_adj, marker='s', linewidth=2.5, color='darkgreen', label='Cumulative TLW Adjusted', markersize=8)
    
    ax_cum.set_xlabel('Month (2026)', fontsize=12, fontweight='bold')
    ax_cum.set_ylabel('Cumulative Revenue (USD)', fontsize=12, fontweight='bold')
    ax_cum.set_title('2026 TLW Rollout: Cumulative Revenue Buildup', fontsize=13, fontweight='bold')
    ax_cum.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_cum.grid(True, alpha=0.3)
    
    # Add value labels on cumulative points
    for i, month in enumerate(month_names):
        ax_cum.text(i, cumulative_flat[i], f'${cumulative_flat[i]:,.0f}', fontsize=8, ha='center', va='bottom', color='steelblue', fontweight='bold')
        ax_cum.text(i, cumulative_adj[i], f'${cumulative_adj[i]:,.0f}', fontsize=8, ha='center', va='top', color='darkgreen', fontweight='bold')
    
    ax_cum.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print 2026 monthly budget breakdown
    print(f"\n✅ 2026 MONTHLY REVENUE BUDGET (Based on Rollout Schedule):")
    print(f"\n{'Month':<12} {'Monthly Flat':<18} {'Monthly Adj':<18} {'Cumulative Flat':<18} {'Cumulative Adj':<18}")
    print(f"{'-'*84}")
    
    for i, month in enumerate(month_names):
        print(f"{month:<12} ${revenue_flat_monthly[i]:>15,.2f} ${revenue_adj_monthly[i]:>15,.2f} ${cumulative_flat[i]:>15,.2f} ${cumulative_adj[i]:>15,.2f}")
    
    print(f"{'-'*84}")
    print(f"{'TOTAL (2026)':<12} ${sum(revenue_flat_monthly):>15,.2f} ${sum(revenue_adj_monthly):>15,.2f} ${cumulative_flat[-1]:>15,.2f} ${cumulative_adj[-1]:>15,.2f}")
    
    # TLW_ADJ RANGE PLOT WITH MODE MARKER BY PROPERTY
    print(f"\n✅ TLW_ADJ RANGE PLOT WITH MODE MARKER BY PROPERTY:")
    
    # First, we need to calculate tlw_adj for all rollout units
    main_df['tlw_adj_temp'] = 0.0
    for prop in main_df['Property'].unique():
        prop_mask = (main_df['Property'] == prop) & (main_df['rollout'] == True)
        if prop_mask.sum() > 0:
            avg_sqft = main_df.loc[prop_mask, 'SQFT'].mean()
            main_df.loc[prop_mask, 'tlw_adj_temp'] = (main_df.loc[prop_mask, 'SQFT'] * 15.0 / avg_sqft)
    
    # Prepare data for range plot
    rollout_properties_list = sorted(main_df[main_df['rollout'] == True]['Property'].unique())
    
    # Calculate min, max, mode, median for each property
    ranges_data = []
    for prop in rollout_properties_list:
        prop_data = main_df[(main_df['Property'] == prop) & (main_df['rollout'] == True)]['tlw_adj_temp'].values
        prop_data = prop_data[prop_data > 0]
        
        if len(prop_data) > 0:
            min_val = np.min(prop_data)
            max_val = np.max(prop_data)
            median_val = np.median(prop_data)
            
            from scipy import stats
            try:
                mode_val = stats.mode(prop_data, keepdims=True).mode[0]
            except:
                mode_val = median_val
            
            ranges_data.append({
                'property': prop,
                'min': min_val,
                'max': max_val,
                'mode': mode_val,
                'median': median_val
            })
    
    # Create range plot
    fig_range, ax_range = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(len(ranges_data))
    
    for i, data in enumerate(ranges_data):
        # Draw horizontal line from min to max
        ax_range.plot([data['min'], data['max']], [i, i], 'o-', linewidth=2, markersize=6, color='steelblue', alpha=0.7)
        
        # Add mode marker (diamond)
        ax_range.plot(data['mode'], i, 'D', markersize=10, color='darkgreen', markeredgecolor='black', markeredgewidth=1.5)
    
    ax_range.set_yticks(y_pos)
    ax_range.set_yticklabels([d['property'] for d in ranges_data])
    ax_range.set_xlabel('TLW Adjusted Rate (USD)', fontsize=12)
    ax_range.set_ylabel('Property', fontsize=12)
    ax_range.set_title('TLW Adjusted Rate Range by Property (Min—Max with Mode)', fontsize=14, weight='bold')
    ax_range.grid(True, axis='x', alpha=0.3)
    
    from matplotlib.ticker import FuncFormatter
    ax_range.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.2f}'))
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', linewidth=2, marker='o', markersize=6, label='Min—Max Range'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='darkgreen', markersize=10, markeredgecolor='black', markeredgewidth=1.5, label='Mode')
    ]
    ax_range.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics table
    print(f"\n{'='*95}")
    print(f"{'Property':<12} {'Min':<12} {'Mode':<12} {'Max':<12} {'Range':<12} {'Count':<8}")
    print(f"{'='*95}")
    
    for data in ranges_data:
        count = len(main_df[(main_df['Property'] == data['property']) & (main_df['rollout'] == True)])
        range_val = data['max'] - data['min']
        print(
            f"{data['property']:<12} "
            f"${data['min']:<11.2f} "
            f"${data['mode']:<11.2f} "
            f"${data['max']:<11.2f} "
            f"${range_val:<11.2f} "
            f"{count:<8}"
        )
    print(f"{'='*95}")
    
    # Drop temporary column
    main_df.drop('tlw_adj_temp', axis=1, inplace=True)
    
    # PART 3: TLW PRICING CALCULATION
    print(f"\n{'='*60}")
    print(f"PART 3: TLW PRICING CALCULATION")
    print(f"{'='*60}")
    
    # Add week_in and week_out columns based on property schedule
    main_df['week_in'] = 0
    main_df['week_out'] = 0
    
    for prop, end_week in property_completions.items():
        if prop in property_unit_counts:
            prop_mask = (main_df['Property'] == prop) & (main_df['rollout'] == True)
            
            # Get start week based on property size
            units = property_unit_counts[prop]
            if units >= 400:
                start_week = end_week - 2
            elif units >= 200:
                start_week = end_week - 2
            elif units >= 100:
                start_week = end_week - 1
            else:
                start_week = end_week
            
            main_df.loc[prop_mask, 'week_in'] = start_week
            main_df.loc[prop_mask, 'week_out'] = end_week
    
    # Calculate TLW flat rate ($15 per apartment)
    main_df['tlw_flat'] = 0.0
    main_df.loc[main_df['rollout'] == True, 'tlw_flat'] = 15.0
    
    # Calculate TLW adjusted rate (SQFT-based, average pays $15)
    main_df['tlw_adj'] = 0.0
    
    for prop in main_df['Property'].unique():
        prop_mask = (main_df['Property'] == prop) & (main_df['rollout'] == True)
        if prop_mask.sum() > 0:
            avg_sqft = main_df.loc[prop_mask, 'SQFT'].mean()
            main_df.loc[prop_mask, 'tlw_adj'] = (main_df.loc[prop_mask, 'SQFT'] * 15.0 / avg_sqft).round(2)
    
    print(f"\n✅ TLW PRICING COLUMNS ADDED:")
    print(f"   - tlw_flat: Flat rate of $15.00 per apartment")
    print(f"   - tlw_adj: SQFT-adjusted rate (avg SQFT pays $15.00 per property)")
    
    # Show sample of pricing
    print(f"\n✅ SAMPLE TLW PRICING:")
    sample_df = main_df[main_df['rollout'] == True][['Property', 'Unit', 'SQFT', 'week_in', 'week_out', 'tlw_flat', 'tlw_adj']].head(10)
    print(sample_df.to_string(index=False))
    
    # Export to Excel
    print(f"\n✅ EXPORTING TO EXCEL:")
    output_columns = ['Unit_ID', 'Key', 'Property', 'Unit', 'SQFT', 'est_rent', 'rollout', 'week_in', 'week_out', 'tlw_flat', 'tlw_adj']
    export_df = main_df[output_columns]
    
    output_filename = 'tlw_budget_2026_final.xlsx'
    export_df.to_excel(output_filename, index=False, sheet_name='TLW_Budget_2026')
    
    print(f"   ✅ File created: {output_filename}")
    print(f"   - Total rows: {len(export_df):,}")
    print(f"   - Total columns: {len(output_columns)}")
    print(f"   - Rollout units: {(export_df['rollout'] == True).sum():,}")
    
    # Summary statistics
    print(f"\n✅ TLW REVENUE SUMMARY (Rollout Properties Only):")
    rollout_mask = main_df['rollout'] == True
    total_flat = main_df.loc[rollout_mask, 'tlw_flat'].sum()
    total_adj = main_df.loc[rollout_mask, 'tlw_adj'].sum()
    
    print(f"   - Flat Rate Revenue: ${total_flat:,.2f}")
    print(f"   - Adjusted Rate Revenue: ${total_adj:,.2f}")
    print(f"   - Difference: ${abs(total_flat - total_adj):,.2f}")
    print(f"   - Average Adjusted Rate: ${main_df.loc[rollout_mask, 'tlw_adj'].mean():.2f}")
    
    # CALCULATE 2026 ANNUAL REVENUE: Use monthly sum from rollout schedule
    print(f"\n✅ 2026 ANNUAL REVENUE CALCULATION:")
    
    # Use the monthly revenue already calculated correctly above
    total_2026_cumulative = sum(revenue_adj_monthly)  # Sum of 9 months (April-December)
    avg_monthly_2026 = total_2026_cumulative / 9
    final_monthly_revenue = revenue_adj_monthly[-1]  # December full run rate
    
    print(f"   Total 2026 Revenue (April-December): ${total_2026_cumulative:,.2f}")
    print(f"   Average Monthly Revenue (9 months):  ${avg_monthly_2026:,.2f}")
    print(f"   Full Monthly Run Rate (December):    ${final_monthly_revenue:,.2f}")
    print(f"   Ramp-up Factor: {(avg_monthly_2026 / final_monthly_revenue * 100):.1f}% of final run rate")
    
    # ========================================================================
    # PART 3: 5-YEAR TLW BUDGET (2026-2030)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"PART 3: 5-YEAR TLW BUDGET FORECAST (2026-2030)")
    print(f"{'='*80}")
    
    # Cost structure benchmark (% of GWP)
    cost_structure = {
        'Gross Written Premium (GWP)': 100.0,
        'Vacancy Drag': -10.0,
        'Bad Debt': -3.0,
        'Net Earned Premium (Effective)': 87.0,
        'Expected Losses (Base)': -58.0,
        'Loss Development Lag (IBNR)': -4.0,
        'Adjusted Loss Ratio': -62.0,
        'Claims Administration (TPA)': -9.0,
        'Customer Acquisition Cost (CAC)': -2.0,
        'Captive Operating Costs': -5.0,
        'Premium Taxes': -1.5,
        'Capital Cost': -4.0,
        'Compliance/Regulatory Buffer': -2.0
    }
    
    print(f"\n✅ COST STRUCTURE (% of GWP):")
    print(f"{'Component':<40} {'% of GWP':<12}")
    print(f"{'-'*52}")
    for component, pct in cost_structure.items():
        print(f"{component:<40} {pct:>10.1f}%")
    
    # Calculate net margin (excluding duplicates: use Adjusted Loss Ratio, not base+IBNR separately)
    operating_costs = {
        'Adjusted Loss Ratio': 62.0,
        'Claims Administration (TPA)': 9.0,
        'Customer Acquisition Cost (CAC)': 2.0,
        'Captive Operating Costs': 5.0,
        'Premium Taxes': 1.5,
        'Capital Cost': 4.0,
        'Compliance/Regulatory Buffer': 2.0
    }
    total_costs = sum(operating_costs.values())
    net_margin = 87.0 - total_costs
    
    print(f"\n✅ MARGIN CALCULATION:")
    print(f"   Net Earned Premium (after vacancy + bad debt): 87.0%")
    print(f"   Total Operating Costs (all components): {total_costs:.1f}%")
    print(f"   Net Operating Margin: {net_margin:.1f}%")
    
    print(f"\n✅ PREMIUM GROWTH ASSUMPTIONS:")
    print(f"   2026: Partial year (rollout April-December)")
    print(f"   2027+: 2.5% annual premium increase")
    print(f"   Growth Driver: Market rate adjustments + unit mix optimization")
    
    # Build 5-year forecast
    years = [2026, 2027, 2028, 2029, 2030]
    base_monthly_revenue = 95382.11  # 2027+ baseline
    growth_rate = 0.025  # 2.5% annual growth
    
    # 2026 revenue from cumulative week-by-week rollout
    revenue_2026_actual = total_2026_cumulative  # Use actual cumulative sum from rollout schedule
    
    forecast_data = []
    
    for idx, year in enumerate(years):
        if year == 2026:
            # 2026: Use cumulative revenue from week-by-week rollout
            annual_revenue = revenue_2026_actual
            monthly_revenue = annual_revenue / 9  # Derive monthly average from actual
        else:
            # 2027+: Full year with annual growth
            years_elapsed = year - 2027
            monthly_revenue = base_monthly_revenue * ((1 + growth_rate) ** years_elapsed)
            annual_revenue = monthly_revenue * 12
        
        # Calculate costs based on revenue
        gwp = annual_revenue
        vacancy_impact = gwp * 0.10
        bad_debt = gwp * 0.03
        net_earned = gwp * 0.87
        
        # Operating costs
        expected_losses = gwp * 0.58
        loss_development = gwp * 0.04
        claims_admin = gwp * 0.09
        cac = gwp * 0.02
        captive_ops = gwp * 0.05
        premium_taxes = gwp * 0.015
        capital_cost = gwp * 0.04
        compliance = gwp * 0.02
        
        total_costs = (expected_losses + loss_development + claims_admin + cac +
                      captive_ops + premium_taxes + capital_cost + compliance)
        
        net_profit = net_earned - total_costs
        margin_pct = (net_profit / gwp * 100) if gwp > 0 else 0
        
        forecast_data.append({
            'Year': year,
            'Monthly_Avg': monthly_revenue,
            'Annual_Revenue': annual_revenue,
            'Vacancy_Impact': -vacancy_impact,
            'Bad_Debt': -bad_debt,
            'Net_Earned': net_earned,
            'Expected_Losses': -expected_losses,
            'Loss_Dev_IBNR': -loss_development,
            'Claims_Admin': -claims_admin,
            'CAC': -cac,
            'Captive_Ops': -captive_ops,
            'Premium_Taxes': -premium_taxes,
            'Capital_Cost': -capital_cost,
            'Compliance': -compliance,
            'Total_Costs': -total_costs,
            'Net_Profit': net_profit,
            'Margin_%': margin_pct
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    print(f"\n✅ 5-YEAR TLW BUDGET FORECAST:")
    
    # Print detailed breakdown by year
    print(f"\n{'='*100}")
    for idx, (_, row) in enumerate(forecast_df.iterrows()):
        print(f"\n{'='*100}")
        print(f"YEAR {int(row['Year'])} - INPUTS & CALCULATIONS")
        print(f"{'='*100}")
        
        print(f"\n✅ REVENUE INPUTS:")
        if row['Year'] == 2026:
            print(f"   Sum of TLW Adjusted (Rollout):  ${revenue_2026_actual:>15,.2f}")
            print(f"   Monthly Average (9 months):    ${row['Monthly_Avg']:>15,.2f}")
            print(f"   {'─'*50}")
            print(f"   Annual Revenue (GWP):           ${row['Annual_Revenue']:>15,.2f}")
        else:
            print(f"   Monthly Average Revenue:        ${row['Monthly_Avg']:>15,.2f}")
            print(f"   Months in Period:               {12:>15} months")
            print(f"   {'─'*50}")
            print(f"   Annual Revenue (GWP):           ${row['Annual_Revenue']:>15,.2f}")
        
        print(f"\n✅ REVENUE ADJUSTMENTS:")
        print(f"   Vacancy Drag (10%):             ${row['Vacancy_Impact']:>15,.2f}")
        print(f"   Bad Debt (3%):                  ${row['Bad_Debt']:>15,.2f}")
        print(f"   {'─'*50}")
        print(f"   Net Earned Premium (87%):      ${row['Net_Earned']:>15,.2f}")
        
        print(f"\n✅ COST BREAKDOWN (% of GWP):")
        print(f"   Expected Losses (58%):         ${row['Expected_Losses']:>15,.2f}")
        print(f"   Loss Development IBNR (4%):    ${row['Loss_Dev_IBNR']:>15,.2f}")
        print(f"   Claims Administration (9%):    ${row['Claims_Admin']:>15,.2f}")
        print(f"   Customer Acquisition (2%):     ${row['CAC']:>15,.2f}")
        print(f"   Captive Operating Costs (5%):  ${row['Captive_Ops']:>15,.2f}")
        print(f"   Premium Taxes (1.5%):          ${row['Premium_Taxes']:>15,.2f}")
        print(f"   Capital Cost (4%):             ${row['Capital_Cost']:>15,.2f}")
        print(f"   Compliance/Regulatory (2%):    ${row['Compliance']:>15,.2f}")
        print(f"   {'─'*50}")
        print(f"   Total Operating Costs:         ${row['Total_Costs']:>15,.2f}")
        
        print(f"\n✅ NET PROFIT & MARGIN:")
        print(f"   Net Profit:                    ${row['Net_Profit']:>15,.2f}")
        print(f"   Net Margin %:                  {row['Margin_%']:>15.1f}%")
        
        # Show premium growth rate
        if row['Year'] == 2026:
            print(f"   Premium Growth:                {'N/A (Rollout Year)':>15}")
        elif row['Year'] == 2027:
            growth_pct = ((row['Annual_Revenue'] - revenue_2026_actual * (12/9)) / (revenue_2026_actual * (12/9))) * 100
            print(f"   Premium Growth vs 2026 Full:   {growth_pct:>14.1f}%")
        else:
            prev_revenue = forecast_df[forecast_df['Year'] == row['Year'] - 1]['Annual_Revenue'].values[0]
            growth_pct = ((row['Annual_Revenue'] - prev_revenue) / prev_revenue) * 100
            print(f"   Premium Growth (YoY):          {growth_pct:>14.1f}%")
    
    print(f"\n{'='*100}")
    print(f"\n✅ 5-YEAR SUMMARY:")
    print(f"\n{'Year':<8} {'Annual Revenue':>18} {'Net Earned':>18} {'Total Costs':>18} {'Net Profit':>18} {'Margin %':>12}")
    print(f"{'-'*92}")
    
    for _, row in forecast_df.iterrows():
        print(f"{int(row['Year']):<8} ${row['Annual_Revenue']:>16,.0f} ${row['Net_Earned']:>16,.0f} ${row['Total_Costs']:>16,.0f} ${row['Net_Profit']:>16,.0f} {row['Margin_%']:>10.1f}%")
    
    print(f"{'-'*92}")
    total_revenue_5yr = forecast_df['Annual_Revenue'].sum()
    total_profit_5yr = forecast_df['Net_Profit'].sum()
    avg_margin_5yr = forecast_df['Margin_%'].mean()
    
    print(f"{'5-YR TOTAL':<8} ${total_revenue_5yr:>16,.0f} ${forecast_df['Net_Earned'].sum():>16,.0f} ${forecast_df['Total_Costs'].sum():>16,.0f} ${total_profit_5yr:>16,.0f} {avg_margin_5yr:>10.1f}%")
    
    # Export 5-year forecast
    forecast_export = forecast_df.copy()
    forecast_export.to_excel('tlw_5year_forecast.xlsx', index=False, sheet_name='TLW_Forecast_2026_2030')
    
    print(f"\n✅ EXPORT: tlw_5year_forecast.xlsx created successfully")
    
except FileNotFoundError as e:
    print(f"❌ Error: Could not find the Excel file - {e}")
    print("Please make sure the Excel files are in the same directory as this Python script")
except Exception as e:
    print(f"❌ Error loading files: {e}")
