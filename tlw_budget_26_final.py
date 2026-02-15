"""
TLW Budget 26 Final - Complete Rollout Budget Analysis for 2026
==============================================================

Script: tlw_budget_26_final.py
Date: February 14, 2026
Author: Budget Analysis Team

OBJECTIVE:
---------
Estimate comprehensive budget for TLW (Tenant Laundry Works) complete rollout 
across all Acento properties in 2026, including timeline analysis, property 
rollout scheduling, and detailed P&L projections.

INPUT FILES:
-----------
1. acento_apartments_fixed.xlsx
   - Complete apartment inventory with unit details and SQFT data
   - 7,231 units across 30 properties with perfect data alignment
   - Columns: Unit_ID, Key, Property, Unit, SQFT

2. Property Base rent SQFT.xlsx
   - Base rent and pricing data per square foot by property
   - Market rate analysis and rental income benchmarks
   - Property-specific economic indicators

STRATEGY:
---------
Generate optimized rollout schedule algorithmically to avoid key alignment issues.
Use clean acento_apartments_fixed.xlsx as single source of truth for all 7,231 units.

DELIVERABLES:
------------
1. 📊 Apartment Units Timeline Graphic
   - Monthly rollout schedule visualization
   - Property-by-property implementation timeline
   - Unit deployment phases and milestones

2. 📋 Properties Gantt Chart for Rollout
   - Interactive Gantt chart showing rollout sequence
   - Resource allocation and timeline dependencies
   - Critical path analysis for 2026 implementation

3. 💰 Final P&L Analysis
   - Comprehensive profit and loss projections
   - ROI calculations and payback period analysis
   - Revenue forecasting and cost optimization

METHODOLOGY:
-----------
- CLEAN DATA APPROACH: Use acento_apartments_fixed.xlsx as single source (avoids key mismatches)
- ALGORITHMIC SCHEDULING: Generate optimized rollout timeline starting April 2026
- RESOURCE OPTIMIZATION: Minimize parallel deployments by property size
- STATISTICAL ANALYSIS: Unit economics from Property Base rent SQFT.xlsx
- FINANCIAL MODELING: NPV, IRR, payback analysis with scenario planning
- VISUAL ANALYTICS: Timeline graphics, Gantt charts, and P&L dashboards

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def load_and_examine_files():
    """
    PART 1: UPLOAD AND EXPLORATION
    Load and examine the two primary Excel files for TLW budget analysis.
    
    Returns:
    --------
    tuple: (acento_df, base_rent_df) - loaded DataFrames
    """
    print("PART 1: UPLOAD AND EXPLORATION")
    print("="*40)
    
    # Load acento apartments fixed data
    print("\n1. LOADING ACENTO APARTMENTS FIXED:")
    try:
        acento_df = pd.read_excel('acento_apartments_fixed.xlsx')
        print(f"   ✅ Successfully loaded")
        print(f"   Records: {len(acento_df):,}")
        print(f"   Columns: {list(acento_df.columns)}")
        print(f"   Properties: {acento_df['Property'].nunique()}")
        print(f"   Units with SQFT: {acento_df['SQFT'].notna().sum():,}/{len(acento_df):,}")
        
        print(f"\n   Sample Records:")
        sample_cols = ['Property', 'Unit', 'SQFT'] if 'SQFT' in acento_df.columns else ['Property', 'Unit']
        print(acento_df[sample_cols].head(3).to_string(index=False))
        
    except Exception as e:
        print(f"   ❌ Error loading acento_apartments_fixed.xlsx: {e}")
        return None, None
    
    # Load Property Base rent SQFT data
    print("\n\n2. LOADING PROPERTY BASE RENT SQFT:")
    try:
        base_rent_df = pd.read_excel('Property Base rent SQFT.xlsx')
        print(f"   ✅ Successfully loaded")
        print(f"   Records: {len(base_rent_df):,}")
        print(f"   Columns: {list(base_rent_df.columns)}")
        
        print(f"\n   Sample Records:")
        print(base_rent_df.head(3).to_string(index=False))
        
        # Quick analysis
        if 'Min Rent' in base_rent_df.columns and 'Max Rent' in base_rent_df.columns:
            print(f"\n   Rent Range: ${base_rent_df['Min Rent'].min():,.0f} - ${base_rent_df['Max Rent'].max():,.0f}")
        if 'Min SQFT' in base_rent_df.columns and 'Max SQFT' in base_rent_df.columns:
            print(f"   SQFT Range: {base_rent_df['Min SQFT'].min():.0f} - {base_rent_df['Max SQFT'].max():.0f} sq ft")
        
    except Exception as e:
        print(f"   ❌ Error loading Property Base rent SQFT.xlsx: {e}")
        return acento_df, None
    
    return acento_df, base_rent_df

def examine_data_quality(acento_df, base_rent_df):
    """
    Check data quality and identify any issues before proceeding with analysis.
    
    Parameters:
    -----------
    acento_df : DataFrame
        Acento apartments data
    base_rent_df : DataFrame
        Property base rent data
    """
    print("\n\nDATA QUALITY EXAMINATION")
    print("="*35)
    
    if acento_df is not None:
        print("\nACENTO APARTMENTS ANALYSIS:")
        
        # Check for missing values
        missing_data = acento_df.isnull().sum()
        if missing_data.sum() > 0:
            print("   ⚠️ Missing Values:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"      - {col}: {count:,} missing ({count/len(acento_df)*100:.1f}%)")
                
                # Show details of missing records
                if col == 'SQFT' and count > 0:
                    missing_records = acento_df[acento_df[col].isnull()]
                    print(f"      Missing {col} Records:")
                    display_cols = ['Unit_ID', 'Key', 'Property', 'Unit'] if 'Unit_ID' in missing_records.columns else ['Key', 'Property', 'Unit']
                    for _, row in missing_records.iterrows():
                        record_info = " | ".join([f"{col}: {row[col]}" for col in display_cols if col in row])
                        print(f"         - {record_info}")
        else:
            print("   ✅ No missing values found")
        
        # Property distribution
        print(f"\n   Property Distribution:")
        prop_counts = acento_df['Property'].value_counts().head(10)
        for prop, count in prop_counts.items():
            print(f"      - {prop}: {count:,} units")
        
        if len(prop_counts) == 10:
            remaining = len(acento_df['Property'].value_counts()) - 10
            if remaining > 0:
                print(f"      - ... and {remaining} more properties")
    
    if base_rent_df is not None:
        print(f"\nBASE RENT DATA ANALYSIS:")
        
        # Check property alignment
        if acento_df is not None:
            acento_props = set(acento_df['Property'].unique())
            rent_props = set(base_rent_df['Property'].unique())
            
            matching = acento_props.intersection(rent_props)
            missing_in_rent = acento_props - rent_props
            extra_in_rent = rent_props - acento_props
            
            print(f"   Property Alignment:")
            print(f"      - Matching: {len(matching)} properties")
            print(f"      - Missing in rent data: {len(missing_in_rent)} properties")
            print(f"      - Extra in rent data: {len(extra_in_rent)} properties")
            
            if missing_in_rent:
                print(f"      ⚠️ Missing: {', '.join(list(missing_in_rent)[:5])}{'...' if len(missing_in_rent) > 5 else ''}")
    
    print("\n✅ Data examination complete.")

def create_working_dataframe(acento_df, base_rent_df):
    """
    PART 2: CREATE WORKING DATAFRAME
    Start with 5 columns from acento apartments, add 6th column "Rent" calculated 
    by multiplying each unit's SQFT × property-specific slope.
    
    Parameters:
    -----------
    acento_df : DataFrame
        Acento apartments fixed data (5 columns)
    base_rent_df : DataFrame
        Property base rent data with slopes (stays untouched)
    
    Returns:
    --------
    DataFrame: Working dataframe with exactly 6 columns including calculated Rent
    """
    print("\n\nPART 2: CREATE WORKING DATAFRAME")
    print("="*45)
    
    # Start with acento apartments 5 columns
    print("1. STARTING WITH ACENTO APARTMENTS (5 COLUMNS):")
    working_df = acento_df.copy()
    print(f"   ✅ Base dataframe: {len(working_df):,} units")
    print(f"   Columns: {list(working_df.columns)}")
    
    # Create property slope lookup dictionary
    print("\n2. CREATING PROPERTY SLOPE LOOKUP:")
    slope_lookup = dict(zip(base_rent_df['Property'], base_rent_df['Slope']))
    print(f"   ✅ Loaded slope data for {len(slope_lookup)} properties")
    print(f"   Sample slopes: {dict(list(slope_lookup.items())[:3])}")
    
    # Calculate rent for each unit: Unit_SQFT × Property_Slope
    print("\n3. CALCULATING RENT COLUMN:")
    print("   Formula: Rent = Unit_SQFT × Property_Slope")
    
    def calculate_unit_rent(row):
        if pd.isna(row['SQFT']):
            return np.nan
        
        property_slope = slope_lookup.get(row['Property'])
        if property_slope is None:
            return np.nan
            
        return row['SQFT'] * property_slope
    
    working_df['Rent'] = working_df.apply(calculate_unit_rent, axis=1)
    
    # Round to nearest dollar
    working_df['Rent'] = working_df['Rent'].round(0)
    
    # Analysis results
    valid_rent_count = working_df['Rent'].notna().sum()
    print(f"   ✅ Calculated rent for {valid_rent_count:,}/{len(working_df):,} units")
    
    if valid_rent_count > 0:
        rent_stats = working_df['Rent'].describe()
        print(f"   Rent Range: ${rent_stats['min']:.0f} - ${rent_stats['max']:.0f}")
        print(f"   Average Rent: ${rent_stats['mean']:.0f}")
    
    print(f"\n   Sample Working DataFrame (6 columns):")
    print(working_df.head(3).to_string(index=False))
    
    print(f"\n✅ Working dataframe completed:")
    print(f"   Total columns: {len(working_df.columns)} (as expected: 6)")
    print(f"   Final columns: {list(working_df.columns)}")
    
    return working_df

def prepare_tlw_portfolio(working_df):
    """
    PART 3: TLW PORTFOLIO PREPARATION
    Exclude specified properties and prepare final TLW rollout portfolio.
    Show excluded and eligible properties with unit counts.
    
    Parameters:
    -----------
    working_df : DataFrame
        Working dataframe with all apartment units and rent calculations
        
    Returns:
    --------
    tuple: (excluded_df, tlw_eligible_df, portfolio_summary)
    """
    print("\n\nPART 3: TLW PORTFOLIO PREPARATION")
    print("="*45)
    
    # Define properties to exclude from TLW rollout
    excluded_properties = ['CWA', 'CCA', 'CHA', 'WEA']
    
    print("1. ORIGINAL PORTFOLIO ANALYSIS:")
    original_properties = working_df['Property'].nunique()
    original_units = len(working_df)
    print(f"   Total Properties: {original_properties}")
    print(f"   Total Units: {original_units:,}")
    
    # Step 1: Create excluded properties table
    print("\n2. EXCLUDED PROPERTIES ANALYSIS:")
    
    excluded_df = working_df[working_df['Property'].isin(excluded_properties)].copy()
    
    if len(excluded_df) > 0:
        excluded_summary = (excluded_df.groupby('Property')
                           .size()
                           .reset_index(name='Units')
                           .sort_values('Units', ascending=False))
        
        print(f"   ❌ Properties to exclude: {len(excluded_properties)}")
        print(f"\n   EXCLUDED PROPERTIES TABLE:")
        print("   " + "="*45)
        print(f"   {'Property':<12} {'Units':<10} {'Percentage':<12}")
        print("   " + "-"*45)
        
        total_excluded_units = 0
        for _, row in excluded_summary.iterrows():
            percentage = (row['Units'] / original_units) * 100
            print(f"   {row['Property']:<12} {row['Units']:<10,} {percentage:<12.1f}%")
            total_excluded_units += row['Units']
        
        print("   " + "-"*45)
        print(f"   {'TOTAL':<12} {total_excluded_units:<10,} {(total_excluded_units/original_units)*100:<12.1f}%")
        print("   " + "="*45)
        
    else:
        print("   No properties found matching exclusion criteria")
        excluded_summary = pd.DataFrame()
        total_excluded_units = 0
    
    # Step 2: Create TLW eligible properties table  
    print("\n3. TLW ELIGIBLE PORTFOLIO:")
    
    tlw_eligible_df = working_df[~working_df['Property'].isin(excluded_properties)].copy()
    
    if len(tlw_eligible_df) > 0:
        tlw_summary = (tlw_eligible_df.groupby('Property')
                      .agg({
                          'Unit': 'count',
                          'Rent': 'sum'
                      })
                     .rename(columns={'Unit': 'Units', 'Rent': 'Total_Rent_Roll'})
                     .sort_values('Units', ascending=False)
                     .reset_index())
        
        print(f"   ✅ TLW Eligible Properties: {len(tlw_summary)}")
        print(f"\n   TLW ELIGIBLE PROPERTIES TABLE:")
        print("   " + "="*60)
        print(f"   {'#':<3} {'Property':<25} {'Units':<8} {'Rent Roll/Month':<15}")
        print("   " + "-"*60)
        
        total_tlw_units = 0
        total_tlw_revenue = 0
        
        # Show all properties
        for i, (_, row) in enumerate(tlw_summary.iterrows(), 1):
            rent_roll = row['Total_Rent_Roll'] if pd.notna(row['Total_Rent_Roll']) else 0
            print(f"   {i:<3} {row['Property']:<25} {row['Units']:<8,} ${rent_roll:<15,.0f}")
            total_tlw_units += row['Units']
            total_tlw_revenue += rent_roll
        
        print("   " + "-"*60)
        print(f"   {'TOTAL TLW':<29} {total_tlw_units:<8,} ${total_tlw_revenue:<15,.0f}")
        print("   " + "="*60)
        
        # Calculate average units per property
        avg_units_per_property = total_tlw_units / len(tlw_summary)
        print(f"\n   Average units per TLW property: {avg_units_per_property:.1f}")
        
    else:
        print("   ❌ No properties remaining after exclusions")
        tlw_summary = pd.DataFrame()
        total_tlw_units = 0
        total_tlw_revenue = 0
    
    # Step 3: Verification and Summary
    print("\n4. PORTFOLIO VERIFICATION:")
    print("   " + "="*50)
    
    calculated_total = total_excluded_units + total_tlw_units
    
    print(f"   {'Description':<25} {'Units':<10} {'Percentage':<12}")
    print("   " + "-"*50)
    print(f"   {'Original Total Units:':<25} {original_units:<10,} {'100.0%':<12}")
    print(f"   {'❌ Excluded Units:':<25} {total_excluded_units:<10,} {(total_excluded_units/original_units)*100:<12.1f}%")  
    print(f"   {'✅ TLW Eligible Units:':<25} {total_tlw_units:<10,} {(total_tlw_units/original_units)*100:<12.1f}%")
    print("   " + "-"*50)
    print(f"   {'Calculated Total:':<25} {calculated_total:<10,} {'100.0%':<12}")
    print("   " + "="*50)
    
    if calculated_total == original_units:
        print(f"   ✅ VERIFICATION PASSED: Totals match ({original_units:,} units)")
    else:
        difference = abs(calculated_total - original_units)
        print(f"   ⚠️ VERIFICATION WARNING: {difference:,} unit difference")
    
    # Portfolio summary
    portfolio_summary = {
        'original_properties': original_properties,
        'original_units': original_units,
        'excluded_properties': len(excluded_properties),
        'excluded_units': total_excluded_units,
        'tlw_properties': len(tlw_summary) if len(tlw_summary) > 0 else 0,
        'tlw_units': total_tlw_units,
        'tlw_monthly_revenue': total_tlw_revenue,
        'avg_units_per_property': avg_units_per_property if total_tlw_units > 0 else 0
    }
    
    print(f"\n   FINAL TLW PORTFOLIO:")
    print(f"      - Properties for TLW: {portfolio_summary['tlw_properties']}")
    print(f"      - Units for TLW: {portfolio_summary['tlw_units']:,}")  
    print(f"      - Total Rent Roll: ${portfolio_summary['tlw_monthly_revenue']:,.0f}/month")
    print(f"      - Average units per property: {portfolio_summary['avg_units_per_property']:.1f}")
    
    return excluded_df, tlw_eligible_df, portfolio_summary

def calculate_tlw_rollout_timeline(tlw_eligible_df, portfolio_summary):
    """
    PART 4: TLW ROLLOUT TIMELINE CALCULATION
    Calculate week-by-week rollout schedule for TLW deployment across eligible properties.
    Simple distribution across available weeks 15-52 (April-December 2026).
    
    Parameters:
    -----------
    tlw_eligible_df : DataFrame
        TLW eligible apartments dataframe
    portfolio_summary : dict
        Portfolio summary statistics
        
    Returns:
    --------
    DataFrame: Enhanced dataframe with rollout timeline columns
    """
    print("\n\nPART 4: TLW ROLLOUT TIMELINE CALCULATION")
    print("="*50)
    
    # Timeline parameters - CORRECTED for realistic 2026 rollout
    timeline_params = {
        'current_week': 7,           # Today is Feb 15, Week 7
        'rollout_start_week': 15,    # Start in April (Week 15)
        'rollout_end_week': 52,      # Finish by end of year (Week 52)
        'available_weeks': 37,       # 52 - 15 = 37 weeks available
        'total_units': portfolio_summary['tlw_units'],
        'target_units_per_week': portfolio_summary['tlw_units'] // 37  # Simple division
    }
    
    print("1. TIMELINE PARAMETERS:")
    print(f"   Current Week: {timeline_params['current_week']} (Feb 15, 2026)")
    print(f"   Rollout Period: Week {timeline_params['rollout_start_week']} to {timeline_params['rollout_end_week']} (April-December)")
    print(f"   Available Weeks: {timeline_params['available_weeks']} weeks")
    print(f"   Total Units: {timeline_params['total_units']:,} units")
    print(f"   Target Rate: ~{timeline_params['target_units_per_week']} units/week")
    
    # Step 1: Get properties sorted by size (largest first)
    print("\n2. PROPERTY ROLLOUT SCHEDULING:")
    
    property_summary = (tlw_eligible_df.groupby('Property')
                       .agg({'Unit': 'count'})
                       .rename(columns={'Unit': 'Units'})
                       .sort_values('Units', ascending=False)
                       .reset_index())
    
    print(f"   Properties to schedule: {len(property_summary)}")
    
    # Step 2: Distribute properties across weeks 15-52
    rollout_schedule = []
    current_week = timeline_params['rollout_start_week']
    cumulative_units = 0
    
    print(f"\n   WEEKLY ROLLOUT DISTRIBUTION:")
    print("   " + "="*60)
    print(f"   {'Week':<6} {'Property':<12} {'Units':<8} {'Cumulative':<12}")
    print("   " + "-"*60)
    
    for i, (_, prop) in enumerate(property_summary.iterrows()):
        property_name = prop['Property']
        units = prop['Units']
        
        # Assign to current week
        rollout_week = current_week
        cumulative_units += units
        
        rollout_schedule.append({
            'Property': property_name,
            'Units': units,
            'Week': rollout_week,
            'Cumulative_Units': cumulative_units,
            'Sequence': i + 1
        })
        
        print(f"   {rollout_week:<6} {property_name:<12} {units:<8,} {cumulative_units:<12,}")
        
        # Advance week - distribute roughly evenly
        # For 26 properties across 37 weeks, some weeks will have multiple properties
        if i < len(property_summary) - 1:  # Not the last property
            weeks_per_property = timeline_params['available_weeks'] / len(property_summary)
            current_week = min(
                timeline_params['rollout_start_week'] + int((i + 1) * weeks_per_property),
                timeline_params['rollout_end_week']
            )
    
    print("   " + "="*60)
    print(f"   Final Week: {max([p['Week'] for p in rollout_schedule])}")
    print(f"   Total Units: {cumulative_units:,}")
    print(f"   Verification: {'✅ Match' if cumulative_units == timeline_params['total_units'] else '❌ Mismatch'}")
    
    # Step 3: Create cumulative weekly data for visualization
    print("\n3. WEEKLY CUMULATIVE ANALYSIS:")
    
    # Generate week-by-week cumulative data
    weekly_data = []
    for week in range(timeline_params['rollout_start_week'], timeline_params['rollout_end_week'] + 1):
        # Sum all units that roll out by this week
        units_by_week = sum([p['Units'] for p in rollout_schedule if p['Week'] <= week])
        percentage = (units_by_week / timeline_params['total_units']) * 100
        
        weekly_data.append({
            'Week': week,
            'Cumulative_Units': units_by_week,
            'Percentage_Complete': percentage
        })
    
    cumulative_df = pd.DataFrame(weekly_data)
    
    # Show key milestones
    print("   KEY MILESTONES:")
    print("   " + "-"*40)
    milestones = [25, 50, 75, 100]
    for milestone in milestones:
        milestone_data = cumulative_df[cumulative_df['Percentage_Complete'] >= milestone]
        if not milestone_data.empty:
            week = milestone_data['Week'].iloc[0]
            units = milestone_data['Cumulative_Units'].iloc[0]
            month = "April" if week <= 18 else "May" if week <= 22 else "June" if week <= 26 else \
                   "July" if week <= 31 else "August" if week <= 35 else "September" if week <= 39 else \
                   "October" if week <= 44 else "November" if week <= 48 else "December"
            print(f"   {milestone:3d}%: Week {week} ({month}) - {units:,} units")
    
    # Convert rollout schedule to DataFrame
    rollout_df = pd.DataFrame(rollout_schedule)
    
    print(f"\n4. TIMELINE SUMMARY:")
    print("   " + "-"*40)
    print(f"   Duration: Week {timeline_params['rollout_start_week']} to {rollout_df['Week'].max()}")
    print(f"   Properties: {len(rollout_df)}")
    print(f"   Total Units: {timeline_params['total_units']:,}")
    print(f"   Average per week: {timeline_params['total_units'] // timeline_params['available_weeks']}")
    print("   ✅ Timeline calculation complete")
    
    return rollout_df, cumulative_df, timeline_params

def visualize_tlw_rollout_timeline(rollout_df, cumulative_df, timeline_params, portfolio_summary):
    """
    Create TLW rollout visualizations using working code structure from tlw_budget_26.py
    """
    print("\n\nPHASE 3: WEEKLY CUMULATIVE ENROLLMENT VISUALIZATION")
    print("="*80)
    
    # Create cumulative enrollment chart matching original format
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    # Plot cumulative enrollment
    ax2.plot(cumulative_df['Week'], cumulative_df['Cumulative_Units'], 
             linewidth=4, color='#2E86AB', marker='o', markersize=4)
    
    # Fill area under curve
    ax2.fill_between(cumulative_df['Week'], cumulative_df['Cumulative_Units'], 
                     alpha=0.3, color='lightblue', label='Enrollment Progress')
    
    # Add milestone annotations
    for pct in [25, 50, 75]:
        target_units = int(portfolio_summary['tlw_units'] * pct / 100)
        milestone_row = cumulative_df[cumulative_df['Cumulative_Units'] >= target_units]
        if not milestone_row.empty:
            week = milestone_row['Week'].iloc[0]
            units = milestone_row['Cumulative_Units'].iloc[0]
            
            # Add milestone marker
            ax2.plot(week, units, marker='s', markersize=10, color='orange', markeredgecolor='black')
            
            # Add label
            ax2.annotate(f'Week {week}\n{units:,} units', (week, units), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    # Add final point
    final_week = cumulative_df['Week'].max()
    final_units = cumulative_df['Cumulative_Units'].max()
    ax2.annotate(f'End: Week {final_week}\n{final_units:,} units', 
                (final_week, final_units), 
                xytext=(-50, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    # Add milestone reference lines with colors
    milestone_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    milestone_labels = []
    for i, pct in enumerate([25, 50, 75]):
        target_units = portfolio_summary['tlw_units'] * pct / 100
        color = milestone_colors[i]
        ax2.axhline(y=target_units, color=color, linestyle='--', alpha=0.8, linewidth=2)
        milestone_labels.append(f'{pct}% Milestone ({int(target_units):,} units)')
    
    # Customize chart
    ax2.set_xlabel('Timeline - April to December 2026 (Week 15 to 52)', fontsize=12)
    ax2.set_ylabel('Cumulative Apartments Enrolled', fontsize=12)
    ax2.set_title(f'{portfolio_summary["tlw_units"]:,} Apartment Units Timeline (April-December)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(timeline_params['rollout_start_week'] - 1, 52 + 2)
    ax2.set_ylim(0, portfolio_summary['tlw_units'] * 1.05)
    
    # Add current week marker
    ax2.axvline(x=timeline_params['current_week'], color='red', linestyle='-', linewidth=2, 
               label='Today: Week 7 (Feb 15)')
    
    # Create legend with milestone colors
    legend_elements = [plt.Line2D([0], [0], color='#2E86AB', linewidth=4, label='Enrollment Progress')]
    legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=2, label='Today: Week 7 (Feb 15)'))
    for i, (pct, color) in enumerate(zip([25, 50, 75], milestone_colors)):
        target_units = int(portfolio_summary['tlw_units'] * pct / 100)
        legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='--', linewidth=2, 
                                        label=f'{pct}% Target ({target_units:,} units)'))
    
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Weekly cumulative enrollment graph generated successfully!")
    
    print("\n\nPHASE 4: PROPERTY ENROLLMENT GANTT CHART")
    print("="*80)
    
    # Display property schedule table first
    print("PROPERTY ENROLLMENT SCHEDULE:")
    print("-" * 95)
    print(f"{'Property':<12} {'Units':<8} {'Start':<12} {'End':<12} {'Duration':<10} {'Weeks':<10}")
    print("-" * 95)
    
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
    
    for _, row in rollout_df.iterrows():
        prop_name = row['Property']
        prop_units = row['Units']
        week_start = row['Week']
        
        # Calculate duration based on property size (1-3 weeks)
        if prop_units >= 400:
            duration_weeks = 3
            category = 'Large (400+ units)'
        elif prop_units >= 200:
            duration_weeks = 2
            category = 'Medium (200-399 units)'  
        elif prop_units >= 100:
            duration_weeks = 2
            category = 'Small (100-199 units)'
        else:
            duration_weeks = 1
            category = 'Micro (<100 units)'
        
        # Convert week to month name
        start_month = "April" if week_start <= 18 else \
                     "May" if week_start <= 22 else \
                     "June" if week_start <= 26 else \
                     "July" if week_start <= 31 else \
                     "August" if week_start <= 35 else \
                     "September" if week_start <= 39 else \
                     "October" if week_start <= 44 else \
                     "November" if week_start <= 48 else "December"
        
        end_week = week_start + duration_weeks - 1
        end_month = "April" if end_week <= 18 else \
                   "May" if end_week <= 22 else \
                   "June" if end_week <= 26 else \
                   "July" if end_week <= 31 else \
                   "August" if end_week <= 35 else \
                   "September" if end_week <= 39 else \
                   "October" if end_week <= 44 else \
                   "November" if end_week <= 48 else "December"
        
        properties.append(prop_name)
        start_weeks.append(week_start)
        durations_weeks.append(duration_weeks)
        units.append(prop_units)
        colors.append(color_map[category])
        
        print(f"{prop_name:<12} {prop_units:<8} {start_month:<12} {end_month:<12} {duration_weeks:<10} {duration_weeks:<10}")
    
    print("-" * 95)
    
    # Create Gantt chart with larger figure size
    fig, ax = plt.subplots(figsize=(19, max(12, len(properties) * 0.5)))
    
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
        ax.text(end_week + 2, y_pos, f'W{start_weeks[i]}-{end_week}', 
                ha='left', va='center', fontsize=8, fontweight='bold')
    
    # Customize chart
    ax.set_yticks(y_positions)
    ax.set_yticklabels(properties)
    ax.set_xlabel('Timeline - April to December 2026 (Week Numbers)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Properties', fontweight='bold')
    fig.suptitle('TLW Property Enrollment Schedule - Gantt Chart\nProperty-by-Property Timeline', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Add month boundaries
    months = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_start_weeks = [15, 19, 23, 27, 31, 35, 39, 44, 48]
    
    for i, (month, week) in enumerate(zip(months, month_start_weeks)):
        ax.axvline(x=week, color='gray', linestyle='--', alpha=0.5)
        ax.text(week + 1, len(properties), month, 
               rotation=45, ha='left', va='bottom', fontsize=9, fontweight='bold')
    
    # Add Today's week marker
    today_week = 7  # Feb 15, 2026
    ax.axvline(x=today_week, color='red', linestyle='-', linewidth=2, alpha=0.8, 
              label=f'Today (Week {today_week})')
    
    # Format x-axis
    ax.set_xlim(5, 57)
    
    # Add grid
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
                                    label=f'Today: Week {today_week} (Feb 15)'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.subplots_adjust(bottom=0.18, top=0.85, left=0.08, right=0.95)
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
    
    print(f"\n✅ Property Gantt chart generated successfully!")
    
    return fig2, fig

def main():
    """
    Main function to execute TLW budget analysis.
    """
    print("TLW BUDGET 26 FINAL ANALYSIS")
    print("="*50)
    print(f"Analysis Date: {datetime.now().strftime('%B %d, %Y')}")
    print("Objective: Complete TLW rollout budget estimation for 2026\n")
    
    # Step 1: Load and examine data files
    acento_df, base_rent_df = load_and_examine_files()
    
    if acento_df is None:
        print("❌ Cannot proceed without acento_apartments_fixed.xlsx")
        return
    
    # Step 2: Examine data quality
    examine_data_quality(acento_df, base_rent_df)
    
    # Step 3: Create working dataframe with estimated rent
    working_df = create_working_dataframe(acento_df, base_rent_df)
    
    return acento_df, base_rent_df, working_df

if __name__ == "__main__":
    acento_df, base_rent_df, working_df = main()
    
    # Continue with TLW portfolio preparation
    if working_df is not None:
        print("\n" + "="*70)
        print("CONTINUING TO TLW PORTFOLIO PREPARATION...")
        print("="*70)
        
        # Prepare TLW portfolio by excluding specified properties
        excluded_df, tlw_eligible_df, portfolio_summary = prepare_tlw_portfolio(working_df)
        
        # Calculate TLW rollout timeline
        rollout_df, cumulative_df, timeline_params = calculate_tlw_rollout_timeline(tlw_eligible_df, portfolio_summary)
        
        # Create rollout visualizations
        rollout_fig = visualize_tlw_rollout_timeline(rollout_df, cumulative_df, timeline_params, portfolio_summary)
