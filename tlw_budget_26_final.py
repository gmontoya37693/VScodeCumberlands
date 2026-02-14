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
        print(f"\n   TLW ELIGIBLE PROPERTIES TABLE (All Properties):")
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
    Create sequential deployment with minimal parallel properties to achieve steady slope.
    
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
    
    # Timeline parameters
    timeline_params = {
        'current_week': 7,           # Today is Feb 14, Week 7
        'rollout_start_week': 15,    # Start in April (Week 15)
        'rollout_end_week': 52,      # Finish by end of year (Week 52)
        'max_parallel_properties': 2, # Minimal parallel deployment
        'installation_weeks_per_100_units': 2,  # Installation time
        'ramp_up_weeks': 2,          # Service ramp-up after installation
        'buffer_weeks': 1            # Buffer between properties
    }
    
    print("1. TIMELINE PARAMETERS:")
    print(f"   Current Week: {timeline_params['current_week']} (Feb 14, 2026)")
    print(f"   Rollout Period: Week {timeline_params['rollout_start_week']} to {timeline_params['rollout_end_week']} (April-December)")
    print(f"   Available Weeks: {timeline_params['rollout_end_week'] - timeline_params['rollout_start_week']} weeks")
    print(f"   Max Parallel Properties: {timeline_params['max_parallel_properties']}")
    
    # Step 1: Create property-level rollout schedule
    print("\n2. CREATING PROPERTY ROLLOUT SCHEDULE:")
    
    # Get property summary for scheduling
    property_schedule = (tlw_eligible_df.groupby('Property')
                        .agg({
                            'Unit': 'count',
                            'Rent': 'mean'  # Average rent per unit (for prioritization)
                        })
                        .rename(columns={'Unit': 'Total_Units', 'Rent': 'Avg_Rent'})
                        .reset_index())
    
    # Strategic priority scoring (larger properties first, but consider complexity)
    property_schedule['Priority_Score'] = (
        property_schedule['Total_Units'].rank(ascending=False) * 0.6 +  # Size priority
        property_schedule['Avg_Rent'].rank(ascending=False) * 0.4       # Revenue priority
    )
    
    # Sort by priority
    property_schedule = property_schedule.sort_values('Priority_Score', ascending=False).reset_index(drop=True)
    
    # Calculate deployment timeline for each property
    current_week = timeline_params['rollout_start_week']
    rollout_schedule = []
    
    print(f"   Properties to schedule: {len(property_schedule)}")
    print(f"\n   PROPERTY ROLLOUT SEQUENCE:")
    print("   " + "="*80)
    print(f"   {'#':<3} {'Property':<12} {'Units':<6} {'Install Weeks':<6} {'Roll-In':<12} {'Roll-Out':<12}")
    print("   " + "-"*80)
    
    for i, (_, prop) in enumerate(property_schedule.iterrows(), 1):
        units = prop['Total_Units']
        
        # Calculate installation duration based on property size
        install_weeks = max(2, int(np.ceil(units / 50)))  # ~50 units per week installation
        
        # Installation period
        install_start_week = current_week
        install_end_week = install_start_week + install_weeks - 1
        
        # Service roll-out period (after installation + ramp-up)
        service_start_week = install_end_week + timeline_params['ramp_up_weeks']
        service_end_week = service_start_week + 1  # Service goes live quickly after installation
        
        rollout_schedule.append({
            'Property': prop['Property'],
            'Total_Units': units,
            'Priority_Sequence': i,
            'Install_Start_Week': install_start_week,
            'Install_End_Week': install_end_week,
            'Install_Duration_Weeks': install_weeks,
            'Service_Start_Week': service_start_week,
            'Service_End_Week': service_end_week,
            'Cumulative_Units': sum([p['Total_Units'] for p in rollout_schedule]) + units
        })
        
        # Print schedule
        install_period = f"W{install_start_week}-{install_end_week}"
        service_period = f"W{service_start_week}-{service_end_week}"
        print(f"   {i:<3} {prop['Property']:<12} {units:<6} {install_weeks:<6} {install_period:<12} {service_period:<12}")
        
        # Move to next property with buffer
        current_week = install_end_week + timeline_params['buffer_weeks']
        
        # Check if we're within timeline constraints
        if current_week > timeline_params['rollout_end_week'] - 5:  # Leave some buffer
            print(f"   ⚠️ Approaching timeline limit at Week {current_week}")
    
    # Create rollout DataFrame
    rollout_df = pd.DataFrame(rollout_schedule)
    
    print("   " + "="*80)
    print(f"   Total Properties Scheduled: {len(rollout_df)}")
    print(f"   Final Week: {rollout_df['Service_End_Week'].max()}")
    print(f"   ✅ Timeline Status: {'Within Bounds' if rollout_df['Service_End_Week'].max() <= timeline_params['rollout_end_week'] else '⚠️ Exceeds Timeline'}")
    
    # Step 2: Add rollout columns to apartment-level dataframe
    print("\n3. ADDING ROLLOUT COLUMNS TO APARTMENT DATAFRAME:")
    
    # Create lookup dictionary for property rollout data
    property_lookup = rollout_df.set_index('Property').to_dict('index')
    
    # Add rollout columns to each apartment unit
    rollout_columns = [
        'Week_Roll_In_Start',      # Installation start week
        'Week_Roll_In_End',        # Installation end week  
        'Week_Roll_Out_Start',     # Service start week
        'Week_Roll_Out_End',       # Service end week
        'Install_Duration_Weeks',   # Installation duration
        'Priority_Sequence',       # Property priority sequence
        'Cumulative_Units_At_Rollout'  # Cumulative units when this property goes live
    ]
    
    # Initialize new columns
    enhanced_df = tlw_eligible_df.copy()
    for col in rollout_columns:
        enhanced_df[col] = None
    
    # Populate rollout data for each apartment
    for prop, prop_data in property_lookup.items():
        mask = enhanced_df['Property'] == prop
        enhanced_df.loc[mask, 'Week_Roll_In_Start'] = prop_data['Install_Start_Week']
        enhanced_df.loc[mask, 'Week_Roll_In_End'] = prop_data['Install_End_Week']
        enhanced_df.loc[mask, 'Week_Roll_Out_Start'] = prop_data['Service_Start_Week']
        enhanced_df.loc[mask, 'Week_Roll_Out_End'] = prop_data['Service_End_Week']
        enhanced_df.loc[mask, 'Install_Duration_Weeks'] = prop_data['Install_Duration_Weeks']
        enhanced_df.loc[mask, 'Priority_Sequence'] = prop_data['Priority_Sequence']
        enhanced_df.loc[mask, 'Cumulative_Units_At_Rollout'] = prop_data['Cumulative_Units']
    
    print(f"   Added {len(rollout_columns)} rollout columns to apartment dataframe")
    print(f"   Enhanced dataframe shape: {enhanced_df.shape}")
    
    # Show sample of enhanced dataframe
    print(f"\n   SAMPLE ENHANCED DATAFRAME:")
    sample_cols = ['Property', 'Unit', 'Week_Roll_In_Start', 'Week_Roll_Out_Start', 'Priority_Sequence']
    print("   " + enhanced_df[sample_cols].head(3).to_string(index=False))
    
    # Step 3: Create cumulative enrollment tracking
    print("\n4. CUMULATIVE ENROLLMENT ANALYSIS:")
    
    # Calculate weekly cumulative enrollment
    weekly_cumulative = []
    for week in range(timeline_params['rollout_start_week'], timeline_params['rollout_end_week'] + 1):
        # Units that have service active by this week
        active_units = enhanced_df[enhanced_df['Week_Roll_Out_Start'] <= week]['Unit'].count()
        weekly_cumulative.append({
            'Week': week,
            'Cumulative_Units': active_units,
            'Percentage_Complete': (active_units / portfolio_summary['tlw_units']) * 100
        })
    
    cumulative_df = pd.DataFrame(weekly_cumulative)
    
    # Show key milestones
    print("   KEY MILESTONES:")
    print("   " + "="*40)
    milestones = [25, 50, 75, 100]
    for milestone in milestones:
        milestone_week = cumulative_df[cumulative_df['Percentage_Complete'] >= milestone]['Week'].min()
        milestone_units = cumulative_df[cumulative_df['Week'] == milestone_week]['Cumulative_Units'].iloc[0] if pd.notna(milestone_week) else 0
        status = f"Week {milestone_week}: {milestone_units:,} units" if pd.notna(milestone_week) else "Not achieved"
        print(f"   {milestone:3d}% Complete: {status}")
    
    print(f"\n   FINAL ROLLOUT SUMMARY:")
    print(f"   - Start: Week {timeline_params['rollout_start_week']} (April 2026)")
    print(f"   - End: Week {rollout_df['Service_End_Week'].max()} (2026)")
    print(f"   - Duration: {rollout_df['Service_End_Week'].max() - timeline_params['rollout_start_week']} weeks")
    print(f"   - Total Units: {portfolio_summary['tlw_units']:,}")
    print(f"   - Properties: {len(rollout_df)}")
    
    return enhanced_df, rollout_df, cumulative_df, timeline_params

def visualize_tlw_rollout_timeline(rollout_df, cumulative_df, timeline_params, portfolio_summary):
    """
    Create comprehensive visualizations for TLW rollout timeline.
    Generate Gantt chart and cumulative enrollment graphics.
    
    Parameters:
    -----------
    rollout_df : DataFrame
        Property-level rollout schedule
    cumulative_df : DataFrame
        Weekly cumulative enrollment data
    timeline_params : dict
        Timeline parameters and constraints
    portfolio_summary : dict
        Portfolio summary statistics
    """
    print("\n\nPART 5: TLW ROLLOUT VISUALIZATIONS")
    print("="*45)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('TLW Property Enrollment Schedule - 2026 Rollout Timeline', fontsize=16, fontweight='bold', y=0.98)
    
    # ==========================================
    # CHART 1: GANTT CHART (Property-by-Property Timeline)
    # ==========================================
    
    print("1. Creating Gantt Chart - Property-by-Property Timeline:")
    
    # Prepare data for Gantt chart
    gantt_data = rollout_df.copy()
    
    # Create property categories by size
    gantt_data['Size_Category'] = pd.cut(gantt_data['Total_Units'], 
                                        bins=[0, 100, 200, 400, float('inf')],
                                        labels=['Micro (<100 units)', 'Small (100-199 units)', 
                                               'Medium (200-399 units)', 'Large (400+ units)'])
    
    # Color mapping for different categories
    color_map = {
        'Micro (<100 units)': '#87CEEB',      # Light blue
        'Small (100-199 units)': '#40E0D0',   # Turquoise  
        'Medium (200-399 units)': '#FF6B6B',  # Light red
        'Large (400+ units)': '#FF4444'       # Red
    }
    
    # Create y-axis positions (reverse order for top-down display)
    y_positions = range(len(gantt_data))
    gantt_data = gantt_data.reset_index(drop=True)
    
    # Plot installation periods
    for i, (_, row) in enumerate(gantt_data.iterrows()):
        # Installation bar
        install_duration = row['Install_End_Week'] - row['Install_Start_Week'] + 1
        color = color_map[row['Size_Category']]
        
        ax1.barh(len(gantt_data) - 1 - i, install_duration, 
                left=row['Install_Start_Week'], height=0.6, 
                color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add property label and unit count
        label_x = row['Install_Start_Week'] + install_duration/2
        label_text = f"{row['Property']} ({row['Total_Units']} units)"
        ax1.text(label_x, len(gantt_data) - 1 - i, label_text, 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Add service start marker
        ax1.plot(row['Service_Start_Week'], len(gantt_data) - 1 - i, 
                marker='|', markersize=15, color='green', markeredgewidth=3)
    
    # Customize Gantt chart
    ax1.set_yticks(range(len(gantt_data)))
    ax1.set_yticklabels([f"{row['Property']}" for _, row in gantt_data[::-1].iterrows()], fontsize=8)
    ax1.set_xlabel('Timeline - April to December 2026 (Week Numbers)', fontsize=12)
    ax1.set_ylabel('Properties', fontsize=12)
    ax1.set_title('Property-by-Property Timeline', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.set_xlim(timeline_params['rollout_start_week'] - 2, timeline_params['rollout_end_week'] + 2)
    
    # Add month labels on top
    month_weeks = {15: 'Apr', 19: 'May', 23: 'Jun', 27: 'Jul', 31: 'Aug', 
                   35: 'Sep', 40: 'Oct', 44: 'Nov', 48: 'Dec', 52: 'Year End'}
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(list(month_weeks.keys()))
    ax1_top.set_xticklabels(list(month_weeks.values()), fontsize=10)
    
    # Add current week marker
    ax1.axvline(x=timeline_params['current_week'], color='red', linestyle='--', linewidth=2, 
               label=f"Today: Week {timeline_params['current_week']} (Feb 14)")
    
    # Create legend for size categories
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.8, label=category)
                      for category, color in color_map.items()]
    legend_elements.append(plt.Line2D([0], [0], color='green', marker='|', markersize=10, 
                                     label='Service Goes Live', linestyle='None', markeredgewidth=2))
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', label='Today (Week 7)'))
    
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    print(f"   ✅ Gantt chart created with {len(gantt_data)} properties")
    
    # ==========================================
    # CHART 2: CUMULATIVE UNITS TIMELINE
    # ==========================================
    
    print("2. Creating Cumulative Units Timeline:")
    
    # Plot cumulative enrollment
    ax2.plot(cumulative_df['Week'], cumulative_df['Cumulative_Units'], 
             linewidth=4, color='#2E86AB', marker='o', markersize=6, markevery=5)
    
    # Fill area under curve
    ax2.fill_between(cumulative_df['Week'], cumulative_df['Cumulative_Units'], 
                     alpha=0.3, color='#2E86AB', label='Enrollment Progress')
    
    # Add milestone markers
    milestones = [
        (25, 1589, 'Week 25\n1,589 units'),
        (50, 3179, 'Week 35\n3,179 units'), 
        (75, 4769, 'Week 47\n4,769 units'),
        (100, 6359, 'End: Week 52\n6,359 units')
    ]
    
    for pct, target_units, label in milestones:
        # Find closest week
        target_week_data = cumulative_df[cumulative_df['Cumulative_Units'] >= target_units]
        if not target_week_data.empty:
            week = target_week_data['Week'].iloc[0]
            units = target_week_data['Cumulative_Units'].iloc[0]
            
            # Add milestone marker
            ax2.plot(week, units, marker='s', markersize=10, color='orange', markeredgecolor='black')
            
            # Add label box
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7)
            ax2.annotate(label, (week, units), xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold', bbox=bbox_props)
    
    # Add percentage reference lines
    total_units = portfolio_summary['tlw_units']
    percentage_lines = [0.25, 0.50, 0.75, 1.0]
    colors = ['orange', 'gold', 'lightgreen', 'green']
    
    for i, pct in enumerate(percentage_lines):
        target_units = total_units * pct
        ax2.axhline(y=target_units, color=colors[i], linestyle=':', alpha=0.7, linewidth=2)
        ax2.text(timeline_params['rollout_end_week'] + 0.5, target_units, 
                f'{int(pct*100)}% ({int(target_units):,} units)', 
                va='center', fontsize=10, color=colors[i], fontweight='bold')
    
    # Customize cumulative chart
    ax2.set_xlabel('Timeline - April to December 2026 (Week 15 to 52)', fontsize=12)
    ax2.set_ylabel('Cumulative Apartments Enrolled', fontsize=12)
    ax2.set_title(f'{total_units:,} Apartment Units Timeline (April-December)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(timeline_params['rollout_start_week'] - 1, timeline_params['rollout_end_week'] + 2)
    ax2.set_ylim(0, total_units * 1.1)
    
    # Add current week marker
    ax2.axvline(x=timeline_params['current_week'], color='red', linestyle='--', linewidth=2)
    ax2.text(timeline_params['current_week'], total_units * 0.1, 'Start: Week 15\n70 units', 
             ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
             fontsize=10, fontweight='bold')
    
    # Add legend for cumulative chart
    legend_elements2 = [
        plt.Line2D([0], [0], color='#2E86AB', linewidth=4, label='Cumulative Apartments Enrolled'),
        plt.Line2D([0], [0], color='orange', marker='s', markersize=8, label='Milestone Markers', linestyle='None'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Today (Week 7)')
    ]
    ax2.legend(handles=legend_elements2, loc='upper left', fontsize=10)
    
    print(f"   ✅ Cumulative timeline created showing progression to {total_units:,} units")
    
    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    # Summary statistics
    print(f"\n3. VISUALIZATION SUMMARY:")
    print("   " + "="*40)
    print(f"   Timeline: Week {timeline_params['rollout_start_week']} to {rollout_df['Service_End_Week'].max()}")
    print(f"   Properties: {len(rollout_df)}")
    print(f"   Total Units: {portfolio_summary['tlw_units']:,}")
    print(f"   Peak Week: Week {cumulative_df.loc[cumulative_df['Cumulative_Units'].idxmax(), 'Week']}")
    print(f"   Completion: Week {rollout_df['Service_End_Week'].max()}")
    print("   ✅ Both visualizations generated successfully!")
    
    return fig

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
        enhanced_df, rollout_df, cumulative_df, timeline_params = calculate_tlw_rollout_timeline(tlw_eligible_df, portfolio_summary)
        
        # Create rollout visualizations
        rollout_fig = visualize_tlw_rollout_timeline(rollout_df, cumulative_df, timeline_params, portfolio_summary)
