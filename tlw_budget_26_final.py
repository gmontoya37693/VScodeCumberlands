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
    print("📂 PART 1: UPLOAD AND EXPLORATION")
    print("="*40)
    
    # Load acento apartments fixed data
    print("\n1️⃣ LOADING ACENTO APARTMENTS FIXED:")
    try:
        acento_df = pd.read_excel('acento_apartments_fixed.xlsx')
        print(f"   ✅ Successfully loaded")
        print(f"   📊 Records: {len(acento_df):,}")
        print(f"   📋 Columns: {list(acento_df.columns)}")
        print(f"   🏢 Properties: {acento_df['Property'].nunique()}")
        print(f"   🏠 Units with SQFT: {acento_df['SQFT'].notna().sum():,}/{len(acento_df):,}")
        
        print(f"\n   📋 Sample Records:")
        sample_cols = ['Property', 'Unit', 'SQFT'] if 'SQFT' in acento_df.columns else ['Property', 'Unit']
        print(acento_df[sample_cols].head(3).to_string(index=False))
        
    except Exception as e:
        print(f"   ❌ Error loading acento_apartments_fixed.xlsx: {e}")
        return None, None
    
    # Load Property Base rent SQFT data
    print("\n\n2️⃣ LOADING PROPERTY BASE RENT SQFT:")
    try:
        base_rent_df = pd.read_excel('Property Base rent SQFT.xlsx')
        print(f"   ✅ Successfully loaded")
        print(f"   📊 Records: {len(base_rent_df):,}")
        print(f"   📋 Columns: {list(base_rent_df.columns)}")
        
        print(f"\n   📋 Sample Records:")
        print(base_rent_df.head(3).to_string(index=False))
        
        # Quick analysis
        if 'Min Rent' in base_rent_df.columns and 'Max Rent' in base_rent_df.columns:
            print(f"\n   💰 Rent Range: ${base_rent_df['Min Rent'].min():,.0f} - ${base_rent_df['Max Rent'].max():,.0f}")
        if 'Min SQFT' in base_rent_df.columns and 'Max SQFT' in base_rent_df.columns:
            print(f"   📐 SQFT Range: {base_rent_df['Min SQFT'].min():.0f} - {base_rent_df['Max SQFT'].max():.0f} sq ft")
        
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
    print("\n\n🔍 DATA QUALITY EXAMINATION")
    print("="*35)
    
    if acento_df is not None:
        print("\n📊 ACENTO APARTMENTS ANALYSIS:")
        
        # Check for missing values
        missing_data = acento_df.isnull().sum()
        if missing_data.sum() > 0:
            print("   ⚠️ Missing Values:")
            for col, count in missing_data[missing_data > 0].items():
                print(f"      • {col}: {count:,} missing ({count/len(acento_df)*100:.1f}%)")
                
                # Show details of missing records
                if col == 'SQFT' and count > 0:
                    missing_records = acento_df[acento_df[col].isnull()]
                    print(f"      🔍 Missing {col} Records:")
                    display_cols = ['Unit_ID', 'Key', 'Property', 'Unit'] if 'Unit_ID' in missing_records.columns else ['Key', 'Property', 'Unit']
                    for _, row in missing_records.iterrows():
                        record_info = " | ".join([f"{col}: {row[col]}" for col in display_cols if col in row])
                        print(f"         - {record_info}")
        else:
            print("   ✅ No missing values found")
        
        # Property distribution
        print(f"\n   🏢 Property Distribution:")
        prop_counts = acento_df['Property'].value_counts().head(10)
        for prop, count in prop_counts.items():
            print(f"      • {prop}: {count:,} units")
        
        if len(prop_counts) == 10:
            remaining = len(acento_df['Property'].value_counts()) - 10
            if remaining > 0:
                print(f"      • ... and {remaining} more properties")
    
    if base_rent_df is not None:
        print(f"\n📊 BASE RENT DATA ANALYSIS:")
        
        # Check property alignment
        if acento_df is not None:
            acento_props = set(acento_df['Property'].unique())
            rent_props = set(base_rent_df['Property'].unique())
            
            matching = acento_props.intersection(rent_props)
            missing_in_rent = acento_props - rent_props
            extra_in_rent = rent_props - acento_props
            
            print(f"   🔗 Property Alignment:")
            print(f"      • Matching: {len(matching)} properties")
            print(f"      • Missing in rent data: {len(missing_in_rent)} properties")
            print(f"      • Extra in rent data: {len(extra_in_rent)} properties")
            
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
    print("\n\n💼 PART 2: CREATE WORKING DATAFRAME")
    print("="*45)
    
    # Start with acento apartments 5 columns
    print("1️⃣ STARTING WITH ACENTO APARTMENTS (5 COLUMNS):")
    working_df = acento_df.copy()
    print(f"   ✅ Base dataframe: {len(working_df):,} units")
    print(f"   📋 Columns: {list(working_df.columns)}")
    
    # Create property slope lookup dictionary
    print("\n2️⃣ CREATING PROPERTY SLOPE LOOKUP:")
    slope_lookup = dict(zip(base_rent_df['Property'], base_rent_df['Slope']))
    print(f"   ✅ Loaded slope data for {len(slope_lookup)} properties")
    print(f"   📋 Sample slopes: {dict(list(slope_lookup.items())[:3])}")
    
    # Calculate rent for each unit: Unit_SQFT × Property_Slope
    print("\n3️⃣ CALCULATING RENT COLUMN:")
    print("   📐 Formula: Rent = Unit_SQFT × Property_Slope")
    
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
        print(f"   💰 Rent Range: ${rent_stats['min']:.0f} - ${rent_stats['max']:.0f}")
        print(f"   💰 Average Rent: ${rent_stats['mean']:.0f}")
    
    print(f"\n   📋 Sample Working DataFrame (6 columns):")
    print(working_df.head(3).to_string(index=False))
    
    print(f"\n✅ Working dataframe completed:")
    print(f"   📊 Total columns: {len(working_df.columns)} (as expected: 6)")
    print(f"   📋 Final columns: {list(working_df.columns)}")
    
    return working_df

def main():
    """
    Main function to execute TLW budget analysis.
    """
    print("🚀 TLW BUDGET 26 FINAL ANALYSIS")
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
