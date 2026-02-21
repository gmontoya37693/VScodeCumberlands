# Budget 26 Version 5 Linear - Renewal-Based Enrollment Model
# Started: February 21, 2026  
# Model Type: Linear monthly enrollment based on lease renewal cycles
# 
# CHANGE LOG:
# Version 5.0 - NEW APPROACH: Renewal-based enrollment model with parameter sensitivity
#   - Linear enrollment model: same monthly capacity across all months
#   - Uses RealPage renewal data as base for calculations
#   - Two enrollment streams: renewal enrollments + current tenant enrollments
#   - Property exclusions: WEA, CCA, CHA, CWA
#   - Built-in parameter sensitivity framework
#
# ============================================================================
# 📊 PARAMETER TABLE - SENSITIVITY ANALYSIS INPUTS
# ============================================================================

# ENROLLMENT PARAMETERS
RENEWAL_OPTOUT_RATE = 0.20          # 20% of renewal tenants opt-out for own H04
CURRENT_TENANT_ENROLLMENT_RATE = 0.15  # 15% of current tenants enroll voluntarily
TIMELINE_START_MONTH = 4            # April 2026
TIMELINE_END_MONTH = 12             # December 2026

# PROPERTY PARAMETERS  
EXCLUDED_PROPERTIES = ['WEA', 'CCA', 'CHA', 'CWA']  # Properties excluded from TLW

# PRICING PARAMETERS
TLW_FLAT_RATE = 15.00              # $15.00 per unit flat rate
TLW_SQFT_BASED = True              # Use SQFT-adjusted pricing (avg pays $15)
ANNUAL_PREMIUM_GROWTH = 0.025      # 2.5% annual premium increase (2027+)

# OPERATIONAL PARAMETERS
MONTHLY_RENEWAL_DISTRIBUTION = 1.0/12.0  # Even distribution (100%/12 months)
ENROLLMENT_START_DELAY = 0         # No delay in enrollment recognition
IMMEDIATE_REVENUE_RECOGNITION = True

# COST STRUCTURE (% of GWP) - From previous analysis
COST_STRUCTURE = {
    'Vacancy_Drag': 10.0,
    'Bad_Debt': 3.0, 
    'Expected_Losses': 58.0,
    'Loss_Development_IBNR': 4.0,
    'Claims_Administration': 9.0,
    'Customer_Acquisition_Cost': 2.0,
    'Captive_Operating_Costs': 5.0,
    'Premium_Taxes': 1.5,
    'Capital_Cost': 4.0,
    'Compliance_Regulatory': 2.0
}

# SENSITIVITY SCENARIOS
SCENARIOS = {
    'CONSERVATIVE': {'optout_rate': 0.30, 'current_enrollment': 0.10},
    'BASE_CASE':    {'optout_rate': 0.20, 'current_enrollment': 0.15}, 
    'OPTIMISTIC':   {'optout_rate': 0.15, 'current_enrollment': 0.25}
}

print("📊 TLW BUDGET V5 LINEAR - RENEWAL-BASED ENROLLMENT MODEL")
print("="*70)
print(f"🎯 BASE CASE PARAMETERS:")
print(f"   Renewal Opt-out Rate: {RENEWAL_OPTOUT_RATE:.1%}")
print(f"   Current Tenant Enrollment: {CURRENT_TENANT_ENROLLMENT_RATE:.1%}")
print(f"   Timeline: {TIMELINE_START_MONTH}/2026 - {TIMELINE_END_MONTH}/2026")
print(f"   Excluded Properties: {', '.join(EXCLUDED_PROPERTIES)}")
print(f"   TLW Pricing: ${TLW_FLAT_RATE:.2f} flat, SQFT-adjusted: {TLW_SQFT_BASED}")
print("="*70)

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# 📋 REALPAGE PROPERTY DATA - Expiration and Renewal Activity 
# Source: Acento Real Estate Partners, LLC (01/01/2025 - 12/31/2025)
# ============================================================================

# Property data: [Property_Name, Expiration_Count, Retention_Rate]
REALPAGE_PROPERTY_DATA = [
    ['Ashland Towne Square', 218, 61.9],
    ['Burnham Woods', 168, 64.3], 
    ['Camden Hills', 152, 60.5],
    ['Carlyle Landing Apartments', 172, 55.8],
    ['Center Pointe', 151, 68.9],
    ['Centre at Silver Spring', 241, 52.3], 
    ['Chatham Gardens', 417, 74.6],  # CHA - Will be excluded
    ['Chelsea Park', 71, 78.9],
    ['Crystal Woods', 343, 67.6],    # CWA - Will be excluded
    ['Eaton Square', 416, 59.4],
    ['Governor Square', 232, 74.1],
    ['Hamilton Gardens', 75, 49.3],
    ['Indigo', 346, 62.7],
    ['Jefferson Hall', 115, 67.0],
    ['Lakeshore at Hampton Center', 385, 47.5],
    ['Lakeview', 563, 49.4],
    ['Linden Park Apartments', 199, 69.3],
    ['Lockwood', 107, 54.2],
    ['Northampton Reserve', 593, 55.5],
    ['Old Orchard', 182, 69.8],
    ['Park Gardens', 51, 54.9],
    ['Terrace Green', 97, 39.2],
    ['The Commons at Cowan Boulevard', 238, 60.1],  # CCA - Will be excluded
    ['The Grove At Alban', 280, 70.7],
    ['The Landing at Oyster Point', 508, 57.5],
    ['The Westwinds', 146, 67.8],    # CWA mistake - this should be different
    ['Towne Crest Apartments', 103, 72.8],
    ['Townsend Square', 214, 51.4],
    ['Wellington Woods', 109, 59.6],  # WEA - Will be excluded
    ['Woodgate Apartments', 258, 66.7]
]

# Convert to DataFrame
realpage_df = pd.DataFrame(REALPAGE_PROPERTY_DATA, 
                          columns=['Property_Name', 'Total_Units', 'Retention_Rate'])

print(f"\n📋 REALPAGE PROPERTY DATA LOADED:")
print(f"   Total Properties: {len(realpage_df)}")
print(f"   Total Units: {realpage_df['Total_Units'].sum():,}")
print(f"   Average Retention Rate: {realpage_df['Retention_Rate'].mean():.1f}%")

# ============================================================================
# 🚫 APPLY PROPERTY EXCLUSIONS
# ============================================================================

# Apply exclusions - Mapping RealPage names to property codes
PROPERTY_NAME_TO_CODE = {
    'Wellington Woods': 'WEA',                    # WEA = Wellington
    'The Commons at Cowan Boulevard': 'CCA',      # CCA = Commons Cowan 
    'Chatham Gardens': 'CHA',                     # CHA = Chatham
    'Crystal Woods': 'CWA'                        # CWA = Crystal Woods
}

# Flag excluded properties
realpage_df['excluded'] = False
for prop_name, prop_code in PROPERTY_NAME_TO_CODE.items():
    if prop_code in EXCLUDED_PROPERTIES:
        realpage_df.loc[realpage_df['Property_Name'] == prop_name, 'excluded'] = True

# Create rollout dataset (excluding flagged properties)
rollout_properties_df = realpage_df[realpage_df['excluded'] == False].copy()

print(f"\n🚫 EXCLUSION RESULTS:")
excluded_df = realpage_df[realpage_df['excluded'] == True]
if len(excluded_df) > 0:
    print(f"   Excluded Properties: {len(excluded_df)}")
    for _, row in excluded_df.iterrows():
        code = PROPERTY_NAME_TO_CODE.get(row['Property_Name'], 'Unknown')
        print(f"     - {row['Property_Name']} ({code}): {row['Total_Units']} units")
    print(f"   Excluded Units: {excluded_df['Total_Units'].sum():,}")
else:
    print(f"   No properties excluded from RealPage data")

print(f"\n✅ ROLLOUT PROPERTIES:")
print(f"   Total Properties Available: {len(realpage_df)}")
print(f"   Excluded Properties: {len(excluded_df)}")
print(f"   Rollout Properties: {len(rollout_properties_df)}")
print(f"   Rollout Units: {rollout_properties_df['Total_Units'].sum():,}")
print(f"   Average Retention: {rollout_properties_df['Retention_Rate'].mean():.1f}%")

# ============================================================================
# 📊 MONTHLY ENROLLMENT CALCULATIONS
# ============================================================================

# Calculate monthly components for each rollout property
rollout_properties_df = rollout_properties_df.copy()

# Calculate annual renewals based on retention rate (retained = renewed)
rollout_properties_df['Annual_Renewals'] = (rollout_properties_df['Total_Units'] * 
                                           rollout_properties_df['Retention_Rate'] / 100).round(0)

# Calculate monthly renewals (even distribution throughout year)
rollout_properties_df['Monthly_Renewals'] = (rollout_properties_df['Annual_Renewals'] * 
                                            MONTHLY_RENEWAL_DISTRIBUTION).round(1)

# Calculate current residents (total minus those renewing this month)
rollout_properties_df['Current_Residents'] = (rollout_properties_df['Total_Units'] - 
                                             rollout_properties_df['Monthly_Renewals']).round(0)

# Calculate monthly enrollment streams
# Stream 1: Renewal enrollments (automatic minus opt-outs)
rollout_properties_df['Monthly_Renewal_Enrollments'] = (
    rollout_properties_df['Monthly_Renewals'] * (1 - RENEWAL_OPTOUT_RATE)
).round(1)

# Stream 2: Current tenant enrollments (voluntary sign-ups)
rollout_properties_df['Monthly_Current_Enrollments'] = (
    rollout_properties_df['Current_Residents'] * CURRENT_TENANT_ENROLLMENT_RATE
).round(1)

# Total monthly enrollments per property
rollout_properties_df['Total_Monthly_Enrollments'] = (
    rollout_properties_df['Monthly_Renewal_Enrollments'] + 
    rollout_properties_df['Monthly_Current_Enrollments']
).round(1)

print(f"\n📊 MONTHLY ENROLLMENT CALCULATIONS SUMMARY:")
print(f"   Total Monthly Renewal Stream: {rollout_properties_df['Monthly_Renewal_Enrollments'].sum():.1f} enrollments/month")
print(f"   Total Monthly Current Stream: {rollout_properties_df['Monthly_Current_Enrollments'].sum():.1f} enrollments/month") 
print(f"   Total Monthly Enrollments: {rollout_properties_df['Total_Monthly_Enrollments'].sum():.1f} enrollments/month")
print(f"   Annual Run Rate: {rollout_properties_df['Total_Monthly_Enrollments'].sum() * 12:.0f} enrollments/year")

# ============================================================================
# 📋 PROPERTY-BY-PROPERTY ENROLLMENT TABLE
# ============================================================================

print(f"\n📋 PROPERTY-BY-PROPERTY MONTHLY ENROLLMENT BREAKDOWN:")
print(f"\n{'=' * 120}")
print(f"{'Property':<35} {'Units':<8} {'Retention':<10} {'Monthly':<8} {'Current':<8} {'Renewal':<8} {'Current':<8} {'Total':<8}")
print(f"{'Name':<35} {'Total':<8} {'Rate %':<10} {'Renewals':<8} {'Residents':<8} {'Enrolls':<8} {'Enrolls':<8} {'Monthly':<8}")
print(f"{'=' * 120}")

for _, row in rollout_properties_df.iterrows():
    prop_name = row['Property_Name'][:32] + "..." if len(row['Property_Name']) > 32 else row['Property_Name']
    print(f"{prop_name:<35} "
          f"{row['Total_Units']:<8.0f} "
          f"{row['Retention_Rate']:<10.1f} "
          f"{row['Monthly_Renewals']:<8.1f} "
          f"{row['Current_Residents']:<8.0f} "
          f"{row['Monthly_Renewal_Enrollments']:<8.1f} "
          f"{row['Monthly_Current_Enrollments']:<8.1f} "
          f"{row['Total_Monthly_Enrollments']:<8.1f}")

print(f"{'=' * 120}")
totals_row = rollout_properties_df[['Total_Units', 'Monthly_Renewals', 'Current_Residents', 
                                   'Monthly_Renewal_Enrollments', 'Monthly_Current_Enrollments', 
                                   'Total_Monthly_Enrollments']].sum()
avg_retention = rollout_properties_df['Retention_Rate'].mean()

print(f"{'TOTALS':<35} "
      f"{totals_row['Total_Units']:<8.0f} "
      f"{avg_retention:<10.1f} "
      f"{totals_row['Monthly_Renewals']:<8.1f} "
      f"{totals_row['Current_Residents']:<8.0f} "
      f"{totals_row['Monthly_Renewal_Enrollments']:<8.1f} "
      f"{totals_row['Monthly_Current_Enrollments']:<8.1f} "
      f"{totals_row['Total_Monthly_Enrollments']:<8.1f}")

# ============================================================================
# 📊 STEP 2: TWO REVENUE STREAMS WITH ACTUAL MONTHLY DATA 
# ============================================================================

print(f"\n📊 STEP 2: TWO REVENUE STREAMS WITH ACTUAL MONTHLY DATA")
print("="*70)

# STREAM 1: Contract Ending Revenue (Using actual monthly renewal patterns)
print(f"\n🔄 STREAM 1: CONTRACT ENDING (Actual Monthly Patterns)")

stream1_monthly = {}
for i, month in enumerate(month_names):
    month_col = rollout_months[i]
    monthly_renewals = rollout_monthly_df[month_col].sum()
    monthly_tlw_enrollments = monthly_renewals * (1 - RENEWAL_OPTOUT_RATE)
    stream1_monthly[month] = monthly_tlw_enrollments
    print(f"   {month}: {monthly_renewals} renewals → {monthly_tlw_enrollments:.0f} TLW enrollments")

# STREAM 2: Current Tenant Conversion (April-June only, using March remaining data)
print(f"\n📋 STREAM 2: CURRENT TENANT CONVERSION (April-June)")

# Use March remaining data as starting point for current tenants
march_remaining = rollout_monthly_df['Mar_Remain'].sum()
current_tenant_pool = march_remaining

# Calculate conversion numbers
current_tenant_enrollments = current_tenant_pool * CURRENT_TENANT_OPTIN_RATE
monthly_current_conversion = current_tenant_enrollments / 3  # Spread over 3 months

print(f"   March remaining tenants: {march_remaining:,.0f}")
print(f"   Total TLW conversions available: {current_tenant_enrollments:,.0f} ({CURRENT_TENANT_OPTIN_RATE:.1%})")
print(f"   Monthly conversion (Apr-Jun): {monthly_current_conversion:,.0f}")

# COMBINED MONTHLY TIMELINE
print(f"\n📅 COMBINED REVENUE TIMELINE (April-December 2026)")
print("="*80)

cumulative_enrollments = 0
timeline_data = []

print(f"{'Month':<12} {'Stream 1':<10} {'Stream 2':<10} {'Total':<10} {'Cumulative':<12} {'Revenue':<12}")
print(f"{'2026':<12} {'Renewals':<10} {'Current':<10} {'Monthly':<10} {'Enrollments':<12} {'Monthly':<12}")
print("-" * 80)

for i, month in enumerate(month_names):
    # Stream 1: Actual renewal data
    stream1_enrollments = stream1_monthly[month]
    
    # Stream 2: Current tenant conversion (only April-June)
    stream2_enrollments = monthly_current_conversion if i < 3 else 0
    
    # Total monthly enrollments
    total_monthly = stream1_enrollments + stream2_enrollments
    cumulative_enrollments += total_monthly
    
    # Monthly revenue
    monthly_revenue = cumulative_enrollments * TLW_FLAT_RATE
    
    timeline_data.append({
        'Month': month,
        'Stream1': stream1_enrollments,
        'Stream2': stream2_enrollments,
        'Total_Monthly': total_monthly,
        'Cumulative': cumulative_enrollments,
        'Revenue': monthly_revenue
    })
    
    print(f"{month:<12} "
          f"{stream1_enrollments:<10.0f} "
          f"{stream2_enrollments:<10.0f} "
          f"{total_monthly:<10.0f} "
          f"{cumulative_enrollments:<12.0f} "
          f"${monthly_revenue:<11,.0f}")

print("-" * 80)
total_stream1 = sum(stream1_monthly.values())
total_stream2 = monthly_current_conversion * 3

print(f"{'TOTAL 2026':<12} "
      f"{total_stream1:<10.0f} "
      f"{total_stream2:<10.0f} "
      f"{cumulative_enrollments:<10.0f} "
      f"{cumulative_enrollments:<12.0f} "
      f"${monthly_revenue:<11,.0f}")

print(f"\n✅ ACTUAL DATA MODEL SUMMARY:")
print(f"   Stream 1 (Actual Renewals): {total_stream1:.0f} enrollments")
print(f"   Stream 2 (Current Conversion): {total_stream2:.0f} enrollments") 
print(f"   Total 2026 Enrollments: {cumulative_enrollments:.0f}")
print(f"   Final Monthly Revenue: ${monthly_revenue:,.0f}")
print(f"   Annual Run Rate: ${monthly_revenue * 12:,.0f}")

print(f"\n🎯 KEY INSIGHTS FROM ACTUAL DATA:")
peak_month = max(stream1_monthly, key=stream1_monthly.get)
low_month = min(stream1_monthly, key=stream1_monthly.get)
print(f"   Peak renewal month: {peak_month} ({stream1_monthly[peak_month]:.0f} enrollments)")
print(f"   Lowest renewal month: {low_month} ({stream1_monthly[low_month]:.0f} enrollments)")
print(f"   Renewal pattern variation: {stream1_monthly[peak_month]/stream1_monthly[low_month]:.1f}x difference")
