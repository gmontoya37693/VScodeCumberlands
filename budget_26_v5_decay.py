# TLW BUDGET V5 DECAY - Step by Step Build
# Date: February 24, 2026
# 
# ============================================================================
# PARAMETER SETUP - FINANCIAL & OPERATIONAL ASSUMPTIONS
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("TLW BUDGET V5 DECAY - Step by Step Build")
print("="*60)
print(f"PARAMETER SETUP - Financial & Operational Assumptions")
print(f"{'='*60}\n")

# ============================================================================
# ADDITIONAL PARAMETERS
# ============================================================================

active_rate = 0.48357
no_reply_rate = 0.50
tlw_price = 15.00  # USD
tlw_default_rate = 0.70  # 70% of new tenants default to TLW

old_tlw_rate = (1.0 - active_rate) * no_reply_rate

print(f"\nADDITIONAL PARAMETERS")
print(f"{'='*60}")
print(f"Active Rate: {active_rate:.5f}")
print(f"No Reply Rate: {no_reply_rate:.2%}")
print(f"TLW Price: ${tlw_price:,.2f}")
print(f"TLW Default Rate: {tlw_default_rate:.2%}")
print(f"Old TLW Rate: {old_tlw_rate:.5f}")
print(f"{'='*60}\n")

# Parameter definitions with both Suggested and Alex scenarios (as % of GWP)
PARAMETERS_DATA = [
    ['Gross Written Premium (GWP)', 100.0, 100.0],
    ['Vacancy Drag', -10.0, -7.20],
    ['Bad Debt', -3.0, 0.00],
    ['NET EARNED PREMIUM', None, None],
    ['Expected Losses (Base)', -58.0, -58.00],
    ['Loss Development Lag (IBNR)', -4.0, -4.00],
    ['ADJUSTED LOSS RATIO', None, None],
    ['Claims Administration (TPA)', -9.0, -9.00],
    ['Customer Acquisition Cost (CAC)', -2.0, 0.00],
    ['Captive Operating Cost (COC)', -5.0, -5.00],
    ['Premium Taxes', -1.5, -1.5],
    ['Capital Cost', -4.0, 0.00],
    ['Compliance/Regulatory Buffer', -2.0, -2.00],
]

# Create parameters dataframe
params_data = []
for row in PARAMETERS_DATA:
    description = row[0]
    suggested = row[1]
    alex = row[2]
    
    params_data.append({
        'Description': description,
        'Suggested': suggested,
        'Alex': alex
    })

params_df = pd.DataFrame(params_data)

# Calculate NET EXPECTED PREMIUM dynamically (GWP + Vacancy Drag + Bad Debt)
gwp = 100.0
vacancy_drag_suggested = params_df.loc[1, 'Suggested']
bad_debt_suggested = params_df.loc[2, 'Suggested']
vacancy_drag_alex = params_df.loc[1, 'Alex']
bad_debt_alex = params_df.loc[2, 'Alex']

net_expected_premium_suggested = gwp + vacancy_drag_suggested + bad_debt_suggested
net_expected_premium_alex = gwp + vacancy_drag_alex + bad_debt_alex

params_df.loc[3, 'Suggested'] = net_expected_premium_suggested
params_df.loc[3, 'Alex'] = net_expected_premium_alex

# Extract loss parameters BEFORE calculating adjusted loss ratio
expected_losses_suggested = params_df.loc[4, 'Suggested']
loss_dev_suggested = params_df.loc[5, 'Suggested']
expected_losses_alex = params_df.loc[4, 'Alex']
loss_dev_alex = params_df.loc[5, 'Alex']

# Calculate ADJUSTED LOSS RATIO dynamically
adjusted_loss_ratio_suggested = expected_losses_suggested + loss_dev_suggested
adjusted_loss_ratio_alex = expected_losses_alex + loss_dev_alex

params_df.loc[6, 'Suggested'] = adjusted_loss_ratio_suggested
params_df.loc[6, 'Alex'] = adjusted_loss_ratio_alex

# Extract operating costs
tpa_suggested = params_df.loc[7, 'Suggested']
cac_suggested = params_df.loc[8, 'Suggested']
coc_suggested = params_df.loc[9, 'Suggested']
premium_taxes_suggested = params_df.loc[10, 'Suggested']
capital_cost_suggested = params_df.loc[11, 'Suggested']
compliance_suggested = params_df.loc[12, 'Suggested']

tpa_alex = params_df.loc[7, 'Alex']
cac_alex = params_df.loc[8, 'Alex']
coc_alex = params_df.loc[9, 'Alex']
premium_taxes_alex = params_df.loc[10, 'Alex']
capital_cost_alex = params_df.loc[11, 'Alex']
compliance_alex = params_df.loc[12, 'Alex']

# Calculate PROFIT % dynamically
profit_suggested = net_expected_premium_suggested + adjusted_loss_ratio_suggested + tpa_suggested + cac_suggested + coc_suggested + premium_taxes_suggested + capital_cost_suggested + compliance_suggested
profit_alex = net_expected_premium_alex + adjusted_loss_ratio_alex + tpa_alex + cac_alex + coc_alex + premium_taxes_alex + capital_cost_alex + compliance_alex

print(f"✅ PARAMETER SET: SUGGESTED vs ALEX SCENARIO")
print(f"   (Waterfall % of GWP = 100%)")
print(f"{'-'*70}")
print(params_df.to_string(index=False))
print(f"{'-'*70}")

# Add profit calculation row
print(f"\n{'PROFIT %':<35} {profit_suggested:>12.2f}%  {profit_alex:>12.2f}%")
print(f"{'-'*70}")

# Validation
print(f"\n✅ PARAMETER VALIDATION:")
print(f"   ✅ Suggested scenario profit: {profit_suggested:.2f}%")
print(f"   ✅ Alex scenario profit: {profit_alex:.2f}%")
print(f"   ✅ Using ALEX scenario for model")
print(f"   ✅ Parameters loaded successfully")

# Extract rates for budget calculations (Alex scenario, converted to decimal)
vac_drag_rate = vacancy_drag_alex / 100.0
exp_loss_rate = expected_losses_alex / 100.0
ibnr_rate = loss_dev_alex / 100.0
tpa_rate = tpa_alex / 100.0
coc_rate = coc_alex / 100.0
prem_tax_rate = premium_taxes_alex / 100.0
buffer_rate = compliance_alex / 100.0

print(f"\n✅ BUDGET RATES (Alex Scenario):")
print(f"   Vacancy Drag Rate: {vac_drag_rate:.4f}")
print(f"   Expected Loss Rate: {exp_loss_rate:.4f}")
print(f"   IBNR Rate: {ibnr_rate:.4f}")
print(f"   TPA Rate: {tpa_rate:.4f}")
print(f"   COC Rate: {coc_rate:.4f}")
print(f"   Premium Tax Rate: {prem_tax_rate:.4f}")
print(f"   Buffer Rate: {buffer_rate:.4f}")

print(f"\n{'='*60}")
print(f"✅ PARAMETER SECTION COMPLETE")
print(f"{'='*60}\n")

# ============================================================================
# STEP 1: PROPERTY COUNT DATA UPLOAD AND POPULATION
# ============================================================================

print(f"\nSTEP 1: Property Portfolio Data Upload")
print(f"{'='*60}\n")

# Property expiration count data (exp_count = unit count for each property)
PROPERTY_EXPIRATION_DATA = [
    ['Ashland Towne Square', 218],
    ['Burnham Woods', 168],
    ['Camden Hills', 152],
    ['Carlyle Landing Apartments', 172],
    ['Center Pointe', 151],
    ['Centre at Silver Spring', 241],
    ['Chatham Gardens', 417],
    ['Chelsea Park', 71],
    ['Crystal Woods', 343],
    ['Eaton Square', 416],
    ['Governor Square', 232],
    ['Hamilton Gardens', 75],
    ['Indigo', 346],
    ['Jefferson Hall', 115],
    ['Lakeshore at Hampton Center', 385],
    ['Lakeview', 563],
    ['Linden Park Apartments', 199],
    ['Lockwood', 107],
    ['Northampton Reserve', 593],
    ['Old Orchard', 182],
    ['Park Gardens', 51],
    ['Terrace Green', 97],
    ['The Commons at Cowan Boulevard', 238],
    ['The Grove At Alban', 280],
    ['The Landing at Oyster Point', 508],
    ['The Westwinds', 146],
    ['Towne Crest Apartments', 103],
    ['Townsend Square', 214],
    ['Wellington Woods', 109],
    ['Woodgate Apartments', 258]
]

# Create DataFrame
properties_df = pd.DataFrame(PROPERTY_EXPIRATION_DATA, 
                           columns=['Property_Name', 'exp_count'])

print(f"\nSTEP 1: Property Portfolio Data Upload")
print(f"   Total Properties: {len(properties_df)}")
print(f"   Total Units: {properties_df['exp_count'].sum():,}")

print(f"\n✅ STEP 1 Complete: Property dataframe created")
print(f"\n✅ Properties DataFrame:")
print(properties_df.to_string(index=False))

# Add totals row
print("\n" + "="*50)
print(f"{'TOTALS':<35} {properties_df['exp_count'].sum():>6}")
print("="*50)

# Validation checks
total_properties = len(properties_df)
total_units = properties_df['exp_count'].sum()

print(f"\n✅ STEP 1 VALIDATION:")
print(f"   ✅ Properties loaded: {total_properties}")
print(f"   ✅ Total units: {total_units:,}")
print(f"   ✅ Dataframe created successfully")

print(f"\n\n{'='*60}")
print(f"STEP 2: LINEAR ROLLOVER - 12 MONTH SPLIT (Apr-2026 to Mar-2027)")
print(f"{'='*60}\n")

# ============================================================================
# STEP 2: LINEAR ROLLOVER - 12 MONTH SPLIT (Apr-2026 to Mar-2027)
# ============================================================================

# Define months (Apr-2026 = month 1, May-2026 = month 2, ..., Mar-2027 = month 12)
months = ['Apr-26', 'May-26', 'Jun-26', 'Jul-26', 'Aug-26', 'Sep-26', 
          'Oct-26', 'Nov-26', 'Dec-26', 'Jan-27', 'Feb-27', 'Mar-27']

# Create new and old columns for each month
for month_num, month in enumerate(months, start=1):
    new_col = f'{month}_new'
    old_col = f'{month}_old'
    
    # new = int(exp_count/12) * n
    properties_df[new_col] = (properties_df['exp_count'] // 12) * month_num
    
    # old = exp_count - new
    properties_df[old_col] = properties_df['exp_count'] - properties_df[new_col]

print(f"✅ Created 24 columns (12 months × 2: new/old)")
print(f"   Months: {months}")
print(f"\n✅ Sample rows (first 3 properties, showing Apr-26 and Dec-26):")

# Show sample data
sample_cols = ['Property_Name', 'exp_count', 'Apr-26_new', 'Apr-26_old', 'Dec-26_new', 'Dec-26_old']
print(properties_df[sample_cols].head(3).to_string(index=False))

print(f"\n✅ STEP 2 VALIDATION:")
print(f"   ✅ 24 columns created (12 months × new/old)")
print(f"   ✅ Formula applied: new = int(exp_count/12) × month_number")
print(f"   ✅ Formula applied: old = exp_count - new")

print(f"\n{'='*60}")
print(f"✅ STEP 2 COMPLETE - Ready for STEP 3")
print(f"{'='*60}\n")

# ============================================================================
# STEP 3: CREATE ROLLOVER GRAPHIC
# ============================================================================

print(f"\nSTEP 3: GENERATING ROLLOVER CHART")
print(f"{'='*60}\n")

# Calculate cumulative new and old for each month
cumulative_new = []
cumulative_old = []

for month in months:
    new_col = f'{month}_new'
    old_col = f'{month}_old'
    
    cum_new = properties_df[new_col].sum()
    cum_old = properties_df[old_col].sum()
    
    cumulative_new.append(cum_new)
    cumulative_old.append(cum_old)

# Create chart
plt.figure(figsize=(14, 8))

x_pos = np.arange(len(months))

plt.plot(x_pos, cumulative_new, 
         marker='o', linewidth=3, markersize=10, 
         color='#4472C4', label='Roll Over Accumulation', zorder=3)

plt.plot(x_pos, cumulative_old, 
         marker='o', linewidth=3, markersize=10, 
         color='#C5504B', label='Old Rentroll', zorder=3)

# Add data labels
for i, (new_val, old_val) in enumerate(zip(cumulative_new, cumulative_old)):
    plt.text(i, new_val + 100, str(new_val), ha='center', va='bottom', 
             fontweight='bold', fontsize=10, color='#4472C4')
    plt.text(i, old_val - 150, str(old_val), ha='center', va='top', 
             fontweight='bold', fontsize=10, color='#C5504B')

plt.title('Acento Turn-over - 7150 units', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('Number of Tenants', fontsize=12, fontweight='bold')
plt.xticks(x_pos, months, rotation=45)
plt.ylim(0, 8000)
plt.grid(True, alpha=0.3, zorder=0)
plt.legend(loc='upper left', fontsize=12, frameon=True)
plt.tight_layout()

chart_filename = 'acento_turnover_7150_units.png'
plt.savefig(chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"✅ Chart generated and saved as: {chart_filename}")

# Print summary table
print(f"\n✅ ROLLOVER SUMMARY:")
print(f"{'Month':<10} {'New Cumul':<12} {'Old Remains':<12} {'Total':<8}")
print(f"{'-'*45}")
for month, new, old in zip(months, cumulative_new, cumulative_old):
    print(f"{month:<10} {new:<12} {old:<12} {new+old:<8}")

print(f"\n✅ STEP 3 COMPLETE - Graphics generated")
print(f"{'='*60}\n")

# ============================================================================
# STEP 4: TOTAL UNITS DATAFRAME - MONTHLY BREAKDOWN
# ============================================================================

print(f"\nSTEP 4: TOTAL UNITS MONTHLY BREAKDOWN")
print(f"{'='*60}\n")

# Create dataframe with monthly totals
units_totals_data = []

for i, month in enumerate(months):
    new_total = cumulative_new[i]
    old_total = cumulative_old[i]
    
    # Calculate new_tlw (new tenants defaulting to TLW)
    new_tlw = int(new_total * tlw_default_rate)
    
    # Calculate new_Ins (new tenants with insurance)
    new_ins = new_total - new_tlw
    
    # Calculate old splits
    old_ins = int(old_total * active_rate)
    old_tlw = int(old_total * old_tlw_rate)
    drag = old_total - old_ins - old_tlw
    
    # Calculate totals
    tlw_total = new_tlw + old_tlw
    total_ins = new_ins + old_ins
    
    units_totals_data.append({
        'Month': month,
        'new': new_total,
        'old': old_total,
        'new_tlw': new_tlw,
        'new_Ins': new_ins,
        'old_ins': old_ins,
        'old_tlw': old_tlw,
        'drag': drag,
        'tlw_total': tlw_total,
        'total_ins': total_ins
    })

units_totals_df = pd.DataFrame(units_totals_data)

print(f"✅ UNITS TOTALS DATAFRAME:")
print(units_totals_df.to_string(index=False))

print(f"\n✅ STEP 4 VALIDATION:")
print(f"   ✅ Total months: {len(units_totals_df)}")
print(f"   ✅ Final month new units: {units_totals_df.iloc[-1]['new']:,}")
print(f"   ✅ Final month old units: {units_totals_df.iloc[-1]['old']:,}")
print(f"   ✅ TLW Default Rate applied: {tlw_default_rate:.0%}")

print(f"\n{'='*60}")
print(f"✅ STEP 4 COMPLETE - Units totals dataframe created")
print(f"{'='*60}\n")

# ============================================================================
# STEP 5: TENANT COVERAGE TRANSITION CHART (TLW vs P&C)
# ============================================================================

print(f"\nSTEP 5: TENANT COVERAGE TRANSITION CHART")
print(f"{'='*60}\n")

# Create the stacked area chart
fig, ax = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(units_totals_df))

# Stack the areas (order matters - bottom to top)
ax.fill_between(x_pos, 0, units_totals_df['new_tlw'], 
                 label='New TLW', color='#4472C4', alpha=0.9)

ax.fill_between(x_pos, units_totals_df['new_tlw'], 
                 units_totals_df['new_tlw'] + units_totals_df['new_Ins'],
                 label='New P&C', color='#C5504B', alpha=0.9)

ax.fill_between(x_pos, 
                 units_totals_df['new_tlw'] + units_totals_df['new_Ins'],
                 units_totals_df['new_tlw'] + units_totals_df['new_Ins'] + units_totals_df['old_ins'],
                 label='Old P&C', color='#A9D08E', alpha=0.9)

ax.fill_between(x_pos, 
                 units_totals_df['new_tlw'] + units_totals_df['new_Ins'] + units_totals_df['old_ins'],
                 units_totals_df['new_tlw'] + units_totals_df['new_Ins'] + units_totals_df['old_ins'] + units_totals_df['old_tlw'],
                 label='Old TLW', color='#9E7CB5', alpha=0.9)

ax.fill_between(x_pos, 
                 units_totals_df['new_tlw'] + units_totals_df['new_Ins'] + units_totals_df['old_ins'] + units_totals_df['old_tlw'],
                 units_totals_df['new_tlw'] + units_totals_df['new_Ins'] + units_totals_df['old_ins'] + units_totals_df['old_tlw'] + units_totals_df['drag'],
                 label='Drag', color='#70ADC7', alpha=0.9)

# Formatting
ax.set_title('Tenant Coverage Transition 2026-2027\nTLW vs P&C', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Tenants', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(units_totals_df['Month'], rotation=45, ha='right')
ax.set_ylim(0, 7500)
ax.legend(loc='upper left', fontsize=11, frameon=True)
ax.grid(True, alpha=0.3, axis='y', zorder=0)

plt.tight_layout()

coverage_chart_filename = 'tenant_coverage_transition_2026_2027.png'
plt.savefig(coverage_chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"✅ Chart generated and saved as: {coverage_chart_filename}")

print(f"\n{'='*60}")
print(f"✅ STEP 5 COMPLETE - Coverage transition chart created")
print(f"{'='*60}\n")

# ============================================================================
# STEP 6: SIMPLIFIED COVERAGE CHART (TLW vs Insurance with Data Labels)
# ============================================================================

print(f"\nSTEP 6: SIMPLIFIED COVERAGE CHART WITH DATA LABELS")
print(f"{'='*60}\n")

# Create the simplified stacked area chart
fig, ax = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(units_totals_df))

# Stack the three areas (order matters - bottom to top)
ax.fill_between(x_pos, 0, units_totals_df['tlw_total'], 
                 label='TLW total', color='#4472C4', alpha=0.9)

ax.fill_between(x_pos, units_totals_df['tlw_total'], 
                 units_totals_df['tlw_total'] + units_totals_df['total_ins'],
                 label='P&C total', color='#C5504B', alpha=0.9)

ax.fill_between(x_pos, 
                 units_totals_df['tlw_total'] + units_totals_df['total_ins'],
                 units_totals_df['tlw_total'] + units_totals_df['total_ins'] + units_totals_df['drag'],
                 label='Drag', color='#A9D08E', alpha=0.9)

# Add data labels for each segment
for i in range(len(units_totals_df)):
    # TLW total label (middle of TLW segment)
    tlw_y = units_totals_df['tlw_total'].iloc[i] / 2
    ax.text(i, tlw_y, str(units_totals_df['tlw_total'].iloc[i]), 
            ha='center', va='center', fontweight='bold', fontsize=10, color='black')
    
    # P&C total label (middle of P&C segment)
    pc_y = units_totals_df['tlw_total'].iloc[i] + (units_totals_df['total_ins'].iloc[i] / 2)
    ax.text(i, pc_y, str(units_totals_df['total_ins'].iloc[i]), 
            ha='center', va='center', fontweight='bold', fontsize=10, color='black')
    
    # Drag label (middle of Drag segment)
    drag_y = units_totals_df['tlw_total'].iloc[i] + units_totals_df['total_ins'].iloc[i] + (units_totals_df['drag'].iloc[i] / 2)
    ax.text(i, drag_y, str(units_totals_df['drag'].iloc[i]), 
            ha='center', va='center', fontweight='bold', fontsize=10, color='black')

# Formatting
ax.set_title('Tenant Coverage Transition 2026-2027\nTLW vs Insurance', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('', fontsize=12, fontweight='bold')
ax.set_ylabel('', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(units_totals_df['Month'], rotation=0, ha='center')
ax.set_ylim(0, 7500)
ax.legend(loc='lower left', fontsize=11, frameon=True)
ax.grid(False)

plt.tight_layout()

simplified_chart_filename = 'tenant_coverage_tlw_insurance_simplified.png'
plt.savefig(simplified_chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"✅ Chart generated and saved as: {simplified_chart_filename}")

print(f"\n{'='*60}")
print(f"✅ STEP 6 COMPLETE - Simplified coverage chart with labels created")
print(f"{'='*60}\n")

# ============================================================================
# STEP 7: MONTHLY BUDGET DATAFRAME
# ============================================================================

print(f"\nSTEP 7: MONTHLY BUDGET CALCULATIONS")
print(f"{'='*60}\n")

# Create budget dataframe
budget_data = []

for i, row in units_totals_df.iterrows():
    month = row['Month']
    tlw_total = row['tlw_total']
    
    # Calculate budget columns (rates are already signed, e.g., -0.072 for -7.2%)
    gwp = tlw_total * tlw_price
    vac_drag = gwp * vac_drag_rate  # negative value
    net_ear_p = gwp + vac_drag  # add the negative vac_drag to get net
    exp_loss = gwp * exp_loss_rate  # negative value
    ibnr = gwp * ibnr_rate  # negative value
    claim_admin = gwp * tpa_rate  # negative value
    capt_op_cost = gwp * coc_rate  # negative value
    prem_tax = gwp * prem_tax_rate  # negative value
    comp_buff = gwp * buffer_rate  # negative value
    
    # Profit is the sum (note: costs are already negative values)
    profit = gwp + vac_drag + exp_loss + ibnr + claim_admin + capt_op_cost + prem_tax + comp_buff
    
    # Profit margin
    profit_mar = profit / gwp if gwp > 0 else 0
    
    budget_data.append({
        'Month': month,
        'gwp': gwp,
        'vac_drag': vac_drag,
        'net_ear_p': net_ear_p,
        'exp_loss': exp_loss,
        'ibnr': ibnr,
        'claim_admin': claim_admin,
        'capt_op_cost': capt_op_cost,
        'prem_tax': prem_tax,
        'comp_buff': comp_buff,
        'profit': profit,
        'profit_mar': profit_mar
    })

budget_df = pd.DataFrame(budget_data)

print(f"✅ MONTHLY BUDGET DATAFRAME:")
print(budget_df.to_string(index=False))

print(f"\n✅ STEP 7 VALIDATION:")
print(f"   ✅ Total months: {len(budget_df)}")
print(f"   ✅ Final month GWP: ${budget_df.iloc[-1]['gwp']:,.2f}")
print(f"   ✅ Final month profit: ${budget_df.iloc[-1]['profit']:,.2f}")
print(f"   ✅ Final month profit margin: {budget_df.iloc[-1]['profit_mar']:.2%}")

print(f"\n{'='*60}")
print(f"✅ STEP 7 COMPLETE - Monthly budget dataframe created")
print(f"{'='*60}\n")

# ============================================================================
# STEP 8: BUDGET SUMMARY TABLE GRAPHIC
# ============================================================================

print(f"\nSTEP 8: CREATING BUDGET SUMMARY TABLE GRAPHIC")
print(f"{'='*60}\n")

# Prepare data for table (transpose budget_df)
table_data = []

# Define row labels and corresponding column names
row_labels = [
    'Gross Written Premium',
    'Vacancy Drag',
    'Net Earned Premium',
    'Expected Losses',
    'Loss Development Lag',
    'Claims Administration',
    'Captive Operating Costs',
    'Premium Taxes',
    'Compliance/Regulatory Buff',
    'Profit'
]

column_keys = [
    'gwp',
    'vac_drag',
    'net_ear_p',
    'exp_loss',
    'ibnr',
    'claim_admin',
    'capt_op_cost',
    'prem_tax',
    'comp_buff',
    'profit'
]

# Build table data with percentage rates in first column
rate_values = [
    '100.0%',
    f'{vac_drag_rate*100:.1f}%',
    f'{(1 + vac_drag_rate)*100:.1f}%',
    f'{exp_loss_rate*100:.1f}%',
    f'{ibnr_rate*100:.1f}%',
    f'{tpa_rate*100:.1f}%',
    f'{coc_rate*100:.1f}%',
    f'{prem_tax_rate*100:.1f}%',
    f'{buffer_rate*100:.1f}%',
    f'{profit_alex:.2f}%'  # Profit margin from Alex scenario
]

for idx, (label, key, rate) in enumerate(zip(row_labels, column_keys, rate_values)):
    row = [label, rate]
    for month_data in budget_df.itertuples():
        value = getattr(month_data, key)
        row.append(f'${value:,.2f}')
    table_data.append(row)

# Add accumulated rows
# Calculate cumulative sums
cumulative_gwp = budget_df['gwp'].cumsum()
cumulative_profit = budget_df['profit'].cumsum()

# Add Accumulated GWP row
accum_gwp_row = ['Accumulated GWP', '']
for value in cumulative_gwp:
    accum_gwp_row.append(f'${value:,.2f}')
table_data.append(accum_gwp_row)

# Add Accumulated Profit row
accum_profit_row = ['Accumulated Profit', '']
for value in cumulative_profit:
    accum_profit_row.append(f'${value:,.2f}')
table_data.append(accum_profit_row)

# Prepare units split table data
units_split_labels = [
    'New on TLW',
    'New on Insurance',
    'Old Active Insurance',
    'Old on TLW',
    'Drag',
    'Total'
]

units_split_keys = [
    'new_tlw',
    'new_Ins',
    'old_ins',
    'old_tlw',
    'drag',
    None  # Total will be calculated
]

# Calculate drag rate (what remains from old after active and tlw)
drag_rate = 1.0 - active_rate - old_tlw_rate

units_split_rates = [
    f'{tlw_default_rate*100:.0f}%',
    f'{(1-tlw_default_rate)*100:.0f}%',
    f'{active_rate*100:.2f}%',
    f'{old_tlw_rate*100:.2f}%',
    f'{drag_rate*100:.2f}%',
    ''  # Total has no rate
]

units_split_data = []
for label, key, rate in zip(units_split_labels, units_split_keys, units_split_rates):
    row = [label, rate]
    if key is None:  # Total row
        for month_data in units_totals_df.itertuples():
            total = month_data.new_tlw + month_data.new_Ins + month_data.old_tlw + month_data.old_ins + month_data.drag
            row.append(f'{total:,.0f}')
    else:
        for month_data in units_totals_df.itertuples():
            value = getattr(month_data, key)
            row.append(f'{value:,.0f}')
    units_split_data.append(row)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# ===== TOP TABLE: UNITS SPLIT =====
ax1.axis('off')

units_col_labels = ['Unit Split', '%'] + list(units_totals_df['Month'])

units_table = ax1.table(cellText=units_split_data, colLabels=units_col_labels,
                        cellLoc='center', loc='center',
                        colWidths=[0.18, 0.04] + [0.0615]*12,
                        bbox=[0, 0, 1, 1])

units_table.auto_set_font_size(False)
units_table.set_fontsize(9)
units_table.scale(1, 2)

# Style the header row
for i in range(len(units_col_labels)):
    cell = units_table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style the data rows
for i in range(len(units_split_data)):
    cell = units_table[(i+1, 0)]
    cell.set_facecolor('#D9E2F3')
    cell.set_text_props(weight='bold', ha='left')
    
    cell = units_table[(i+1, 1)]
    cell.set_facecolor('#D9E2F3')
    cell.set_text_props(weight='bold')
    
    # Color Total row differently
    if i == len(units_split_data) - 1:  # Total row
        for j in range(len(units_col_labels)):
            cell = units_table[(i+1, j)]
            if j >= 2:  # Data columns only
                cell.set_facecolor('#A9D08E')
                cell.set_text_props(weight='bold')

# ===== BOTTOM TABLE: BUDGET P&L =====
ax2.axis('off')

# Column headers
col_labels = ['Monthly P&L', '%'] + list(budget_df['Month'])

# Create table
table = ax2.table(cellText=table_data, colLabels=col_labels,
                  cellLoc='center', loc='center',
                  colWidths=[0.18, 0.04] + [0.0615]*12,
                  bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style the header row
for i in range(len(col_labels)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style the data rows
for i in range(len(table_data)):
    # Row label (first column)
    cell = table[(i+1, 0)]
    cell.set_facecolor('#D9E2F3')
    cell.set_text_props(weight='bold', ha='left')
    
    # Rate column (second column)
    cell = table[(i+1, 1)]
    cell.set_facecolor('#D9E2F3')
    cell.set_text_props(weight='bold')
    
    # Color profit row differently
    if i == len(row_labels) - 1:  # Profit row only
        for j in range(len(col_labels)):
            cell = table[(i+1, j)]
            if j >= 2:  # Data columns only
                cell.set_facecolor('#F4B084')
                cell.set_text_props(weight='bold')
    
    # Color accumulated rows differently
    if i >= len(row_labels):  # Accumulated GWP and Accumulated Profit rows
        for j in range(len(col_labels)):
            cell = table[(i+1, j)]
            if j >= 2:  # Data columns only
                cell.set_facecolor('#A9D08E')
                cell.set_text_props(weight='bold')

plt.tight_layout()

budget_table_filename = 'budget_summary_table.png'
plt.savefig(budget_table_filename, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"✅ Budget summary table created and saved as: {budget_table_filename}")

print(f"\n{'='*60}")
print(f"✅ STEP 8 COMPLETE - Budget summary table graphic created")
print(f"{'='*60}\n")

# ============================================================================
# STEP 9: GWP & PROFIT CHART WITH DUAL AXES
# ============================================================================

print(f"\nSTEP 9: CREATING GWP & PROFIT CHART")
print(f"{'='*60}\n")

# Calculate accumulated values
cumulative_gwp = budget_df['gwp'].cumsum()
cumulative_profit = budget_df['profit'].cumsum()

# Create figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(months))

# Primary y-axis: Monthly GWP bar chart
ax1.bar(x_pos, budget_df['gwp'], color='#4472C4', alpha=0.8, label='Gross Written Premium')
ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
ax1.set_ylabel('Monthly GWP ($)', fontsize=12, fontweight='bold', color='#4472C4')
ax1.tick_params(axis='y', labelcolor='#4472C4')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(months, rotation=45, ha='right')

# Format primary y-axis with thousand separators
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))

# Add data labels on bars
for i, (value, month) in enumerate(zip(budget_df['gwp'], months)):
    ax1.text(i, value/2, f'${value:,.2f}', ha='center', va='center', 
             fontweight='bold', fontsize=8, color='white')
    # Month label on top of bar - closer to bar end
    ax1.text(i, value + 500, month, ha='center', va='bottom', 
             fontweight='bold', fontsize=7, color='black')

# Secondary y-axis: Accumulated GWP and Profit lines
ax2 = ax1.twinx()

# Plot accumulated GWP line
ax2.plot(x_pos, cumulative_gwp, color='#C5504B', linewidth=3, 
         marker='o', markersize=7, label='Gross Written Premium Acum.')

# Plot accumulated Profit line  
ax2.plot(x_pos, cumulative_profit, color='#A9D08E', linewidth=3, 
         marker='o', markersize=7, label='Profit Acum.')

ax2.set_ylabel('Accumulated ($)', fontsize=12, fontweight='bold', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Format secondary y-axis with thousand separators
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))

# Add data labels for accumulated GWP - positioned to avoid overlap
for i, value in enumerate(cumulative_gwp):
    # Alternate positioning slightly to reduce overlap
    offset = 15000 if i % 2 == 0 else 18000
    ax2.text(i, value + offset, f'${value:,.2f}', ha='center', va='bottom', 
             fontweight='bold', fontsize=7, color='#C5504B')

# Add data labels for accumulated Profit - positioned to avoid overlap
for i, value in enumerate(cumulative_profit):
    # Position below the line with some offset
    offset = 8000 if i % 2 == 0 else 5000
    ax2.text(i, value - offset, f'${value:,.2f}', ha='center', va='top', 
             fontweight='bold', fontsize=7, color='white')

# Title
plt.title('Gross Written Premium & Profit from TLW by Lease Contract Roll-over', 
          fontsize=14, fontweight='bold', pad=20)

# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# Grid
ax1.grid(True, alpha=0.3, axis='y')

# Adjust layout to prevent label cutoff
plt.tight_layout()

gwp_profit_chart_filename = 'gwp_profit_rollover_chart.png'
plt.savefig(gwp_profit_chart_filename, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"✅ GWP & Profit chart created and saved as: {gwp_profit_chart_filename}")

print(f"\n{'='*60}")
print(f"✅ STEP 9 COMPLETE - GWP & Profit chart created")
print(f"{'='*60}\n")
