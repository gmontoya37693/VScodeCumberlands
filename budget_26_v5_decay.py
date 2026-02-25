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

print(f"📋 PARAMETER SET: SUGGESTED vs ALEX SCENARIO")
print(f"   (Waterfall % of GWP = 100%)")
print(f"{'-'*70}")
print(params_df.to_string(index=False))
print(f"{'-'*70}")

# Add profit calculation row
print(f"\n{'PROFIT %':<35} {profit_suggested:>12.2f}%  {profit_alex:>12.2f}%")
print(f"{'-'*70}")

# Validation
print(f"\n📊 PARAMETER VALIDATION:")
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

print(f"\n📊 BUDGET RATES (Alex Scenario):")
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
print(f"\n📊 Properties DataFrame:")
print(properties_df.to_string(index=False))

# Add totals row
print("\n" + "="*50)
print(f"{'TOTALS':<35} {properties_df['exp_count'].sum():>6}")
print("="*50)

# Validation checks
total_properties = len(properties_df)
total_units = properties_df['exp_count'].sum()

print(f"\n📋 STEP 1 VALIDATION:")
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
print(f"\n📊 Sample rows (first 3 properties, showing Apr-26 and Dec-26):")

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
print(f"\n📊 ROLLOVER SUMMARY:")
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

print(f"📊 UNITS TOTALS DATAFRAME:")
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
# STEP 6: SIMPLIFIED COVERAGE CHART (TLW vs HO4 with Data Labels)
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
ax.set_title('Tenant Coverage Transition 2026-2027\nTLW vs HO4', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('', fontsize=12, fontweight='bold')
ax.set_ylabel('', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(units_totals_df['Month'], rotation=0, ha='center')
ax.set_ylim(0, 7500)
ax.legend(loc='lower left', fontsize=11, frameon=True)
ax.grid(False)

plt.tight_layout()

simplified_chart_filename = 'tenant_coverage_tlw_ho4_simplified.png'
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

print(f"📊 MONTHLY BUDGET DATAFRAME:")
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
    'Compliance/Regulatory Buffer',
    'Profit',
    'Profit Margin'
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
    'profit',
    'profit_mar'
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
    '',  # Profit doesn't have a fixed rate
    ''   # Profit margin is calculated
]

for idx, (label, key, rate) in enumerate(zip(row_labels, column_keys, rate_values)):
    row = [label, rate]
    for month_data in budget_df.itertuples():
        value = getattr(month_data, key)
        # Format profit margin as percentage
        if key == 'profit_mar':
            row.append(f'{value*100:.2f}%')
        else:
            row.append(f'${value:,.0f}')
    table_data.append(row)

# Create figure
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('off')

# Column headers
col_labels = ['Monthly', ''] + list(budget_df['Month'])

# Create table
table = ax.table(cellText=table_data, colLabels=col_labels,
                cellLoc='right', loc='center',
                colWidths=[0.15, 0.04] + [0.065]*12)

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
    
    # Color profit rows differently
    if i >= len(table_data) - 2:  # Profit and Profit Margin rows
        for j in range(len(col_labels)):
            cell = table[(i+1, j)]
            if j >= 2:  # Data columns only
                cell.set_facecolor('#F4B084')
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
# STEP 9: ACCUMULATED BUDGET SUMMARY TABLE GRAPHIC
# ============================================================================

print(f"\nSTEP 9: CREATING ACCUMULATED BUDGET SUMMARY TABLE")
print(f"{'='*60}\n")

# Create cumulative dataframes
units_accum_df = units_totals_df.copy()
budget_accum_df = budget_df.copy()

# Calculate cumulative sums for relevant columns
for col in ['new', 'old', 'new_tlw', 'new_Ins', 'old_ins', 'old_tlw', 'drag', 'tlw_total', 'total_ins']:
    units_accum_df[col] = units_accum_df[col].cumsum()

for col in ['gwp', 'vac_drag', 'net_ear_p', 'exp_loss', 'ibnr', 'claim_admin', 
            'capt_op_cost', 'prem_tax', 'comp_buff', 'profit']:
    budget_accum_df[col] = budget_accum_df[col].cumsum()

# Recalculate profit margin for cumulative
budget_accum_df['profit_mar'] = budget_accum_df['profit'] / budget_accum_df['gwp']

# Calculate ratios for units table
units_accum_df['tlw_ratio'] = (units_accum_df['tlw_total'] / 
                                (units_accum_df['tlw_total'] + units_accum_df['total_ins'] + units_accum_df['drag']))

# Create figure with more height for two tables
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# ===== TABLE 1: ACCUMULATED PROPERTY SPLIT =====
ax1.axis('off')

# Prepare property split data
prop_split_labels = [
    'Unit Count Total Properties',
    'New Contracts',
    'Old Contracts'
]

prop_split_data = []

# Row 1: Total Units
row = ['Unit Count Total Properties', '']
for month_data in units_accum_df.itertuples():
    total = month_data.tlw_total + month_data.total_ins + month_data.drag
    row.append(f'{total:,.0f}')
prop_split_data.append(row)

# Row 2: New Contracts
row = ['New Contracts', '']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.new:,.0f}')
prop_split_data.append(row)

# Row 3: Old Contracts  
row = ['Old Contracts', '']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.old:,.0f}')
prop_split_data.append(row)

# Add blank row
prop_split_data.append(['', ''] + ['']*12)

# Detailed breakdown rows
row = ['New Contracts on TLW', '70%']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.new_tlw:,.0f}')
prop_split_data.append(row)

row = ['New Contracts on HO4', '30%']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.new_Ins:,.0f}')
prop_split_data.append(row)

# Add blank row
prop_split_data.append(['', ''] + ['']*12)

row = ['Old Contracts with Active HO4', f'{active_rate*100:.2f}%']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.old_ins:,.0f}')
prop_split_data.append(row)

row = ['Old Contracts on TLW', f'{old_tlw_rate*100:.2f}%']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.old_tlw:,.0f}')
prop_split_data.append(row)

row = ['Drag', '']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.drag:,.0f}')
prop_split_data.append(row)

# Add blank rows
prop_split_data.append(['', ''] + ['']*12)
prop_split_data.append(['', ''] + ['']*12)

# Summary rows
row = ['TLW Total', '']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.tlw_total:,.0f}')
prop_split_data.append(row)

row = ['HO4 Total', '']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.total_ins:,.0f}')
prop_split_data.append(row)

row = ['Drag', '']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.drag:,.0f}')
prop_split_data.append(row)

# Add blank row
prop_split_data.append(['', ''] + ['']*12)

# TLW total ratio
row = ['TLW-total Ratio', '']
for month_data in units_accum_df.itertuples():
    row.append(f'{month_data.tlw_ratio*100:.2f}%')
prop_split_data.append(row)

col_labels1 = ['Acum. Property Split', ''] + list(budget_accum_df['Month'])

table1 = ax1.table(cellText=prop_split_data, colLabels=col_labels1,
                   cellLoc='right', loc='center',
                   colWidths=[0.15, 0.04] + [0.065]*12)

table1.auto_set_font_size(False)
table1.set_fontsize(8)
table1.scale(1, 1.5)

# Style header
for i in range(len(col_labels1)):
    cell = table1[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(len(prop_split_data)):
    cell = table1[(i+1, 0)]
    cell.set_facecolor('#D9E2F3')
    cell.set_text_props(weight='bold', ha='left')
    cell = table1[(i+1, 1)]
    cell.set_facecolor('#D9E2F3')
    cell.set_text_props(weight='bold')

# ===== TABLE 2: ACCUMULATED P&L =====
ax2.axis('off')

pnl_data = []
for idx, (label, key, rate) in enumerate(zip(row_labels, column_keys, rate_values)):
    row = [label, rate]
    for month_data in budget_accum_df.itertuples():
        value = getattr(month_data, key)
        if key == 'profit_mar':
            row.append(f'{value*100:.2f}%')
        else:
            row.append(f'${value:,.0f}')
    pnl_data.append(row)

col_labels2 = ['Acum P&L', ''] + list(budget_accum_df['Month'])

table2 = ax2.table(cellText=pnl_data, colLabels=col_labels2,
                   cellLoc='right', loc='center',
                   colWidths=[0.15, 0.04] + [0.065]*12)

table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1, 1.5)

# Style header
for i in range(len(col_labels2)):
    cell = table2[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(len(pnl_data)):
    cell = table2[(i+1, 0)]
    cell.set_facecolor('#D9E2F3')
    cell.set_text_props(weight='bold', ha='left')
    cell = table2[(i+1, 1)]
    cell.set_facecolor('#D9E2F3')
    cell.set_text_props(weight='bold')
    
    # Highlight profit rows
    if i >= len(pnl_data) - 2:
        for j in range(len(col_labels2)):
            cell = table2[(i+1, j)]
            if j >= 2:
                cell.set_facecolor('#F4B084')
                cell.set_text_props(weight='bold')

plt.tight_layout()

accum_table_filename = 'budget_accumulated_summary.png'
plt.savefig(accum_table_filename, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"✅ Accumulated budget summary created and saved as: {accum_table_filename}")

print(f"\n{'='*60}")
print(f"✅ STEP 9 COMPLETE - Accumulated budget summary table created")
print(f"{'='*60}\n")
