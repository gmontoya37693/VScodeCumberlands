#!/usr/bin/env python3
"""Analyze unit count differences between MODIVES and floorplan data"""

import pandas as pd
import tlw_budget_26 as tlw

def analyze_unit_differences():
    # Load original MODIVES data
    df_original, _, _ = tlw.main()
    original_counts = df_original.groupby('Property').size().sort_index()

    # Load our cleaned floorplan data  
    df_floorplan = pd.read_csv('floorplan.csv')
    floorplan_counts = df_floorplan.groupby('Property').size().sort_index()

    print('UNIT COUNT COMPARISON:')
    print('=' * 70)
    print(f'{"Property":<8} {"MODIVES":<10} {"Floorplan":<10} {"Difference":<12} {"%Change":<8}')
    print('-' * 70)

    total_original = 0
    total_floorplan = 0
    properties_with_increase = []

    for prop in sorted(set(original_counts.index) | set(floorplan_counts.index)):
        orig = original_counts.get(prop, 0)
        floor = floorplan_counts.get(prop, 0) 
        diff = floor - orig
        pct_change = (diff / orig * 100) if orig > 0 else 0
        
        total_original += orig
        total_floorplan += floor
        
        if diff > 0:
            properties_with_increase.append((prop, diff, pct_change))
        
        status = '+' if diff > 0 else (' ' if diff == 0 else '-')
        print(f'{prop:<8} {orig:<10} {floor:<10} {status}{abs(diff):<11} {pct_change:>6.1f}%')

    print('-' * 70)
    print(f'{"TOTALS":<8} {total_original:<10} {total_floorplan:<10} +{total_floorplan - total_original:<11}')

    print(f'\nSUMMARY:')
    print(f'Original MODIVES units: {total_original:,}')
    print(f'Current floorplan units: {total_floorplan:,}') 
    net_increase = total_floorplan - total_original
    print(f'Net increase: +{net_increase:,} units ({net_increase / total_original * 100:.1f}%)')

    print(f'\nBIGGEST INCREASES (Top 5):')
    for prop, increase, pct in sorted(properties_with_increase, key=lambda x: x[1], reverse=True)[:5]:
        print(f'{prop}: +{increase} units ({pct:.1f}% increase)')
    
    print(f'\nPOSSIBLE REASONS FOR +{net_increase:,} UNITS:')
    print('1. Rent roll has more recent/updated data than MODIVES')
    print('2. Properties expanded or added units since MODIVES data')
    print('3. Different counting methods (MODIVES vs rent roll)')
    print('4. Rent roll includes units not in original MODIVES scope')

if __name__ == "__main__":
    analyze_unit_differences()