#!/usr/bin/env python3
"""Check for office units in the rent roll data"""

import pandas as pd
import tlw_budget_26 as tlw
from floorplan import FloorplanProcessor

def check_office_units():
    processor = FloorplanProcessor()
    processor.load_original_property_data()
    processor.load_excel_structure()
    processor.identify_data_columns()
    processor.process_all_properties()
    processor.clean_to_match_original()
    
    df = processor.floorplan_database_cleaned
    
    # Find all office units
    office_units = df[df['Unit'].str.contains('OFC|OFFICE|Office', case=False, na=False)]
    
    print('OFFICE UNITS FOUND:')
    print('='*50)
    if len(office_units) > 0:
        print(f'Total office units: {len(office_units)}')
        print()
        for _, unit in office_units.iterrows():
            sqft_status = 'INVALID' if unit['SQFT'] <= 0 else 'Valid'
            print(f'{unit["Property"]} - {unit["Unit"]} | SQFT: {unit["SQFT"]} ({sqft_status})')
    else:
        print('No office units found')
    
    # Check for other non-residential patterns
    other_patterns = ['STOR', 'COMM', 'LAUNDRY', 'MAINT', 'CLUB', 'POOL', 'GYM', 'GARAGE']
    other_units = df[df['Unit'].str.contains('|'.join(other_patterns), case=False, na=False)]
    
    print(f'\nOTHER NON-RESIDENTIAL UNITS: {len(other_units)}')
    if len(other_units) > 0:
        for _, unit in other_units.head(10).iterrows():
            print(f'{unit["Property"]} - {unit["Unit"]} | SQFT: {unit["SQFT"]}')
    
    return office_units, other_units

if __name__ == "__main__":
    check_office_units()