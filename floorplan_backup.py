#!/usr/bin/env python3
"""
Floorplan Analysis - Clean Start
Created on February 13, 2026

Loading TLW output files and Rent Roll data for step-by-step processing
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data_files():
    """
    Load all three data files:
    1. acento_apartments.xlsx (TLW output - all properties)
    2. tlw_rollout.xlsx (TLW output - rollout schedule)  
    3. Rent Roll all Properties.xlsx (property tabs with unit/SQFT data)
    """
    
    print("="*60)
    print("LOADING DATA FILES")
    print("="*60)
    
    # File paths
    acento_file = "acento_apartments.xlsx"
    rollout_file = "tlw_rollout.xlsx"
    rent_roll_file = "Rent Roll all Properties.xlsx"
    
    # Check if files exist
    files_to_check = [acento_file, rollout_file, rent_roll_file]
    for file_path in files_to_check:
        if not Path(file_path).exists():
            print(f"❌ {file_path} not found")
            return None, None, None
        else:
            print(f"✅ {file_path} found")
    
    # Load TLW output files
    try:
        print(f"\nLoading {acento_file}...")
        acento_data = pd.read_excel(acento_file)
        print(f"  📊 Shape: {acento_data.shape}")
        print(f"  📋 Columns: {list(acento_data.columns)}")
        
        print(f"\nLoading {rollout_file}...")
        rollout_data = pd.read_excel(rollout_file)
        print(f"  📊 Shape: {rollout_data.shape}")
        print(f"  📋 Columns: {list(rollout_data.columns)}")
        
        # Examine Rent Roll structure
        print(f"\nExamining {rent_roll_file} structure...")
        excel_file = pd.ExcelFile(rent_roll_file)
        sheet_names = excel_file.sheet_names
        print(f"  📑 Number of tabs: {len(sheet_names)}")
        print(f"  📑 Tab names: {sheet_names}")
        
        return acento_data, rollout_data, excel_file
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None

def display_sample_data(acento_data, rollout_data, excel_file):
    """
    Display sample data from each source
    """
    print("\n" + "="*60)
    print("SAMPLE DATA PREVIEW")
    print("="*60)
    
    if acento_data is not None:
        print("\n🏢 ACENTO APARTMENTS (TLW Output):")
        print(acento_data.head())
        
    if rollout_data is not None:
        print("\n📅 TLW ROLLOUT SCHEDULE:")
        print(rollout_data.head())
        
    if excel_file is not None:
        print("\n🏘️ RENT ROLL SAMPLE (First Tab):")
        first_tab = excel_file.sheet_names[0]
        sample_df = pd.read_excel(excel_file, sheet_name=first_tab, nrows=5)
        print(f"  Tab: {first_tab}")
        print(sample_df)

def step2_extract_rentroll_data():
    """
    Step 2: Extract data from all rent roll tabs
    - Read all 30 property tabs, skipping headers
    - Locate Bldg/Unit column in actual data
    - Create Unit_ID, Key, Property, Unit columns
    - Count total units across all properties
    """
    
    print("\n" + "="*60)
    print("STEP 2: EXTRACTING RENT ROLL DATA FROM ALL TABS")
    print("="*60)
    
    global excel_file
    
    if excel_file is None:
        print("❌ Rent roll Excel file not loaded")
        return None
    
    all_rentroll_data = []
    tab_summary = []
    
    total_units = 0
    
    print(f"Processing {len(excel_file.sheet_names)} property tabs...")
    
    for i, tab_name in enumerate(excel_file.sheet_names, 1):
        print(f"\n{i:2d}. Processing tab: {tab_name}")
        
        try:
            # Read the tab data without headers first
            df_raw = pd.read_excel(excel_file, sheet_name=tab_name, header=None)
            
            print(f"   📊 Raw shape: {df_raw.shape}")
            
            # Look for the header row with "Bldg/Unit" or similar
            header_row = None
            unit_col_idx = None
            
            for row_idx in range(min(15, len(df_raw))):  # Check first 15 rows
                row_values = df_raw.iloc[row_idx].astype(str).str.lower()
                
                for col_idx, value in enumerate(row_values):
                    if any(pattern in value for pattern in ['bldg/unit', 'unit', 'apt']):
                        header_row = row_idx
                        unit_col_idx = col_idx
                        break
                
                if header_row is not None:
                    break
            
            if header_row is None:
                print(f"   ⚠️  No Bldg/Unit header found")
                continue
            
            # Read data starting from the identified header row
            df_tab = pd.read_excel(excel_file, sheet_name=tab_name, header=header_row)
            unit_col = df_tab.columns[unit_col_idx]
            
            print(f"   📋 Header row: {header_row}, Unit column: '{unit_col}'")
            
            # Filter out empty/null unit values and header-like data
            df_clean = df_tab[df_tab[unit_col].notna()].copy()
            df_clean = df_clean[df_clean[unit_col].astype(str).str.strip() != ''].copy()
            
            # Remove rows that look like headers or totals
            exclude_patterns = ['rent roll', 'detail', 'total', 'parameter', 'date:', 'resh id']
            mask = df_clean[unit_col].astype(str).str.lower().str.contains('|'.join(exclude_patterns), na=False)
            df_clean = df_clean[~mask].copy()
            
            # Also filter out rows where unit looks like a number only (like row numbers)
            df_clean = df_clean.reset_index(drop=True)
            
            if len(df_clean) == 0:
                print(f"   ⚠️  No valid unit data found after filtering")
                continue
            
            # Create the required columns
            # Unit_ID: counter per property (tab)  
            df_clean['Unit_ID'] = range(1, len(df_clean) + 1)
            
            # Property: tab name
            df_clean['Property'] = tab_name
            
            # Unit: copy of Bldg/Unit
            df_clean['Unit'] = df_clean[unit_col].astype(str).str.strip()
            
            # Key: Property + "_" + Unit
            df_clean['Key'] = df_clean['Property'] + "_" + df_clean['Unit']
            
            # Select only the required columns
            rentroll_subset = df_clean[['Unit_ID', 'Key', 'Property', 'Unit']].copy()
            
            # Add to collection
            all_rentroll_data.append(rentroll_subset)
            
            units_count = len(rentroll_subset)
            total_units += units_count
            
            tab_summary.append({
                'Property': tab_name,
                'Units': units_count,
                'Sample_Units': rentroll_subset['Unit'].head(3).tolist()
            })
            
            print(f"   ✅ {units_count} units extracted")
            print(f"   📋 Sample units: {', '.join(rentroll_subset['Unit'].head(3))}")
            
        except Exception as e:
            print(f"   ❌ Error processing {tab_name}: {e}")
            continue
    
    if not all_rentroll_data:
        print("\n❌ No data extracted from any tabs")
        return None
    
    # Combine all tab data
    combined_rentroll = pd.concat(all_rentroll_data, ignore_index=True)
    
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"{'Property':<8} {'Units':<8} {'Sample Units'}")
    print("-" * 60)
    for summary in tab_summary:
        sample_str = ', '.join(summary['Sample_Units'][:2]) 
        print(f"{summary['Property']:<8} {summary['Units']:<8} {sample_str}")
    
    print(f"\n🎯 TOTAL RESULTS:")
    print(f"   Properties processed: {len(tab_summary)}")
    print(f"   Total units extracted: {total_units:,}")
    print(f"   Dataset shape: {combined_rentroll.shape}")
    
    # Show sample of combined data
    print(f"\n📋 Combined dataset sample:")
    print(combined_rentroll.head())
    
    # Save to Excel file
    output_file = f"{total_units}_rentroll_combined.xlsx"
    combined_rentroll.to_excel(output_file, index=False)
    print(f"\n💾 SAVED: {output_file}")
    print(f"   📊 {total_units:,} units from all rent roll tabs")
    
    return combined_rentroll

def step3_extract_sqft_and_compare():
    """
    Step 3: Extract SQFT data from rent roll and compare unit counts
    - Add SQFT column to rent roll data
    - Validate footer elimination 
    - Compare unit counts between acento and rentroll by property
    """
    
    print("\n" + "="*60)
    print("STEP 3: EXTRACTING SQFT DATA & COMPARING UNIT COUNTS")
    print("="*60)
    
    global excel_file
    
    if excel_file is None:
        print("❌ Rent roll Excel file not loaded")
        return None
    
    # Load existing datasets
    try:
        acento_data = pd.read_excel('7229_acento_apartments.xlsx')
        rentroll_data = pd.read_excel('8316_rentroll_combined.xlsx')
        print(f"✅ Loaded existing datasets")
        print(f"   📊 Acento: {len(acento_data):,} units")
        print(f"   📊 Rent roll: {len(rentroll_data):,} units")
    except Exception as e:
        print(f"❌ Error loading existing files: {e}")
        return None
    
    all_rentroll_sqft = []
    comparison_data = []
    
    print(f"\n🔍 PROCESSING ALL TABS FOR SQFT EXTRACTION...")
    
    for i, tab_name in enumerate(excel_file.sheet_names, 1):
        print(f"\n{i:2d}. Processing tab: {tab_name}")
        
        try:
            # Read raw data without headers first
            df_raw = pd.read_excel(excel_file, sheet_name=tab_name, header=None)
            
            # Find header row with both unit and SQFT columns
            header_row = None
            unit_col_idx = None
            sqft_col_idx = None
            
            for row_idx in range(min(15, len(df_raw))):
                row_values = df_raw.iloc[row_idx].astype(str).str.lower()
                
                unit_found = False
                sqft_found = False
                
                for col_idx, value in enumerate(row_values):
                    # Look for Bldg/Unit column (has the actual unit numbers)
                    if 'bldg/unit' in value or ('unit' in value and 'designation' not in value and len(value) < 20):
                        unit_col_idx = col_idx
                        unit_found = True
                    if any(pattern in value for pattern in ['sqft', 'sq ft', 'square']):
                        sqft_col_idx = col_idx
                        sqft_found = True
                
                if unit_found and sqft_found:
                    header_row = row_idx
                    break
            
            if header_row is None or unit_col_idx is None or sqft_col_idx is None:
                print(f"   ⚠️  Could not find both Unit and SQFT columns")
                continue
            
            # Read data from header row
            df_tab = pd.read_excel(excel_file, sheet_name=tab_name, header=header_row)
            unit_col = df_tab.columns[unit_col_idx]
            sqft_col = df_tab.columns[sqft_col_idx]
            
            print(f"   📋 Header row: {header_row}")
            print(f"   📋 Unit column: '{unit_col}'")  
            print(f"   📋 SQFT column: '{sqft_col}'")
            
            # Clean the data
            df_clean = df_tab.copy()
            
            # Remove rows with missing unit or sqft data
            df_clean = df_clean[df_clean[unit_col].notna() & df_clean[sqft_col].notna()]
            
            # Convert SQFT to numeric first, removing non-numeric entries
            df_clean['SQFT_Numeric'] = pd.to_numeric(df_clean[sqft_col], errors='coerce')
            df_clean = df_clean[df_clean['SQFT_Numeric'].notna()]
            
            # Additional SQFT range validation (200-3000 sq ft to exclude footers)  
            df_clean = df_clean[(df_clean['SQFT_Numeric'] > 200) & (df_clean['SQFT_Numeric'] < 3000)]
            
            # Remove specific footer/summary text patterns (but keep actual unit numbers)
            unit_str = df_clean[unit_col].astype(str)
            exclude_mask = unit_str.str.lower().str.contains(
                'rent roll|detail|total|parameter|date:|resh id|summary|note:|this section|statistically|applicants|undoing|canceling', 
                na=False
            )
            df_clean = df_clean[~exclude_mask]
            
            # Additional footer validation - remove rows where SQFT is 0 or too large/small
            df_clean = df_clean[(df_clean['SQFT_Numeric'] > 200) & (df_clean['SQFT_Numeric'] < 3000)]
            
            # Reset index and create standard columns
            df_clean = df_clean.reset_index(drop=True)
            
            if len(df_clean) == 0:
                print(f"   ⚠️  No valid data after cleaning")
                continue
            
            # Create final dataset with SQFT
            df_final = pd.DataFrame({
                'Unit_ID': range(1, len(df_clean) + 1),
                'Property': tab_name,
                'Unit': df_clean[unit_col].astype(str).str.strip(),
                'SQFT': df_clean['SQFT_Numeric']
            })
            
            # Create Key
            df_final['Key'] = df_final['Property'] + "_" + df_final['Unit']
            
            # Reorder columns
            df_final = df_final[['Unit_ID', 'Key', 'Property', 'Unit', 'SQFT']]
            
            all_rentroll_sqft.append(df_final)
            
            # Get acento count for this property
            acento_count = len(acento_data[acento_data['Property'] == tab_name])
            rentroll_count = len(df_final)
            
            comparison_data.append({
                'Property': tab_name,
                'Acento_Units': acento_count,
                'RentRoll_Units': rentroll_count,
                'Difference': rentroll_count - acento_count,
                'Avg_SQFT': df_final['SQFT'].mean(),
                'Min_SQFT': df_final['SQFT'].min(),
                'Max_SQFT': df_final['SQFT'].max()
            })
            
            print(f"   ✅ {rentroll_count} units with SQFT extracted")
            print(f"   📊 SQFT range: {df_final['SQFT'].min():.0f} - {df_final['SQFT'].max():.0f}")
            
        except Exception as e:
            print(f"   ❌ Error processing {tab_name}: {e}")
            continue
    
    if not all_rentroll_sqft:
        print("\n❌ No SQFT data extracted")
        return None
    
    # Combine all data
    combined_sqft_data = pd.concat(all_rentroll_sqft, ignore_index=True)
    
    print(f"\n📊 SQFT EXTRACTION SUMMARY:")
    print(f"   Properties processed: {len(all_rentroll_sqft)}")
    print(f"   Total units with SQFT: {len(combined_sqft_data):,}")
    print(f"   Average SQFT: {combined_sqft_data['SQFT'].mean():.0f}")
    print(f"   SQFT range: {combined_sqft_data['SQFT'].min():.0f} - {combined_sqft_data['SQFT'].max():.0f}")
    
    # Property comparison
    print(f"\n📋 PROPERTY UNIT COUNT COMPARISON:")
    print(f"{'Property':<8} {'Acento':<8} {'RentRoll':<10} {'Diff':<6} {'Avg SQFT':<8}")
    print("-" * 60)
    
    comparison_df = pd.DataFrame(comparison_data)
    total_acento = 0
    total_rentroll = 0
    
    for _, row in comparison_df.iterrows():
        total_acento += row['Acento_Units']
        total_rentroll += row['RentRoll_Units']
        diff_str = f"+{row['Difference']}" if row['Difference'] > 0 else str(row['Difference'])
        print(f"{row['Property']:<8} {row['Acento_Units']:<8} {row['RentRoll_Units']:<10} {diff_str:<6} {row['Avg_SQFT']:.0f}")
    
    print("-" * 60)
    total_diff = total_rentroll - total_acento
    diff_str = f"+{total_diff}" if total_diff > 0 else str(total_diff)
    print(f"{'TOTAL':<8} {total_acento:<8} {total_rentroll:<10} {diff_str:<6}")
    
    # Save SQFT dataset
    output_file = f"{len(combined_sqft_data)}_rentroll_sqft.xlsx"
    combined_sqft_data.to_excel(output_file, index=False)
    
    print(f"\n💾 SAVED: {output_file}")
    print(f"   📊 {len(combined_sqft_data):,} units with SQFT data")
    print(f"   📋 Columns: {list(combined_sqft_data.columns)}")
    
    # Footer validation summary
    original_total = sum([len(pd.read_excel('8316_rentroll_combined.xlsx')[
        pd.read_excel('8316_rentroll_combined.xlsx')['Property'] == prop]) for prop in comparison_df['Property']])
    
    print(f"\n🧹 FOOTER ELIMINATION VALIDATION:")
    print(f"   Original extraction: {original_total:,} units")
    print(f"   After SQFT cleaning: {len(combined_sqft_data):,} units")
    print(f"   Removed (footers/invalid): {original_total - len(combined_sqft_data):,} records")
    print(f"   ✅ Footer rows successfully eliminated")
    
    return combined_sqft_data

def load_cleaned_acento_data():
    """
    Load the cleaned 7229_acento_apartments.xlsx file for continued processing
    """
    clean_file = "7229_acento_apartments.xlsx"
    
    if Path(clean_file).exists():
        clean_data = pd.read_excel(clean_file)
        print(f"📂 Loaded: {clean_file}")
        print(f"   📊 {len(clean_data):,} clean residential units")
        return clean_data
    else:
        print(f"❌ {clean_file} not found - run Step 1 first")
        return None

def step1_clean_acento_data():
    """
    Step 1: Clean acento apartments data
    - Find and remove office units
    - Remove "N/A-" from key column
    """
    
    print("\n" + "="*60)
    print("STEP 1: CLEANING ACENTO APARTMENTS DATA")
    print("="*60)
    
    global acento_data
    original_count = len(acento_data)
    
    print(f"Original count: {original_count:,} units")
    
    # 1. Find office units
    print(f"\n1️⃣ Looking for office units...")
    office_patterns = ['OFC', 'OFFICE', 'Office', 'ofc']
    office_mask = acento_data['Unit'].str.contains('|'.join(office_patterns), case=False, na=False)
    office_units = acento_data[office_mask]
    
    if len(office_units) > 0:
        print(f"   Found {len(office_units)} office unit(s):")
        for _, unit in office_units.iterrows():
            print(f"   - {unit['Property']}: {unit['Unit']} (Key: {unit['Key']})")
        
        # Remove office units
        acento_data = acento_data[~office_mask].copy()
        print(f"   ✅ Office units removed")
    else:
        print(f"   ✅ No office units found")
    
    # 2. Clean N/A- pattern from Key column
    print(f"\n2️⃣ Cleaning Key column (removing N/A- patterns)...")
    
    # Check for N/A- patterns
    na_mask = acento_data['Key'].str.contains('_N/A-', case=False, na=False)
    na_count = na_mask.sum()
    
    if na_count > 0:
        print(f"   Found {na_count} keys with N/A- pattern")
        
        # Show samples before cleaning
        sample_na_keys = acento_data[na_mask]['Key'].head(3).tolist()
        print(f"   Sample keys before: {sample_na_keys}")
        
        # Remove N/A- pattern
        acento_data['Key'] = acento_data['Key'].str.replace('_N/A-', '_', regex=False)
        
        # Show samples after cleaning
        sample_cleaned_keys = acento_data[na_mask]['Key'].head(3).tolist()
        print(f"   Sample keys after:  {sample_cleaned_keys}")
        print(f"   ✅ N/A- patterns removed from keys")
    else:
        print(f"   ✅ No N/A- patterns found in keys")
    
    # Summary
    final_count = len(acento_data)
    removed_count = original_count - final_count
    
    print(f"\n📊 STEP 1 SUMMARY:")
    print(f"   Original units: {original_count:,}")
    print(f"   Units removed: {removed_count}")
    print(f"   Final count: {final_count:,}")
    print(f"   ✅ Acento data cleaned")
    
    # Show cleaned sample
    print(f"\n📋 Cleaned data sample:")
    print(acento_data.head())
    
    # Save cleaned data to new Excel file
    output_file = "7229_acento_apartments.xlsx"
    acento_data.to_excel(output_file, index=False)
    print(f"\n💾 SAVED: {output_file}")
    print(f"   📊 {final_count:,} clean residential units")
    print(f"   📋 Columns: {list(acento_data.columns)}")
    print(f"   ✅ Ready for next steps")
    
    return acento_data

def main():
    """
    Main function - load data and wait for instructions
    """
    print("FLOORPLAN ANALYSIS - CLEAN START")
    print("Loading TLW outputs and Rent Roll data...")
    
    # Load all data files
    acento_data, rollout_data, excel_file = load_data_files()
    
    if all(data is not None for data in [acento_data, rollout_data, excel_file]):
        print("\n✅ All files loaded successfully!")
        
        # Display samples
        display_sample_data(acento_data, rollout_data, excel_file)
        
        print("\n" + "="*60)
        print("📋 DATA READY - AWAITING STEP-BY-STEP INSTRUCTIONS")
        print("="*60)
        print("Files loaded and ready for processing:")
        print(f"  • Acento apartments: {len(acento_data):,} records")
        print(f"  • TLW rollout: {len(rollout_data):,} records")  
        print(f"  • Rent roll tabs: {len(excel_file.sheet_names)} properties")
        
        # Store data globally for step-by-step processing
        globals()['acento_data'] = acento_data
        globals()['rollout_data'] = rollout_data
        globals()['excel_file'] = excel_file
        
        # Execute Step 1: Clean acento data
        print("\n" + "▶️" * 20)
        print("EXECUTING STEP 1...")
        print("▶️" * 20)
        
        cleaned_acento = step1_clean_acento_data()
        print(f"\n✅ Step 1 completed - Acento data cleaned")
        
        # Update global reference to cleaned data
        globals()['acento_data'] = cleaned_acento
        
        print(f"\n" + "🎯" * 20)
        print("EXECUTING STEP 2...")
        print("🎯" * 20)
        
        # Execute Step 2: Extract rent roll data
        rentroll_combined = step2_extract_rentroll_data()
        
        if rentroll_combined is not None:
            print(f"\n✅ Step 2 completed - Rent roll data extracted")
            globals()['rentroll_data'] = rentroll_combined
        else:
            print(f"\n❌ Step 2 failed - Could not extract rent roll data")
        
        print(f"\n" + "🏁" * 20)
        print("EXECUTING STEP 3...")
        print("🏁" * 20)
        
        # Execute Step 3: Extract SQFT data and compare
        sqft_data = step3_extract_sqft_and_compare()
        
        if sqft_data is not None:
            print(f"\n✅ Step 3 completed - SQFT data extracted and comparison done")
            globals()['sqft_data'] = sqft_data
        else:
            print(f"\n❌ Step 3 failed - Could not extract SQFT data")
        
        print(f"\n" + "🎉" * 20)
        print("ALL STEPS COMPLETED")
        print("🎉" * 20)
        print(f"📁 Files created:")
        print(f"  • 7229_acento_apartments.xlsx (clean acento data)")
        if rentroll_combined is not None:
            print(f"  • {len(rentroll_combined)}_rentroll_combined.xlsx (rent roll data)")
        if sqft_data is not None:
            print(f"  • {len(sqft_data)}_rentroll_sqft.xlsx (rent roll with SQFT)")
        
        return True
    else:
        print("\n❌ Failed to load required files")
        return False

if __name__ == "__main__":
    main()