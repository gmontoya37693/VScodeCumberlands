#!/usr/bin/env python3
"""
FLOORPLAN ANALYSIS - CLEAN START
Step-by-step processing of TLW outputs and Rent Roll data

Only loads the original Excel files and waits for step-by-step instructions.
"""

import pandas as pd
from pathlib import Path

def load_data_files():
    """
    Load the three main Excel files for processing:
    1. acento_apartments.xlsx (TLW output)
    2. tlw_rollout.xlsx (TLW rollout schedule)
    3. Rent Roll all Properties.xlsx (Property rent roll data)
    """
    
    print("=" * 60)
    print("LOADING DATA FILES")
    print("=" * 60)
    
    # File paths
    acento_file = "acento_apartments.xlsx"
    rollout_file = "tlw_rollout.xlsx"
    rentroll_file = "Rent Roll all Properties.xlsx"
    
    # Check all files exist
    files_to_check = [acento_file, rollout_file, rentroll_file]
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✅ {file_path} found")
        else:
            print(f"❌ {file_path} not found")
            return None, None, None
    
    try:
        # Load acento apartments data
        print(f"\nLoading {acento_file}...")
        acento_data = pd.read_excel(acento_file)
        print(f"  📊 Shape: {acento_data.shape}")
        print(f"  📋 Columns: {list(acento_data.columns)}")
        
        # Load rollout data
        print(f"\nLoading {rollout_file}...")
        rollout_data = pd.read_excel(rollout_file)
        print(f"  📊 Shape: {rollout_data.shape}")
        print(f"  📋 Columns: {list(rollout_data.columns)}")
        
        # Load rent roll Excel file (multi-tab)
        print(f"\nExamining {rentroll_file} structure...")
        excel_file = pd.ExcelFile(rentroll_file)
        print(f"  📑 Number of tabs: {len(excel_file.sheet_names)}")
        print(f"  📑 Tab names: {excel_file.sheet_names}")
        
        return acento_data, rollout_data, excel_file
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return None, None, None

def display_sample_data(acento_data, rollout_data, excel_file):
    """
    Display sample data from all loaded files for verification
    """
    
    print("\n" + "=" * 60)
    print("SAMPLE DATA PREVIEW")
    print("=" * 60)
    
    # Show acento sample
    print(f"\n🏢 ACENTO APARTMENTS (TLW Output):")
    print(acento_data.head())
    
    # Show rollout sample
    print(f"\n📅 TLW ROLLOUT SCHEDULE:")
    print(rollout_data.head())
    
    # Show rent roll sample (first tab)
    print(f"\n🏘️ RENT ROLL SAMPLE (First Tab):")
    first_tab = excel_file.sheet_names[0]
    print(f"  Tab: {first_tab}")
    sample_data = pd.read_excel(excel_file, sheet_name=first_tab).head()
    print(sample_data)

def step1_extract_rentroll_data():
    """
    Step 1: Extract rent roll data from all 30 property tabs
    
    Strategy:
    - Skip first 5 rows (headers/metadata) 
    - Use row 6 as column headers (Bldg/Unit, SQFT, etc.)
    - Extract rows 7+ with valid apartment data
    - Filter out footer rows (totals, summaries, notes)
    - Create Unit_ID, Key, Property, Unit, SQFT structure
    """
    
    print("\n" + "=" * 60)
    print("STEP 1: EXTRACTING RENT ROLL DATA")
    print("=" * 60)
    
    global excel_file
    
    if excel_file is None:
        print("❌ Rent roll Excel file not loaded")
        return None
    
    all_rentroll_data = []
    extraction_summary = []
    
    print(f"Processing {len(excel_file.sheet_names)} property tabs...")
    
    for i, property_code in enumerate(excel_file.sheet_names, 1):
        print(f"\n{i:2d}. Processing property: {property_code}")
        
        try:
            # Read the tab, skipping first 5 rows, using row 6 as header
            df_tab = pd.read_excel(excel_file, sheet_name=property_code, header=5)
            
            print(f"   📊 Raw data shape: {df_tab.shape}")
            print(f"   📋 Columns: {list(df_tab.columns)[:5]}...")  # Show first 5 columns
            
            # Find Bldg/Unit and SQFT columns
            bldg_unit_col = None
            sqft_col = None
            
            for col in df_tab.columns:
                col_str = str(col).lower()
                # Look for Bldg/Unit column (primary) or Unit column (fallback for LHC)
                if ('bldg' in col_str and 'unit' in col_str) or col_str == 'unit':
                    bldg_unit_col = col
                elif 'sqft' in col_str:
                    sqft_col = col
            
            if bldg_unit_col is None or sqft_col is None:
                print(f"   ⚠️  Could not find Bldg/Unit or SQFT columns")
                continue
                
            print(f"   📋 Unit column: '{bldg_unit_col}'")
            print(f"   📋 SQFT column: '{sqft_col}'")
            
            # Clean the data
            df_clean = df_tab.copy()
            
            # Remove rows with missing unit or sqft data
            df_clean = df_clean[df_clean[bldg_unit_col].notna() & df_clean[sqft_col].notna()]
            
            # Convert SQFT to numeric, filter out non-numeric entries (footers)
            df_clean['SQFT_Numeric'] = pd.to_numeric(df_clean[sqft_col], errors='coerce')
            df_clean = df_clean[df_clean['SQFT_Numeric'].notna()]
            
            # Filter out footer rows by content and SQFT range
            unit_str = df_clean[bldg_unit_col].astype(str)
            
            # Remove footer text patterns
            footer_patterns = [
                'total', 'summary', 'note:', 'this section', 'statistically', 
                'rent roll', 'detail', 'parameter', 'applicants', 'undoing', 'canceling'
            ]
            footer_mask = unit_str.str.lower().str.contains('|'.join(footer_patterns), na=False)
            df_clean = df_clean[~footer_mask]
            
            # Filter by reasonable SQFT range (200-3000 sq ft)
            df_clean = df_clean[(df_clean['SQFT_Numeric'] >= 200) & (df_clean['SQFT_Numeric'] <= 3000)]
            
            # Reset index
            df_clean = df_clean.reset_index(drop=True)
            
            if len(df_clean) == 0:
                print(f"   ⚠️  No valid apartment data found after filtering")
                continue
            
            # Create the target structure
            rentroll_data = pd.DataFrame({
                'Unit_ID': range(1, len(df_clean) + 1),
                'Key': property_code + "_" + df_clean[bldg_unit_col].astype(str).str.strip(),
                'Property': property_code,
                'Unit': df_clean[bldg_unit_col].astype(str).str.strip(),
                'SQFT': df_clean['SQFT_Numeric']
            })
            
            all_rentroll_data.append(rentroll_data)
            
            # Summary for this property
            extraction_summary.append({
                'Property': property_code,
                'Units': len(rentroll_data),
                'SQFT_Min': int(rentroll_data['SQFT'].min()),
                'SQFT_Max': int(rentroll_data['SQFT'].max()),
                'SQFT_Avg': int(rentroll_data['SQFT'].mean()),
                'Sample_Units': rentroll_data['Unit'].head(3).tolist()
            })
            
            print(f"   ✅ {len(rentroll_data)} apartment units extracted")
            print(f"   📊 SQFT range: {int(rentroll_data['SQFT'].min())}-{int(rentroll_data['SQFT'].max())}")
            print(f"   📋 Sample units: {', '.join(rentroll_data['Unit'].head(3))}")
            
        except Exception as e:
            print(f"   ❌ Error processing {property_code}: {e}")
            continue
    
    if not all_rentroll_data:
        print("\n❌ No data extracted from any property tabs")
        return None
    
    # Combine all property data
    combined_rentroll = pd.concat(all_rentroll_data, ignore_index=True)
    
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"{'Property':<8} {'Units':<6} {'SQFT Range':<12} {'Avg':<5} {'Sample Units'}")
    print("-" * 65)
    
    total_units = 0
    for summary in extraction_summary:
        total_units += summary['Units']
        sample_str = ', '.join(summary['Sample_Units'][:2])
        sqft_range = f"{summary['SQFT_Min']}-{summary['SQFT_Max']}"
        print(f"{summary['Property']:<8} {summary['Units']:<6} {sqft_range:<12} {summary['SQFT_Avg']:<5} {sample_str}")
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Properties processed: {len(extraction_summary)}")
    print(f"   Total apartment units: {len(combined_rentroll):,}")
    print(f"   Overall SQFT range: {int(combined_rentroll['SQFT'].min())}-{int(combined_rentroll['SQFT'].max())}")
    print(f"   Average SQFT: {int(combined_rentroll['SQFT'].mean())}")
    
    print(f"\n📋 Sample combined data:")
    print(combined_rentroll.head(10))
    
    # Save to Excel file
    output_filename = f"{len(combined_rentroll)}_rentroll_with_sqft.xlsx"
    combined_rentroll.to_excel(output_filename, index=False)
    
    print(f"\n💾 SAVED: {output_filename}")
    print(f"   📊 {len(combined_rentroll):,} apartment units with SQFT")
    print(f"   📋 Columns: {list(combined_rentroll.columns)}")
    print(f"   ✅ Ready for matching with TLW rollout data")
    
    return combined_rentroll

def main():
    """
    Main function - load data files and wait for step-by-step instructions
    """
    print("FLOORPLAN ANALYSIS - CLEAN START")
    print("Loading TLW outputs and Rent Roll data...")
    
    # Load all data files
    acento_data, rollout_data, excel_file = load_data_files()
    
    if all(data is not None for data in [acento_data, rollout_data, excel_file]):
        print("\n✅ All files loaded successfully!")
        
        # Display sample data for verification
        display_sample_data(acento_data, rollout_data, excel_file)
        
        print("\n" + "=" * 60)
        print("📋 DATA READY - AWAITING STEP-BY-STEP INSTRUCTIONS")
        print("=" * 60)
        print("Files loaded and ready for processing:")
        print(f"  • Acento apartments: {len(acento_data):,} records")
        print(f"  • TLW rollout: {len(rollout_data):,} records")  
        print(f"  • Rent roll tabs: {len(excel_file.sheet_names)} properties")
        
        # Store data globally for step-by-step processing
        globals()['acento_data'] = acento_data
        globals()['rollout_data'] = rollout_data
        globals()['excel_file'] = excel_file
        
        print(f"\n🎯 Ready for step-by-step instructions!")
        print(f"   Data is loaded in memory and ready for processing.")
        
        return True
    else:
        print("\n❌ Failed to load required files")
        return False

if __name__ == "__main__":
    main()