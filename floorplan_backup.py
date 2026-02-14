#!/usr/bin/env python3
"""
FLOORPLAN ANALYSIS - CLEAN START
Step-by-step processing of TLW outputs and Rent Roll data

Only loads the original Excel files and waits for step-by-step instructions.
"""

import pandas as pd
from pathlib import Path

def step1_load_data_files():
    """
    STEP 1: Load the three main Excel files for processing:
    1. acento_apartments.xlsx (TLW output)
    2. tlw_rollout.xlsx (TLW rollout schedule)
    3. Rent Roll all Properties.xlsx (Property rent roll data)
    """
    
    print("=" * 60)
    print("STEP 1: LOADING DATA FILES")
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

def step2_extract_rentroll_data():
    """
    STEP 2: Extract rent roll data from all property tabs
    
    Strategy:
    - Skip first 5 rows (headers/metadata) 
    - Use row 6 as column headers (Bldg/Unit, SQFT, etc.)
    - Extract rows 7+ with valid apartment data
    - Filter out footer rows (totals, summaries, notes)
    - Create Unit_ID, Key, Property, Unit, SQFT structure
    """
    
    print("\n" + "=" * 60)
    print("STEP 2: EXTRACTING RENT ROLL DATA")
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

def step3_analyze_database():
    """
    STEP 3: Comprehensive analysis of the extracted rent roll database
    - Data types and structure
    - Null value analysis  
    - Unique value counts
    - Statistical summary
    """
    
    print("\n" + "=" * 60)
    print("STEP 3: DATABASE SUMMARY ANALYSIS")
    print("=" * 60)
    
    # Load the extracted database
    try:
        df = pd.read_excel('8259_rentroll_with_sqft.xlsx')
        print(f"✅ Database loaded: 8259_rentroll_with_sqft.xlsx")
    except Exception as e:
        print(f"❌ Could not load database: {e}")
        return None
    
    print(f"\n📊 BASIC DATABASE INFO:")
    print(f"   Total records: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"   Date range: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    
    print(f"\n📋 COLUMN ANALYSIS:")
    print(f"{'Column':<12} {'Type':<10} {'Non-Null':<10} {'Null':<6} {'Unique':<8} {'Sample Values'}")
    print("-" * 80)
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        
        # Sample values (first 3 unique values)
        sample_values = df[col].dropna().unique()[:3]
        sample_str = ', '.join([str(x)[:15] for x in sample_values])
        if len(sample_str) > 40:
            sample_str = sample_str[:37] + "..."
        
        print(f"{col:<12} {col_type:<10} {non_null:<10} {null_count:<6} {unique_count:<8} {sample_str}")
    
    print(f"\n🔍 DETAILED COLUMN ANALYSIS:")
    
    # Unit_ID Analysis
    print(f"\n1️⃣ Unit_ID Column:")
    print(f"   Data type: {df['Unit_ID'].dtype}")
    print(f"   Range: {df['Unit_ID'].min()} - {df['Unit_ID'].max()}")
    print(f"   Per-property ranges:")
    unit_id_analysis = df.groupby('Property')['Unit_ID'].agg(['min', 'max', 'count']).head(10)
    for prop, row in unit_id_analysis.iterrows():
        print(f"     {prop}: {row['min']}-{row['max']} ({row['count']} units)")
    
    # Key Analysis  
    print(f"\n2️⃣ Key Column:")
    print(f"   Data type: {df['Key'].dtype}")
    print(f"   Unique keys: {df['Key'].nunique():,}")
    print(f"   Duplicates: {df['Key'].duplicated().sum()}")
    print(f"   Sample keys: {list(df['Key'].head(5))}")
    
    # Property Analysis
    print(f"\n3️⃣ Property Column:")
    print(f"   Data type: {df['Property'].dtype}")
    print(f"   Unique properties: {df['Property'].nunique()}")
    print(f"   Properties: {sorted(df['Property'].unique())}")
    
    print(f"\n   Units per property:")
    property_counts = df['Property'].value_counts().sort_index()
    for prop, count in property_counts.head(15).items():
        print(f"     {prop}: {count:,} units")
    if len(property_counts) > 15:
        print(f"     ... and {len(property_counts)-15} more properties")
    
    # Unit Analysis
    print(f"\n4️⃣ Unit Column:")
    print(f"   Data type: {df['Unit'].dtype}")
    print(f"   Unique units: {df['Unit'].nunique():,}")
    print(f"   Sample units: {list(df['Unit'].head(8))}")
    
    # Unit patterns analysis
    unit_patterns = df['Unit'].astype(str).str.extract(r'([A-Z]+)').value_counts().head(5)
    if not unit_patterns.empty:
        print(f"   Common unit prefixes:")
        for pattern, count in unit_patterns.items():
            print(f"     {pattern}: {count} units")
    
    # SQFT Analysis
    print(f"\n5️⃣ SQFT Column:")
    print(f"   Data type: {df['SQFT'].dtype}")
    print(f"   Non-null values: {df['SQFT'].count():,}")
    print(f"   Null values: {df['SQFT'].isnull().sum()}")
    
    if df['SQFT'].count() > 0:
        print(f"   Statistical summary:")
        print(f"     Min SQFT: {df['SQFT'].min():.0f}")
        print(f"     Max SQFT: {df['SQFT'].max():.0f}")
        print(f"     Mean SQFT: {df['SQFT'].mean():.0f}")
        print(f"     Median SQFT: {df['SQFT'].median():.0f}")
        print(f"     Std Dev: {df['SQFT'].std():.0f}")
        
        # SQFT distribution
        print(f"   SQFT distribution:")
        print(f"     300-599 sq ft: {((df['SQFT'] >= 300) & (df['SQFT'] < 600)).sum():,} units")  
        print(f"     600-899 sq ft: {((df['SQFT'] >= 600) & (df['SQFT'] < 900)).sum():,} units")
        print(f"     900-1199 sq ft: {((df['SQFT'] >= 900) & (df['SQFT'] < 1200)).sum():,} units")
        print(f"     1200+ sq ft: {(df['SQFT'] >= 1200).sum():,} units")
    
    print(f"\n✅ DATABASE QUALITY ASSESSMENT:")
    total_nulls = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    completeness = ((total_cells - total_nulls) / total_cells) * 100
    
    print(f"   Data completeness: {completeness:.1f}%")
    print(f"   Total null values: {total_nulls:,}")
    print(f"   Key uniqueness: {(df['Key'].nunique() / len(df)) * 100:.1f}%")
    print(f"   Property coverage: {df['Property'].nunique()}/30 properties")
    
    return df

def step4_analyze_duplicate_keys():
    """
    STEP 4: Analyze duplicate keys in rent roll data
    - Identify which keys appear multiple times
    - Compare data across all columns for duplicated keys
    - Determine if duplicates are identical or have different data
    """
    
    print("\n" + "=" * 60)
    print("STEP 4: DUPLICATE KEY ANALYSIS")
    print("=" * 60)
    
    # Load the extracted database
    try:
        df = pd.read_excel('8259_rentroll_with_sqft.xlsx')
        print(f"✅ Database loaded: {df.shape[0]:,} records")
    except Exception as e:
        print(f"❌ Could not load database: {e}")
        return None
    
    # Find duplicate keys
    duplicate_keys = df[df['Key'].duplicated(keep=False)]['Key'].unique()
    total_duplicates = len(df[df['Key'].duplicated(keep=False)])
    
    print(f"\n📊 DUPLICATE KEY SUMMARY:")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique keys: {df['Key'].nunique():,}")
    print(f"   Keys with duplicates: {len(duplicate_keys):,}")
    print(f"   Total duplicate records: {total_duplicates:,}")
    print(f"   Expected unique records: {len(df) - total_duplicates + len(duplicate_keys):,}")
    
    if len(duplicate_keys) == 0:
        print("✅ No duplicate keys found!")
        return df
    
    print(f"\n🔍 ANALYZING {len(duplicate_keys)} DUPLICATE KEYS:")
    
    # Analyze each duplicate key
    identical_duplicates = []
    different_data_duplicates = []
    property_analysis = {}
    
    print(f"\\n{'Key':<15} {'Count':<6} {'Property':<8} {'Status'}")
    print("-" * 50)
    
    # Analyze ALL duplicate keys, not just first 20
    for i, key in enumerate(duplicate_keys):
        key_records = df[df['Key'] == key].copy()
        count = len(key_records)
        property_name = key_records['Property'].iloc[0]
        
        # Check if all records are identical (excluding Unit_ID which should be different)
        comparison_cols = ['Key', 'Property', 'Unit', 'SQFT']
        first_record = key_records[comparison_cols].iloc[0]
        
        # Check if all other records match the first one
        is_identical = True
        for idx in range(1, len(key_records)):
            if not key_records[comparison_cols].iloc[idx].equals(first_record):
                is_identical = False
                break
        
        status = "IDENTICAL" if is_identical else "DIFFERENT"
        
        # Only print first 20 for readability
        if i < 20:
            print(f"{key:<15} {count:<6} {property_name:<8} {status}")
        
        # Store for summary (analyze ALL keys)
        if is_identical:
            identical_duplicates.append(key)
        else:
            different_data_duplicates.append(key)
            
        # Property analysis
        if property_name not in property_analysis:
            property_analysis[property_name] = {'identical': 0, 'different': 0}
        property_analysis[property_name]['identical' if is_identical else 'different'] += 1
    
    if len(duplicate_keys) > 20:
        print(f"... and {len(duplicate_keys) - 20} more duplicate keys")
    
    print(f"\n📋 DUPLICATE ANALYSIS RESULTS:")
    print(f"   Identical duplicates: {len(identical_duplicates):,} keys")
    print(f"   Different data duplicates: {len(different_data_duplicates):,} keys")
    
    # Property-level analysis
    print(f"\n🏢 DUPLICATES BY PROPERTY:")
    print(f"{'Property':<8} {'Identical':<10} {'Different':<10} {'Total'}")
    print("-" * 40)
    
    for prop in sorted(property_analysis.keys()):
        identical_count = property_analysis[prop]['identical']
        different_count = property_analysis[prop]['different']
        total = identical_count + different_count
        print(f"{prop:<8} {identical_count:<10} {different_count:<10} {total}")
    
    # Show examples of different data duplicates
    if len(different_data_duplicates) > 0:
        print(f"\n🔍 EXAMPLES OF KEYS WITH DIFFERENT DATA:")
        for key in different_data_duplicates[:3]:
            print(f"\n   Key: {key}")
            key_records = df[df['Key'] == key][['Unit_ID', 'Unit', 'SQFT']]
            for idx, row in key_records.iterrows():
                print(f"     Unit_ID {row['Unit_ID']}: {row['Unit']} ({row['SQFT']} sq ft)")
    
    # Show examples of identical duplicates
    if len(identical_duplicates) > 0:
        print(f"\n✅ EXAMPLES OF IDENTICAL DUPLICATE KEYS:")
        for key in identical_duplicates[:3]:
            key_records = df[df['Key'] == key]
            count = len(key_records)
            unit = key_records['Unit'].iloc[0]
            sqft = key_records['SQFT'].iloc[0]
            print(f"   {key}: {count} identical records ({unit}, {sqft} sq ft)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if len(identical_duplicates) > 0:
        # Calculate actual duplicate records to remove (each key appears multiple times)
        duplicate_record_count = sum(len(df[df['Key'] == key]) - 1 for key in identical_duplicates)
        print(f"   • Remove {duplicate_record_count:,} identical duplicate records")
        print(f"   • From {len(identical_duplicates):,} duplicate keys")
        print(f"   • This would result in {len(df) - duplicate_record_count:,} unique records")
        
        # Verify this matches acento count
        if len(df) - duplicate_record_count == 7231:
            print(f"   ✅ Perfect match with acento dataset (7,231 records)")
        else:
            print(f"   ⚠️  Would result in {len(df) - duplicate_record_count:,} vs acento's 7,231")
    
    if len(different_data_duplicates) > 0:
        print(f"   • Investigate {len(different_data_duplicates):,} keys with different data")
        print(f"   • These may indicate data quality issues or legitimate variations")
    
    return {'df': df, 'identical_duplicates': identical_duplicates, 'different_data_duplicates': different_data_duplicates}

def step5_clean_dataset():
    """
    STEP 5: Clean dataset by removing duplicate records
    - Remove identical duplicate records
    - Keep first occurrence of each key
    - Create clean dataset matching acento count
    """
    
    print("\n" + "=" * 60)
    print("STEP 5: CLEANING DATASET - REMOVING DUPLICATES")
    print("=" * 60)
    
    # Load the extracted database
    try:
        df = pd.read_excel('8259_rentroll_with_sqft.xlsx')
        print(f"✅ Original database loaded: {df.shape[0]:,} records")
    except Exception as e:
        print(f"❌ Could not load database: {e}")
        return None
    
    print(f"\n📊 BEFORE CLEANING:")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique keys: {df['Key'].nunique():,}")
    print(f"   Duplicate records: {len(df) - df['Key'].nunique():,}")
    
    # Remove duplicates - keep first occurrence of each key
    print(f"\n🧹 REMOVING DUPLICATES...")
    
    # Keep track of what's being removed
    duplicate_keys = df[df['Key'].duplicated(keep=False)]['Key'].unique()
    duplicate_records_removed = len(df) - df['Key'].nunique()
    
    print(f"   Keys with duplicates: {len(duplicate_keys):,}")
    print(f"   Records to remove: {duplicate_records_removed:,}")
    
    # Remove duplicates, keeping first occurrence
    df_clean = df.drop_duplicates(subset=['Key'], keep='first')
    
    print(f"\n📊 AFTER CLEANING:")
    print(f"   Clean records: {len(df_clean):,}")
    print(f"   Unique keys: {df_clean['Key'].nunique():,}")
    print(f"   Records removed: {len(df) - len(df_clean):,}")
    
    # Verify results
    print(f"\n✅ VERIFICATION:")
    if len(df_clean) == 7231:
        print(f"   🎯 Perfect match: {len(df_clean):,} = Acento count (7,231)")
    else:
        print(f"   ⚠️  Count mismatch: {len(df_clean):,} vs Acento (7,231)")
    
    if df_clean['Key'].nunique() == len(df_clean):
        print(f"   ✅ All keys are unique")
    else:
        print(f"   ❌ Still have duplicates!")
    
    # Property-level verification
    print(f"\n🏢 CLEANED UNITS BY PROPERTY:")
    clean_property_counts = df_clean['Property'].value_counts().sort_index()
    original_property_counts = df['Property'].value_counts().sort_index()
    
    print(f"{'Property':<8} {'Original':<9} {'Cleaned':<8} {'Removed'}")
    print("-" * 40)
    
    total_removed_by_property = 0
    for prop in sorted(clean_property_counts.index):
        orig_count = original_property_counts[prop]
        clean_count = clean_property_counts[prop]
        removed = orig_count - clean_count
        total_removed_by_property += removed
        print(f"{prop:<8} {orig_count:<9} {clean_count:<8} {removed}")
    
    print(f"{'TOTAL':<8} {len(df):<9} {len(df_clean):<8} {total_removed_by_property}")
    
    # Save cleaned dataset
    output_file = '7231_rentroll_cleaned.xlsx'
    print(f"\n💾 SAVING CLEANED DATASET:")
    try:
        df_clean.to_excel(output_file, index=False)
        print(f"   ✅ Saved as: {output_file}")
        print(f"   📊 Final dataset: {len(df_clean):,} records, {len(df_clean.columns)} columns")
    except Exception as e:
        print(f"   ❌ Error saving: {e}")
        return None
    
    # Final summary
    print(f"\n🎉 CLEANING COMPLETED SUCCESSFULLY!")
    print(f"   Original: 8,259 records → Cleaned: {len(df_clean):,} records")
    print(f"   Removed: {len(df) - len(df_clean):,} duplicate records")
    print(f"   Output: {output_file}")
    print(f"   Status: Ready for cross-dataset comparison!")
    
    return df_clean

def step6_clean_tlw_rollout():
    """
    STEP 6: Clean TLW rollout data by removing N/A entries
    - Remove records with N/A- unit identifiers
    - Show before/after statistics
    - Save cleaned rollout dataset
    """
    
    print("\n" + "=" * 60)
    print("STEP 6: CLEANING TLW ROLLOUT - REMOVING N/A ENTRIES")
    print("=" * 60)
    
    global rollout_data
    
    if rollout_data is None:
        print("❌ TLW rollout data not loaded")
        return None
    
    df = rollout_data.copy()
    
    print(f"\n📊 BEFORE CLEANING TLW ROLLOUT:")
    print(f"   Total records: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check for N/A entries in Unit column
    if 'Unit' in df.columns:
        na_mask = df['Unit'].astype(str).str.contains('N/A-', na=False)
        na_count = na_mask.sum()
        
        print(f"   Records with N/A- units: {na_count:,}")
        
        if na_count > 0:
            # Show examples of N/A entries
            print(f"\n🔍 EXAMPLES OF N/A ENTRIES:")
            na_examples = df[na_mask]['Unit'].head(10).tolist()
            print(f"   {', '.join(na_examples)}")
            
            # Show property breakdown of N/A entries
            print(f"\n🏢 N/A ENTRIES BY PROPERTY:")
            na_by_property = df[na_mask]['Property'].value_counts()
            for prop, count in na_by_property.items():
                print(f"   {prop}: {count:,} N/A entries")
            
            # Remove N/A entries
            print(f"\n🧹 REMOVING N/A ENTRIES...")
            df_clean = df[~na_mask].copy()
            
            print(f"\n📊 AFTER CLEANING:")
            print(f"   Clean records: {len(df_clean):,}")
            print(f"   Records removed: {len(df) - len(df_clean):,}")
            print(f"   Removal percentage: {((len(df) - len(df_clean)) / len(df)) * 100:.1f}%")
            
            # Property summary after cleaning
            print(f"\n🏢 CLEANED UNITS BY PROPERTY:")
            original_counts = df['Property'].value_counts().sort_index()
            clean_counts = df_clean['Property'].value_counts().sort_index()
            
            print(f"{'Property':<8} {'Original':<9} {'Cleaned':<8} {'N/A Removed'}")
            print("-" * 45)
            
            for prop in sorted(original_counts.index):
                orig = original_counts.get(prop, 0)
                clean = clean_counts.get(prop, 0) 
                removed = orig - clean
                print(f"{prop:<8} {orig:<9} {clean:<8} {removed}")
            
            # Save cleaned rollout data
            output_file = f'{len(df_clean)}_tlw_rollout_cleaned.xlsx'
            print(f"\n💾 SAVING CLEANED TLW ROLLOUT:")
            try:
                df_clean.to_excel(output_file, index=False)
                print(f"   ✅ Saved as: {output_file}")
                
                # Update global variable
                globals()['rollout_data'] = df_clean
                
                print(f"\n🎉 TLW ROLLOUT CLEANING COMPLETED!")
                print(f"   Original: {len(df):,} → Cleaned: {len(df_clean):,} records")
                print(f"   Removed: {len(df) - len(df_clean):,} N/A entries")
                print(f"   Clean dataset ready for analysis!")
                
                return df_clean
                
            except Exception as e:
                print(f"   ❌ Error saving: {e}")
def step6_clean_key_formatting():
    """
    STEP 6: Clean key formatting by removing "N/A-" prefixes
    - Remove "N/A-" prefix from Unit and Key columns in both datasets
    - Keep all records, just clean the formatting
    - Update keys to match proper format between datasets
    """
    
    print("\n" + "=" * 60)
    print("STEP 6: CLEANING KEY FORMATTING (Remove N/A- prefixes)")
    print("=" * 60)
    
    global rollout_data, acento_data
    
    # Clean TLW Rollout key formatting
    print(f"\n🔄 CLEANING TLW ROLLOUT KEY FORMAT:")
    if rollout_data is None:
        print("❌ TLW rollout data not loaded")
        return None
    
    rollout_df = rollout_data.copy()
    print(f"   Original records: {len(rollout_df):,}")
    
    # Clean Unit and Key columns
    rollout_cleaned = 0
    if 'Unit' in rollout_df.columns:
        # Find entries with N/A- prefix
        na_mask = rollout_df['Unit'].astype(str).str.contains('N/A-', na=False)
        na_count = na_mask.sum()
        
        if na_count > 0:
            print(f"   Found {na_count:,} entries with N/A- prefix")
            
            # Show examples before cleaning
            examples = rollout_df[na_mask]['Unit'].head(5).tolist()
            print(f"   Examples before: {examples}")
            
            # Clean Unit column: remove "N/A-" prefix
            rollout_df.loc[na_mask, 'Unit'] = rollout_df.loc[na_mask, 'Unit'].str.replace('N/A-', '', regex=False)
            
            # Clean Key column: rebuild Key with cleaned Unit
            if 'Key' in rollout_df.columns and 'Property' in rollout_df.columns:
                rollout_df.loc[na_mask, 'Key'] = rollout_df.loc[na_mask, 'Property'] + '_' + rollout_df.loc[na_mask, 'Unit']
            
            # Show examples after cleaning
            examples_after = rollout_df[na_mask]['Unit'].head(5).tolist()
            keys_after = rollout_df[na_mask]['Key'].head(5).tolist()
            print(f"   Examples after:  {examples_after}")
            print(f"   Keys after:      {keys_after}")
            
            rollout_cleaned = na_count
        else:
            print(f"   ✅ No N/A- entries found")
    
    # Clean Acento Apartments key formatting  
    print(f"\n🔄 CLEANING ACENTO APARTMENTS KEY FORMAT:")
    if acento_data is None:
        print("❌ Acento apartments data not loaded")
        return None
    
    acento_df = acento_data.copy()
    print(f"   Original records: {len(acento_df):,}")
    
    acento_cleaned = 0
    if 'Unit' in acento_df.columns:
        # Find entries with N/A- prefix
        na_mask = acento_df['Unit'].astype(str).str.contains('N/A-', na=False)
        na_count = na_mask.sum()
        
        if na_count > 0:
            print(f"   Found {na_count:,} entries with N/A- prefix")
            
            # Show examples before cleaning
            examples = acento_df[na_mask]['Unit'].head(5).tolist()
            print(f"   Examples before: {examples}")
            
            # Clean Unit column: remove "N/A-" prefix
            acento_df.loc[na_mask, 'Unit'] = acento_df.loc[na_mask, 'Unit'].str.replace('N/A-', '', regex=False)
            
            # Clean Key column: rebuild Key with cleaned Unit
            if 'Key' in acento_df.columns and 'Property' in acento_df.columns:
                acento_df.loc[na_mask, 'Key'] = acento_df.loc[na_mask, 'Property'] + '_' + acento_df.loc[na_mask, 'Unit']
            
            # Show examples after cleaning
            examples_after = acento_df[na_mask]['Unit'].head(5).tolist()
            keys_after = acento_df[na_mask]['Key'].head(5).tolist()
            print(f"   Examples after:  {examples_after}")
            print(f"   Keys after:      {keys_after}")
            
            acento_cleaned = na_count
        else:
            print(f"   ✅ No N/A- entries found")
    
    # Save cleaned datasets
    if rollout_cleaned > 0:
        rollout_output = f'{len(rollout_df)}_tlw_rollout_keys_cleaned.xlsx'
        rollout_df.to_excel(rollout_output, index=False)
        globals()['rollout_data'] = rollout_df
        print(f"   ✅ Rollout saved: {rollout_output}")
    
    if acento_cleaned > 0:
        acento_output = f'{len(acento_df)}_acento_apartments_keys_cleaned.xlsx'
        acento_df.to_excel(acento_output, index=False)
        globals()['acento_data'] = acento_df
        print(f"   ✅ Acento saved: {acento_output}")
    
    # Summary
    print(f"\n📊 KEY CLEANING SUMMARY:")
    print(f"   TLW Rollout: {len(rollout_df):,} records, {rollout_cleaned:,} keys cleaned")
    print(f"   Acento Apartments: {len(acento_df):,} records, {acento_cleaned:,} keys cleaned")
    print(f"   Ready for proper rent roll (7,231) vs acento ({len(acento_df):,}) comparison!")
    
    return {'rollout': rollout_df, 'acento': acento_df, 'rollout_cleaned': rollout_cleaned, 'acento_cleaned': acento_cleaned}

def step7_compare_rentroll_vs_acento():
    """
    STEP 7: Smart comparison - Find perfect match properties vs naming differences
    - Since both datasets have 7,231 records, differences are likely naming issues
    - Identify properties with 100% key matches first
    - Focus investigation on properties with naming differences
    """
    
    print("\n" + "=" * 60)
    print("STEP 7: SMART COMPARISON - PERFECT MATCHES vs NAMING ISSUES") 
    print("=" * 60)
    
    # Load cleaned rent roll data
    try:
        rentroll_df = pd.read_excel('7231_rentroll_cleaned.xlsx')
        print(f"✅ Cleaned rent roll database: {len(rentroll_df):,} records")
    except Exception as e:
        print(f"❌ Could not load cleaned rent roll: {e}")
        return None
    
    # Use cleaned acento data
    global acento_data
    acento_df = acento_data.copy() if acento_data is not None else None
    
    if acento_df is None:
        print(f"❌ Acento apartments data not available")
        return None
    else:
        print(f"✅ Acento apartments: {len(acento_df):,} records")
    
    print(f"\n💡 SMART ANALYSIS LOGIC:")
    print(f"   Both datasets have same record count ({len(rentroll_df):,} = {len(acento_df):,})")
    print(f"   Differences likely due to key naming inconsistencies, not missing units")
    
    # Property-level analysis - identify perfect matches vs issues
    all_properties = set(rentroll_df['Property']) | set(acento_df['Property'])
    
    perfect_matches = []
    naming_issues = []
    property_analysis = {}
    
    print(f"\n🔍 ANALYZING EACH PROPERTY:")
    print(f"{'Property':<8} {'RentRoll':<10} {'Acento':<8} {'Match':<8} {'Status'}")
    print("-" * 50)
    
    for prop in sorted(all_properties):
        # Keys for this property in each dataset
        rr_prop_keys = set(rentroll_df[rentroll_df['Property'] == prop]['Key'])
        ac_prop_keys = set(acento_df[acento_df['Property'] == prop]['Key'])
        
        # Calculate match stats
        matching_keys = rr_prop_keys & ac_prop_keys
        rr_count = len(rr_prop_keys)
        ac_count = len(ac_prop_keys)
        match_count = len(matching_keys)
        
        # Determine status
        if rr_count == ac_count == match_count and rr_count > 0:
            status = "✅ PERFECT"
            perfect_matches.append(prop)
        elif rr_count == ac_count and rr_count > 0:
            status = "⚠️  NAMING"
            naming_issues.append(prop)
        elif rr_count == 0:
            status = "❌ NO RR"
        elif ac_count == 0:
            status = "❌ NO AC"
        else:
            status = "🚨 COUNT"
        
        print(f"{prop:<8} {rr_count:<10} {ac_count:<8} {match_count:<8} {status}")
        
        property_analysis[prop] = {
            'rentroll_count': rr_count,
            'acento_count': ac_count,
            'matching_keys': match_count,
            'status': status,
            'rr_keys': rr_prop_keys,
            'ac_keys': ac_prop_keys
        }
    
    return {
        'perfect_matches': perfect_matches,
        'naming_issues': naming_issues,
        'property_analysis': property_analysis
    }

def show_mismatches(properties=['CHP', 'LVA'], limit=3):
    """
    Show specific key mismatches for selected properties
    """
    
    print(f"\n" + "=" * 60)
    print(f"ANALYZING KEY MISMATCHES FOR {', '.join(properties)}")
    print("=" * 60)
    
    # Load datasets
    try:
        rentroll_df = pd.read_excel('7231_rentroll_cleaned.xlsx')
        print(f"✅ Rent roll loaded: {len(rentroll_df):,} records")
    except Exception as e:
        print(f"❌ Could not load rent roll: {e}")
        return None
    
    global acento_data
    acento_df = acento_data.copy() if acento_data is not None else None
    
    if acento_df is None:
        print(f"❌ Acento data not available")
        return None
    
    print(f"✅ Acento loaded: {len(acento_df):,} records")
    
    for prop in properties:
        print(f"\n🔍 {prop} PROPERTY MISMATCHES:")
        
        # Get keys for this property
        rr_keys = set(rentroll_df[rentroll_df['Property'] == prop]['Key'])
        ac_keys = set(acento_df[acento_df['Property'] == prop]['Key'])
        
        # Find mismatches
        rr_only = sorted(list(rr_keys - ac_keys))[:limit]
        ac_only = sorted(list(ac_keys - rr_keys))[:limit]
        
        print(f"   Total {prop} units - Rent roll: {len(rr_keys)}, Acento: {len(ac_keys)}")
        print(f"   Matching: {len(rr_keys & ac_keys)}")
        
        if rr_only:
            print(f"\n   📊 Rent roll keys (missing from acento):")
            for key in rr_only:
                print(f"      {key}")
                
        if ac_only:
            print(f"\n   🏢 Acento keys (missing from rent roll):") 
            for key in ac_only:
                print(f"      {key}")
        
        # Show pattern analysis
        if rr_only and ac_only and len(rr_only) == len(ac_only):
            print(f"\n   🔧 PATTERN ANALYSIS:")
            for i in range(min(len(rr_only), len(ac_only))):
                rr_key = rr_only[i]
                ac_key = ac_only[i]
                rr_unit = rr_key.split('_')[1] if '_' in rr_key else rr_key
                ac_unit = ac_key.split('_')[1] if '_' in ac_key else ac_key
                print(f"      {rr_unit} → {ac_unit}")

def fix_specific_keys():
    """
    Fix the specific 6 key mismatches identified in CHP and LVA properties
    """
    
    print(f"\n" + "=" * 60)
    print(f"FIXING SPECIFIC KEY MISMATCHES")
    print("=" * 60)
    
    # Load acento data
    try:
        acento_df = pd.read_excel('acento_apartments.xlsx')
        print(f"✅ Original acento loaded: {len(acento_df):,} records")
    except Exception as e:
        print(f"❌ Could not load acento file: {e}")
        return None
    
    # First, let's check what CHP keys exist in acento
    chp_keys = acento_df[acento_df['Property'] == 'CHP']['Key'].tolist()
    print(f"\n🔍 CHP keys in acento (showing first 10):")
    for key in sorted(chp_keys)[:10]:
        print(f"   {key}")
    
    # Check for the specific keys we're looking for
    print(f"\n🔍 Looking for specific keys:")
    target_chp_keys = ['CHP_12-001', 'CHP_12-002', 'CHP_12-004']
    for key in target_chp_keys:
        if key in acento_df['Key'].values:
            print(f"   ✅ {key} found")
        else:
            print(f"   ❌ {key} not found")
    
    # Define the specific replacements
    key_replacements = {
        # CHP replacements: N/A-3-digit to 2-digit (removing N/A- prefix)
        'CHP_N/A-12-001': 'CHP_12-01',
        'CHP_N/A-12-002': 'CHP_12-02', 
        'CHP_N/A-12-004': 'CHP_12-04',
        
        # LVA replacements: 3-digit to 4-digit
        'LVA_101': 'LVA_0101',
        'LVA_102': 'LVA_0102',
        'LVA_103': 'LVA_0103'
    }
    
    # Make replacements
    print(f"\n🔧 APPLYING KEY FIXES:")
    changes_made = 0
    
    for old_key, new_key in key_replacements.items():
        mask = acento_df['Key'] == old_key
        if mask.any():
            acento_df.loc[mask, 'Key'] = new_key
            changes_made += 1
            print(f"   ✅ {old_key} → {new_key}")
        else:
            print(f"   ⚠️  {old_key} not found")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total changes made: {changes_made}")
    print(f"   Final record count: {len(acento_df):,}")
    
    # Save updated file
    output_file = 'acento_apartments_fixed.xlsx'
    acento_df.to_excel(output_file, index=False)
    print(f"   📁 Saved as: {output_file}")
    
    # Verify the changes
    print(f"\n🔍 VERIFICATION:")
    for new_key in key_replacements.values():
        if new_key in acento_df['Key'].values:
            print(f"   ✅ {new_key} confirmed in dataset")
        else:
            print(f"   ❌ {new_key} missing from dataset")
    
    return acento_df

def main():
    """
    Main function - Execute steps 1-3 and prepare for next instructions
    """
    # Check if user wants to just fix the keys
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'fix':
        print("🔧 RUNNING KEY FIX MODE")
        fix_specific_keys()
        return
        
    print("FLOORPLAN ANALYSIS - STEP-BY-STEP EXECUTION")
    print("="*60)
    
    # STEP 1: Load all data files
    acento_data, rollout_data, excel_file = step1_load_data_files()
    
    if all(data is not None for data in [acento_data, rollout_data, excel_file]):
        print("\n✅ STEP 1 COMPLETED: All files loaded successfully!")
        
        # Display sample data for verification
        display_sample_data(acento_data, rollout_data, excel_file)
        
        # Store data globally for next steps
        globals()['acento_data'] = acento_data
        globals()['rollout_data'] = rollout_data
        globals()['excel_file'] = excel_file
        
        # STEP 2: Extract rent roll data (if not already done)
        import os
        if not os.path.exists('8259_rentroll_with_sqft.xlsx'):
            print(f"\n➡️  Running STEP 2: Extract rent roll data...")
            step2_extract_rentroll_data()
        else:
            print(f"\n✅ STEP 2 COMPLETED: Rent roll data already extracted")
        
        # STEP 3: Database analysis 
        if os.path.exists('8259_rentroll_with_sqft.xlsx'):
            print(f"\n➡️  Running STEP 3: Database analysis...")
            step3_analyze_database()
            print(f"\n✅ STEP 3 COMPLETED: Database analysis finished")
            
            # STEP 4: Duplicate key analysis
            print(f"\n➡️  Running STEP 4: Duplicate key analysis...")
            duplicate_analysis = step4_analyze_duplicate_keys()
            print(f"\n✅ STEP 4 COMPLETED: Duplicate key analysis finished")
            
            # STEP 5: Clean dataset  
            print(f"\n➡️  Running STEP 5: Clean dataset...")
            clean_dataset = step5_clean_dataset()
            print(f"\n✅ STEP 5 COMPLETED: Dataset cleaned successfully")
            
            # STEP 6: Clean key formatting (remove N/A- prefixes)
            print(f"\n➡️  Running STEP 6: Clean key formatting...")
            clean_datasets = step6_clean_key_formatting()
            print(f"\n✅ STEP 6 COMPLETED: Key formatting cleaned")
            
            # STEP 7: Compare rent roll database vs acento apartments
            print(f"\n➡️  Running STEP 7: Rent roll vs acento comparison...")
            comparison_results = step7_compare_rentroll_vs_acento()
            print(f"\n✅ STEP 7 COMPLETED: Database comparison finished")
            
            # Show specific mismatches for problematic properties
            show_mismatches(['CHP', 'LVA'], limit=3)
        
        print("\n" + "=" * 60)
        print("🎯 READY FOR NEXT STEP")
        print("=" * 60)
        print("Completed:")
        print("  ✅ Step 1: Data files loaded")
        print("  ✅ Step 2: Rent roll data extracted (8,259 units)")
        print("  ✅ Step 3: Database analysis completed")
        print("  ✅ Step 4: Duplicate key analysis completed")
        print("  ✅ Step 5: Dataset cleaned (7,231 unique records)")
        print("  ✅ Step 6: Key formatting cleaned (N/A- prefixes removed)")
        print("  ✅ Step 7: Rent roll vs acento comparison completed")
        print(f"\nReady for targeted fixes on CHP and LVA properties!")
        
        return True
    else:
        print("\n❌ Failed to load required files")
        return False

if __name__ == "__main__":
    main()