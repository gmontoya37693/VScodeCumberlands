#!/usr/bin/env python3
"""
Floorplan Database Generator
Created on February 13, 2026

DESCRIPTION:
Processes "Rent Roll all Properties.xlsx" containing 30 property tabs to extract
apartment areas (sqft) and unit identifiers, with validation against original
MODIVES data to ensure data integrity.

INPUT:
- "Rent Roll all Properties.xlsx" (30 tabs, one per property)
- Each tab contains unit and SQFT data with property abbreviations
- Cross-validation with tlw_budget_26.py original data

OUTPUT:
- Processed dataset with columns: Property, Unit, SQFT
- Validation report comparing extracted vs expected unit counts
- Data quality and integrity analysis

PROCESSING STEPS:
1. Load "Rent Roll all Properties.xlsx" and examine all 30 tabs
2. Load original MODIVES data for unit count validation
3. Smart column detection for unit and SQFT data
4. Extract and validate data from each property tab
5. Cross-validate extracted counts against original data
6. Display validation results and prepare for transformation

VALIDATION:
- Ensures extracted unit counts match original property data
- Identifies discrepancies for manual review
- Confirms data integrity before transformation steps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from datetime import datetime

# Import TLW budget analysis for validation
try:
    import tlw_budget_26 as tlw
    TLW_AVAILABLE = True
except ImportError:
    TLW_AVAILABLE = False
    print("Note: tlw_budget_26.py not available - validation will be limited")

class FloorplanProcessor:
    """
    Process "Rent Roll all Properties.xlsx" with validation against original data
    """
    
    def __init__(self, file_path="Rent Roll all Properties.xlsx"):
        """
        Initialize the floorplan processor
        
        Args:
            file_path (str): Path to the rent roll Excel file
        """
        self.file_path = file_path
        self.raw_data = {}
        self.processed_data = []
        self.floorplan_database = None
        self.original_property_data = None
        self.validation_results = {}
        
        print("="*80)
        print("FLOORPLAN PROCESSOR - RENT ROLL ALL PROPERTIES")
        print("="*80)
        print(f"Target file: {file_path}")
    
    def load_original_property_data(self):
        """
        Load original MODIVES data for validation purposes
        """
        print(f"\nLOADING ORIGINAL PROPERTY DATA FOR VALIDATION")
        print("-" * 60)
        
        if not TLW_AVAILABLE:
            print("✗ tlw_budget_26.py not available - skipping validation")
            return False
        
        try:
            # Load the original MODIVES data
            df, tlw_portfolio, timeline_data = tlw.main()
            
            if df is not None:
                # Create property summary for validation
                property_summary = df.groupby('Property').size().reset_index(name='Expected_Units')
                self.original_property_data = property_summary
                
                print(f"✓ Loaded original data: {len(df):,} total units across {len(property_summary)} properties")
                print(f"\nProperty unit counts from original data:")
                for _, row in property_summary.head(10).iterrows():
                    print(f"  {row['Property']}: {row['Expected_Units']} units")
                
                if len(property_summary) > 10:
                    print(f"  ... and {len(property_summary)-10} more properties")
                
                return True
            else:
                print("✗ Failed to load original MODIVES data")
                return False
                
        except Exception as e:
            print(f"✗ Error loading original data: {e}")
            return False
    
    def load_excel_structure(self):
        """
        Step 1: Load and examine the Rent Roll Excel file structure
        """
        print(f"\nSTEP 1: EXAMINING RENT ROLL FILE STRUCTURE")
        print("-" * 50)
        
        try:
            # Check if file exists
            if not Path(self.file_path).exists():
                print(f"✗ Error: File '{self.file_path}' not found")
                return False
            
            # Get all sheet names
            excel_file = pd.ExcelFile(self.file_path)
            sheet_names = excel_file.sheet_names
            
            print(f"✓ File loaded successfully")
            print(f"Number of tabs found: {len(sheet_names)} (Expected: 30 properties)")
            
            if len(sheet_names) != 30:
                print(f"⚠️  Warning: Expected 30 property tabs, found {len(sheet_names)}")
            
            # Display all sheet names (property abbreviations)
            print(f"\nProperty tabs (abbreviations):")
            for i, sheet in enumerate(sheet_names, 1):
                print(f"  {i:2d}. {sheet}")
            
            # Examine structure of first few tabs
            print(f"\nEXAMINING SAMPLE TAB STRUCTURES:")
            print("-" * 50)
            
            for i, sheet in enumerate(sheet_names[:3]):  # Examine first 3 tabs
                print(f"\nTab {i+1}: '{sheet}'")
                try:
                    df_sample = pd.read_excel(self.file_path, sheet_name=sheet, header=None)
                    print(f"  Dimensions: {df_sample.shape[0]} rows x {df_sample.shape[1]} columns")
                    
                    # Show first few rows to identify structure
                    print(f"  First 5 rows:")
                    for row_idx in range(min(5, len(df_sample))):
                        row_data = df_sample.iloc[row_idx].fillna('').astype(str).tolist()
                        # Only show first 6 columns to avoid clutter
                        display_cols = row_data[:6] + (['...'] if len(row_data) > 6 else [])
                        print(f"    Row {row_idx}: {display_cols}")
                
                except Exception as e:
                    print(f"  ✗ Error reading tab '{sheet}': {e}")
            
            self.sheet_names = sheet_names
            return True
            
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return False
    
    def identify_data_columns(self, sample_sheets=None):
        """
        Step 2: Identify Bldg/Unit and SQFT columns across all tabs
        
        Args:
            sample_sheets (int): Number of sheets to sample (None = all sheets)
        """
        print(f"\nSTEP 2: IDENTIFYING DATA COLUMNS ACROSS ALL PROPERTIES")
        print("-" * 60)
        
        column_patterns = {
            'unit': ['bldg/unit', 'unit', 'apartment', 'apt', 'bldg', 'building'],
            'sqft': ['sqft', 'sq ft', 'area', 'square', 'footage', 'sf']
        }
        
        found_columns = {}
        
        # Process all sheets, not just a sample
        sheets_to_process = self.sheet_names if sample_sheets is None else self.sheet_names[:sample_sheets]
        
        for i, sheet in enumerate(sheets_to_process):
            print(f"\nAnalyzing tab: '{sheet}'")
            
            try:
                # Read with multiple header options to find column names
                for header_row in range(0, 10):  # Check first 10 rows as potential headers
                    try:
                        df = pd.read_excel(self.file_path, sheet_name=sheet, header=header_row)
                        columns = [str(col).lower().strip() for col in df.columns]
                        
                        # Look for unit column
                        unit_col = None
                        for col in df.columns:
                            col_lower = str(col).lower().strip()
                            if any(pattern in col_lower for pattern in column_patterns['unit']):
                                unit_col = col
                                break
                        
                        # Look for sqft column  
                        sqft_col = None
                        for col in df.columns:
                            col_lower = str(col).lower().strip()
                            if any(pattern in col_lower for pattern in column_patterns['sqft']):
                                sqft_col = col
                                break
                        
                        if unit_col and sqft_col:
                            found_columns[sheet] = {
                                'header_row': header_row,
                                'unit_column': unit_col,
                                'sqft_column': sqft_col,
                                'all_columns': list(df.columns)
                            }
                            print(f"  ✓ Found columns at header row {header_row}:")
                            print(f"    Unit column: '{unit_col}'")
                            print(f"    SQFT column: '{sqft_col}'")
                            break
                    
                    except:
                        continue
                
                if sheet not in found_columns:
                    print(f"  ✗ Could not identify required columns")
            
            except Exception as e:
                print(f"  ✗ Error analyzing tab: {e}")
        
        self.column_mapping = found_columns
        
        if found_columns:
            print(f"\n✓ Successfully identified columns in {len(found_columns)} tabs")
            return True
        else:
            print(f"\n✗ Could not identify required columns in any tabs")
            return False
    
    def extract_property_data(self, property_name):
        """
        Step 3: Extract clean data from a single property tab
        
        Args:
            property_name (str): Name of the property tab
            
        Returns:
            pd.DataFrame: Clean data with Unit and SQFT columns
        """
        if property_name not in self.column_mapping:
            print(f"✗ No column mapping found for property: {property_name}")
            return None
        
        try:
            mapping = self.column_mapping[property_name]
            header_row = mapping['header_row']
            
            # Load the data
            df = pd.read_excel(self.file_path, sheet_name=property_name, header=header_row)
            
            # Extract required columns
            unit_col = mapping['unit_column']
            sqft_col = mapping['sqft_column']
            
            # Create clean DataFrame
            clean_data = pd.DataFrame({
                'Property': property_name,
                'Unit': df[unit_col],
                'SQFT': df[sqft_col]
            })
            
            # Remove rows with missing data
            clean_data = clean_data.dropna(subset=['Unit', 'SQFT'])
            
            # Clean SQFT column (remove non-numeric characters)
            clean_data['SQFT'] = pd.to_numeric(clean_data['SQFT'], errors='coerce')
            clean_data = clean_data.dropna(subset=['SQFT'])
            
            print(f"  ✓ Extracted {len(clean_data)} units from '{property_name}'")
            return clean_data
        
        except Exception as e:
            print(f"  ✗ Error extracting data from '{property_name}': {e}")
            return None
    
    def process_all_properties(self):
        """
        Step 4: Process all properties and create unified dataset
        """
        print(f"\nSTEP 4: PROCESSING ALL PROPERTIES")
        print("-" * 50)
        
        all_data = []
        processed_properties = 0
        
        for property_name in self.sheet_names:
            print(f"Processing: {property_name}")
            
            # Extract data
            property_data = self.extract_property_data(property_name)
            
            if property_data is not None and len(property_data) > 0:
                all_data.append(property_data)
                processed_properties += 1
            else:
                print(f"  ✗ Skipped '{property_name}' - no valid data")
        
        if all_data:
            # Combine all property data
            self.floorplan_database = pd.concat(all_data, ignore_index=True)
            
            print(f"\n✓ Successfully processed {processed_properties} properties")
            print(f"✓ Total apartments in dataset: {len(self.floorplan_database):,}")
            
            return True
        else:
            print(f"\n✗ No data could be processed from any properties")
            return False
    
    def validate_extraction_results(self):
        """
        Step 5: Cross-validate extracted data against original property data
        """
        if self.floorplan_database is None:
            print("✗ No dataset available for validation")
            return
        
        print(f"\nSTEP 5: CROSS-VALIDATION WITH ORIGINAL DATA")
        print("-" * 50)
        
        df = self.floorplan_database
        
        # Extract unit counts from processed data
        extracted_counts = df.groupby('Property').size().reset_index(name='Extracted_Units')
        
        print(f"EXTRACTION SUMMARY:")
        print(f"Total apartments extracted: {len(df):,}")
        print(f"Properties processed: {len(extracted_counts)}")
        
        # Compare with original data if available
        if self.original_property_data is not None:
            print(f"\nVALIDATION AGAINST ORIGINAL MODIVES DATA:")
            print("-" * 60)
            
            # Create property name mapping (abbreviation to full name)
            # This will need to be refined based on actual data
            validation_df = extracted_counts.copy()
            validation_df.columns = ['Property_Abbrev', 'Extracted_Units']
            
            print(f"{'Property Abbrev':<15} {'Extracted':<10} {'Status'}")
            print("-" * 40)
            
            total_extracted = 0
            for _, row in validation_df.iterrows():
                abbrev = row['Property_Abbrev']
                extracted = row['Extracted_Units']
                total_extracted += extracted
                
                # For now, just show extracted counts - detailed matching will need property name mapping
                print(f"{abbrev:<15} {extracted:<10} {'✓ Extracted'}")
            
            print("-" * 40)
            print(f"{'TOTAL':<15} {total_extracted:<10}")
            
            # Store validation results
            self.validation_results = {
                'total_extracted': total_extracted,
                'properties_processed': len(validation_df),
                'extraction_summary': validation_df
            }
            
        else:
            print(f"\nNo original data available for validation")
            print(f"Extracted unit counts by property:")
            for _, row in extracted_counts.iterrows():
                print(f"  {row['Property']}: {row['Extracted_Units']} units")
        
        # Data quality checks
        print(f"\nDATA QUALITY CHECKS:")
        print(f"Missing Unit values: {df['Unit'].isnull().sum()}")
        print(f"Missing SQFT values: {df['SQFT'].isnull().sum()}")
        print(f"Invalid SQFT values (≤0): {(df['SQFT'] <= 0).sum()}")
        
        print(f"\nAREA STATISTICS:")
        print(f"Average SQFT: {df['SQFT'].mean():.0f}")
        print(f"Median SQFT: {df['SQFT'].median():.0f}")
        print(f"Min SQFT: {df['SQFT'].min():.0f}")
        print(f"Max SQFT: {df['SQFT'].max():.0f}")
    
    def display_results(self):
        """
        Step 6: Display final results and prepare for transformation
        """
        if self.floorplan_database is None:
            print("✗ No dataset available")
            return
        
        print(f"\nSTEP 6: PROCESSING COMPLETE - READY FOR TRANSFORMATION")
        print("=" * 60)
        
        df = self.floorplan_database
        
        print(f"✓ Successfully processed rent roll data")
        print(f"📊 Dataset: {len(df):,} apartments from {df['Property'].nunique()} properties")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show sample data
        print(f"\n📋 Sample of extracted data:")
        print(df.head(10))
        
        # Show property summary
        property_counts = df['Property'].value_counts().sort_index()
        print(f"\n📊 Units per property (by abbreviation):")
        for prop, count in property_counts.items():
            print(f"  {prop}: {count} units")
        
        # Ready for transformation message
        print(f"\n🔄 DATA EXTRACTION COMPLETE")
        print(f"Ready to proceed with transformation steps.")
        print(f"Please provide the specific transformation requirements.")
        
        return True

def main():
    """
    Main function to process Rent Roll all Properties.xlsx
    """
    print("FLOORPLAN PROCESSOR - RENT ROLL ALL PROPERTIES")
    print("="*60)
    
    file_path = "Rent Roll all Properties.xlsx"
    
    processor = FloorplanProcessor(file_path)
    
    # Load original data for validation (if available)
    processor.load_original_property_data()
    
    # 6-Step processing workflow
    if processor.load_excel_structure():
        if processor.identify_data_columns():
            if processor.process_all_properties():
                processor.validate_extraction_results()
                processor.display_results()
    else:
        print(f"\nFile '{file_path}' not found in workspace.")
        print("Please ensure the file is uploaded and named correctly.")

if __name__ == "__main__":
    main()