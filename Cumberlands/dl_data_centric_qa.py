#!/usr/bin/env python3
"""
2026 Spring - Deep Learning (MSDS-534-M40)
Extra Credit Challenge: Data-Centric AI & Quality Assurance

Professor: Dr. Intisar Rizwan I. Haque
Student: German Montoya

Objective:
Build a data-centric QA workflow for 30-day readmission prediction by:
1) Integrating demographics (CSV) and outcomes (JSON),
2) Auditing and fixing three feature corruptions,
3) Auditing target labels for contradictions and class imbalance.

Inputs:
- patient_demographics.csv  (patient demographics from legacy system A)
- medical_records.json      (medical outcomes from legacy system B)

Planned Outputs (update at completion):
- Unified merged DataFrame
- Three documented feature-level corruption fixes
- Contradictory label rows removed
- Class imbalance percentages and written interpretation
- Final cleaned dataset shape summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: LOAD RAW DATA
# ------------------------------------------------------------
# We load both source files into separate DataFrames and keep
# raw copies so we can compare before/after at each step.
# The demographics come from a CSV export and the medical
# outcomes come from a JSON export of a different legacy system.
# ============================================================

demographics_raw = pd.read_csv("patient_demographics.csv")
records_raw      = pd.read_json("medical_records.json")

print("=== STEP 1: Raw Data Loaded ===")
print(f"\nDemographics shape : {demographics_raw.shape}")
print(f"Columns            : {list(demographics_raw.columns)}")
print(demographics_raw.head(3))

print(f"\nMedical records shape : {records_raw.shape}")
print(f"Columns               : {list(records_raw.columns)}")
print(records_raw.head(3))

# FINDING: Demographics has 3 columns — patient_id, age, weight.
# Medical records has 3 columns — patient_id, resting_bp, readmitted.
# Both share patient_id as the natural join key.


# ============================================================
# STEP 2: DATA INTEGRATION — MERGE ON patient_id
# ------------------------------------------------------------
# Before merging we validate the join key in both sources:
#   - Check for nulls in the key
#   - Check for duplicate keys (one-to-one vs one-to-many)
#   - Check for unmatched keys between the two sources
# We use an inner join so only patients with records in BOTH
# systems are included — this gives us complete, supervised rows.
# ============================================================

key = "patient_id"

print("\n=== STEP 2: Key Validation Before Merge ===")
print(f"Demographics  | nulls in key   : {demographics_raw[key].isna().sum()}")
print(f"Demographics  | duplicate keys : {demographics_raw[key].duplicated().sum()}")
print(f"Medical recs  | nulls in key   : {records_raw[key].isna().sum()}")
print(f"Medical recs  | duplicate keys : {records_raw[key].duplicated().sum()}")

unmatched_in_demo = set(demographics_raw[key]) - set(records_raw[key])
unmatched_in_recs = set(records_raw[key]) - set(demographics_raw[key])
print(f"\nKeys in demographics with no medical record : {len(unmatched_in_demo)}")
print(f"Keys in medical records with no demographic : {len(unmatched_in_recs)}")

# FINDING: records_raw has duplicate patient_id entries (e.g. PID_1042 appears
# twice with conflicting readmitted labels). This will be fixed in Step 4.
# For now we merge all rows including duplicates and address them after merge.

merged_df = demographics_raw.merge(records_raw, on=key, how="inner")
df = merged_df.copy()  # working copy — all fixes applied to df

print(f"\nShape before merge : demographics {demographics_raw.shape}, records {records_raw.shape}")
print(f"Shape after inner join : {df.shape}")

# FINDING: Inner join keeps only patients present in both systems.
# Any unmatched keys are excluded and logged above for transparency.


# ============================================================
# STEP 3: INITIAL FEATURE AUDIT — Run Descriptive Statistics
# ------------------------------------------------------------
# Before fixing anything, we profile all numeric columns to
# surface anomalies visually and statistically.
# We use .describe(), .info(), missing value counts, and
# histograms — the standard first-look toolkit.
# ============================================================

print("\n=== STEP 3: Initial Feature Audit ===")
print("\n--- .info() ---")
df.info()

print("\n--- .describe() ---")
print(df.describe())

print("\n--- Missing values per column ---")
print(df.isna().sum())

# Plot raw distributions before any cleaning
df[["age", "weight", "resting_bp"]].hist(bins=40, figsize=(14, 4), color="steelblue")
plt.suptitle("Step 3: Raw Distributions (Before Any Fixes)")
plt.tight_layout()
plt.savefig("step3_raw_distributions.png")
plt.show()

# FINDING (age): max age is 999 — a classic legacy database placeholder
#   for missing data. No human lives to age 999. This is Bug #1.
# FINDING (weight): most values cluster 40–115 kg but there is a
#   separate cluster above 130 reaching ~243. Those outliers match
#   common human weights in POUNDS (lb), not kilograms.
#   Normal adult weights: 40–150 kg. In lbs: 88–330 lb.
#   Values like 178, 213, 225 make sense as lbs but not kg. This is Bug #2.
# FINDING (resting_bp): contains negative values (e.g. -120 mmHg).
#   A resting blood pressure cannot be negative — this defies physiology.
#   This is Bug #3.


# ============================================================
# STEP 3a: FIX BUG #1 — Placeholder in `age`
# ------------------------------------------------------------
# The value 999 is used as a sentinel for "unknown age" in the
# legacy CSV export. It is numerically impossible as a human age.
# Fix: replace 999 with NaN so the column is properly treated as
# missing data rather than a valid extreme value.
# ============================================================

print("\n=== STEP 3a: Fix Bug #1 — Placeholder in age ===")
placeholder_mask = df["age"] == 999
print(f"Rows with age == 999 (placeholder): {placeholder_mask.sum()}")
print(f"  Sample PIDs: {df.loc[placeholder_mask, key].values[:5]}")

df.loc[placeholder_mask, "age"] = np.nan

print(f"After fix: NaN count in age = {df['age'].isna().sum()}")
print(df["age"].describe())

# FINDING: X rows had age=999 and are now NaN.
# The age distribution now reflects a realistic hospital patient
# population (roughly 20–100 years old).


# ============================================================
# STEP 3b: FIX BUG #2 — Unit Mismatch in `weight`
# ------------------------------------------------------------
# The weight column has a bimodal distribution. Values above ~130
# do not belong to the kg cluster — they are weights recorded in
# pounds (lbs) by a subset of staff using the imperial system.
# Evidence: 178.2, 213.7, 225.9, 242.7 are realistic human weights
#   in lbs (80–110 kg) but would represent severe obesity in kg.
# Fix: identify rows where weight > 130 and convert lbs → kg
#   using the factor: 1 lb = 0.453592 kg
# ============================================================

print("\n=== STEP 3b: Fix Bug #2 — Unit Mismatch in weight ===")
print(f"Before fix — weight stats:\n{df['weight'].describe()}")

lbs_mask = df["weight"] > 130
print(f"\nRows with weight > 130 (recorded in lbs): {lbs_mask.sum()}")
print(f"  Sample values: {sorted(df.loc[lbs_mask, 'weight'].values)[:8]}")

df.loc[lbs_mask, "weight"] = (df.loc[lbs_mask, "weight"] * 0.453592).round(1)

print(f"\nAfter fix — weight stats:\n{df['weight'].describe()}")

# FINDING: X rows were in lbs and have been converted to kg.
# The distribution is now unimodal and consistent with a clinical
# patient population (roughly 40–115 kg).


# ============================================================
# STEP 3c: FIX BUG #3 — Impossible Value in `resting_bp`
# ------------------------------------------------------------
# Resting blood pressure (mmHg) cannot be a negative number.
# The laws of physiology require BP > 0 for a living person.
# The specific corrupt value found in the data is -120, which
# appears to be a sign-error (the magnitude 120 is a normal BP).
# Fix: drop all rows where resting_bp < 0 since the intended
#   true value cannot be recovered with certainty.
# ============================================================

print("\n=== STEP 3c: Fix Bug #3 — Impossible resting_bp values ===")
impossible_mask = df["resting_bp"] < 0
print(f"Rows with resting_bp < 0 (impossible): {impossible_mask.sum()}")
print(f"  Corrupt values found : {df.loc[impossible_mask, 'resting_bp'].unique()}")
print(f"  Sample PIDs          : {df.loc[impossible_mask, key].values[:5]}")

rows_before = df.shape[0]
df = df[~impossible_mask].reset_index(drop=True)
print(f"\nRows removed: {rows_before - df.shape[0]}")
print(f"Shape after removal: {df.shape}")

# FINDING: X rows with resting_bp = -120 were identified and dropped.
# These are physiologically impossible and cannot be imputed from context.


# ============================================================
# STEP 3 SUMMARY: Post-fix distribution review
# ------------------------------------------------------------
# After all three feature fixes, we re-plot the distributions to
# visually confirm the corrections produced clean, coherent data.
# ============================================================

df[["age", "weight", "resting_bp"]].hist(bins=40, figsize=(14, 4), color="seagreen")
plt.suptitle("After Feature QA Fixes (age placeholder, weight units, impossible BP)")
plt.tight_layout()
plt.savefig("step3_clean_distributions.png")
plt.show()


# ============================================================
# STEP 4: TARGET LABEL AUDITING — Contradictory Labels
# ------------------------------------------------------------
# A supervised model requires that every training example has a
# single, trustworthy label. If the same patient (same inputs)
# appears multiple times with conflicting readmitted values,
# the model receives contradictory ground truth and cannot learn.
#
# Strategy: group rows by all input feature values and check
# whether any group contains more than one unique readmitted value.
# We also check for duplicate patient_id entries directly, since
# the same patient should have one canonical record.
# ============================================================

print("\n=== STEP 4: Target Label Auditing ===")

# Check for duplicate patient_id entries in the merged dataset
dup_pid = df[df.duplicated(subset=[key], keep=False)].sort_values(key)
print(f"Rows with duplicate patient_id: {len(dup_pid)}")
print(dup_pid[[key, "age", "weight", "resting_bp", "readmitted"]])

# Among duplicates, find which ones have CONFLICTING labels
input_cols = ["age", "weight", "resting_bp"]
conflict_groups = df.groupby(input_cols)["readmitted"].nunique()
conflicting = conflict_groups[conflict_groups > 1]
print(f"\nGroups with identical inputs but conflicting readmitted labels: {len(conflicting)}")

# FINDING: PID_1042 appears twice in medical_records.json.
# First entry: readmitted=0. Second (appended at end of file): readmitted=1.
# All other feature values are identical. This is a contradictory label.
# Fix: keep the first occurrence (more likely the original record),
# drop the second (likely a data entry error appended later).

rows_before_dedup = df.shape[0]
df = df.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
print(f"\nRows removed by deduplication: {rows_before_dedup - df.shape[0]}")
print(f"Shape after label dedup: {df.shape}")

# Verify no contradictions remain
remaining_conflicts = df.groupby(input_cols)["readmitted"].nunique()
print(f"Contradictory groups remaining after fix: {(remaining_conflicts > 1).sum()}")


# ============================================================
# STEP 5: CLASS IMBALANCE ANALYSIS
# ------------------------------------------------------------
# Before feeding data into any model we must understand the
# distribution of the target variable. If one class dominates,
# a naive model will simply predict the majority class for every
# input and appear highly accurate while being clinically useless.
# ============================================================

print("\n=== STEP 5: Class Imbalance Analysis ===")

counts  = df["readmitted"].value_counts().sort_index()
pct     = (df["readmitted"].value_counts(normalize=True) * 100).sort_index().round(2)
balance = pd.DataFrame({"Label": ["Not Readmitted (0)", "Readmitted (1)"],
                         "Count": counts.values,
                         "Percentage %": pct.values})
print(balance.to_string(index=False))

# Plot class distribution
counts.plot(kind="bar", color=["steelblue", "tomato"], figsize=(6, 4))
plt.title("Readmitted Class Distribution")
plt.xlabel("Readmitted (0 = No, 1 = Yes)")
plt.ylabel("Patient Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("step5_class_distribution.png")
plt.show()

# MARKDOWN INTERPRETATION (will become a markdown cell in the notebook):
# ---------------------------------------------------------------------
# The dataset shows significant class imbalance: approximately X% of
# patients were NOT readmitted (class 0) versus Y% who WERE readmitted
# (class 1). This imbalance creates a serious trap for standard neural
# networks.
#
# If we feed this unbalanced data directly into a model without any
# adjustment, the model can achieve ~X% accuracy by doing nothing more
# than predicting "not readmitted" for every single patient. From a
# raw accuracy metric, this looks excellent — but the model has learned
# nothing useful. It will completely fail to identify the patients who
# ARE at risk of readmission (class 1), which is the entire clinical
# purpose of building the model.
#
# This is the "accuracy paradox" caused by class imbalance. The model
# optimizes for the majority class and ignores the minority, resulting
# in near-zero recall for the readmitted group. In a hospital setting
# this is not just useless — it is dangerous, as high-risk patients
# would go unidentified.
#
# Mitigation strategies before modeling:
#   1. Use class_weight='balanced' in the model
#   2. Oversample the minority class (SMOTE)
#   3. Undersample the majority class
#   4. Use precision-recall AUC instead of accuracy as the evaluation metric


# ============================================================
# STEP 6: FINAL QA SUMMARY
# ------------------------------------------------------------
# Log every transformation applied, rows affected, and the
# final state of the cleaned dataset ready for modeling.
# ============================================================

print("\n=== STEP 6: Final QA Summary ===")
print(f"  Original demographics rows    : {demographics_raw.shape[0]}")
print(f"  Original medical records rows : {records_raw.shape[0]}")
print(f"  After inner join              : {merged_df.shape[0]}")
print(f"  Bug #1 (age placeholder 999)  : {placeholder_mask.sum()} values set to NaN")
print(f"  Bug #2 (weight unit mismatch) : {lbs_mask.sum()} rows converted lbs → kg")
print(f"  Bug #3 (impossible resting_bp): {rows_before - df.shape[0] + (rows_before_dedup - df.shape[0])} rows dropped (negative BP)")
print(f"  Label dedup (contradictory)   : {rows_before_dedup - df.shape[0]} rows dropped")
print(f"  Final cleaned dataset shape   : {df.shape}")
print(f"\n  Final class distribution:")
print(balance.to_string(index=False))
print("\n  Cleaned dataset saved to df — ready for notebook export.")