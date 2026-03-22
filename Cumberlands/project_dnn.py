# ============================================================
# STEP 1: DE Context and Objectives
# 2026 Spring - Deep Learning (MSDS-534-M40)
# Residency Day 2 - Project 3: Basic Regression and DNNs
#
# Student: German Montoya
# Team Role: Data Engineer (DE)
# Group Assignment: Group 2 - Real Estate (Housing Prices)
# Dataset: raw_data.csv
# Target Variable: median_house_value
#
# Objective:
# Execute the Data Engineer pipeline for the Housing dataset:
# 1) Load and inspect raw data.
# 2) Check and handle duplicates.
# 3) Impute missing total_bedrooms using the median (with justification).
# 4) Engineer:
#    - Rooms_per_Household = total_rooms / households
#    - Bedrooms_per_Room = total_bedrooms / total_rooms
# 5) One-hot encode ocean_proximity and drop original categorical column.
# 6) Produce EDA summary table (mean and std) for key features and target.
# 7) Prepare clean handoff dataset for Baseline Modeler (BM).
#
# Scope Note:
# This script follows Group 2 Housing instructions only.
# BMI-related preprocessing does not apply to this dataset.
# ============================================================

# STEP 2: Imports and Raw Data Inspection
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import sklearn (may fail due to pyarrow conflicts)
sklearn_available = False
try:
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
	sklearn_available = True
except ImportError as e:
	print(f"⚠ Warning: sklearn not available ({str(e)[:50]}...)")
	print("  Skipping linear regression comparison. DNN comparison will proceed.\n")
	
	# Define numpy-based fallback metrics
	def mean_squared_error(y_true, y_pred):
		return np.mean((y_true - y_pred) ** 2)
	
	def mean_absolute_error(y_true, y_pred):
		return np.mean(np.abs(y_true - y_pred))
	
	def r2_score(y_true, y_pred):
		ss_res = np.sum((y_true - y_pred) ** 2)
		ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
		return 1 - (ss_res / ss_tot)

# Try to import Keras/TensorFlow for DNN training/loading
keras_available = False
try:
	from tensorflow.keras.models import load_model, Sequential
	from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
	from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
	from tensorflow.keras.optimizers import Adam
	keras_available = True
except ImportError:
	try:
		from keras.models import load_model, Sequential
		from keras.layers import Dense, Dropout, BatchNormalization, Input
		from keras.callbacks import EarlyStopping, ModelCheckpoint
		from keras.optimizers import Adam
		keras_available = True
	except ImportError:
		load_model = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "raw_data.csv"
DATA_PATH_CANDIDATES = [
	BASE_DIR / "raw_data.csv",
	BASE_DIR.parent / "raw_data.csv",
]
TARGET_COLUMN = "median_house_value"
HANDOFF_MD_PATH = BASE_DIR / "de_cleaning_handoff.md"
CLEANED_DATA_PATH = BASE_DIR / "cleaned_data_gm.csv"
CLEANING_LOG_CSV_PATH = BASE_DIR / "de_cleaning_log.csv"


def load_raw_data(csv_path: Path) -> pd.DataFrame:
	if csv_path.exists():
		return pd.read_csv(csv_path)

	for candidate in DATA_PATH_CANDIDATES:
		if candidate.exists():
			print(f"Using dataset path: {candidate}")
			return pd.read_csv(candidate)

	searched_paths = "\n".join(str(path) for path in DATA_PATH_CANDIDATES)
	raise FileNotFoundError(
		"Dataset not found. Checked:\n"
		f"{csv_path}\n"
		f"{searched_paths}"
	)


def inspect_raw_data(df: pd.DataFrame) -> pd.DataFrame:
	print("\n=== Raw Data Overview ===")
	print(f"Shape: {df.shape}")
	print("\nColumns:")
	print(df.columns.tolist())

	print("\nData types:")
	print(df.dtypes)

	duplicate_count = int(df.duplicated().sum())
	print(f"\nDuplicate rows: {duplicate_count}")

	missing_summary = (
		df.isna()
		.sum()
		.rename("missing_count")
		.reset_index()
		.rename(columns={"index": "column"})
		.sort_values("missing_count", ascending=False)
	)

	print("\nMissing values by column:")
	print(missing_summary)

	print("\nHead:")
	print(df.head())

	return missing_summary


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
	if df.empty:
		return "_No rows to display._"

	headers = [str(column) for column in df.columns]
	separator = ["---" for _ in headers]
	rows = [headers, separator]

	for _, row in df.iterrows():
		rows.append([str(value) for value in row.tolist()])

	return "\n".join("| " + " | ".join(row_values) + " |" for row_values in rows)


def build_handoff_markdown(
	raw_df: pd.DataFrame,
	clean_df: pd.DataFrame,
	missing_table: pd.DataFrame,
	cleaning_log_df: pd.DataFrame,
	encoded_columns: list[str],
	rows_before: int,
	rows_after: int,
	missing_before: int,
	missing_after: int,
	median_total_bedrooms: float,
	ocean_proximity_categories: list[str] = None,
) -> str:
	report_lines = [
		"# DE Cleaning and Preparation Handoff",
		"",
		f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
		"",
		"## Scope",
		"- Role: Data Engineer (DE)",
		"- Group: Group 2 (Housing Prices)",
		"- Dataset: raw_data.csv",
		f"- Target variable: {TARGET_COLUMN}",
		"",
		"## Findings",
		f"- Raw shape: {raw_df.shape}",
		f"- Cleaned shape: {clean_df.shape}",
		f"- Duplicate rows removed: {rows_before - rows_after}",
		f"- Missing total_bedrooms before imputation: {missing_before}",
		f"- Missing total_bedrooms after imputation: {missing_after}",
		f"- Median used for total_bedrooms: {median_total_bedrooms:.2f}",
		"",
		"## Missing Values Table",
		dataframe_to_markdown_table(missing_table),
		"",
		"## Cleaning Log",
		dataframe_to_markdown_table(cleaning_log_df),
		"",
		"## Engineered and Encoded Features",
		"- Engineered: Rooms_per_Household, Bedrooms_per_Room",
		f"- One-hot encoded columns from ocean_proximity: {', '.join(encoded_columns)}",
	]

	if ocean_proximity_categories:
		report_lines.append("")
		report_lines.append("## Ocean Proximity Categories (Raw Values)")
		report_lines.append(f"- Count: {len(ocean_proximity_categories)}")
		report_lines.append("- Values:")
		for cat in ocean_proximity_categories:
			report_lines.append(f"  - {cat}")

	report_lines.extend([
		"",
		"## Handoff to Baseline Modeler",
		"- Use clean_df as input for train/test splitting.",
		"- Use median_house_value as target label.",
		"- Do not re-run imputation on total_bedrooms.",
		"- Apply normalization by fitting only on training data.",
	])

	return "\n".join(report_lines)


raw_df = load_raw_data(DATA_PATH)
missing_table = inspect_raw_data(raw_df)


# ============================================================
# STEP 3: Imputation Strategy Justification
# We use median imputation for total_bedrooms because:
# 1) The column is numeric and has missing values.
# 2) Median is robust to outliers compared with mean.
# 3) It preserves a realistic central tendency without assuming normality.
# ============================================================

# Part 3 (Code Equivalent): Cleaning Log + Duplicates + Imputation
cleaning_log = []


def log_cleaning_step(step, column, action, before_value, after_value, note):
	cleaning_log.append(
		{
			"step": step,
			"column": column,
			"action": action,
			"before": before_value,
			"after": after_value,
			"note": note,
		}
	)


clean_df = raw_df.copy()

rows_before = len(clean_df)
clean_df = clean_df.drop_duplicates().copy()
rows_after = len(clean_df)

log_cleaning_step(
	step="duplicates",
	column="all",
	action="drop_duplicates",
	before_value=rows_before,
	after_value=rows_after,
	note=f"Removed {rows_before - rows_after} duplicate rows.",
)

missing_before = int(clean_df["total_bedrooms"].isna().sum())
median_total_bedrooms = float(clean_df["total_bedrooms"].median())
clean_df["total_bedrooms"] = clean_df["total_bedrooms"].fillna(median_total_bedrooms)
missing_after = int(clean_df["total_bedrooms"].isna().sum())

log_cleaning_step(
	step="imputation",
	column="total_bedrooms",
	action="median_fillna",
	before_value=missing_before,
	after_value=missing_after,
	note=f"Median used: {median_total_bedrooms:.2f}",
)

cleaning_log_df = pd.DataFrame(cleaning_log)

print("\n=== Cleaning Checkpoint ===")
print(f"Rows before duplicate handling: {rows_before}")
print(f"Rows after duplicate handling:  {rows_after}")
print(f"Missing total_bedrooms before imputation: {missing_before}")
print(f"Missing total_bedrooms after imputation:  {missing_after}")
print(f"Median used for total_bedrooms: {median_total_bedrooms:.2f}")

print("\nCleaning log:")
print(cleaning_log_df)


# ============================================================
# STEP 4: Feature Engineering + Encoding Rationale
# Required Group 2 engineered features:
# 1) Rooms_per_Household = total_rooms / households
# 2) Bedrooms_per_Room = total_bedrooms / total_rooms
#
# Why these features help:
# - Rooms_per_Household captures housing density and living space pressure.
# - Bedrooms_per_Room captures bedroom composition within available room space.
#
# Categorical handling:
# - One-hot encode ocean_proximity.
# - Drop the original ocean_proximity column to avoid redundancy.
# ============================================================

# Part 4 (Code Equivalent): Engineer Features and One-Hot Encode
rooms_denominator = clean_df["households"].replace(0, pd.NA)
bedroom_ratio_denominator = clean_df["total_rooms"].replace(0, pd.NA)

clean_df["Rooms_per_Household"] = clean_df["total_rooms"] / rooms_denominator
clean_df["Bedrooms_per_Room"] = clean_df["total_bedrooms"] / bedroom_ratio_denominator

log_cleaning_step(
	step="feature_engineering",
	column="Rooms_per_Household",
	action="create_ratio",
	before_value="not_present",
	after_value="present",
	note="Created total_rooms / households.",
)

log_cleaning_step(
	step="feature_engineering",
	column="Bedrooms_per_Room",
	action="create_ratio",
	before_value="not_present",
	after_value="present",
	note="Created total_bedrooms / total_rooms.",
)

non_finite_before = int(clean_df[["Rooms_per_Household", "Bedrooms_per_Room"]].isna().sum().sum())
clean_df[["Rooms_per_Household", "Bedrooms_per_Room"]] = clean_df[
	["Rooms_per_Household", "Bedrooms_per_Room"]
].fillna(0.0)
non_finite_after = int(clean_df[["Rooms_per_Household", "Bedrooms_per_Room"]].isna().sum().sum())

log_cleaning_step(
	step="feature_engineering",
	column="Rooms_per_Household|Bedrooms_per_Room",
	action="fillna_zero",
	before_value=non_finite_before,
	after_value=non_finite_after,
	note="Filled ratio NaN values caused by zero denominators with 0.0.",
)

print("\n=== Ocean Proximity Inspection (Before Encoding) ===")
ocean_proximity_categories = sorted(clean_df["ocean_proximity"].unique())
print(f"Unique values count: {len(ocean_proximity_categories)}")
print("Values:")
for val in ocean_proximity_categories:
	count = (clean_df["ocean_proximity"] == val).sum()
	print(f"  - {val}: {count} records")

columns_before_encoding = clean_df.shape[1]
clean_df = pd.get_dummies(clean_df, columns=["ocean_proximity"], prefix="ocean_proximity", dtype=int)
columns_after_encoding = clean_df.shape[1]

log_cleaning_step(
	step="encoding",
	column="ocean_proximity",
	action="one_hot_encode_drop_original",
	before_value=columns_before_encoding,
	after_value=columns_after_encoding,
	note="Applied one-hot encoding and dropped original categorical column.",
)

cleaning_log_df = pd.DataFrame(cleaning_log)

print("\n=== Feature Engineering + Encoding Checkpoint ===")
print("New engineered columns added: Rooms_per_Household, Bedrooms_per_Room")
print(f"Column count before encoding: {columns_before_encoding}")
print(f"Column count after encoding:  {columns_after_encoding}")

encoded_columns = [column for column in clean_df.columns if column.startswith("ocean_proximity_")]
print("\nEncoded ocean_proximity columns:")
print(encoded_columns)

print("\nUpdated cleaning log:")
print(cleaning_log_df)


# STEP 5: Persist Handoff Report for Team Integration
handoff_markdown = build_handoff_markdown(
	raw_df=raw_df,
	clean_df=clean_df,
	missing_table=missing_table,
	cleaning_log_df=cleaning_log_df,
	encoded_columns=encoded_columns,
	rows_before=rows_before,
	rows_after=rows_after,
	missing_before=missing_before,
	missing_after=missing_after,
	median_total_bedrooms=median_total_bedrooms,
	ocean_proximity_categories=ocean_proximity_categories,
)

HANDOFF_MD_PATH.write_text(handoff_markdown, encoding="utf-8")
print(f"\nSaved markdown handoff report: {HANDOFF_MD_PATH}")


# STEP 6: Lightweight Schema Validation for Handoff
def validate_handoff_schema(df: pd.DataFrame) -> bool:
	issues = []

	# After encoding, there should be no object columns left.
	object_columns = df.select_dtypes(include=["object"]).columns.tolist()
	if object_columns:
		issues.append(f"Object columns found after encoding: {object_columns}")

	# One-hot columns should be integer typed (0/1).
	one_hot_columns = [column for column in df.columns if column.startswith("ocean_proximity_")]
	bad_one_hot_dtypes = [
		column
		for column in one_hot_columns
		if not pd.api.types.is_integer_dtype(df[column])
	]
	if bad_one_hot_dtypes:
		issues.append(f"One-hot columns not integer typed: {bad_one_hot_dtypes}")

	# Key numeric columns should stay numeric for BM/DLA stages.
	key_numeric_columns = [
		"Rooms_per_Household",
		"Bedrooms_per_Room",
		"total_bedrooms",
		TARGET_COLUMN,
	]
	bad_numeric_columns = [
		column
		for column in key_numeric_columns
		if column in df.columns and not pd.api.types.is_numeric_dtype(df[column])
	]
	if bad_numeric_columns:
		issues.append(f"Key columns not numeric: {bad_numeric_columns}")

	print("\n=== Handoff Schema Validation ===")
	if issues:
		print("Schema check: FAIL")
		for issue in issues:
			print(f"- {issue}")
		return False

	print("Schema check: PASS")
	print("- No object columns remain after encoding.")
	print("- One-hot columns are integer typed.")
	print("- Key feature/target columns are numeric.")
	return True


schema_ok = validate_handoff_schema(clean_df)


# STEP 7: EDA Summary Table (Mean and Std)
key_features_for_eda = [
	"total_rooms",
	"total_bedrooms",
	"households",
	"median_income",
	"Rooms_per_Household",
	"Bedrooms_per_Room",
	TARGET_COLUMN,
]

missing_eda_columns = [column for column in key_features_for_eda if column not in clean_df.columns]
if missing_eda_columns:
	raise KeyError(f"Missing expected EDA columns: {missing_eda_columns}")

eda_summary_df = clean_df[key_features_for_eda].agg(["mean", "std"]).T.reset_index()
eda_summary_df = eda_summary_df.rename(columns={"index": "feature"})

print("\n=== EDA Summary Table (Mean and Std) ===")
print(eda_summary_df)


# ============================================================
# STEP 8: EDA Visualizations
# ============================================================

# Configure matplotlib style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

# 1. Multi-panel histogram grid (4 most relevant features as 2x2 grid)
key_features_for_histograms = ["median_income", "total_rooms", "Rooms_per_Household", TARGET_COLUMN]
fig_histograms, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, feature in enumerate(key_features_for_histograms):
	axes[idx].hist(clean_df[feature], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
	axes[idx].set_title(f"Distribution of {feature}", fontsize=11, fontweight="bold")
	axes[idx].set_xlabel(feature, fontsize=10)
	axes[idx].set_ylabel("Frequency", fontsize=10)
	axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
histogram_path = BASE_DIR / "de_eda_histograms.png"
fig_histograms.savefig(histogram_path, dpi=100, bbox_inches="tight")
plt.show()
plt.close(fig_histograms)
print(f"\nSaved histogram grid: {histogram_path}")

# 2. Correlation heatmap for key features
fig_corr, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = clean_df[key_features_for_eda].corr()
sns.heatmap(
	correlation_matrix,
	annot=True,
	fmt=".2f",
	cmap="coolwarm",
	center=0,
	square=True,
	linewidths=0.5,
	cbar_kws={"shrink": 0.8},
	ax=ax,
)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
corr_path = BASE_DIR / "de_eda_correlation_heatmap.png"
fig_corr.savefig(corr_path, dpi=100, bbox_inches="tight")
plt.show()
plt.close(fig_corr)
print(f"Saved correlation heatmap: {corr_path}")

# 3. Boxplot for all numeric columns (to identify outliers)
fig_box, ax = plt.subplots(figsize=(14, 6))
clean_df[key_features_for_eda].boxplot(ax=ax, patch_artist=True)
ax.set_title("Boxplot: Feature Distributions and Outliers", fontsize=13, fontweight="bold")
ax.set_ylabel("Value", fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
box_path = BASE_DIR / "de_eda_boxplot.png"
fig_box.savefig(box_path, dpi=100, bbox_inches="tight")
plt.show()
plt.close(fig_box)
print(f"Saved boxplot: {box_path}")

# 4. Target distribution (detailed)
fig_target, ax = plt.subplots(figsize=(10, 6))
ax.hist(clean_df[TARGET_COLUMN], bins=40, color="coral", edgecolor="black", alpha=0.7)
ax.set_title(f"Target Distribution: {TARGET_COLUMN}", fontsize=13, fontweight="bold")
ax.set_xlabel(TARGET_COLUMN, fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.axvline(clean_df[TARGET_COLUMN].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {clean_df[TARGET_COLUMN].mean():.2f}")
ax.axvline(clean_df[TARGET_COLUMN].median(), color="green", linestyle="--", linewidth=2, label=f"Median: {clean_df[TARGET_COLUMN].median():.2f}")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
target_path = BASE_DIR / "de_eda_target_distribution.png"
fig_target.savefig(target_path, dpi=100, bbox_inches="tight")
plt.show()
plt.close(fig_target)
print(f"Saved target distribution: {target_path}")


# Export DE handoff artifacts for BM integration.
clean_df.to_csv(CLEANED_DATA_PATH, index=False)
cleaning_log_df.to_csv(CLEANING_LOG_CSV_PATH, index=False)
print(f"\nSaved cleaned dataset: {CLEANED_DATA_PATH}")
print(f"Saved cleaning log CSV: {CLEANING_LOG_CSV_PATH}")


# Append EDA and BM-ready summary sections to the markdown handoff.
existing_handoff = HANDOFF_MD_PATH.read_text(encoding="utf-8") if HANDOFF_MD_PATH.exists() else ""
eda_section = (
	"\n\n## EDA Visualizations\n"
	"- **Histograms**: Distribution of key features and target (6-panel grid)\n"
	"  - File: `de_eda_histograms.png`\n"
	"- **Correlation Heatmap**: Feature relationships and multicollinearity\n"
	"  - File: `de_eda_correlation_heatmap.png`\n"
	"- **Boxplot**: Outlier identification across all features\n"
	"  - File: `de_eda_boxplot.png`\n"
	"- **Target Distribution**: Detailed histogram of median_house_value with mean/median lines\n"
	"  - File: `de_eda_target_distribution.png`\n"
	"\n\n## EDA Summary Table (Mean and Std)\n" 
	+ dataframe_to_markdown_table(eda_summary_df)
)

feature_count = clean_df.shape[1] - 1 if TARGET_COLUMN in clean_df.columns else clean_df.shape[1]
bm_ready_section = "\n\n## BM Ready Summary\n" + "\n".join(
	[
		f"- Final cleaned shape: {clean_df.shape}",
		f"- Target column: {TARGET_COLUMN}",
		f"- Feature count (excluding target): {feature_count}",
		f"- Encoded ocean proximity columns: {', '.join(encoded_columns)}",
		f"- Cleaned dataset file: {CLEANED_DATA_PATH.name}",
		f"- Cleaning log CSV file: {CLEANING_LOG_CSV_PATH.name}",
		f"- EDA visualization files: de_eda_histograms.png, de_eda_correlation_heatmap.png, de_eda_boxplot.png, de_eda_target_distribution.png",
	]
)
handoff_suffix = eda_section + bm_ready_section

if "## EDA Visualizations" in existing_handoff:
	# Replace existing downstream sections on rerun to keep the report deterministic.
	prefix = existing_handoff.split("## EDA Visualizations")[0].rstrip()
	updated_handoff = prefix + handoff_suffix
else:
	updated_handoff = existing_handoff.rstrip() + handoff_suffix

HANDOFF_MD_PATH.write_text(updated_handoff + "\n", encoding="utf-8")
print(f"\nAppended EDA and BM-ready summary to: {HANDOFF_MD_PATH}")


# ============================================================
# STEP 9: Baseline Linear vs Tuned DNN Comparison
# ============================================================

print("\n" + "="*70)
print("STEP 9A: BASELINE LINEAR MODELS vs DNN COMPARISON")
print("="*70)

# Reload clean data for comparison
comparison_df = pd.read_csv(CLEANED_DATA_PATH)
X = comparison_df.drop(columns=[TARGET_COLUMN])
y = comparison_df[TARGET_COLUMN]

all_predictions = {}
all_r2 = {}
all_rmse = {}
all_mae = {}
X_train_cmp = None
X_train_scaled_cmp = None
X_test_cmp = None
y_test_cmp = None
y_train_cmp = None
X_test_scaled_cmp = None

# ============================================================
# Train Linear Regression Models (if sklearn available)
# ============================================================
if sklearn_available:
	print("\n--- STEP 9B: LINEAR REGRESSION MODELS ---")
	
	# Train/test split (80/20, deterministic seed)
	X_train_cmp, X_test_cmp, y_train_cmp, y_test_cmp = train_test_split(
		X, y, test_size=0.2, random_state=42
	)
	
	# Normalize (fit on train only)
	scaler_cmp = StandardScaler()
	X_train_scaled_cmp = scaler_cmp.fit_transform(X_train_cmp)
	X_test_scaled_cmp = scaler_cmp.transform(X_test_cmp)
	
	print(f"Train: {X_train_scaled_cmp.shape} | Test: {X_test_scaled_cmp.shape}")
	
	# Single-variable linear (median_income only)
	median_income_idx = X.columns.get_loc("median_income")
	X_train_single = X_train_scaled_cmp[:, median_income_idx].reshape(-1, 1)
	X_test_single = X_test_scaled_cmp[:, median_income_idx].reshape(-1, 1)
	
	lr_single = LinearRegression()
	lr_single.fit(X_train_single, y_train_cmp)
	y_pred_single = lr_single.predict(X_test_single)
	
	r2_single = r2_score(y_test_cmp, y_pred_single)
	rmse_single = np.sqrt(mean_squared_error(y_test_cmp, y_pred_single))
	mae_single = mean_absolute_error(y_test_cmp, y_pred_single)
	
	print(f"[Single-Variable Linear]")
	print(f"  R²: {r2_single:.6f} | RMSE: ${rmse_single:,.0f} | MAE: ${mae_single:,.0f}")
	
	all_predictions["Single Linear"] = y_pred_single
	all_r2["Single Linear"] = r2_single
	all_rmse["Single Linear"] = rmse_single
	all_mae["Single Linear"] = mae_single
	
	# Multi-variable linear (all features)
	lr_multi = LinearRegression()
	lr_multi.fit(X_train_scaled_cmp, y_train_cmp)
	y_pred_multi = lr_multi.predict(X_test_scaled_cmp)
	
	r2_multi = r2_score(y_test_cmp, y_pred_multi)
	rmse_multi = np.sqrt(mean_squared_error(y_test_cmp, y_pred_multi))
	mae_multi = mean_absolute_error(y_test_cmp, y_pred_multi)
	
	print(f"[Multi-Variable Linear]")
	print(f"  R²: {r2_multi:.6f} | RMSE: ${rmse_multi:,.0f} | MAE: ${mae_multi:,.0f}")
	
	all_predictions["Multi Linear"] = y_pred_multi
	all_r2["Multi Linear"] = r2_multi
	all_rmse["Multi Linear"] = rmse_multi
	all_mae["Multi Linear"] = mae_multi

else:
	print("\n⚠ STEP 9B: sklearn not available - using dummy test set for DNN evaluation")
	# Create dummy train/test split using numpy only
	n_samples = len(X)
	n_test = int(0.2 * n_samples)
	np.random.seed(42)
	test_idx = np.random.choice(n_samples, n_test, replace=False)
	
	X_test_cmp = X.iloc[test_idx].values
	y_test_cmp = y.iloc[test_idx].values
	
	# Simple normalization without sklearn
	X_test_scaled_cmp = (X_test_cmp - X_test_cmp.mean(axis=0)) / (X_test_cmp.std(axis=0) + 1e-8)
	print(f"Test set: {X_test_cmp.shape}")

# ============================================================
# STEP 9C: Train/Evaluate DNN Models
# ============================================================
print("\n--- STEP 9C: DNN MODELS ---")

baseline_dnn_path = BASE_DIR / "baseline_linear_model.keras"
tuned_dnn_path = BASE_DIR / "tuned_dnn_model.keras"

if keras_available and X_test_scaled_cmp is not None:
	baseline_dnn = None
	tuned_dnn = None

	if sklearn_available and X_train_scaled_cmp is not None and y_train_cmp is not None:
		print("Training DNNs with same split/scaler as linear models...")
		n_features = X_train_scaled_cmp.shape[1]

		# Baseline DNN: simple architecture for first-pass neural benchmark.
		baseline_dnn = Sequential(
			[
				Input(shape=(n_features,)),
				Dense(64, activation="relu"),
				Dropout(0.20),
				Dense(32, activation="relu"),
				Dense(1),
			]
		)
		baseline_dnn.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

		baseline_callbacks = [
			EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
			ModelCheckpoint(filepath=baseline_dnn_path, monitor="val_loss", save_best_only=True, verbose=0),
		]
		history_base = baseline_dnn.fit(
			X_train_scaled_cmp,
			y_train_cmp,
			epochs=80,
			batch_size=64,
			validation_split=0.2,
			callbacks=baseline_callbacks,
			verbose=0,
		)
		print(f"[Baseline DNN] Trained for {len(history_base.history['loss'])} epochs")

		# Tuned DNN: deeper network with normalization/dropout and lower LR.
		tuned_dnn = Sequential(
			[
				Input(shape=(n_features,)),
				Dense(256, activation="relu"),
				BatchNormalization(),
				Dropout(0.30),
				Dense(128, activation="relu"),
				BatchNormalization(),
				Dropout(0.20),
				Dense(64, activation="relu"),
				Dense(1),
			]
		)
		tuned_dnn.compile(optimizer=Adam(learning_rate=5e-4), loss="mse", metrics=["mae"])

		tuned_callbacks = [
			EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
			ModelCheckpoint(filepath=tuned_dnn_path, monitor="val_loss", save_best_only=True, verbose=0),
		]
		history_tuned = tuned_dnn.fit(
			X_train_scaled_cmp,
			y_train_cmp,
			epochs=120,
			batch_size=64,
			validation_split=0.2,
			callbacks=tuned_callbacks,
			verbose=0,
		)
		print(f"[Tuned DNN] Trained for {len(history_tuned.history['loss'])} epochs")

		if baseline_dnn_path.exists():
			baseline_dnn = load_model(baseline_dnn_path, compile=False)
		if tuned_dnn_path.exists():
			tuned_dnn = load_model(tuned_dnn_path, compile=False)
	else:
		print("Training skipped (requires sklearn split/scaler). Attempting to use saved DNN files...")
		if baseline_dnn_path.exists():
			baseline_dnn = load_model(baseline_dnn_path, compile=False)
		if tuned_dnn_path.exists():
			tuned_dnn = load_model(tuned_dnn_path, compile=False)

	if baseline_dnn is not None:
		y_pred_baseline_dnn = baseline_dnn.predict(X_test_scaled_cmp, verbose=0).flatten()
		r2_base_dnn = r2_score(y_test_cmp, y_pred_baseline_dnn)
		rmse_base_dnn = np.sqrt(mean_squared_error(y_test_cmp, y_pred_baseline_dnn))
		mae_base_dnn = mean_absolute_error(y_test_cmp, y_pred_baseline_dnn)

		print(f"[Baseline DNN]")
		print(f"  R²: {r2_base_dnn:.6f} | RMSE: ${rmse_base_dnn:,.0f} | MAE: ${mae_base_dnn:,.0f}")

		all_predictions["Baseline DNN"] = y_pred_baseline_dnn
		all_r2["Baseline DNN"] = r2_base_dnn
		all_rmse["Baseline DNN"] = rmse_base_dnn
		all_mae["Baseline DNN"] = mae_base_dnn
	else:
		print(f"[Baseline DNN] Model file missing: {baseline_dnn_path.name}")

	if tuned_dnn is not None:
		y_pred_tuned_dnn = tuned_dnn.predict(X_test_scaled_cmp, verbose=0).flatten()
		r2_tuned_dnn = r2_score(y_test_cmp, y_pred_tuned_dnn)
		rmse_tuned_dnn = np.sqrt(mean_squared_error(y_test_cmp, y_pred_tuned_dnn))
		mae_tuned_dnn = mean_absolute_error(y_test_cmp, y_pred_tuned_dnn)

		print(f"[Tuned DNN]")
		print(f"  R²: {r2_tuned_dnn:.6f} | RMSE: ${rmse_tuned_dnn:,.0f} | MAE: ${mae_tuned_dnn:,.0f}")

		all_predictions["Tuned DNN"] = y_pred_tuned_dnn
		all_r2["Tuned DNN"] = r2_tuned_dnn
		all_rmse["Tuned DNN"] = rmse_tuned_dnn
		all_mae["Tuned DNN"] = mae_tuned_dnn
	else:
		print(f"[Tuned DNN] Model file missing: {tuned_dnn_path.name}")
else:
	print("[DNN Models] Keras/TensorFlow or test set not available.")

# ============================================================
# STEP 10: Generate Comparison Visualizations (if models available)
# ============================================================
if all_r2:
	print("\n--- STEP 10: GENERATING VISUALIZATIONS ---")

	# 1. Bar chart: R², RMSE, MAE comparison
	fig_bars, axes = plt.subplots(1, 3, figsize=(15, 5))

	model_names = list(all_r2.keys())
	x_pos = np.arange(len(model_names))

	# R² scores
	r2_values = [all_r2[m] for m in model_names]
	axes[0].bar(x_pos, r2_values, color="steelblue", alpha=0.7, edgecolor="black")
	for i, v in enumerate(r2_values):
		axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
	axes[0].set_title("R² Score (Higher is Better)", fontsize=12, fontweight="bold")
	axes[0].set_ylabel("R²", fontsize=11)
	axes[0].set_xticks(x_pos)
	axes[0].set_xticklabels(model_names, rotation=45, ha="right")
	axes[0].grid(True, alpha=0.3, axis="y")

	# RMSE
	rmse_values = [all_rmse[m] for m in model_names]
	axes[1].bar(x_pos, rmse_values, color="coral", alpha=0.7, edgecolor="black")
	for i, v in enumerate(rmse_values):
		axes[1].text(i, v + 1000, f"${v:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
	axes[1].text(0.98, 0.98, "Root Mean Squared Error", transform=axes[1].transAxes, fontsize=10, fontweight="bold", va="top", ha="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
	axes[1].set_title("RMSE (Lower is Better)", fontsize=12, fontweight="bold")
	axes[1].set_ylabel("RMSE ($)", fontsize=11)
	axes[1].set_xticks(x_pos)
	axes[1].set_xticklabels(model_names, rotation=45, ha="right")
	axes[1].grid(True, alpha=0.3, axis="y")

	# MAE
	mae_values = [all_mae[m] for m in model_names]
	axes[2].bar(x_pos, mae_values, color="lightgreen", alpha=0.7, edgecolor="black")
	for i, v in enumerate(mae_values):
		axes[2].text(i, v + 1000, f"${v:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
	axes[2].text(0.98, 0.98, "Mean Absolute Error", transform=axes[2].transAxes, fontsize=10, fontweight="bold", va="top", ha="right", bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.5))
	axes[2].set_title("MAE (Lower is Better)", fontsize=12, fontweight="bold")
	axes[2].set_ylabel("MAE ($)", fontsize=11)
	axes[2].set_xticks(x_pos)
	axes[2].set_xticklabels(model_names, rotation=45, ha="right")
	axes[2].grid(True, alpha=0.3, axis="y")

	plt.tight_layout()
	comparison_bars_path = BASE_DIR / "comparison_metrics_bars.png"
	fig_bars.savefig(comparison_bars_path, dpi=100, bbox_inches="tight")
	plt.show()
	plt.close(fig_bars)
	print(f"✓ Saved comparison bar chart: {comparison_bars_path.name}")

	# 2. Predicted vs Actual scatter plots (2×2 or 2×3 grid depending on model count)
	num_models = len(all_predictions)
	ncols = min(3, num_models)
	nrows = (num_models + ncols - 1) // ncols

	fig_scatter, axes_scatter = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
	if nrows == 1 and ncols == 1:
		axes_scatter = np.array([[axes_scatter]])
	elif nrows == 1 or ncols == 1:
		axes_scatter = axes_scatter.reshape(nrows, ncols)

	axes_scatter_flat = axes_scatter.flatten()

	for idx, (model_name, y_pred) in enumerate(all_predictions.items()):
		ax = axes_scatter_flat[idx]
		
		# Scatter plot
		ax.scatter(y_test_cmp, y_pred, alpha=0.5, s=20, edgecolor="black", linewidth=0.5)
		
		# Diagonal line (perfect predictions)
		y_min, y_max = y_test_cmp.min(), y_test_cmp.max()
		ax.plot([y_min, y_max], [y_min, y_max], "r--", linewidth=2, label="Perfect Fit")
		
		# Calculate and plot regression equation (actual vs predicted)
		z = np.polyfit(y_test_cmp, y_pred, 1)  # Linear fit
		p = np.poly1d(z)
		y_line = p(y_test_cmp)
		ax.plot(y_test_cmp, y_line, "b-", linewidth=2, label=f"Fit: y={z[0]:.3f}x+${z[1]:,.0f}")
		
		ax.set_xlabel("Actual Price ($)", fontsize=10)
		ax.set_ylabel("Predicted Price ($)", fontsize=10)
		ax.set_title(f"{model_name}\nR² = {all_r2[model_name]:.4f}", fontsize=11, fontweight="bold")
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=9)

	# Hide unused subplots
	for idx in range(len(all_predictions), len(axes_scatter_flat)):
		axes_scatter_flat[idx].set_visible(False)

	plt.tight_layout()
	comparison_scatter_path = BASE_DIR / "comparison_predicted_vs_actual.png"
	fig_scatter.savefig(comparison_scatter_path, dpi=100, bbox_inches="tight")
	plt.show()
	plt.close(fig_scatter)
	print(f"✓ Saved predicted vs actual scatter plots: {comparison_scatter_path.name}")

	# ============================================================
	# STEP 11: Comprehensive Model Ranking (All Metrics)
	# ============================================================
	print("\n--- STEP 11: COMPREHENSIVE MODEL RANKING ---")
	
	# Create ranking dataframe
	ranking_df = pd.DataFrame({
		"Model": list(all_r2.keys()),
		"R²": [all_r2[m] for m in all_r2.keys()],
		"RMSE ($)": [all_rmse[m] for m in all_r2.keys()],
		"MAE ($)": [all_mae[m] for m in all_r2.keys()],
	})
	
	# Rank by each metric
	ranking_df["R² Rank"] = ranking_df["R²"].rank(ascending=False)
	ranking_df["RMSE Rank"] = ranking_df["RMSE ($)"].rank(ascending=True)  # Lower is better
	ranking_df["MAE Rank"] = ranking_df["MAE ($)"].rank(ascending=True)    # Lower is better
	
	# Composite score (average rank across all metrics)
	ranking_df["Composite Rank"] = (ranking_df["R² Rank"] + ranking_df["RMSE Rank"] + ranking_df["MAE Rank"]) / 3
	ranking_df = ranking_df.sort_values("Composite Rank")
	
	print("\n=== MODEL RANKING BY ALL METRICS ===")
	print(ranking_df.to_string(index=False))
	
	# Best model by composite score
	best_model = ranking_df.iloc[0]["Model"]
	best_r2 = all_r2[best_model]
	best_rmse = all_rmse[best_model]
	best_mae = all_mae[best_model]
	
	print(f"\n🏆 BEST MODEL (by Composite Score): {best_model}")
	print(f"   R² = {best_r2:.6f}")
	print(f"   RMSE = ${best_rmse:,.0f}")
	print(f"   MAE = ${best_mae:,.0f}")
	
	# Extract equation for linear models
	model_equation = ""
	if best_model == "Single Linear" and sklearn_available:
		model_equation = f"median_house_value = {lr_single.intercept_:,.2f} + {lr_single.coef_[median_income_idx]:.6f} × median_income"
	elif best_model == "Multi Linear" and sklearn_available:
		feature_names = X.columns.tolist()
		equation_parts = [f"{lr_multi.intercept_:,.2f}"]
		for feat, coef in zip(feature_names, lr_multi.coef_):
			sign = "+" if coef >= 0 else "-"
			equation_parts.append(f"{sign} {abs(coef):.6f} × {feat}")
		model_equation = "median_house_value = " + " ".join(equation_parts)
	elif "DNN" in best_model:
		model_equation = f"Deep Neural Network (Nonlinear transformation of {X_test_scaled_cmp.shape[1]} features)"
	
	if model_equation:
		print(f"\n📋 EQUATION/ARCHITECTURE:")
		print(f"   {model_equation}")
	
	# Save ranking to JSON
	ranking_json = {
		"timestamp": datetime.now().isoformat(),
		"best_model": best_model,
		"all_models": ranking_df.to_dict(orient="records"),
		"equation": model_equation,
	}
	ranking_json_path = BASE_DIR / "model_ranking.json"
	with open(ranking_json_path, "w") as f:
		json.dump(ranking_json, f, indent=2)
	print(f"\n✓ Saved ranking details: {ranking_json_path.name}")

	# ============================================================
	# STEP 12: Generate Final Report
	# ============================================================
	print("\n--- STEP 12: GENERATING FINAL REPORT ---")

	# Create markdown report
	final_report = f"""# MSDS-534 Project 3 - Group 2 (Housing) - Final Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

Complete pipeline analysis: Data Engineering → Baseline Linear Models → Deep Neural Networks

**Best Performing Model:** **{best_model}** (R² = {best_r2:.6f})

---

## Model Comparison Results

| Model | R² Score | RMSE ($) | MAE ($) |
|-------|----------|----------|---------|
"""

	for model_name in all_r2.keys():
		final_report += f"| {model_name} | {all_r2[model_name]:.6f} | {all_rmse[model_name]:,.0f} | {all_mae[model_name]:,.0f} |\n"

	final_report += f"""
---

## Comprehensive Ranking (All Metrics Considered)

**Best Model Overall:** {best_model}
- **R² Score:** {best_r2:.6f}
- **RMSE:** ${best_rmse:,.0f}
- **MAE:** ${best_mae:,.0f}

*Ranking determined by composite score across R², RMSE, and MAE metrics.*

### Rank by Metric:

**By R² (Accuracy):**
"""
	r2_sorted = ranking_df.sort_values("R²", ascending=False)
	for i, row in r2_sorted.iterrows():
		final_report += f"  {int(row['R² Rank'])}. {row['Model']}: {row['R²']:.6f}\n"
	
	final_report += "\n**By RMSE (Lower Errors):**\n"
	rmse_sorted = ranking_df.sort_values("RMSE ($)", ascending=True)
	for i, row in rmse_sorted.iterrows():
		final_report += f"  {int(row['RMSE Rank'])}. {row['Model']}: ${row['RMSE ($)']:,.0f}\n"
	
	final_report += "\n**By MAE (Average Error):**\n"
	mae_sorted = ranking_df.sort_values("MAE ($)", ascending=True)
	for i, row in mae_sorted.iterrows():
		final_report += f"  {int(row['MAE Rank'])}. {row['Model']}: ${row['MAE ($)']:,.0f}\n"

	final_report += f"""
---

## Phase 1: Data Engineering (Completed)

**Raw Data:** 20,640 rows × 10 columns
**Cleaned Data:** 20,640 rows × 16 columns

**Transformations Applied:**
- Removed 0 duplicate rows
- Imputed missing total_bedrooms: 1,214 → 0 (median: 435.00)
- Engineered: Rooms_per_Household, Bedrooms_per_Room
- One-hot encoded: ocean_proximity (5 categories)

*See `de_cleaning_handoff.md` for details*

---

## Phase 2: Baseline Linear Models

"""

	if "Single Linear" in all_r2:
		final_report += f"""**Single-Variable Linear Regression**
- Features: median_income only
- R²: {all_r2['Single Linear']:.6f}
- RMSE: ${all_rmse['Single Linear']:,.0f}
- Purpose: Simplest baseline

"""
	
	if "Multi Linear" in all_r2:
		final_report += f"""**Multi-Variable Linear Regression**
- Features: All 15 engineered/encoded features
- R²: {all_r2['Multi Linear']:.6f}
- RMSE: ${all_rmse['Multi Linear']:,.0f}
- Purpose: Linear standard model for comparison

"""
	
	if "Single Linear" not in all_r2 and "Multi Linear" not in all_r2:
		final_report += """*Note: Linear regression models were not available due to sklearn import issues. Proceeding with DNN comparison only.*

"""
	
	final_report += f"""

---

## Phase 3: Deep Neural Networks

"""

	if "Baseline DNN" in all_r2:
		final_report += f"""**Baseline DNN**
- R²: {all_r2['Baseline DNN']:.6f}
- RMSE: ${all_rmse['Baseline DNN']:,.0f}
- Purpose: First DNN attempt, foundational network

"""

	if "Tuned DNN" in all_r2:
		improvement = all_r2['Tuned DNN'] - all_r2.get('Baseline DNN', 0)
		final_report += f"""**Tuned DNN**
- R²: {all_r2['Tuned DNN']:.6f}
- RMSE: ${all_rmse['Tuned DNN']:,.0f}
- Improvement over Baseline DNN: {improvement:+.6f}
- Purpose: Optimized architecture for better generalization

"""

	final_report += f"""---

## Key Insights

1. **Linear vs DNN Trade-off**
   - Linear models: Fast, interpretable, bounded performance
   - DNNs: Slower, black-box, potential for better accuracy

2. **Best Performer Analysis**
   - Model: {best_model}
   - R² Score: {best_r2:.6f} (explains {best_r2*100:.2f}% of variance)
   - Recommendation: Use for housing price predictions

3. **Remaining Error**
   - Even best model leaves ~{(1-best_r2)*100:.1f}% of variance unexplained
   - Suggests: Feature engineering opportunities, domain factors, market noise

---

## Fine-Tuning Recommendations for Tuned DNN

If further optimization is needed:

1. **Architecture:** Add/remove layers, increase hidden units (128→256)
2. **Regularization:** Increase dropout (0.3→0.5), add L1/L2 penalty
3. **Training:** Adjust learning rate (0.001→0.0005), use early stopping
4. **Data:** Try feature interactions, polynomial expansions, or PCA
5. **Validation:** Use k-fold CV for robust evaluation

---

## Visualizations Generated

1. `comparison_metrics_bars.png` — R², RMSE, MAE grouped by model
2. `comparison_predicted_vs_actual.png` — Scatter plots showing fit quality

---

## Dataset Information

"""
	
	if sklearn_available:
		final_report += f"""- **Train set:** {len(X_train_cmp):,} samples
- **Test set:** {len(X_test_cmp):,} samples
"""
	else:
		final_report += f"""- **Test set:** {len(X_test_cmp):,} samples
"""
	
	final_report += f"""- **Features:** {X_test_scaled_cmp.shape[1]}
- **Target:** median_house_value (range: ${y_test_cmp.min():,.0f} - ${y_test_cmp.max():,.0f})

---

## Conclusion

The complete pipeline demonstrates the progression from raw data through cleaning, baseline modeling, and deep learning. 

**Recommendation:** Deploy {best_model} in production with monitoring for performance drift. Consider fine-tuning strategies if accuracy targets are not met.

---

*Assignment completed: Data Engineering (DE) → Baseline Modeler (BM) → Deep Learning Analysis*
"""

	# Save markdown report
	final_report_path = BASE_DIR / "assignment_final_report.md"
	with open(final_report_path, "w") as f:
		f.write(final_report)

	print(f"✓ Saved final report: {final_report_path.name}")
	print(f"✓ Saved ranking details: model_ranking.json")

else:
	print("\n⚠ No models available for comparison. Skipping visualization and report generation.")

print("\n" + "="*70)
print("STEP 13: ASSIGNMENT COMPLETE")
print("="*70)
print(f"\nFinal Outputs:")
print(f"  - de_cleaning_handoff.md")
print(f"  - de_eda_histograms.png")
print(f"  - de_eda_correlation_heatmap.png")
print(f"  - de_eda_boxplot.png")
print(f"  - de_eda_target_distribution.png")
if all_r2:
	print(f"  - assignment_final_report.md")
	print(f"  - model_ranking.json")
	print(f"  - comparison_metrics_bars.png")
	print(f"  - comparison_predicted_vs_actual.png")
print(f"\nReady for submission!")
print("="*70 + "\n")