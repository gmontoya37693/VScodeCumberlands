# ============================================================
# MSDS-534 PROJECT 3 - GROUP 2 (HOUSING)
# ASSIGNMENT COMPARISON: BASELINE LINEAR vs TUNED DNN
# ============================================================
# Student: German Montoya
# Role: Comparative Analysis
# Goal: Compare all 4 models (single linear, multi linear, 
#       baseline DNN, tuned DNN) on same test set
# ============================================================

# Part 1: Setup & Imports
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
	from tensorflow.keras.models import load_model
except ImportError:
	from keras.models import load_model

# Configuration
BASE_DIR = Path(__file__).resolve().parent
CLEANED_DATA_PATH = BASE_DIR / "cleaned_data_gm.csv"
BASELINE_DNN_PATH = BASE_DIR / "baseline_linear_model.h5"
TUNED_DNN_PATH = BASE_DIR / "tuned_dnn_model.h5"
TARGET_COLUMN = "median_house_value"

# ============================================================
# Part 2: Load Data & Prepare Train/Test Split
# ============================================================
print("\n" + "="*70)
print("LOADING AND PREPARING DATA")
print("="*70)

# Load cleaned data
clean_df = pd.read_csv(CLEANED_DATA_PATH)
print(f"✓ Loaded cleaned dataset: {clean_df.shape}")

# Separate features and target
X = clean_df.drop(columns=[TARGET_COLUMN])
y = clean_df[TARGET_COLUMN]

# Train/test split (80/20, deterministic seed)
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)
print(f"✓ Train set: {X_train.shape} | Test set: {X_test.shape}")

# Normalize using StandardScaler (fit on train only, apply to test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Normalized features (Z-score): train {X_train_scaled.shape}, test {X_test_scaled.shape}")

# ============================================================
# Part 3: Train Linear Regression Models
# ============================================================
print("\n" + "="*70)
print("TRAINING LINEAR REGRESSION MODELS")
print("="*70)

# Model 1: Single-variable linear (median_income only)
print("\n[Single-Variable Linear - median_income only]")
lr_single = LinearRegression()
X_train_single = X_train_scaled[:, X.columns.get_loc("median_income")].reshape(-1, 1)
X_test_single = X_test_scaled[:, X.columns.get_loc("median_income")].reshape(-1, 1)
lr_single.fit(X_train_single, y_train)

y_pred_single = lr_single.predict(X_test_single)
r2_single = r2_score(y_test, y_pred_single)
rmse_single = np.sqrt(mean_squared_error(y_test, y_pred_single))
mae_single = mean_absolute_error(y_test, y_pred_single)

print(f"  R² Score:  {r2_single:.6f}")
print(f"  RMSE:      ${rmse_single:,.2f}")
print(f"  MAE:       ${mae_single:,.2f}")

# Model 2: Multi-variable linear (all features)
print("\n[Multi-Variable Linear - all features]")
lr_multi = LinearRegression()
lr_multi.fit(X_train_scaled, y_train)

y_pred_multi = lr_multi.predict(X_test_scaled)
r2_multi = r2_score(y_test, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_multi))
mae_multi = mean_absolute_error(y_test, y_pred_multi)

print(f"  R² Score:  {r2_multi:.6f}")
print(f"  RMSE:      ${rmse_multi:,.2f}")
print(f"  MAE:       ${mae_multi:,.2f}")

# ============================================================
# Part 4: Load DNN Models
# ============================================================
print("\n" + "="*70)
print("LOADING DNN MODELS")
print("="*70)

baseline_dnn = None
tuned_dnn = None

if BASELINE_DNN_PATH.exists():
	baseline_dnn = load_model(BASELINE_DNN_PATH)
	print(f"✓ Loaded baseline DNN from {BASELINE_DNN_PATH.name}")
else:
	print(f"✗ Baseline DNN not found at {BASELINE_DNN_PATH}")

if TUNED_DNN_PATH.exists():
	tuned_dnn = load_model(TUNED_DNN_PATH)
	print(f"✓ Loaded tuned DNN from {TUNED_DNN_PATH.name}")
else:
	print(f"✗ Tuned DNN not found at {TUNED_DNN_PATH}")

# ============================================================
# Part 5: Evaluate All Models on Test Set
# ============================================================
print("\n" + "="*70)
print("EVALUATING ALL MODELS ON TEST SET")
print("="*70)

results = {
	"single_linear": {
		"name": "Single-Variable Linear",
		"r2": float(r2_single),
		"rmse": float(rmse_single),
		"mae": float(mae_single),
		"model_type": "sklearn.LinearRegression",
		"features": 1,
	},
	"multi_linear": {
		"name": "Multi-Variable Linear",
		"r2": float(r2_multi),
		"rmse": float(rmse_multi),
		"mae": float(mae_multi),
		"model_type": "sklearn.LinearRegression",
		"features": X_train_scaled.shape[1],
	}
}

if baseline_dnn is not None:
	print("\n[Baseline DNN]")
	y_pred_baseline_dnn = baseline_dnn.predict(X_test_scaled, verbose=0).flatten()
	r2_baseline_dnn = r2_score(y_test, y_pred_baseline_dnn)
	rmse_baseline_dnn = np.sqrt(mean_squared_error(y_test, y_pred_baseline_dnn))
	mae_baseline_dnn = mean_absolute_error(y_test, y_pred_baseline_dnn)
	
	print(f"  R² Score:  {r2_baseline_dnn:.6f}")
	print(f"  RMSE:      ${rmse_baseline_dnn:,.2f}")
	print(f"  MAE:       ${mae_baseline_dnn:,.2f}")
	
	results["baseline_dnn"] = {
		"name": "Baseline DNN",
		"r2": float(r2_baseline_dnn),
		"rmse": float(rmse_baseline_dnn),
		"mae": float(mae_baseline_dnn),
		"model_type": "keras.Sequential (baseline)",
		"features": X_train_scaled.shape[1],
	}

if tuned_dnn is not None:
	print("\n[Tuned DNN]")
	y_pred_tuned_dnn = tuned_dnn.predict(X_test_scaled, verbose=0).flatten()
	r2_tuned_dnn = r2_score(y_test, y_pred_tuned_dnn)
	rmse_tuned_dnn = np.sqrt(mean_squared_error(y_test, y_pred_tuned_dnn))
	mae_tuned_dnn = mean_absolute_error(y_test, y_pred_tuned_dnn)
	
	print(f"  R² Score:  {r2_tuned_dnn:.6f}")
	print(f"  RMSE:      ${rmse_tuned_dnn:,.2f}")
	print(f"  MAE:       ${mae_tuned_dnn:,.2f}")
	
	results["tuned_dnn"] = {
		"name": "Tuned DNN",
		"r2": float(r2_tuned_dnn),
		"rmse": float(rmse_tuned_dnn),
		"mae": float(mae_tuned_dnn),
		"model_type": "keras.Sequential (tuned)",
		"features": X_train_scaled.shape[1],
	}

# ============================================================
# Part 6: Generate Comparison Table & Analysis
# ============================================================
print("\n" + "="*70)
print("COMPARISON TABLE: ALL MODELS")
print("="*70)

comparison_df = pd.DataFrame({
	"Model": [results[k]["name"] for k in results.keys()],
	"R² Score": [results[k]["r2"] for k in results.keys()],
	"RMSE ($)": [results[k]["rmse"] for k in results.keys()],
	"MAE ($)": [results[k]["mae"] for k in results.keys()],
	"Type": [results[k]["model_type"] for k in results.keys()],
})

print("\n" + comparison_df.to_string(index=False))

# Identify best model
best_model_key = max(results.keys(), key=lambda k: results[k]["r2"])
best_model = results[best_model_key]

print("\n" + "-"*70)
print(f"BEST PERFORMING MODEL: {best_model['name']}")
print(f"  R² Score: {best_model['r2']:.6f}")
print(f"  RMSE:     ${best_model['rmse']:,.2f}")
print(f"  MAE:      ${best_model['mae']:,.2f}")
print("-"*70)

# ============================================================
# Part 7: Fine-Tuning Recommendations
# ============================================================
print("\n" + "="*70)
print("FINE-TUNING RECOMMENDATIONS FOR TUNED DNN")
print("="*70)

recommendations = """
If further optimization is desired, consider:

1. LEARNING RATE ADJUSTMENT
   - Try: 0.0001, 0.0005, 0.001 (if underfitting, increase)
   - Use learning rate scheduler for adaptive decay

2. BATCH SIZE TUNING
   - Try: 16, 32, 64, 128, 256, 512
   - Smaller batches (32): more frequent updates, higher variance
   - Larger batches (256): smoother gradients, faster computation

3. EPOCHS & EARLY STOPPING
   - Monitor validation loss during training
   - Set patience=10-20 to prevent overfitting
   - Plot train vs val loss to diagnose bias/variance

4. REGULARIZATION TECHNIQUES
   - Increase dropout (0.3 → 0.5): reduce overfitting
   - Add L1/L2 penalty: penalize large weights
   - Consider batch normalization: stabilize learning

5. ARCHITECTURE VARIATIONS
   - Add/remove hidden layers: test depth sensitivity
   - Modify hidden units (64, 128, 256, 512): test capacity
   - Experiment with activation functions (ReLU, ELU, SELU)

6. OPTIMIZER VARIANTS
   - Try: RMSprop, AdaGrad, Nadam in addition to Adam
   - Experiment with momentum (SGD with momentum)
   - Ensemble multiple optimizers on different folds

7. DATA AUGMENTATION / PREPROCESSING
   - Feature scaling alternatives (MinMax, Robust scaler)
   - Polynomial feature interactions (x₁*x₂, x₁²)
   - PCA for dimensionality reduction if overfitting persists

8. CROSS-VALIDATION
   - Use k-fold CV (k=5 or 10) for robust evaluation
   - Reduces variance in performance estimates
   - Helps identify model stability
"""

print(recommendations)

# ============================================================
# Part 8: Export Results
# ============================================================
print("\n" + "="*70)
print("EXPORTING RESULTS")
print("="*70)

# Save to JSON
results_json = {
	"timestamp": datetime.now().isoformat(),
	"dataset": "California Housing (Group 2)",
	"test_set_size": len(X_test),
	"train_set_size": len(X_train),
	"target_column": TARGET_COLUMN,
	"models": results,
	"best_model": {
		"name": best_model["name"],
		"r2": best_model["r2"],
		"rmse": best_model["rmse"],
		"mae": best_model["mae"],
	}
}

json_path = BASE_DIR / "assignment_comparison_results.json"
with open(json_path, "w") as f:
	json.dump(results_json, f, indent=2)

print(f"✓ Saved JSON results: {json_path.name}")

# Generate markdown report
markdown_report = f"""# MSDS-534 Project 3 - Group 2 (Housing) - Final Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

Comparative analysis of baseline linear regression models vs deep neural networks on California Housing dataset.

**Best Performing Model:** {best_model['name']}
- **R² Score:** {best_model['r2']:.6f}
- **RMSE:** ${best_model['rmse']:,.2f}
- **MAE:** ${best_model['mae']:,.2f}

---

## Model Comparison Table

| Model | R² Score | RMSE ($) | MAE ($) | Type |
|-------|----------|----------|---------|------|
"""

for model_key in results.keys():
	model = results[model_key]
	markdown_report += f"| {model['name']} | {model['r2']:.6f} | {model['rmse']:,.2f} | {model['mae']:,.2f} | {model['model_type']} |\n"

markdown_report += f"""
---

## Key Findings

1. **Single-Variable Linear vs Multi-Variable Linear**
   - Single: R² = {results['single_linear']['r2']:.6f}
   - Multi: R² = {results['multi_linear']['r2']:.6f}
   - Delta: {results['multi_linear']['r2'] - results['single_linear']['r2']:.6f}

"""

if "baseline_dnn" in results and "tuned_dnn" in results:
	markdown_report += f"""2. **Baseline DNN vs Tuned DNN**
   - Baseline: R² = {results['baseline_dnn']['r2']:.6f}
   - Tuned: R² = {results['tuned_dnn']['r2']:.6f}
   - Improvement: {results['tuned_dnn']['r2'] - results['baseline_dnn']['r2']:.6f}

3. **Best vs Worst Performer**
   - Best: {best_model['name']} (R² = {best_model['r2']:.6f})
   - Linear models focus: simplicity & interpretability
   - DNN models focus: capacity & non-linear patterns
"""

markdown_report += f"""
---

## Test Set Statistics

- Train set size: {len(X_train)}
- Test set size: {len(X_test)}
- Target variable: {TARGET_COLUMN}
- Features used: {X_train.shape[1]}

---

## Fine-Tuning Recommendations

See printed console output for detailed suggestions on:
- Learning rate adjustment
- Batch size tuning
- Early stopping & epochs
- Regularization techniques
- Architecture variations
- Optimizer selection
- Data preprocessing alternatives
- Cross-validation strategies

---

## Data Engineering Pipeline Summary

**Upstream (DE Phase):**
- Loaded raw dataset: 20,640 rows × 10 columns
- Handled missing values: median imputation for total_bedrooms (1,214 → 0)
- Engineered features: Rooms_per_Household, Bedrooms_per_Room
- One-hot encoded: ocean_proximity (5 binary columns)
- Final cleaned shape: 20,640 × 16

**See:** `de_cleaning_handoff.md` for full DE details

---

## Conclusion

All models were evaluated on the same normalized test set using identical train/test split (80/20, seed=42).

**Recommendation:** 
The {best_model['name']} demonstrates the strongest test-set performance with R² = {best_model['r2']:.6f}.
Consider this as the production candidate, with optional fine-tuning pathways documented above.

"""

report_path = BASE_DIR / "assignment_final_report.md"
with open(report_path, "w") as f:
	f.write(markdown_report)

print(f"✓ Saved markdown report: {report_path.name}")

print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)
print(f"\nOutputs:")
print(f"  1. {json_path.name}")
print(f"  2. {report_path.name}")
print(f"\nReady for submission!")
print("="*70 + "\n")
