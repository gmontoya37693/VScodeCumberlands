# MSDS-534 Project 3 - Group 2 (Housing) - Final Analysis Report

**Generated:** 2026-03-22 08:11:33

---

## Summary

Complete pipeline analysis: Data Engineering → Baseline Linear Models → Deep Neural Networks

**Best Performing Model:** **Tuned DNN** (R² = 0.757423)

---

## Model Comparison Results

| Model | R² Score | RMSE ($) | MAE ($) |
|-------|----------|----------|---------|
| Single Linear | 0.458859 | 84,209 | 62,991 |
| Multi Linear | 0.630251 | 69,608 | 50,398 |
| Baseline DNN | 0.632450 | 69,400 | 49,357 |
| Tuned DNN | 0.757423 | 56,380 | 37,664 |

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

**Single-Variable Linear Regression**
- Features: median_income only
- R²: 0.458859
- RMSE: $84,209
- Purpose: Simplest baseline

**Multi-Variable Linear Regression**
- Features: All 15 engineered/encoded features
- R²: 0.630251
- RMSE: $69,608
- Purpose: Linear standard model for comparison



---

## Phase 3: Deep Neural Networks

**Baseline DNN**
- R²: 0.632450
- RMSE: $69,400
- Purpose: First DNN attempt, foundational network

**Tuned DNN**
- R²: 0.757423
- RMSE: $56,380
- Improvement over Baseline DNN: +0.124974
- Purpose: Optimized architecture for better generalization

---

## Key Insights

1. **Linear vs DNN Trade-off**
   - Linear models: Fast, interpretable, bounded performance
   - DNNs: Slower, black-box, potential for better accuracy

2. **Best Performer Analysis**
   - Model: Tuned DNN
   - R² Score: 0.757423 (explains 75.74% of variance)
   - Recommendation: Use for housing price predictions

3. **Remaining Error**
   - Even best model leaves ~24.3% of variance unexplained
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

- **Train set:** 16,512 samples
- **Test set:** 4,128 samples
- **Features:** 15
- **Target:** median_house_value (range: $14,999 - $500,001)

---

## Conclusion

The complete pipeline demonstrates the progression from raw data through cleaning, baseline modeling, and deep learning. 

**Recommendation:** Deploy Tuned DNN in production with monitoring for performance drift. Consider fine-tuning strategies if accuracy targets are not met.

---

*Assignment completed: Data Engineering (DE) → Baseline Modeler (BM) → Deep Learning Analysis*
