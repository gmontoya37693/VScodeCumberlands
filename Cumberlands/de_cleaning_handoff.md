# DE Cleaning and Preparation Handoff

Generated: 2026-03-22 00:00:36

## Scope
- Role: Data Engineer (DE)
- Group: Group 2 (Housing Prices)
- Dataset: raw_data.csv
- Target variable: median_house_value

## Findings
- Raw shape: (20640, 10)
- Cleaned shape: (20640, 16)
- Duplicate rows removed: 0
- Missing total_bedrooms before imputation: 1214
- Missing total_bedrooms after imputation: 0
- Median used for total_bedrooms: 435.00

## Missing Values Table
| column | missing_count |
| --- | --- |
| total_bedrooms | 1214 |
| longitude | 0 |
| latitude | 0 |
| housing_median_age | 0 |
| total_rooms | 0 |
| population | 0 |
| households | 0 |
| median_income | 0 |
| median_house_value | 0 |
| ocean_proximity | 0 |

## Cleaning Log
| step | column | action | before | after | note |
| --- | --- | --- | --- | --- | --- |
| duplicates | all | drop_duplicates | 20640 | 20640 | Removed 0 duplicate rows. |
| imputation | total_bedrooms | median_fillna | 1214 | 0 | Median used: 435.00 |
| feature_engineering | Rooms_per_Household | create_ratio | not_present | present | Created total_rooms / households. |
| feature_engineering | Bedrooms_per_Room | create_ratio | not_present | present | Created total_bedrooms / total_rooms. |
| feature_engineering | Rooms_per_Household|Bedrooms_per_Room | fillna_zero | 0 | 0 | Filled ratio NaN values caused by zero denominators with 0.0. |
| encoding | ocean_proximity | one_hot_encode_drop_original | 12 | 16 | Applied one-hot encoding and dropped original categorical column. |

## Engineered and Encoded Features
- Engineered: Rooms_per_Household, Bedrooms_per_Room
- One-hot encoded columns from ocean_proximity: ocean_proximity_<1H OCEAN, ocean_proximity_INLAND, ocean_proximity_ISLAND, ocean_proximity_NEAR BAY, ocean_proximity_NEAR OCEAN

## Ocean Proximity Categories (Raw Values)
- Count: 5
- Values:
  - <1H OCEAN
  - INLAND
  - ISLAND
  - NEAR BAY
  - NEAR OCEAN

## Handoff to Baseline Modeler
- Use clean_df as input for train/test splitting.
- Use median_house_value as target label.
- Do not re-run imputation on total_bedrooms.
- Apply normalization by fitting only on training data.

## EDA Visualizations
- **Histograms**: Distribution of key features and target (6-panel grid)
  - File: `de_eda_histograms.png`
- **Correlation Heatmap**: Feature relationships and multicollinearity
  - File: `de_eda_correlation_heatmap.png`
- **Boxplot**: Outlier identification across all features
  - File: `de_eda_boxplot.png`
- **Target Distribution**: Detailed histogram of median_house_value with mean/median lines
  - File: `de_eda_target_distribution.png`


## EDA Summary Table (Mean and Std)
| feature | mean | std |
| --- | --- | --- |
| total_rooms | 2635.7630813953488 | 2181.615251582787 |
| total_bedrooms | 532.59375 | 411.0243253230345 |
| households | 499.5396802325581 | 382.3297528316099 |
| median_income | 3.8706710029069766 | 1.8998217179452732 |
| Rooms_per_Household | 5.428999742190376 | 2.4741731394243205 |
| Bedrooms_per_Room | 0.22020700139600205 | 0.20855615765625343 |
| median_house_value | 206855.81690891474 | 115395.6158744132 |

## BM Ready Summary
- Final cleaned shape: (20640, 16)
- Target column: median_house_value
- Feature count (excluding target): 15
- Encoded ocean proximity columns: ocean_proximity_<1H OCEAN, ocean_proximity_INLAND, ocean_proximity_ISLAND, ocean_proximity_NEAR BAY, ocean_proximity_NEAR OCEAN
- Cleaned dataset file: cleaned_data_gm.csv
- Cleaning log CSV file: de_cleaning_log.csv
- EDA visualization files: de_eda_histograms.png, de_eda_correlation_heatmap.png, de_eda_boxplot.png, de_eda_target_distribution.png
