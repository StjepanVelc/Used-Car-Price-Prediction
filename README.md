# Used Car Price Prediction (Regression)

This project builds a complete machine learning regression workflow to predict used car prices in USD (`priceUSD`) from listing attributes such as make, model, year, mileage, fuel type, engine volume, transmission, drivetrain, and segment.

The repository follows an end-to-end pipeline:

EDA -> data cleaning -> feature engineering -> preprocessing -> model comparison -> final training -> evaluation.

## Project Structure

```text
car-price-prediction/
├── data/
│   ├── cars.csv
│   ├── cars_clean.csv
│   └── cars_features.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_comparison.py
│   ├── model_training.py
│   └── model_evaluation.py
├── models/
│   └── car_price_model.joblib
├── reports/
│   ├── model_comparison.csv
│   ├── training_metrics.json
│   ├── evaluation_metrics.json
│   ├── sample_predictions.csv
│   └── evaluation_examples.csv
├── requirements.txt
└── README.md
```

## Dataset

Input dataset: `cars.csv`  
Target variable: `priceUSD`

Main raw columns include:

- `make`, `model`, `year`, `condition`
- `mileage(kilometers)`, `fuel_type`, `volume(cm3)`
- `color`, `transmission`, `drive_unit`, `segment`

## Implemented Pipeline

1. Data Cleaning (`src/data_cleaning.py`)
- Renames problematic column names (`mileage(kilometers)` -> `mileage_km`, `volume(cm3)` -> `engine_volume_cm3`)
- Normalizes categorical text values
- Converts relevant columns to numeric
- Removes duplicates and invalid rows (year, non-positive prices, negative mileage)
- Applies quantile-based outlier filtering for price and mileage
- Handles missing values in key fields

2. Feature Engineering (`src/feature_engineering.py`)
- `car_age`
- `mileage_per_year`
- `engine_volume_liters`
- `is_newer_car`
- `is_high_mileage`
- `brand_model`

3. Preprocessing (`src/data_preprocessing.py`)
- Numeric pipeline: imputation + scaling
- Categorical pipeline: imputation + one-hot encoding
- Unified transformer through `ColumnTransformer`

4. Model Comparison (`src/model_comparison.py`)
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

5. Final Training (`src/model_training.py`)
- Trains selected model (default: `random_forest`)
- Saves pipeline + model to `models/car_price_model.joblib`
- Saves training metrics and sample predictions

6. Evaluation (`src/model_evaluation.py`)
- Evaluates the saved model on the test split
- Reports MAE, MSE, RMSE, and R2
- Exports real vs predicted examples with error values

## Installation

Use Python 3.11+ (tested in a virtual environment).

```bash
pip install -r requirements.txt
```

## How To Run

Run from the project root:

```bash
python -m src.model_comparison
python -m src.model_training --model random_forest
python -m src.model_evaluation
```

## Evaluation Metrics

The project reports:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R2 (coefficient of determination)

Interpretation example: if `MAE = 1200`, the model prediction is off by about 1200 USD on average.

## Current Results (Latest Run)

Final selected model: `random_forest`

Test-set metrics:

- MAE: `958.57` USD
- MSE: `2,790,035.68`
- RMSE: `1,670.34` USD
- R2: `0.9217`

Model comparison on the same split:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| random_forest | 958.57 | 1670.34 | 0.9217 |
| decision_tree | 1145.90 | 1984.06 | 0.8896 |
| linear_regression | 1623.32 | 2469.63 | 0.8289 |
| ridge | 1625.21 | 2473.09 | 0.8285 |

Result takeaway: `random_forest` achieved the best overall performance (lowest MAE/RMSE and highest R2), so it is kept as the final model in `models/car_price_model.joblib`.

## Output Artifacts

- Trained model: `models/car_price_model.joblib`
- Model comparison: `reports/model_comparison.csv`
- Training metrics: `reports/training_metrics.json`
- Evaluation metrics: `reports/evaluation_metrics.json`
- Prediction examples: `reports/sample_predictions.csv`, `reports/evaluation_examples.csv`

## Reproducibility Notes

- The scripts use a fixed `random_state=42` for consistent splits and model behavior.
- Raw data remains in `data/cars.csv`; intermediate cleaned and engineered files are generated automatically.
- Prefer module execution (`python -m src...`) to ensure package imports resolve correctly.

## Submission Checklist

- Public GitHub repository
- All required scripts and notebook included
- Saved final model (`.joblib`)
- Clear README with run instructions
- Re-runnable project from scratch on another machine
