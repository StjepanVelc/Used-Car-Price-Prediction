from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_cleaning import run_cleaning
from src.data_preprocessing import TARGET_COLUMN, build_preprocessing_artifacts
from src.feature_engineering import run_feature_engineering


def get_model(model_name: str, random_state: int):
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "decision_tree": DecisionTreeRegressor(random_state=random_state, max_depth=18, min_samples_leaf=2),
        "random_forest": RandomForestRegressor(
            n_estimators=150, random_state=random_state, n_jobs=-1, min_samples_leaf=2
        ),
    }
    if model_name not in models:
        valid = ", ".join(models.keys())
        raise ValueError(f"Unknown model '{model_name}'. Valid models: {valid}")
    return models[model_name]


def train_and_save_model(
    df: pd.DataFrame,
    model_name: str,
    model_output_path: Path,
    metrics_output_path: Path,
    predictions_output_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    artifacts = build_preprocessing_artifacts(df, target_column=TARGET_COLUMN)

    X = df[artifacts.feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = get_model(model_name, random_state)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", artifacts.preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "model": model_name,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
        "test_size": test_size,
        "random_state": random_state,
    }

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_output_path)

    with metrics_output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    comparison_df = pd.DataFrame(
        {
            "actual_price": y_test.reset_index(drop=True),
            "predicted_price": pd.Series(predictions),
        }
    )
    comparison_df["error"] = comparison_df["predicted_price"] - comparison_df["actual_price"]
    comparison_df.head(30).to_csv(predictions_output_path, index=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save final car price model.")
    parser.add_argument("--raw-input", default="data/cars.csv", help="Path to raw CSV file.")
    parser.add_argument("--clean-output", default="data/cars_clean.csv", help="Path to save cleaned CSV.")
    parser.add_argument("--features-output", default="data/cars_features.csv", help="Path to save featured CSV.")
    parser.add_argument("--model", default="random_forest", help="Model name.")
    parser.add_argument(
        "--model-output", default="models/car_price_model.joblib", help="Path to save trained model."
    )
    parser.add_argument(
        "--metrics-output",
        default="reports/training_metrics.json",
        help="Path to save training metrics JSON.",
    )
    parser.add_argument(
        "--predictions-output",
        default="reports/sample_predictions.csv",
        help="Path to save sample predictions CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_cleaning(Path(args.raw_input), Path(args.clean_output))
    featured_df = run_feature_engineering(Path(args.clean_output), Path(args.features_output))

    metrics = train_and_save_model(
        df=featured_df,
        model_name=args.model,
        model_output_path=Path(args.model_output),
        metrics_output_path=Path(args.metrics_output),
        predictions_output_path=Path(args.predictions_output),
    )

    print("Model training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
