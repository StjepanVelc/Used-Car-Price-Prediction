from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_cleaning import run_cleaning
from src.data_preprocessing import TARGET_COLUMN, build_preprocessing_artifacts
from src.feature_engineering import run_feature_engineering


def evaluate_model(
    model_path: Path,
    df: pd.DataFrame,
    report_output_path: Path,
    examples_output_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    artifacts = build_preprocessing_artifacts(df, target_column=TARGET_COLUMN)

    X = df[artifacts.feature_columns]
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    examples_output_path.parent.mkdir(parents=True, exist_ok=True)

    with report_output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    examples = pd.DataFrame(
        {
            "actual_price": y_test.reset_index(drop=True),
            "predicted_price": pd.Series(predictions),
        }
    )
    examples["error"] = examples["predicted_price"] - examples["actual_price"]
    examples.head(20).to_csv(examples_output_path, index=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved car price model.")
    parser.add_argument("--raw-input", default="data/cars.csv", help="Path to raw CSV file.")
    parser.add_argument("--clean-output", default="data/cars_clean.csv", help="Path to save cleaned CSV.")
    parser.add_argument("--features-output", default="data/cars_features.csv", help="Path to save featured CSV.")
    parser.add_argument("--model-path", default="models/car_price_model.joblib", help="Path to saved model.")
    parser.add_argument(
        "--report-output",
        default="reports/evaluation_metrics.json",
        help="Path to save evaluation metrics JSON.",
    )
    parser.add_argument(
        "--examples-output",
        default="reports/evaluation_examples.csv",
        help="Path to save evaluation examples CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_cleaning(Path(args.raw_input), Path(args.clean_output))
    featured_df = run_feature_engineering(Path(args.clean_output), Path(args.features_output))

    metrics = evaluate_model(
        model_path=Path(args.model_path),
        df=featured_df,
        report_output_path=Path(args.report_output),
        examples_output_path=Path(args.examples_output),
    )

    print("Model evaluation complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
