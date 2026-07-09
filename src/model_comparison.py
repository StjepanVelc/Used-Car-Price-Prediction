from __future__ import annotations

import argparse
from pathlib import Path

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


def compare_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    artifacts = build_preprocessing_artifacts(df, target_column=TARGET_COLUMN)

    X = df[artifacts.feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    candidate_models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "decision_tree": DecisionTreeRegressor(random_state=random_state, max_depth=18, min_samples_leaf=2),
        "random_forest": RandomForestRegressor(
            n_estimators=150, random_state=random_state, n_jobs=-1, min_samples_leaf=2
        ),
    }

    rows = []
    for name, model in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", artifacts.preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5

        rows.append(
            {
                "model": name,
                "mae": mean_absolute_error(y_test, predictions),
                "mse": mse,
                "rmse": rmse,
                "r2": r2_score(y_test, predictions),
            }
        )

    results = pd.DataFrame(rows).sort_values(by="mae", ascending=True).reset_index(drop=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple regression models.")
    parser.add_argument("--raw-input", default="data/cars.csv", help="Path to raw CSV file.")
    parser.add_argument("--clean-output", default="data/cars_clean.csv", help="Path to save cleaned CSV.")
    parser.add_argument("--features-output", default="data/cars_features.csv", help="Path to save featured CSV.")
    parser.add_argument(
        "--report-output",
        default="reports/model_comparison.csv",
        help="Path to save model comparison report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cleaning(Path(args.raw_input), Path(args.clean_output))
    featured_df = run_feature_engineering(Path(args.clean_output), Path(args.features_output))

    results = compare_models(featured_df)
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_path, index=False)

    print("Model comparison complete:")
    print(results.to_string(index=False))
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
