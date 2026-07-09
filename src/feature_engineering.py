from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional predictive features from base columns."""
    data = df.copy()

    current_year = datetime.now().year
    data["car_age"] = (current_year - data["year"]).clip(lower=0)

    safe_age = data["car_age"].replace(0, 1)
    data["mileage_per_year"] = data["mileage_km"] / safe_age
    data["mileage_per_year"] = data["mileage_per_year"].replace([np.inf, -np.inf], np.nan)
    data["mileage_per_year"] = data["mileage_per_year"].fillna(data["mileage_per_year"].median())

    data["engine_volume_liters"] = data["engine_volume_cm3"] / 1000.0
    data["is_newer_car"] = (data["year"] >= 2018).astype(int)
    data["is_high_mileage"] = (data["mileage_km"] >= 200000).astype(int)
    data["brand_model"] = data["make"].astype(str) + "_" + data["model"].astype(str)

    return data


def run_feature_engineering(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    featured_df = engineer_features(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(output_path, index=False)
    return featured_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature engineering for used car prices.")
    parser.add_argument("--input", default="data/cars_clean.csv", help="Path to cleaned CSV file.")
    parser.add_argument("--output", default="data/cars_features.csv", help="Path to save featured CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    featured_df = run_feature_engineering(Path(args.input), Path(args.output))
    print(f"Feature engineering complete. Rows: {len(featured_df)}, Columns: {featured_df.shape[1]}")


if __name__ == "__main__":
    main()
