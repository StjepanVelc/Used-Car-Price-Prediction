from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


RAW_TO_STANDARD_COLUMNS = {
    "mileage(kilometers)": "mileage_km",
    "volume(cm3)": "engine_volume_cm3",
}


def clean_cars_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the raw cars dataset."""
    data = df.copy()
    data = data.rename(columns=RAW_TO_STANDARD_COLUMNS)

    # Normalize string columns for more consistent categories.
    object_columns = data.select_dtypes(include=["object"]).columns
    for column in object_columns:
        data[column] = data[column].astype(str).str.strip().str.lower()

    numeric_columns = ["priceUSD", "year", "mileage_km", "engine_volume_cm3"]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.drop_duplicates().reset_index(drop=True)

    current_year = datetime.now().year
    data = data[(data["year"] >= 1980) & (data["year"] <= current_year + 1)]
    data = data[(data["priceUSD"] > 0) & (data["mileage_km"] >= 0)]
    data = data[(data["engine_volume_cm3"].isna()) | (data["engine_volume_cm3"] > 0)]

    # Remove extreme outliers in target and mileage using robust quantiles.
    for column in ["priceUSD", "mileage_km"]:
        lower_q = data[column].quantile(0.01)
        upper_q = data[column].quantile(0.99)
        data = data[(data[column] >= lower_q) & (data[column] <= upper_q)]

    for column in ["drive_unit", "segment"]:
        if column in data.columns:
            data[column] = data[column].fillna("unknown")

    if "engine_volume_cm3" in data.columns:
        data["engine_volume_cm3"] = data["engine_volume_cm3"].fillna(data["engine_volume_cm3"].median())

    data = data.dropna(subset=["priceUSD"]).reset_index(drop=True)
    return data


def run_cleaning(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    cleaned_df = clean_cars_data(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    return cleaned_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean used car price dataset.")
    parser.add_argument("--input", default="data/cars.csv", help="Path to raw CSV file.")
    parser.add_argument("--output", default="data/cars_clean.csv", help="Path to save cleaned CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned_df = run_cleaning(Path(args.input), Path(args.output))
    print(f"Cleaning complete. Rows: {len(cleaned_df)}, Columns: {cleaned_df.shape[1]}")


if __name__ == "__main__":
    main()
