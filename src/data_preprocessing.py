from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "priceUSD"


@dataclass
class PreprocessingArtifacts:
    target_column: str
    feature_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    preprocessor: ColumnTransformer


def build_preprocessing_artifacts(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> PreprocessingArtifacts:
    feature_columns = [c for c in df.columns if c != target_column]

    X = df[feature_columns]
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    return PreprocessingArtifacts(
        target_column=target_column,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        preprocessor=preprocessor,
    )
