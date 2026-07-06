"""Baseline models and evaluation split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


@dataclass
class ModelRunResult:
    """Predictions and metadata for one trained model configuration."""

    model_name: str
    y_true: pd.Series
    y_pred: pd.Series
    subjects: pd.Series
    evaluation_mode: str
    window_metadata: pd.DataFrame
    feature_columns: list[str]
    y_probability_stress: pd.Series | None = None


def build_models() -> dict[str, Pipeline]:
    """Create baseline sklearn pipelines."""

    logistic_regression = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def prepare_binary_subset(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Keep only baseline vs stress rows for the first binary demo."""

    subset = feature_table[feature_table["condition"].isin(["baseline", "stress"])].copy()
    if subset.empty:
        raise ValueError("No baseline/stress windows available for binary modeling")
    return subset


def get_model_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Return numeric feature columns used by the binary models."""

    return [
        column
        for column in feature_table.columns
        if column not in {"subject", "condition", "label", "window_start_s", "window_end_s"}
    ]


def prepare_binary_dataset(feature_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare feature matrix, labels, and subject groups for binary modeling."""

    subset = prepare_binary_subset(feature_table)
    feature_columns = [
        column for column in get_model_feature_columns(subset) if pd.api.types.is_numeric_dtype(subset[column])
    ]
    x = subset[feature_columns]
    y = subset["label"].astype(int)
    groups = subset["subject"]
    return x, y, groups


def evaluate_models(feature_table: pd.DataFrame) -> list[ModelRunResult]:
    """Run either subject-aware or internal sanity-check evaluation."""

    subset = prepare_binary_subset(feature_table)
    x, y, groups = prepare_binary_dataset(subset)
    unique_subjects = groups.nunique()
    feature_columns = list(x.columns)
    window_metadata = subset[["subject", "condition", "window_start_s", "window_end_s"]].reset_index(drop=True)

    if unique_subjects == 1:
        # With one subject, this is only a quick internal check, not a generalization test.
        splitter = StratifiedKFold(n_splits=min(3, y.value_counts().min()), shuffle=True, random_state=RANDOM_STATE)
        evaluation_mode = "single-subject stratified cross-validation (internal sanity check only)"
        split_kwargs = {}
    else:
        splitter = LeaveOneGroupOut()
        evaluation_mode = "leave-one-subject-out"
        split_kwargs = {"groups": groups}

    results: list[ModelRunResult] = []
    for model_name, pipeline in build_models().items():
        probabilities = None
        if model_name == "logistic_regression":
            probability_matrix = cross_val_predict(pipeline, x, y, cv=splitter, method="predict_proba", **split_kwargs)
            probabilities = pd.Series(probability_matrix[:, 1])
            predictions = (probabilities >= 0.5).astype(int)
        else:
            predictions = cross_val_predict(pipeline, x, y, cv=splitter, method="predict", **split_kwargs)

        results.append(
            ModelRunResult(
                model_name=model_name,
                y_true=y.reset_index(drop=True),
                y_pred=pd.Series(predictions),
                subjects=groups.reset_index(drop=True),
                evaluation_mode=evaluation_mode,
                window_metadata=window_metadata.copy(),
                feature_columns=feature_columns,
                y_probability_stress=probabilities,
            )
        )
    return results
