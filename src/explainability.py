"""Lightweight explanation and ambiguity reporting for the demo pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .modeling import ModelRunResult, build_models, prepare_binary_dataset
from .subject_profiles import (
    FEATURE_GROUPS,
    build_subject_feature_changes,
    get_feature_group,
    get_feature_group_key,
)


def build_feature_importance_summary(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize logistic-regression feature weights by interpretable feature group."""

    x, y, _ = prepare_binary_dataset(feature_table)
    pipeline = build_models()["logistic_regression"]
    pipeline.fit(x, y)

    classifier = pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    feature_weights = pd.DataFrame(
        {
            "feature": list(x.columns),
            "model_associated_feature_weight": coefficients,
        }
    )
    feature_weights["absolute_model_weight"] = feature_weights["model_associated_feature_weight"].abs()
    feature_weights["feature_group"] = feature_weights["feature"].map(get_feature_group)
    feature_weights["feature_group_key"] = feature_weights["feature"].map(get_feature_group_key)

    rows: list[dict[str, object]] = []
    for group_key in FEATURE_GROUPS:
        group_label = str(FEATURE_GROUPS[group_key]["label"])
        group_frame = feature_weights[feature_weights["feature_group_key"] == group_key]
        if group_frame.empty:
            continue

        top_feature = group_frame.sort_values("absolute_model_weight", ascending=False).iloc[0]
        rows.append(
            {
                "feature_group": group_label,
                "feature_group_key": group_key,
                "number_of_features": len(group_frame),
                "total_absolute_model_weight": group_frame["absolute_model_weight"].sum(),
                "mean_absolute_model_weight": group_frame["absolute_model_weight"].mean(),
                "signed_model_weight_sum": group_frame["model_associated_feature_weight"].sum(),
                "top_weighted_feature": top_feature["feature"],
                "top_weighted_feature_weight": top_feature["model_associated_feature_weight"],
                "contribution_type": "model-associated feature weight",
                "analysis_basis": "Logistic regression coefficients after median imputation and standard scaling.",
                "interpretation_note": "Feature contribution is model-associated, not causal and not clinical.",
            }
        )

    summary = pd.DataFrame(rows).sort_values("total_absolute_model_weight", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def build_uncertain_windows(
    model_results: list[ModelRunResult],
    lower_probability: float = 0.4,
    upper_probability: float = 0.6,
) -> pd.DataFrame:
    """Identify windows with logistic-regression probabilities close to 0.5."""

    columns = [
        "model_name",
        "subject_id",
        "condition",
        "window_start_s",
        "window_end_s",
        "true_label",
        "predicted_label",
        "predicted_probability_stress",
        "uncertainty_band",
        "interpretation_note",
    ]

    logistic_result = next(
        (
            result
            for result in model_results
            if result.model_name == "logistic_regression" and result.y_probability_stress is not None
        ),
        None,
    )
    if logistic_result is None:
        return pd.DataFrame(columns=columns)

    frame = logistic_result.window_metadata.copy()
    frame = frame.rename(columns={"subject": "subject_id"})
    frame.insert(0, "model_name", logistic_result.model_name)
    frame["true_label"] = logistic_result.y_true.map({0: "baseline", 1: "stress"})
    frame["predicted_label"] = logistic_result.y_pred.map({0: "baseline", 1: "stress"})
    frame["predicted_probability_stress"] = logistic_result.y_probability_stress

    mask = frame["predicted_probability_stress"].between(lower_probability, upper_probability, inclusive="both")
    uncertain = frame.loc[mask].copy()
    uncertain["uncertainty_band"] = f"{lower_probability:.2f}-{upper_probability:.2f}"
    uncertain["interpretation_note"] = (
        "Ambiguous model output near the binary decision boundary; this is not a judgment about the user "
        "and is not clinical."
    )

    return uncertain[columns].sort_values(["subject_id", "window_start_s"]).reset_index(drop=True)


def save_explanation_outputs(
    feature_table: pd.DataFrame,
    model_results: list[ModelRunResult],
    tables_dir: Path,
) -> dict[str, object]:
    """Save feature contribution, subject feature-change, and uncertainty tables."""

    tables_dir.mkdir(parents=True, exist_ok=True)

    feature_importance = build_feature_importance_summary(feature_table)
    subject_changes = build_subject_feature_changes(feature_table)
    uncertain_windows = build_uncertain_windows(model_results)

    feature_importance_path = tables_dir / "feature_importance_summary.csv"
    subject_changes_path = tables_dir / "subject_top_feature_changes.csv"
    uncertain_windows_path = tables_dir / "uncertain_windows.csv"

    feature_importance.to_csv(feature_importance_path, index=False)
    subject_changes.to_csv(subject_changes_path, index=False)
    uncertain_windows.to_csv(uncertain_windows_path, index=False)

    return {
        "feature_importance": feature_importance,
        "subject_top_feature_changes": subject_changes,
        "uncertain_windows": uncertain_windows,
        "feature_importance_path": feature_importance_path,
        "subject_changes_path": subject_changes_path,
        "uncertain_windows_path": uncertain_windows_path,
    }
