"""Subject-level state profiles for baseline/stress interpretation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import prepare_binary_subset

METADATA_COLUMNS = {"subject", "condition", "label", "window_start_s", "window_end_s"}

FEATURE_GROUPS = {
    "acc": {
        "prefix": "acc_",
        "label": "ACC / movement-related features",
        "short_label": "movement-related features",
    },
    "bvp": {
        "prefix": "bvp_",
        "label": "BVP / cardiovascular-related features",
        "short_label": "cardiovascular-related features",
    },
    "eda": {
        "prefix": "eda_",
        "label": "EDA / electrodermal-related features",
        "short_label": "electrodermal-related features",
    },
    "temp": {
        "prefix": "temp_",
        "label": "TEMP / skin-temperature-related features",
        "short_label": "skin-temperature-related features",
    },
}


def get_numeric_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding labels and window metadata."""

    return [
        column
        for column in feature_table.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(feature_table[column])
    ]


def get_feature_group(feature_name: str) -> str:
    """Map a feature name to an interpretable sensor feature group."""

    for group_key, group_info in FEATURE_GROUPS.items():
        if feature_name.startswith(str(group_info["prefix"])):
            return str(group_info["label"])
    return "Other feature group"


def get_feature_group_key(feature_name: str) -> str:
    """Map a feature name to its compact group key."""

    for group_key, group_info in FEATURE_GROUPS.items():
        if feature_name.startswith(str(group_info["prefix"])):
            return group_key
    return "other"


def build_standardized_binary_features(feature_table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return baseline/stress rows with z-standardized feature columns.

    Standardization is computed across the available baseline/stress windows in
    the current run. This keeps group summaries comparable while remaining local
    and reproducible.
    """

    binary_table = prepare_binary_subset(feature_table).reset_index(drop=True)
    feature_columns = get_numeric_feature_columns(binary_table)

    standardized = binary_table.copy()
    means = binary_table[feature_columns].mean()
    stds = binary_table[feature_columns].std(ddof=0).replace(0, np.nan)
    standardized[feature_columns] = (binary_table[feature_columns] - means) / stds
    return standardized, feature_columns


def _safe_mean(frame: pd.DataFrame, columns: list[str]) -> float:
    """Compute a skip-na mean across selected columns."""

    if frame.empty or not columns:
        return np.nan
    values = frame[columns].to_numpy(dtype=float)
    if np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


def build_subject_feature_changes(feature_table: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Find the largest descriptive stress-minus-baseline feature changes per subject."""

    binary_table = prepare_binary_subset(feature_table).reset_index(drop=True)
    standardized, feature_columns = build_standardized_binary_features(feature_table)
    rows: list[dict[str, object]] = []

    for subject_id in sorted(binary_table["subject"].unique()):
        raw_subject = binary_table[binary_table["subject"] == subject_id]
        z_subject = standardized[standardized["subject"] == subject_id]

        baseline_raw = raw_subject[raw_subject["condition"] == "baseline"]
        stress_raw = raw_subject[raw_subject["condition"] == "stress"]
        baseline_z = z_subject[z_subject["condition"] == "baseline"]
        stress_z = z_subject[z_subject["condition"] == "stress"]

        if baseline_raw.empty or stress_raw.empty:
            continue

        baseline_raw_means = baseline_raw[feature_columns].mean()
        stress_raw_means = stress_raw[feature_columns].mean()
        baseline_z_means = baseline_z[feature_columns].mean()
        stress_z_means = stress_z[feature_columns].mean()
        delta_z = stress_z_means - baseline_z_means
        delta_raw = stress_raw_means - baseline_raw_means

        ranked_features = delta_z.abs().sort_values(ascending=False).head(top_n).index
        for rank, feature_name in enumerate(ranked_features, start=1):
            rows.append(
                {
                    "subject_id": subject_id,
                    "rank": rank,
                    "feature": feature_name,
                    "feature_group": get_feature_group(feature_name),
                    "baseline_mean_raw": baseline_raw_means[feature_name],
                    "stress_mean_raw": stress_raw_means[feature_name],
                    "stress_minus_baseline_raw": delta_raw[feature_name],
                    "baseline_mean_z": baseline_z_means[feature_name],
                    "stress_mean_z": stress_z_means[feature_name],
                    "stress_minus_baseline_z": delta_z[feature_name],
                    "change_type": "descriptive difference",
                    "interpretation_note": "Descriptive feature change under the WESAD protocol; not causal and not clinical.",
                }
            )

    return pd.DataFrame(rows)


def _format_top_feature_changes(subject_changes: pd.DataFrame) -> str:
    """Format top feature changes for the profile table and markdown cards."""

    if subject_changes.empty:
        return "No complete baseline/stress feature changes available"

    parts = []
    for _, row in subject_changes.sort_values("rank").iterrows():
        delta = row["stress_minus_baseline_z"]
        parts.append(f"{row['feature']} ({delta:+.2f} standardized difference)")
    return "; ".join(parts)


def _build_interpretation(subject_id: str, dominant_group: str | None, has_complete_conditions: bool) -> str:
    """Create a cautious subject-level interpretation sentence."""

    if not has_complete_conditions:
        return (
            f"Under the WESAD protocol, {subject_id} does not have complete baseline/stress windows "
            "for a subject-level baseline/stress distinction; this is not a clinical interpretation."
        )

    group_text = dominant_group or "the available wearable feature groups"
    return (
        f"Under the WESAD protocol, {subject_id} showed a baseline/stress distinction with the largest "
        f"descriptive shift in {group_text}; this may reflect a psychophysiological strain indicator, "
        "not a clinical interpretation."
    )


def build_subject_profiles(feature_table: pd.DataFrame, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one subject-level profile row per subject."""

    binary_table = prepare_binary_subset(feature_table).reset_index(drop=True)
    standardized, feature_columns = build_standardized_binary_features(feature_table)
    top_changes = build_subject_feature_changes(feature_table, top_n=top_n)
    rows: list[dict[str, object]] = []

    for subject_id in sorted(binary_table["subject"].unique()):
        raw_subject = binary_table[binary_table["subject"] == subject_id]
        z_subject = standardized[standardized["subject"] == subject_id]
        baseline_z = z_subject[z_subject["condition"] == "baseline"]
        stress_z = z_subject[z_subject["condition"] == "stress"]

        baseline_windows = int((raw_subject["condition"] == "baseline").sum())
        stress_windows = int((raw_subject["condition"] == "stress").sum())
        has_complete_conditions = baseline_windows > 0 and stress_windows > 0

        profile_row: dict[str, object] = {
            "subject_id": subject_id,
            "baseline_windows": baseline_windows,
            "stress_windows": stress_windows,
        }

        group_deltas: dict[str, float] = {}
        for group_key, group_info in FEATURE_GROUPS.items():
            prefix = str(group_info["prefix"])
            group_columns = [column for column in feature_columns if column.startswith(prefix)]

            baseline_mean = _safe_mean(baseline_z, group_columns)
            stress_mean = _safe_mean(stress_z, group_columns)
            delta = stress_mean - baseline_mean if pd.notna(baseline_mean) and pd.notna(stress_mean) else np.nan
            group_deltas[group_key] = delta

            profile_row[f"baseline_{group_key}_feature_mean_z"] = baseline_mean
            profile_row[f"stress_{group_key}_feature_mean_z"] = stress_mean
            profile_row[f"stress_minus_baseline_{group_key}_feature_mean_z"] = delta

        finite_deltas = {
            key: value for key, value in group_deltas.items() if pd.notna(value)
        }
        if finite_deltas:
            dominant_key = max(finite_deltas, key=lambda key: abs(finite_deltas[key]))
            dominant_group = str(FEATURE_GROUPS[dominant_key]["label"])
        else:
            dominant_key = ""
            dominant_group = None

        subject_changes = top_changes[top_changes["subject_id"] == subject_id]
        profile_row["dominant_changed_feature_group"] = dominant_group or ""
        profile_row["dominant_changed_feature_group_key"] = dominant_key
        profile_row["top_changed_features"] = _format_top_feature_changes(subject_changes)
        profile_row["interpretation"] = _build_interpretation(subject_id, dominant_group, has_complete_conditions)
        rows.append(profile_row)

    return pd.DataFrame(rows), top_changes


def build_subject_profile_cards(profiles: pd.DataFrame) -> str:
    """Create markdown subject profile cards for local reports."""

    lines = [
        "# Subject-Level State Profile Cards",
        "",
        "These cards summarize descriptive baseline/stress distinctions under the WESAD protocol. "
        "They are not clinical or medical conclusions.",
        "",
    ]

    for _, row in profiles.sort_values("subject_id").iterrows():
        lines.extend(
            [
                f"## {row['subject_id']}",
                "",
                f"- Baseline windows: {row['baseline_windows']}",
                f"- Stress windows: {row['stress_windows']}",
                f"- Largest changed feature group: {row['dominant_changed_feature_group'] or 'Not available'}",
                f"- Top changed features: {row['top_changed_features']}",
                f"- Interpretation: {row['interpretation']}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def save_subject_profiles(feature_table: pd.DataFrame, tables_dir: Path, reports_dir: Path) -> dict[str, object]:
    """Save subject profile table and markdown profile cards."""

    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    profiles, top_changes = build_subject_profiles(feature_table)
    profiles_path = tables_dir / "subject_state_profiles.csv"
    cards_path = reports_dir / "subject_profile_cards.md"

    profiles.to_csv(profiles_path, index=False)
    cards_path.write_text(build_subject_profile_cards(profiles), encoding="utf-8")

    return {
        "profiles": profiles,
        "top_feature_changes": top_changes,
        "profiles_path": profiles_path,
        "cards_path": cards_path,
    }
