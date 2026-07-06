"""Generate user-facing feedback card examples from local profile outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _subject_uncertainty_note(subject_id: str, profiles: pd.DataFrame, uncertain_windows: pd.DataFrame) -> str:
    """Summarize ambiguous model outputs for one subject."""

    profile_row = profiles[profiles["subject_id"] == subject_id]
    if profile_row.empty:
        total_windows = None
    else:
        total_windows = int(profile_row.iloc[0]["baseline_windows"] + profile_row.iloc[0]["stress_windows"])

    if uncertain_windows.empty:
        return "No logistic-regression windows for this subject fell in the 0.40-0.60 ambiguity band."

    subject_uncertain = uncertain_windows[uncertain_windows["subject_id"] == subject_id]
    if subject_uncertain.empty:
        return "No logistic-regression windows for this subject fell in the 0.40-0.60 ambiguity band."

    count_text = f"{len(subject_uncertain)}"
    if total_windows:
        count_text = f"{len(subject_uncertain)} of {total_windows}"
    return (
        f"{count_text} baseline/stress windows had ambiguous model output near 0.5; "
        "feedback should be framed cautiously."
    )


def _observed_pattern(subject_id: str, subject_changes: pd.DataFrame, dominant_group: str) -> str:
    """Describe the observed descriptive feature pattern."""

    if subject_changes.empty:
        return (
            f"For {subject_id}, complete baseline/stress feature-change summaries were not available "
            "for this local run."
        )

    groups = subject_changes.sort_values("rank")["feature_group"].drop_duplicates().head(2).tolist()
    group_text = " and ".join(groups) if groups else dominant_group
    top_features = ", ".join(subject_changes.sort_values("rank")["feature"].head(3).tolist())
    return (
        f"{group_text} changed during protocol stress windows compared with baseline windows "
        f"(top descriptive features: {top_features})."
    )


def build_feedback_card_summary(
    profiles: pd.DataFrame,
    subject_top_feature_changes: pd.DataFrame,
    uncertain_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Build one feedback-card example row per subject."""

    rows: list[dict[str, object]] = []
    for _, profile in profiles.sort_values("subject_id").iterrows():
        subject_id = str(profile["subject_id"])
        dominant_group = str(profile.get("dominant_changed_feature_group", "") or "wearable feature groups")
        subject_changes = subject_top_feature_changes[subject_top_feature_changes["subject_id"] == subject_id]

        observed_pattern = _observed_pattern(subject_id, subject_changes, dominant_group)
        possible_interpretation = (
            "This pattern may reflect a higher psychophysiological strain indicator under the experimental "
            "protocol, limited to the baseline/stress distinction."
        )
        uncertainty_note = _subject_uncertainty_note(subject_id, profiles, uncertain_windows)

        rows.append(
            {
                "subject_id": subject_id,
                "dominant_feature_group": dominant_group,
                "observed_pattern": observed_pattern,
                "possible_interpretation": possible_interpretation,
                "confidence_uncertainty_note": uncertainty_note,
                "feedback_style_a_direct_alert": (
                    'Direct alert: "Your body signals suggest a higher-strain moment under this research protocol."'
                ),
                "feedback_style_b_reflective_prompt": (
                    'Reflective prompt: "Would you like to pause and check how you are feeling?"'
                ),
                "feedback_style_c_low_interruption_visual_cue": (
                    "Low-interruption cue: a subtle change in interface color or rhythm."
                ),
                "caveat": (
                    "This is not a clinical or medical conclusion and should be interpreted only as a "
                    "research-demo feedback example."
                ),
            }
        )

    return pd.DataFrame(rows)


def build_feedback_cards_markdown(feedback_summary: pd.DataFrame) -> str:
    """Create markdown feedback cards from summary rows."""

    lines = [
        "# User-Facing Feedback Cards",
        "",
        "These example cards translate descriptive wearable-signal patterns into cautious HCI feedback styles. "
        "They are generated locally from the available WESAD-derived features and are not clinical conclusions.",
        "",
    ]

    for _, row in feedback_summary.sort_values("subject_id").iterrows():
        lines.extend(
            [
                f"## {row['subject_id']}",
                "",
                f"**Observed pattern:** {row['observed_pattern']}",
                "",
                f"**Possible interpretation:** {row['possible_interpretation']}",
                "",
                f"**Confidence / uncertainty note:** {row['confidence_uncertainty_note']}",
                "",
                f"**Feedback style A:** {row['feedback_style_a_direct_alert']}",
                "",
                f"**Feedback style B:** {row['feedback_style_b_reflective_prompt']}",
                "",
                f"**Feedback style C:** {row['feedback_style_c_low_interruption_visual_cue']}",
                "",
                f"**Caveat:** {row['caveat']}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def save_feedback_cards(
    profiles: pd.DataFrame,
    subject_top_feature_changes: pd.DataFrame,
    uncertain_windows: pd.DataFrame,
    tables_dir: Path,
    reports_dir: Path,
) -> dict[str, object]:
    """Save feedback card markdown and summary table."""

    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    feedback_summary = build_feedback_card_summary(profiles, subject_top_feature_changes, uncertain_windows)
    summary_path = tables_dir / "feedback_card_summary.csv"
    cards_path = reports_dir / "user_feedback_cards.md"

    feedback_summary.to_csv(summary_path, index=False)
    cards_path.write_text(build_feedback_cards_markdown(feedback_summary), encoding="utf-8")

    return {
        "feedback_summary": feedback_summary,
        "summary_path": summary_path,
        "cards_path": cards_path,
    }
