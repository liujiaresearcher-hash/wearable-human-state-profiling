"""Command-line entry point for the beginner-friendly WESAD demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import SUPPORTED_TASKS
from .data_io import find_subject_paths, read_subject_e4_data
from .evaluation import (
    ensure_output_dirs,
    save_demo_report,
    save_eda_timeline,
    save_model_outputs,
    save_subject_window_summary,
    save_window_features,
)
from .explainability import save_explanation_outputs
from .feedback_cards import save_feedback_cards
from .features import build_feature_table
from .modeling import evaluate_models
from .preprocessing import add_relative_time_axes, segment_protocol_windows
from .protocol import read_protocol
from .subject_profiles import save_subject_profiles


def build_eda_timeline_table(subject_id: str, signals: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Prepare a compact table for the EDA timeline figure."""

    eda_frame = signals["EDA"]["values"][["time_s", "value"]].copy()
    eda_frame = eda_frame.rename(columns={"value": "eda"})
    eda_frame.insert(0, "subject", subject_id)
    return eda_frame


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run a beginner-friendly WESAD stress detection demo.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to the existing data directory.")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs such as S2 S3 S4.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for exported outputs.")
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS, help="Currently only 'binary' is supported.")
    return parser.parse_args()


def main() -> None:
    """Run the full demo pipeline."""

    args = parse_args()
    subject_paths = find_subject_paths(args.data_dir, subjects=args.subjects)
    tables_dir, figures_dir = ensure_output_dirs(args.output_dir)
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_windows = []
    eda_tables = []

    for subject_info in subject_paths:
        raw_signals = read_subject_e4_data(subject_info)
        timed_signals = add_relative_time_axes(raw_signals)
        protocol = read_protocol(subject_info.quest_path)
        windows = segment_protocol_windows(subject_info.subject_id, timed_signals, protocol)

        all_windows.extend(windows)
        eda_tables.append(build_eda_timeline_table(subject_info.subject_id, timed_signals))

    feature_table = build_feature_table(all_windows)
    if feature_table.empty:
        raise ValueError("No windows were created. Check the protocol timing and signal files.")

    save_window_features(feature_table, tables_dir)
    save_subject_window_summary(feature_table, tables_dir)
    model_results = evaluate_models(feature_table)
    save_model_outputs(model_results, tables_dir, figures_dir)
    profile_outputs = save_subject_profiles(feature_table, tables_dir, reports_dir)
    explanation_outputs = save_explanation_outputs(feature_table, model_results, tables_dir)
    feedback_outputs = save_feedback_cards(
        profile_outputs["profiles"],
        explanation_outputs["subject_top_feature_changes"],
        explanation_outputs["uncertain_windows"],
        tables_dir,
        reports_dir,
    )
    save_eda_timeline(pd.concat(eda_tables, ignore_index=True), figures_dir)
    save_demo_report(
        args.output_dir,
        args.subjects,
        feature_table,
        model_results,
        subject_profiles=profile_outputs["profiles"],
        feature_importance_summary=explanation_outputs["feature_importance"],
        uncertain_windows=explanation_outputs["uncertain_windows"],
        feedback_summary=feedback_outputs["feedback_summary"],
    )

    print("Demo completed successfully.")
    print(f"Subjects: {', '.join(args.subjects)}")
    print(f"Feature rows: {len(feature_table)}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
