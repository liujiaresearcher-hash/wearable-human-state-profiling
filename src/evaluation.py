"""Export tables, figures, and a short markdown demo report."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import gettempdir

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .modeling import ModelRunResult

# Use a writable temporary matplotlib cache and a non-interactive backend.
os.environ.setdefault("MPLCONFIGDIR", str((Path(gettempdir()) / "wesad_matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Create output subdirectories used by the demo."""

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def save_window_features(feature_table: pd.DataFrame, tables_dir: Path) -> Path:
    """Write the full window-level feature table."""

    path = tables_dir / "window_features.csv"
    feature_table.to_csv(path, index=False)
    return path


def save_model_outputs(results: list[ModelRunResult], tables_dir: Path, figures_dir: Path) -> dict[str, Path]:
    """Write model metrics, reports, and a confusion matrix figure."""

    model_rows: list[dict[str, object]] = []
    report_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []

    first_result = results[0]
    first_matrix = confusion_matrix(first_result.y_true, first_result.y_pred, labels=[0, 1])

    for result in results:
        model_rows.append(
            {
                "model_name": result.model_name,
                "evaluation_mode": result.evaluation_mode,
                "accuracy": accuracy_score(result.y_true, result.y_pred),
                "f1_macro": f1_score(result.y_true, result.y_pred, average="macro"),
            }
        )

        report = classification_report(result.y_true, result.y_pred, output_dict=True, zero_division=0)
        report_frame = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "metric"})
        report_frame.insert(0, "model_name", result.model_name)
        report_frame.insert(1, "evaluation_mode", result.evaluation_mode)
        report_frames.append(report_frame)

        matrix = confusion_matrix(result.y_true, result.y_pred, labels=[0, 1])
        matrix_frame = pd.DataFrame(
            matrix,
            index=["true_baseline", "true_stress"],
            columns=["pred_baseline", "pred_stress"],
        ).reset_index().rename(columns={"index": "row"})
        matrix_frame.insert(0, "model_name", result.model_name)
        confusion_frames.append(matrix_frame)

    model_results_path = tables_dir / "model_results.csv"
    classification_report_path = tables_dir / "classification_report.csv"
    confusion_matrix_path = tables_dir / "confusion_matrix.csv"

    pd.DataFrame(model_rows).to_csv(model_results_path, index=False)
    pd.concat(report_frames, ignore_index=True).to_csv(classification_report_path, index=False)
    pd.concat(confusion_frames, ignore_index=True).to_csv(confusion_matrix_path, index=False)

    confusion_figure_path = figures_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(first_matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["baseline", "stress"])
    ax.set_yticks([0, 1], labels=["baseline", "stress"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix: {first_result.model_name}")

    for row_index in range(first_matrix.shape[0]):
        for col_index in range(first_matrix.shape[1]):
            ax.text(col_index, row_index, int(first_matrix[row_index, col_index]), ha="center", va="center")

    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(confusion_figure_path, dpi=150)
    plt.close(fig)

    return {
        "model_results": model_results_path,
        "classification_report": classification_report_path,
        "confusion_matrix": confusion_matrix_path,
        "confusion_matrix_figure": confusion_figure_path,
    }


def save_eda_timeline(eda_table: pd.DataFrame, figures_dir: Path) -> Path:
    """Save a simple EDA timeline plot across available subjects."""

    path = figures_dir / "eda_timeline.png"
    fig, ax = plt.subplots(figsize=(10, 4))

    for subject, group in eda_table.groupby("subject"):
        ax.plot(group["time_s"], group["eda"], label=subject, linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EDA")
    ax.set_title("EDA Timeline")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_demo_report(
    output_dir: Path,
    subjects: list[str],
    feature_table: pd.DataFrame,
    model_results: list[ModelRunResult],
) -> Path:
    """Write a short markdown report describing the demo run."""

    report_path = output_dir / "demo_report.md"
    condition_counts = feature_table["condition"].value_counts().sort_index()
    best_result = model_results[0]

    lines = [
        "# Wearable Stress Detection Demo Report",
        "",
        "## Run Summary",
        "",
        f"- Subjects: {', '.join(subjects)}",
        f"- Number of windows: {len(feature_table)}",
        f"- Evaluation mode: {best_result.evaluation_mode}",
        "",
        "## Window Counts By Condition",
        "",
    ]

    for condition, count in condition_counts.items():
        lines.append(f"- {condition}: {count}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This demo uses raw Empatica E4 CSV data with protocol timing from Sx_quest.csv.",
            "- The binary task keeps only baseline vs stress windows during modeling.",
            "- Single-subject evaluation is only an internal sanity check and not a final generalization result.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
