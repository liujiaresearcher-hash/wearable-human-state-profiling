"""Utilities for locating WESAD subjects and reading raw Empatica E4 data."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .config import SIGNALS


@dataclass
class SubjectPaths:
    """Paths associated with one subject folder."""

    subject_id: str
    subject_dir: Path
    e4_zip_path: Path
    quest_path: Path
    readme_path: Path | None


def find_subject_paths(data_dir: Path, subjects: list[str] | None = None) -> list[SubjectPaths]:
    """Locate subject folders and required files under the existing data layout."""

    data_dir = Path(data_dir)
    requested = set(subjects) if subjects else None
    subject_paths: list[SubjectPaths] = []

    for subject_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        subject_id = subject_dir.name
        if requested and subject_id not in requested:
            continue

        e4_zip_path = subject_dir / f"{subject_id}_E4_Data.zip"
        quest_path = subject_dir / f"{subject_id}_quest.csv"
        readme_path = subject_dir / f"{subject_id}_readme.txt"

        if not e4_zip_path.exists() or not quest_path.exists():
            continue

        subject_paths.append(
            SubjectPaths(
                subject_id=subject_id,
                subject_dir=subject_dir,
                e4_zip_path=e4_zip_path,
                quest_path=quest_path,
                readme_path=readme_path if readme_path.exists() else None,
            )
        )

    if requested:
        found_ids = {item.subject_id for item in subject_paths}
        missing = sorted(requested - found_ids)
        if missing:
            raise FileNotFoundError(f"Could not find required subject files for: {', '.join(missing)}")

    if not subject_paths:
        raise FileNotFoundError(f"No valid subject folders found under {data_dir}")

    return subject_paths


def _read_e4_csv_from_zip(zip_path: Path, member_name: str) -> tuple[float, float, pd.DataFrame]:
    """Read one Empatica E4 CSV file from a zip archive."""

    with ZipFile(zip_path) as archive:
        with archive.open(member_name) as raw_file:
            text = raw_file.read().decode("utf-8")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"{member_name} in {zip_path} does not contain enough rows")

    # The first two rows store metadata rather than signal samples.
    timestamp_values = [value.strip() for value in lines[0].split(",")]
    sampling_values = [value.strip() for value in lines[1].split(",")]

    start_timestamp = float(timestamp_values[0])
    sampling_rate = float(sampling_values[0])

    data_text = "\n".join(lines[2:])
    frame = pd.read_csv(StringIO(data_text), header=None)

    if member_name == "ACC.csv":
        # ACC contains three columns for x, y, and z axes.
        frame.columns = ["x", "y", "z"]
    else:
        frame.columns = ["value"]

    return start_timestamp, sampling_rate, frame


def read_subject_e4_data(paths: SubjectPaths) -> dict[str, dict[str, object]]:
    """Read the selected E4 signals from a subject's raw zip archive."""

    signals: dict[str, dict[str, object]] = {}
    for signal_name in SIGNALS:
        member_name = f"{signal_name}.csv"
        start_timestamp, sampling_rate, values = _read_e4_csv_from_zip(paths.e4_zip_path, member_name)
        signals[signal_name] = {
            "start_timestamp": start_timestamp,
            "sampling_rate": sampling_rate,
            "values": values,
        }
    return signals
