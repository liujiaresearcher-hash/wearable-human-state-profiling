"""Protocol parsing for WESAD questionnaire timing files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CONDITION_MAP = {
    "Base": ("baseline", 0),
    "TSST": ("stress", 1),
    "Fun": ("amusement", 2),
    "Medi 1": ("meditation", 3),
    "Medi 2": ("meditation", 3),
}


def parse_protocol_time(value: str) -> float:
    """Convert WESAD timing strings like 7.08 into seconds."""

    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError("Empty protocol time")

    if "." in cleaned:
        # In these questionnaire files, values are written as minute.second.
        minutes_text, seconds_text = cleaned.split(".", 1)
        return int(minutes_text) * 60 + int(seconds_text)

    return float(cleaned) * 60


def read_protocol(quest_path: Path) -> pd.DataFrame:
    """Read selected protocol conditions from a subject questionnaire file."""

    order: list[str] = []
    starts: list[str] = []
    ends: list[str] = []

    with Path(quest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split(";")]
            if not parts or not parts[0].startswith("#"):
                continue

            key = parts[0]
            values = [value for value in parts[1:] if value]
            if key == "# ORDER":
                order = values
            elif key == "# START":
                starts = values
            elif key == "# END":
                ends = values

    if not order or not starts or not ends:
        raise ValueError(f"Could not parse protocol rows from {quest_path}")

    if not (len(order) == len(starts) == len(ends)):
        raise ValueError(f"Protocol rows do not align in {quest_path}")

    rows = []
    for raw_name, start_text, end_text in zip(order, starts, ends):
        if raw_name not in CONDITION_MAP:
            continue

        condition_name, label = CONDITION_MAP[raw_name]
        start_seconds = parse_protocol_time(start_text)
        end_seconds = parse_protocol_time(end_text)

        rows.append(
            {
                "raw_condition": raw_name,
                "condition": condition_name,
                "label": label,
                "start_s": start_seconds,
                "end_s": end_seconds,
            }
        )

    if not rows:
        raise ValueError(f"No supported protocol conditions found in {quest_path}")

    return pd.DataFrame(rows)
