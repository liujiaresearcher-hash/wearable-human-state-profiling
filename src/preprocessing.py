"""Signal time-axis construction and condition windowing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SIGNALS, WINDOW_SIZE_SECONDS, WINDOW_STEP_SECONDS


@dataclass
class SignalSegment:
    """One signal segment extracted for a time window."""

    signal_name: str
    values: np.ndarray


@dataclass
class WindowSample:
    """A segmented protocol window with all selected signals."""

    subject: str
    condition: str
    label: int
    window_start_s: float
    window_end_s: float
    signals: dict[str, SignalSegment]


def add_relative_time_axes(signals: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Add a relative time axis to each signal based on its sampling rate."""

    enriched: dict[str, dict[str, object]] = {}
    for signal_name, signal_info in signals.items():
        values = signal_info["values"].copy()
        sampling_rate = float(signal_info["sampling_rate"])
        relative_time = np.arange(len(values), dtype=float) / sampling_rate
        values["time_s"] = relative_time
        enriched[signal_name] = {
            **signal_info,
            "values": values,
        }
    return enriched


def _extract_window_values(signal_frame: pd.DataFrame, signal_name: str, start_s: float, end_s: float) -> np.ndarray:
    """Extract raw values for one signal window."""

    mask = (signal_frame["time_s"] >= start_s) & (signal_frame["time_s"] < end_s)
    window_frame = signal_frame.loc[mask]

    if signal_name == "ACC":
        return window_frame[["x", "y", "z"]].to_numpy(dtype=float)

    return window_frame["value"].to_numpy(dtype=float)


def segment_protocol_windows(
    subject_id: str,
    signals: dict[str, dict[str, object]],
    protocol: pd.DataFrame,
    window_size_s: int = WINDOW_SIZE_SECONDS,
    step_s: int = WINDOW_STEP_SECONDS,
) -> list[WindowSample]:
    """Split each protocol condition into overlapping fixed-length windows."""

    windows: list[WindowSample] = []

    for _, row in protocol.iterrows():
        condition_start = float(row["start_s"])
        condition_end = float(row["end_s"])
        current_start = condition_start

        # A window is kept only when the full 60-second span fits inside the condition.
        while current_start + window_size_s <= condition_end:
            current_end = current_start + window_size_s
            window_signals: dict[str, SignalSegment] = {}

            for signal_name in SIGNALS:
                signal_frame = signals[signal_name]["values"]
                window_values = _extract_window_values(signal_frame, signal_name, current_start, current_end)
                window_signals[signal_name] = SignalSegment(signal_name=signal_name, values=window_values)

            windows.append(
                WindowSample(
                    subject=subject_id,
                    condition=str(row["condition"]),
                    label=int(row["label"]),
                    window_start_s=current_start,
                    window_end_s=current_end,
                    signals=window_signals,
                )
            )
            current_start += step_s

    return windows
