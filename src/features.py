"""Feature extraction for segmented wearable signal windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import WindowSample


def compute_acc_magnitude(acc_values: np.ndarray) -> np.ndarray:
    """Convert 3-axis acceleration into a magnitude signal."""

    if acc_values.size == 0:
        return np.array([], dtype=float)
    return np.sqrt((acc_values**2).sum(axis=1))


def compute_slope(values: np.ndarray) -> float:
    """Compute a simple least-squares slope over sample index."""

    if len(values) < 2:
        return np.nan

    x = np.arange(len(values), dtype=float)
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def summarize_signal(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Extract simple descriptive features from a one-dimensional signal."""

    if values.size == 0:
        return {f"{prefix}_{name}": np.nan for name in FEATURE_NAMES}

    q75, q25 = np.percentile(values, [75, 25])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_slope": compute_slope(values),
    }


FEATURE_NAMES = ["mean", "std", "min", "max", "median", "iqr", "p05", "p95", "range", "slope"]


def build_feature_table(windows: list[WindowSample]) -> pd.DataFrame:
    """Create one feature row per segmented window."""

    rows: list[dict[str, object]] = []

    for window in windows:
        acc_magnitude = compute_acc_magnitude(window.signals["ACC"].values)

        row: dict[str, object] = {
            "subject": window.subject,
            "condition": window.condition,
            "label": window.label,
            "window_start_s": window.window_start_s,
            "window_end_s": window.window_end_s,
        }
        row.update(summarize_signal(acc_magnitude, "acc"))
        row.update(summarize_signal(window.signals["BVP"].values, "bvp"))
        row.update(summarize_signal(window.signals["EDA"].values, "eda"))
        row.update(summarize_signal(window.signals["TEMP"].values, "temp"))
        rows.append(row)

    return pd.DataFrame(rows)
