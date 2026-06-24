"""Project configuration defaults."""

from pathlib import Path

SIGNALS = ("ACC", "BVP", "EDA", "TEMP")
SUPPORTED_TASKS = ("binary",)

WINDOW_SIZE_SECONDS = 60
WINDOW_STEP_SECONDS = 30
RANDOM_STATE = 42

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
