# Wearable Stress Detection Demo

This project is a beginner-friendly Python demo for stress detection with the WESAD dataset. The current version starts from the raw Empatica E4 CSV files stored inside each `Sx_E4_Data.zip` archive and uses questionnaire timing metadata from `Sx_quest.csv` to segment the recording into protocol conditions.

## Project Purpose

The goal of this repository is to provide a simple and readable starting point for wearable stress detection experiments. It focuses on a small first version that is easy to inspect, run, and extend rather than a fully optimized research pipeline.

## Pipeline Overview

Raw E4 CSV files -> protocol parsing -> window segmentation -> statistical feature extraction -> baseline/stress classification -> evaluation outputs.

## Current Data Layout

Keep the existing folder structure unchanged:

```text
affective-embodied-interaction-profiler/
|- data/
|  |- S2/
|  |  |- S2_E4_Data.zip
|  |  |- S2_quest.csv
|  |  |- S2_readme.txt
|  |  `- S2.pkl
|  |- S3/
|  |  |- S3_E4_Data.zip
|  |  |- S3_quest.csv
|  |  |- S3_readme.txt
|  |  `- S3.pkl
|  `- ...
`- src/
```

The demo uses:

- `Sx_E4_Data.zip` as the main raw input
- `Sx_quest.csv` for protocol timing
- `Sx_readme.txt` only as an optional reference

The demo does not use `Sx.pkl` as the main input.

## Do Not Commit Data

The WESAD dataset files should be kept locally under `data/` and must not be uploaded to GitHub.

## How To Run The S2 Demo

Run the exact command below from the project root:

```bash
python -m src.run_demo --data_dir data --subjects S2 --output_dir outputs --task binary
```

## Optional Multi-Subject Run

If you want to test multiple subjects later, you can run:

```bash
python -m src.run_demo --data_dir data --subjects S2 S3 S4 S5 --output_dir outputs --task binary
```

## What The Demo Does

1. Finds subject folders under `data/`
2. Reads raw `ACC.csv`, `BVP.csv`, `EDA.csv`, and `TEMP.csv` from each subject's `Sx_E4_Data.zip`
3. Parses condition order and time ranges from `Sx_quest.csv`
4. Segments each condition into 60-second windows with a 30-second step
5. Extracts simple statistical features from each window
6. Trains baseline binary classifiers for `baseline` vs `stress`
7. Exports tables, a confusion matrix figure, an EDA timeline figure, and a short markdown report

## Output Files

The demo writes these files:

```text
outputs/
|- demo_report.md
|- figures/
|  |- confusion_matrix.png
|  `- eda_timeline.png
`- tables/
   |- classification_report.csv
   |- confusion_matrix.csv
   |- model_results.csv
   `- window_features.csv
```

These outputs include the extracted window-level features, model metrics, a detailed classification report, a confusion matrix table, and two beginner-friendly visual summaries.

## Limitations

This raw E4 CSV version uses protocol timings from `Sx_quest.csv` and does not fully solve E4-RespiBAN synchronization. Later versions may use double-tap synchronization or `Sx.pkl` validation.
