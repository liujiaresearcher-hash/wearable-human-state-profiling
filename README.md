# Multimodal Human-State Profiling from Wearable Signals

A raw WESAD Empatica E4 pipeline for stress-state assessment and human factors interpretation

This project is a beginner-friendly Python demo for stress detection with the WESAD dataset. The current version starts from the raw Empatica E4 CSV files stored inside each `Sx_E4_Data.zip` archive and uses questionnaire timing metadata from `Sx_quest.csv` to segment the recording into protocol conditions.

## Project Purpose

The goal of this repository is to provide a simple and readable starting point for wearable stress detection experiments. It focuses on a small first version that is easy to inspect, run, and extend rather than a fully optimized research pipeline.

## Research positioning: from infrastructure state assessment to human-state sensing

My previous work focused on infrastructure sensing, condition assessment, quality evaluation, and decision-oriented analysis. This demo transfers the same assessment logic to human-centered wearable sensing: raw measurements are organized, processed, summarized, and evaluated as evidence of a system state.

The pipeline follows a familiar chain:

```text
raw sensor data -> preprocessing -> feature extraction -> state assessment -> evaluation
```

In this repository, the current state-assessment task is baseline vs stress recognition from wearable Empatica E4 signals. The demo should be read as a baseline research example, not a final deployable stress-recognition system. Its purpose is to show a reproducible path from raw physiological and motion data to a transparent human-state sensing evaluation.

For a fuller explanation of this positioning, see `docs/research_positioning.md`.

## Human factors and ergonomics perspective

This demo is not a physical ergonomics posture-risk demo. It does not estimate posture, joint loading, RULA/REBA scores, or musculoskeletal risk.

Instead, it focuses on cognitive and affective ergonomics: stress, psychophysiological strain, and human-state assessment. The pipeline uses wrist-worn physiological and motion signals to estimate whether a subject is in a baseline or stress condition.

From a human factors perspective, the analysis can be interpreted as a first step toward operator-state monitoring, workload-aware systems, and human-centered sensing. The current result is a baseline demonstration, not a deployable stress-monitoring system.

For a fuller explanation of this human factors framing, see `docs/human_factors_positioning.md`.

## Pipeline Overview

Raw E4 CSV files -> protocol parsing -> window segmentation -> statistical feature extraction -> baseline/stress classification -> evaluation outputs.

In this demo, "multimodal" refers specifically to multimodal wrist-worn Empatica E4 signals: ACC, BVP, EDA, and TEMP. The current repository does not use video, audio, EEG, eye tracking, posture data, or other sensing channels.

## Current Data Layout

Keep the existing folder structure unchanged:

```text
multimodal-human-state-profiling/
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
python -m src.run_demo --data_dir data --subjects S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S13 S14 S15 S16 S17 --output_dir outputs --task binary
```

## Current Results

The current multi-subject baseline was run on these subjects:

- Subjects: `S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17`
- Total valid windows: `1424`
- Binary modeling windows: `897`
- Evaluation: `leave-one-subject-out`
- Logistic regression: accuracy `0.7781`, macro-F1 `0.7677`
- Random forest: accuracy `0.7458`, macro-F1 `0.7090`

This is a baseline demo using raw Empatica E4 CSV data and protocol timings from `Sx_quest.csv`.

Logistic regression performed slightly better than random forest in this baseline setting. The result suggests that simple statistical features from wrist-worn signals contain useful stress-related information. The result should be interpreted as a baseline, not a final stress-recognition system.

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
   |- subject_loso_metrics.csv
   |- subject_window_summary.csv
   `- window_features.csv
```

These outputs include the extracted window-level features, subject-level window counts, model metrics, subject-level leave-one-subject-out metrics, a detailed classification report, a confusion matrix table, and two beginner-friendly visual summaries.

## Limitations

This raw E4 CSV version uses protocol timings from `Sx_quest.csv` and does not fully solve E4-RespiBAN synchronization. It uses only Empatica E4 ACC, BVP, EDA, and TEMP signals; it does not use video, audio, EEG, eye tracking, posture data, RULA/REBA scoring, or musculoskeletal-load assessment. More advanced versions may add double-tap synchronization, HRV features, EDA tonic/phasic decomposition, motion artifact filtering, and optional `Sx.pkl` validation.

The current pipeline is a baseline research demonstration, not a deployable stress-monitoring or occupational health system.

## Dataset Citation

If you use WESAD in your work, please cite:

Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. 2018. Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. ICMI 2018.
