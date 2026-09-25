# Particle Tracking

Computer-vision workflows for detecting and tracking particles in experimental frame sequences. The project combines colour-based annotation, YOLO object detection, BoT-SORT and Kalman tracking, and particle image velocimetry (PIV).

## Included work

- YOLO annotation helpers and selected-frame preparation.
- Kalman and BoT-SORT particle trackers.
- PIV analysis scripts for consecutive frames and collision sequences.
- Metrics and selected figures from 5- and 50-epoch YOLOv8 training runs.

## Results

The `results/` directory contains saved YOLO training curves, a normalized confusion matrix, a validation prediction example, and CSV metrics. The 50-epoch run reached approximately 0.93 mAP@0.5 and 0.56 mAP@0.5 on its configured validation data. Dataset splitting should be improved before treating this as a final generalization estimate.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place non-public source frames under `data/` and update paths at the top of scripts or supply project-relative paths. Raw experimental frames, videos, full annotations, and checkpoints are intentionally excluded from this public repository.

## Layout

- `src/preprocessing/` — image selection and colour-based YOLO-label generation.
- `src/tracking/` — BoT-SORT, Kalman, and green-box tracking approaches.
- `src/piv/` — PIV analysis workflows.
- `configs/` — dataset configuration template.
- `results/` — selected, reproducible training evidence.

## Notes

This repository is a curated snapshot of exploratory research code. Several scripts were developed as standalone experiments; paths and parameters should be adjusted for a new dataset.
