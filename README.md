# FHR Feature Extraction

Feature extraction pipeline for fetal heart rate (FHR) signals.

## Overview
This project extracts clinically relevant features from FHR time-series data.

## Dataset

We use a publicly available fetal monitoring dataset:

https://preana-fo.ece.stonybrook.edu/database.html

## Usage

Make sure dependencies are installed:

```bash
pip install -r requirements.txt
Run the feature extraction pipeline:
python src/feature_extraction.py \
  --signals-dir data/signals \
  --output results/fhr_features.xlsx
```
