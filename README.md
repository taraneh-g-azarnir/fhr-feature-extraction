# FHR Feature Extraction

Python pipeline for extracting clinically relevant features from fetal heart rate (FHR) signals.

---

## Overview

This repository provides a feature extraction pipeline for fetal heart rate time-series data.

The pipeline:
- loads raw FHR CSV files,
- interpolates missing or zero-valued samples,
- segments each signal into fixed-length windows,
- extracts statistical, frequency-domain, and nonlinear features,
- saves the final feature table for downstream modeling.

---

## Repository Structure

```text
fhr-feature-extraction/
├── src/
│   └── fhr_feature_extraction.py
├── requirements.txt
└── README.md
```
---

## Dataset

This project is designed for fetal monitoring signals from the SBU public dataset:

https://preana-fo.ece.stonybrook.edu/database.html 

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/taraneh-g-azarnir/fhr-feature-extraction.git
cd fhr-feature-extraction
pip install -r requirements.txt
```

---

## Usage

Run the pipeline:

```bash
python src/fhr_feature_extraction.py \
  --signals-dir data/signals \
  --output results/fhr_features.xlsx
```
---

## Output

The script generates an Excel file where each row corresponds to a signal segment.

The extracted features include:

### Time-domain features
- baseline_fhr  
- mean_fhr  
- median_fhr  
- std_fhr  
- min_fhr  
- max_fhr  
- range_fhr  
- variance_fhr  
- rmssd  

### Event-based features
- num_accelerations  
- num_decelerations  
- accel_duration_seconds  
- decel_duration_seconds  
- deceleration_segments  

### Frequency-domain features
- peak_frequency  
- lf_power  
- hf_power  
- lf_hf_ratio  

### Nonlinear features
- approx_entropy  
- sample_entropy  
- dfa  

### Statistical distribution features
- iqr_fhr  
- percentile_25  
- percentile_50  
- percentile_75
  
---

## Author

Taraneh Ghanbari Azarnir  
PhD Candidate, Electrical Engineering  
Stony Brook University  
