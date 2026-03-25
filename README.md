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

## Dataset

This project is designed for fetal monitoring signals from the SBU public dataset:

https://preana-fo.ece.stonybrook.edu/database.html

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/taraneh-g-azarnir/fhr-feature-extraction.git
cd fhr-feature-extraction
pip install -r requirements.txt
