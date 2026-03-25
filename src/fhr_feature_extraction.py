"""
FHR Feature Extraction Pipeline

This script processes fetal heart rate (FHR) signals and generates
a feature table for downstream modeling.

Pipeline:
1. Load raw CSV signals
2. Clean and interpolate missing values
3. Segment signals into fixed windows
4. Extract clinical, statistical, frequency, and nonlinear features
5. Save results to disk
"""

from __future__ import annotations

import argparse
from pathlib import Path

import antropy as ant
import numpy as np
import pandas as pd
from scipy.signal import welch


# ============================================================
# Configuration
# ============================================================

FS = 4  # sampling frequency (Hz)
WINDOW_SEC = 10 * 60  # 10-minute segments
OVERLAP_SEC = 0

SIGNAL_COLUMNS = ["samples", "time_sec", "values"]


# ============================================================
# I/O utilities
# ============================================================

def ensure_output_path(path: Path) -> None:
    """Ensure output directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_signals(signals_dir: Path) -> pd.DataFrame:
    """
    Load all CSV files in a directory.

    Returns a dataframe with:
        file_name | signal (raw dataframe)
    """
    if not signals_dir.exists():
        raise FileNotFoundError(f"Directory not found: {signals_dir}")

    records = []
    for file in sorted(signals_dir.glob("*.csv")):
        records.append({
            "file_name": file.name,
            "signal": pd.read_csv(file),
        })

    if not records:
        raise ValueError(f"No CSV files found in {signals_dir}")

    return pd.DataFrame(records)


# ============================================================
# Signal processing
# ============================================================

def interpolate_signal(df: pd.DataFrame) -> np.ndarray:
    """
    Replace zeros with NaN and interpolate missing values.
    """
    if df.shape[1] < 3:
        raise ValueError("Signal must have at least 3 columns.")

    signal = df.iloc[:, :3].copy()
    signal.columns = SIGNAL_COLUMNS

    signal["values"] = pd.to_numeric(signal["values"], errors="coerce")
    signal["values"] = signal["values"].replace(0, np.nan)

    signal["values"] = signal["values"].interpolate()
    signal["values"] = signal["values"].ffill().bfill()

    return signal.to_numpy()


# ============================================================
# Segmentation
# ============================================================

def segment_signal(signal: np.ndarray, meta: dict) -> list[tuple]:
    """Split a signal into fixed-length windows."""
    seg_len = FS * WINDOW_SEC
    step = seg_len - (FS * OVERLAP_SEC)

    segments = []
    start = 0

    while start + seg_len <= len(signal):
        segment = signal[start:start + seg_len]

        segments.append((
            segment,
            meta["label"],
            meta["file_name"],
            meta["mother_person_id"],
            meta["infant_person_id"],
            meta["birth_file"],
        ))

        start += step

    return segments


def segment_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply segmentation to all signals."""
    all_segments = []

    for _, row in df.iterrows():
        meta = {
            "label": row.get("label", "unknown"),
            "file_name": row["file_name"],
            "mother_person_id": row.get("mother_person_id", np.nan),
            "infant_person_id": row.get("infant_person_id", np.nan),
            "birth_file": row.get("birth_file", np.nan),
        }

        all_segments.extend(segment_signal(row["signal"], meta))

    return pd.DataFrame(
        all_segments,
        columns=[
            "segment",
            "label",
            "file_name",
            "mother_person_id",
            "infant_person_id",
            "birth_file",
        ],
    )


# ============================================================
# Feature extraction
# ============================================================

def baseline_fhr(values: np.ndarray) -> float:
    """Estimate baseline FHR from stable portions."""
    smooth = pd.Series(values).rolling(10, min_periods=1).mean()
    delta = smooth.diff().abs()

    stable = smooth[(delta <= 15)]
    return float(np.median(stable)) if not stable.empty else float(np.median(values))


def accel_decel_features(values: np.ndarray, baseline: float):
    """Count accelerations and decelerations."""
    accel, decel = 0, 0
    a_count, d_count = 0, 0

    for v in values:
        if v > baseline + 15:
            a_count += 1
            d_count = 0
        elif v < baseline - 15:
            d_count += 1
            a_count = 0
        else:
            if a_count >= 60:
                accel += 1
            if d_count >= 60:
                decel += 1
            a_count = d_count = 0

    return accel, decel


def extract_features(segment: np.ndarray) -> dict:
    """Extract features from a single segment."""
    values = segment[:, 2].astype(float)

    feats = {}

    # baseline + events
    base = baseline_fhr(values)
    feats["baseline_fhr"] = base

    accel, decel = accel_decel_features(values, base)
    feats["num_accelerations"] = accel
    feats["num_decelerations"] = decel

    # time domain
    feats.update({
        "mean_fhr": float(np.mean(values)),
        "std_fhr": float(np.std(values)),
        "min_fhr": float(np.min(values)),
        "max_fhr": float(np.max(values)),
        "rmssd": float(np.sqrt(np.mean(np.diff(values) ** 2))),
    })

    # frequency domain
    freqs, psd = welch(values, fs=FS)
    feats["peak_frequency"] = float(freqs[np.argmax(psd)])

    # nonlinear
    feats["sample_entropy"] = float(ant.sample_entropy(values))
    feats["dfa"] = float(ant.detrended_fluctuation(values))

    return feats


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for all segments."""
    rows = []

    for _, row in df.iterrows():
        f = extract_features(row["segment"])
        f.update({
            "label": row["label"],
            "file_name": row["file_name"],
        })
        rows.append(f)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="FHR feature extraction")
    parser.add_argument("--signals-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="results/fhr_features.xlsx")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_path(args.output)

    print("Loading signals...")
    df = load_signals(args.signals_dir)

    print("Interpolating...")
    df["signal"] = df["signal"].apply(interpolate_signal)

    print("Segmenting...")
    segments = segment_all(df)

    print(f"Total segments: {len(segments)}")

    print("Extracting features...")
    features = extract_all_features(segments)

    print(f"Saving to {args.output}")
    features.to_excel(args.output, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
