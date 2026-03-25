"""
Extract fetal heart rate (FHR) features from raw CSV signals.

What this script does:
- reads all CSV files in a folder
- treats zeros in the FHR column as missing values
- interpolates the missing samples
- splits each signal into 10-minute windows
- extracts time-domain, frequency-domain, nonlinear, and event-based features
- saves the final feature table as .xlsx or .csv

Example
-------
python src/fhr_feature_extraction.py \
    --signals-dir data/signals \
    --output results/fhr_features.xlsx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import antropy as ant
import numpy as np
import pandas as pd
from scipy.signal import welch


# -----------------------------
# Configuration
# -----------------------------

FS = 4  # samples per second
WINDOW_SECONDS = 10 * 60
OVERLAP_SECONDS = 0

EXPECTED_COLUMNS = ["samples", "time_sec", "values"]

ACCEL_THRESHOLD = 15
DECEL_THRESHOLD = 15
MIN_EVENT_DURATION_SECONDS = 15

LF_RANGE = (0.04, 0.15)
HF_RANGE = (0.15, 0.40)


# -----------------------------
# Command-line arguments
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from fetal heart rate CSV files."
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        required=True,
        help="Folder containing raw FHR CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fhr_features.xlsx"),
        help="Path to the output feature file (.xlsx or .csv).",
    )
    return parser.parse_args()


# -----------------------------
# File handling
# -----------------------------

def load_signal_files(signals_dir: Path) -> pd.DataFrame:
    """Read all CSV files in the input directory."""
    if not signals_dir.exists():
        raise FileNotFoundError(f"Signals directory does not exist: {signals_dir}")

    csv_files = sorted(signals_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files were found in: {signals_dir}")

    rows = []
    for file_path in csv_files:
        rows.append(
            {
                "file_name": file_path.name,
                "signal_df": pd.read_csv(file_path),
            }
        )

    return pd.DataFrame(rows)


def prepare_output_path(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Signal preprocessing
# -----------------------------

def clean_and_interpolate_signal(signal_df: pd.DataFrame) -> np.ndarray:
    """
    Keep the first three columns, rename them, replace zeros with NaN,
    and interpolate the FHR values.
    """
    if signal_df.shape[1] < 3:
        raise ValueError("Each CSV file must have at least 3 columns.")

    signal = signal_df.iloc[:, :3].copy()
    signal.columns = EXPECTED_COLUMNS

    signal["values"] = pd.to_numeric(signal["values"], errors="coerce")
    signal["values"] = signal["values"].replace(0, np.nan)
    signal["values"] = signal["values"].interpolate(method="linear")
    signal["values"] = signal["values"].ffill().bfill()

    if signal["values"].isna().all():
        raise ValueError("The signal has no valid FHR values after interpolation.")

    return signal.to_numpy()


# -----------------------------
# Segmentation
# -----------------------------

def split_into_segments(signal: np.ndarray, file_name: str) -> list[dict]:
    """Split one signal into fixed-length windows."""
    segment_length = FS * WINDOW_SECONDS
    overlap = FS * OVERLAP_SECONDS
    step = segment_length - overlap

    segments = []
    start = 0
    segment_index = 0

    while start + segment_length <= len(signal):
        end = start + segment_length
        segments.append(
            {
                "file_name": file_name,
                "segment_index": segment_index,
                "segment": signal[start:end],
            }
        )
        start += step
        segment_index += 1

    return segments


def build_segment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create one table containing all segments from all signals."""
    all_segments = []

    for _, row in df.iterrows():
        all_segments.extend(
            split_into_segments(row["signal_df"], row["file_name"])
        )

    return pd.DataFrame(all_segments)


# -----------------------------
# Feature helpers
# -----------------------------

def estimate_baseline_fhr(values: np.ndarray, smooth_window: int = 10) -> float:
    """
    Estimate a baseline FHR from relatively stable portions of the signal.
    """
    smoothed = pd.Series(values).rolling(window=smooth_window, min_periods=1).mean()
    change = smoothed.diff().abs()

    stable_values = smoothed[change <= ACCEL_THRESHOLD]

    if stable_values.empty:
        return float(np.median(values))

    return float(np.median(stable_values))


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Return contiguous True regions from a boolean mask.
    Each run is returned as (start_index, end_index), inclusive.
    """
    runs = []
    start = None

    for i, is_true in enumerate(mask):
        if is_true and start is None:
            start = i
        elif not is_true and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(mask) - 1))

    return runs


def detect_accels_and_decels(values: np.ndarray, baseline: float) -> dict:
    """
    Detect accelerations and decelerations relative to the baseline.
    A valid event must last at least MIN_EVENT_DURATION_SECONDS.
    """
    min_samples = FS * MIN_EVENT_DURATION_SECONDS

    accel_mask = values > (baseline + ACCEL_THRESHOLD)
    decel_mask = values < (baseline - DECEL_THRESHOLD)

    accel_runs = find_runs(accel_mask)
    decel_runs = find_runs(decel_mask)

    valid_accels = [
        (start, end) for start, end in accel_runs
        if (end - start + 1) >= min_samples
    ]
    valid_decels = [
        (start, end) for start, end in decel_runs
        if (end - start + 1) >= min_samples
    ]

    accel_duration = sum((end - start + 1) / FS for start, end in valid_accels)
    decel_duration = sum((end - start + 1) / FS for start, end in valid_decels)

    return {
        "num_accelerations": int(len(valid_accels)),
        "num_decelerations": int(len(valid_decels)),
        "accel_duration_seconds": float(accel_duration),
        "decel_duration_seconds": float(decel_duration),
        "deceleration_segments": json.dumps(valid_decels),
    }


def compute_frequency_features(values: np.ndarray) -> dict:
    """Compute PSD-based frequency features using Welch's method."""
    freqs, psd = welch(values, fs=FS)

    peak_frequency = float(freqs[np.argmax(psd)])

    lf_mask = (freqs >= LF_RANGE[0]) & (freqs <= LF_RANGE[1])
    hf_mask = (freqs >= HF_RANGE[0]) & (freqs <= HF_RANGE[1])

    lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
    hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0

    lf_hf_ratio = float(lf_power / hf_power) if hf_power > 0 else np.nan

    return {
        "peak_frequency": peak_frequency,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
    }


def safe_approx_entropy(values: np.ndarray) -> float:
    try:
        return float(ant.app_entropy(values))
    except Exception:
        return np.nan


def safe_sample_entropy(values: np.ndarray) -> float:
    try:
        return float(ant.sample_entropy(values))
    except Exception:
        return np.nan


def safe_dfa(values: np.ndarray) -> float:
    try:
        return float(ant.detrended_fluctuation(values))
    except Exception:
        return np.nan


# -----------------------------
# Main feature extraction
# -----------------------------

def extract_features_from_segment(segment: np.ndarray) -> dict:
    """Extract all features from one signal segment."""
    values = segment[:, 2].astype(float)

    features = {}

    baseline = estimate_baseline_fhr(values)
    features["baseline_fhr"] = baseline

    features.update(detect_accels_and_decels(values, baseline))

    # Time-domain features
    features["mean_fhr"] = float(np.mean(values))
    features["median_fhr"] = float(np.median(values))
    features["std_fhr"] = float(np.std(values))
    features["min_fhr"] = float(np.min(values))
    features["max_fhr"] = float(np.max(values))
    features["range_fhr"] = float(np.ptp(values))
    features["rmssd"] = float(np.sqrt(np.mean(np.diff(values) ** 2)))

    # Frequency-domain features
    features.update(compute_frequency_features(values))

    # Nonlinear features
    features["approx_entropy"] = safe_approx_entropy(values)
    features["sample_entropy"] = safe_sample_entropy(values)
    features["dfa"] = safe_dfa(values)

    # Statistical features
    features["variance_fhr"] = float(np.var(values))
    features["iqr_fhr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
    features["percentile_25"] = float(np.percentile(values, 25))
    features["percentile_50"] = float(np.percentile(values, 50))
    features["percentile_75"] = float(np.percentile(values, 75))

    return features


def extract_features_for_all_segments(segmented_df: pd.DataFrame) -> pd.DataFrame:
    """Run feature extraction on every segment."""
    feature_rows = []

    for _, row in segmented_df.iterrows():
        feature_dict = extract_features_from_segment(row["segment"])
        feature_dict["file_name"] = row["file_name"]
        feature_dict["segment_index"] = row["segment_index"]
        feature_rows.append(feature_dict)

    return pd.DataFrame(feature_rows)


# -----------------------------
# Save output
# -----------------------------

def save_feature_table(features_df: pd.DataFrame, output_path: Path) -> None:
    suffix = output_path.suffix.lower()

    if suffix == ".xlsx":
        features_df.to_excel(output_path, index=False)
    elif suffix == ".csv":
        features_df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output file must end in .xlsx or .csv")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    prepare_output_path(args.output)

    print("Reading signal files...")
    df = load_signal_files(args.signals_dir)

    print("Interpolating signals...")
    df["signal_df"] = df["signal_df"].apply(clean_and_interpolate_signal)

    print("Creating segments...")
    segmented_df = build_segment_table(df)
    print(f"Created {len(segmented_df)} segments.")

    print("Extracting features...")
    features_df = extract_features_for_all_segments(segmented_df)

    print(f"Saving output to {args.output}")
    save_feature_table(features_df, args.output)

    print("Finished.")


if __name__ == "__main__":
    main()
