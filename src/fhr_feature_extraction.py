"""
Feature extraction pipeline for fetal heart rate (FHR) signals.

This script:
1. loads raw FHR CSV files,
2. selects the subset of signals listed in a metadata spreadsheet,
3. interpolates missing or zero-valued samples,
4. segments each signal into fixed-length windows,
5. extracts time-domain, frequency-domain, nonlinear, and statistical features,
6. saves the resulting feature table for downstream modeling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import antropy as ant
import numpy as np
import pandas as pd
from scipy.signal import welch


# ============================================================
# Constants
# ============================================================

SAMPLES_PER_SECOND = 4
SEGMENT_LENGTH_SECONDS = 10 * 60
OVERLAP_LENGTH_SECONDS = 0

SIGNAL_COLUMNS = ["samples", "Time_sec", "values"]
SEGMENT_METADATA_COLUMNS = [
    "segment",
    "label",
    "file_name",
    "mother_person_id",
    "infant_person_id",
    "birth_file",
]


# ============================================================
# Utility functions
# ============================================================

def ensure_directory(path: Path) -> None:
    """Create the parent directory for an output file if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_signal_files(signals_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load all CSV signal files from a directory.

    Parameters
    ----------
    signals_dir : Path
        Directory containing raw FHR signal CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping file names to loaded signal dataframes.
    """
    if not signals_dir.exists():
        raise FileNotFoundError(f"Signals directory not found: {signals_dir}")

    signal_map: dict[str, pd.DataFrame] = {}

    for file_path in sorted(signals_dir.glob("*.csv")):
        signal_map[file_path.name] = pd.read_csv(file_path)

    if not signal_map:
        raise ValueError(f"No CSV files were found in: {signals_dir}")

    return signal_map


def get_selected_filenames(chosen_files_df: pd.DataFrame) -> list[str]:
    """
    Convert the first column of the metadata spreadsheet into expected signal file names.

    The first column is assumed to contain signal identifiers that should match files as:
    signal_<id>.csv

    Parameters
    ----------
    chosen_files_df : pd.DataFrame
        Spreadsheet containing the selected tracing identifiers.

    Returns
    -------
    list[str]
        Ordered list of expected signal file names.
    """
    if chosen_files_df.empty:
        raise ValueError("The chosen files spreadsheet is empty.")

    first_col = chosen_files_df.columns[0]
    file_names = "signal_" + chosen_files_df[first_col].astype(str) + ".csv"
    return file_names.tolist()


def build_selected_signals_dataframe(
    signal_map: dict[str, pd.DataFrame],
    selected_filenames: Iterable[str],
) -> pd.DataFrame:
    """
    Build a dataframe of selected signals, preserving the order in selected_filenames.

    Parameters
    ----------
    signal_map : dict[str, pd.DataFrame]
        Loaded signal files keyed by file name.
    selected_filenames : Iterable[str]
        Ordered collection of desired file names.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: file_name, signal
    """
    records = []

    missing_files = []
    for file_name in selected_filenames:
        if file_name in signal_map:
            records.append(
                {
                    "file_name": file_name,
                    "signal": signal_map[file_name],
                }
            )
        else:
            missing_files.append(file_name)

    if not records:
        raise ValueError("No selected signal files were found in the signal directory.")

    if missing_files:
        print(
            f"Warning: {len(missing_files)} selected files were not found in the signal directory."
        )

    return pd.DataFrame(records)


def interpolate_signal(signal: pd.DataFrame) -> np.ndarray:
    """
    Replace zero values with missing values and linearly interpolate the signal.

    Parameters
    ----------
    signal : pd.DataFrame
        Raw signal dataframe.

    Returns
    -------
    np.ndarray
        Interpolated signal array with columns [samples, Time_sec, values].
    """
    if signal.shape[1] < 3:
        raise ValueError(
            "Each signal file must contain at least 3 columns corresponding to "
            f"{SIGNAL_COLUMNS}."
        )

    signal_df = signal.iloc[:, :3].copy()
    signal_df.columns = SIGNAL_COLUMNS

    signal_df["values"] = pd.to_numeric(signal_df["values"], errors="coerce")
    signal_df["values"] = signal_df["values"].replace(0, np.nan)
    signal_df["values"] = signal_df["values"].interpolate(method="linear")
    signal_df["values"] = signal_df["values"].ffill().bfill()

    return signal_df.to_numpy()


def combine_signal_data_with_info(
    signals_df: pd.DataFrame,
    info_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine selected signals with per-signal metadata.

    The original workflow assumes that the rows in info_df correspond in order
    to the selected signals. This function preserves that behavior.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Dataframe with columns ['file_name', 'signal'].
    info_df : pd.DataFrame
        Metadata dataframe aligned row-by-row with signals_df.

    Returns
    -------
    pd.DataFrame
        Combined dataframe.
    """
    signals_df = signals_df.reset_index(drop=True)
    info_df = info_df.reset_index(drop=True)

    if len(signals_df) != len(info_df):
        raise ValueError(
            "The number of selected signals does not match the number of rows in the info CSV. "
            f"Selected signals: {len(signals_df)}, info rows: {len(info_df)}"
        )

    combined_df = pd.concat([signals_df, info_df], axis=1)

    required_columns = ["label", "mother_person_id", "infant_person_id", "birth_file"]
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    if missing_columns:
        raise ValueError(
            "The info CSV is missing required columns: " + ", ".join(missing_columns)
        )

    return combined_df


# ============================================================
# Segmentation
# ============================================================

def create_segments(
    signal: np.ndarray,
    label: str,
    file_name: str,
    mother_person_id,
    infant_person_id,
    birth_file,
    segment_length: int,
    overlap_length: int,
) -> list[tuple]:
    """
    Segment one signal into fixed-length windows.

    Parameters
    ----------
    signal : np.ndarray
        Signal array with shape (n_samples, 3).
    label : str
        Signal label.
    file_name : str
        Source file name.
    mother_person_id : Any
        Mother identifier.
    infant_person_id : Any
        Infant identifier.
    birth_file : Any
        Birth file reference.
    segment_length : int
        Segment length in samples.
    overlap_length : int
        Overlap between consecutive segments in samples.

    Returns
    -------
    list[tuple]
        List of segment records with metadata.
    """
    segments = []
    start = 0
    step = segment_length - overlap_length

    while start + segment_length <= len(signal):
        end = start + segment_length
        segment = signal[start:end]

        segments.append(
            (
                segment,
                label,
                file_name,
                mother_person_id,
                infant_person_id,
                birth_file,
            )
        )
        start += step

    return segments


def segment_all_signals(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment all signals in the dataframe into fixed 10-minute windows.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined signals and metadata dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe of segmented signals and metadata.
    """
    segment_length = SAMPLES_PER_SECOND * SEGMENT_LENGTH_SECONDS
    overlap_length = SAMPLES_PER_SECOND * OVERLAP_LENGTH_SECONDS

    segmented_data = []

    for _, row in combined_df.iterrows():
        segmented_data.extend(
            create_segments(
                signal=row["signal"],
                label=row["label"],
                file_name=row["file_name"],
                mother_person_id=row["mother_person_id"],
                infant_person_id=row["infant_person_id"],
                birth_file=row["birth_file"],
                segment_length=segment_length,
                overlap_length=overlap_length,
            )
        )

    return pd.DataFrame(segmented_data, columns=SEGMENT_METADATA_COLUMNS)


# ============================================================
# Feature extraction
# ============================================================

def calculate_clinical_baseline(
    values: np.ndarray,
    window_size: int = 10,
    accel_threshold: float = 15.0,
    decel_threshold: float = 15.0,
) -> float:
    """
    Estimate a clinical baseline FHR from stable parts of the signal.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional FHR signal values.
    window_size : int, default=10
        Rolling window size for smoothing.
    accel_threshold : float, default=15.0
        Threshold used to exclude abrupt upward changes.
    decel_threshold : float, default=15.0
        Threshold used to exclude abrupt downward changes.

    Returns
    -------
    float
        Estimated baseline FHR.
    """
    smoothed_values = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
    delta_values = smoothed_values.diff().abs()

    acceleration = delta_values > accel_threshold
    deceleration = delta_values > decel_threshold
    stable_values = smoothed_values[~acceleration & ~deceleration]

    if stable_values.empty:
        return float(np.median(values))

    return float(np.median(stable_values))


def calculate_accel_decel(
    values: np.ndarray,
    baseline_fhr: float,
    accel_threshold: float = 15.0,
    decel_threshold: float = 15.0,
    min_duration: int = 60,
    sampling_rate: int = 4,
) -> tuple[int, int, float, float, list[list[float]]]:
    """
    Count sustained accelerations and decelerations relative to baseline.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional FHR signal values.
    baseline_fhr : float
        Estimated baseline FHR.
    accel_threshold : float, default=15.0
        Threshold above baseline for acceleration detection.
    decel_threshold : float, default=15.0
        Threshold below baseline for deceleration detection.
    min_duration : int, default=60
        Minimum number of consecutive samples required to count an event.
    sampling_rate : int, default=4
        Sampling frequency in Hz.

    Returns
    -------
    tuple
        Acceleration count, deceleration count, acceleration duration in seconds,
        deceleration duration in seconds, and detected deceleration segments.
    """
    accel_count = 0
    decel_count = 0

    sustained_accel = 0
    sustained_decel = 0

    deceleration_segments: list[list[float]] = []
    current_decel_segment: list[float] = []

    for value in values:
        if value > (baseline_fhr + accel_threshold):
            sustained_accel += 1
            sustained_decel = 0

            if current_decel_segment:
                deceleration_segments.append(current_decel_segment)
                current_decel_segment = []

        elif value < (baseline_fhr - decel_threshold):
            sustained_decel += 1
            sustained_accel = 0
            current_decel_segment.append(float(value))

        else:
            if sustained_accel >= min_duration:
                accel_count += 1

            if sustained_decel >= min_duration:
                decel_count += 1
                if current_decel_segment:
                    deceleration_segments.append(current_decel_segment)
                    current_decel_segment = []

            sustained_accel = 0
            sustained_decel = 0
            current_decel_segment = []

    if sustained_accel >= min_duration:
        accel_count += 1

    if sustained_decel >= min_duration:
        decel_count += 1
        if current_decel_segment:
            deceleration_segments.append(current_decel_segment)

    accel_duration_seconds = accel_count * min_duration / sampling_rate
    decel_duration_seconds = decel_count * min_duration / sampling_rate

    return (
        accel_count,
        decel_count,
        accel_duration_seconds,
        decel_duration_seconds,
        deceleration_segments,
    )


def calculate_features(segment: np.ndarray) -> dict:
    """
    Extract features from a single FHR segment.

    Parameters
    ----------
    segment : np.ndarray
        Segment array with columns [samples, Time_sec, values].

    Returns
    -------
    dict
        Dictionary of extracted features.
    """
    values = pd.to_numeric(segment[:, 2], errors="coerce").astype(float)

    features: dict[str, float | list] = {}

    baseline_fhr = calculate_clinical_baseline(values)
    features["baseline_fhr"] = baseline_fhr

    (
        accel_count,
        decel_count,
        accel_duration_seconds,
        decel_duration_seconds,
        deceleration_segments,
    ) = calculate_accel_decel(values, baseline_fhr)

    features["num_accelerations"] = accel_count
    features["num_decelerations"] = decel_count
    features["accel_duration_seconds"] = accel_duration_seconds
    features["decel_duration_seconds"] = decel_duration_seconds
    features["deceleration_segments"] = deceleration_segments

    # Time-domain features
    features["mean_fhr"] = float(np.mean(values))
    features["median_fhr"] = float(np.median(values))
    features["std_fhr"] = float(np.std(values))
    features["min_fhr"] = float(np.min(values))
    features["max_fhr"] = float(np.max(values))
    features["range_fhr"] = float(np.ptp(values))
    features["rmssd"] = float(np.sqrt(np.mean(np.diff(values) ** 2)))

    # Frequency-domain features
    freqs, psd = welch(values, fs=SAMPLES_PER_SECOND)
    features["peak_frequency"] = float(freqs[np.argmax(psd)])

    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    lf_mask = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])

    lf_power = float(np.trapz(psd[lf_mask]))
    hf_power = float(np.trapz(psd[hf_mask]))

    features["lf_power"] = lf_power
    features["hf_power"] = hf_power
    features["lf_hf_ratio"] = float(lf_power / hf_power) if hf_power != 0 else np.nan

    # Nonlinear features
    features["approx_entropy"] = float(ant.app_entropy(values))
    features["sample_entropy"] = float(ant.sample_entropy(values))
    features["dfa"] = float(ant.detrended_fluctuation(values))

    # Statistical features
    features["variance_fhr"] = float(np.var(values))
    features["iqr_fhr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
    features["percentile_25"] = float(np.percentile(values, 25))
    features["percentile_50"] = float(np.percentile(values, 50))
    features["percentile_75"] = float(np.percentile(values, 75))

    return features


def extract_features_from_segments(segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for all segmented FHR windows.

    Parameters
    ----------
    segments_df : pd.DataFrame
        Dataframe containing segmented signals and metadata.

    Returns
    -------
    pd.DataFrame
        Feature table.
    """
    feature_rows = []

    for _, row in segments_df.iterrows():
        features = calculate_features(row["segment"])
        features["label"] = row["label"]
        features["file_name"] = row["file_name"]
        features["mother_person_id"] = row["mother_person_id"]
        features["infant_person_id"] = row["infant_person_id"]
        features["birth_file"] = row["birth_file"]
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract features from fetal heart rate (FHR) signals."
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=Path("data/signals"),
        help="Directory containing raw FHR CSV signal files.",
    )
    parser.add_argument(
        "--chosen-files",
        type=Path,
        default=Path("data/final_chosen_tracings_3551.xlsx"),
        help="Excel file listing the selected signals.",
    )
    parser.add_argument(
        "--info-csv",
        type=Path,
        default=Path("data/Experiment3_info.csv"),
        help="CSV file containing per-signal metadata aligned with the selected signals.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fhr_features_experiment3_10min.xlsx"),
        help="Path to the output Excel file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full FHR feature extraction pipeline."""
    args = parse_args()
    ensure_directory(args.output)

    print("Loading signal files...")
    signal_map = load_signal_files(args.signals_dir)

    print("Reading selected signal list...")
    chosen_files_df = pd.read_excel(args.chosen_files)
    selected_filenames = get_selected_filenames(chosen_files_df)

    print("Building selected signal table...")
    signals_df = build_selected_signals_dataframe(signal_map, selected_filenames)

    print("Reading signal metadata...")
    info_df = pd.read_csv(args.info_csv)

    print("Interpolating missing values...")
    signals_df["signal"] = signals_df["signal"].apply(interpolate_signal)

    expected_segments = len(signals_df) * 3
    print(f"Expected number of 10-minute non-overlapping segments: {expected_segments}")

    print("Combining signals with metadata...")
    combined_df = combine_signal_data_with_info(signals_df, info_df)

    print("Segmenting signals...")
    segments_df = segment_all_signals(combined_df)
    print(f"Total number of extracted segments: {len(segments_df)}")

    print("Extracting features...")
    features_df = extract_features_from_segments(segments_df)

    print(f"Saving feature table to: {args.output}")
    features_df.to_excel(args.output, index=False)

    print("Feature extraction completed successfully.")


if __name__ == "__main__":
    main()
