"""
Bee Acoustic Activity Index (AAI) CLI Tool

Computes a composite acoustic activity index from bee hive audio recordings
and tracks the trend over time.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
CSV_PATH = Path(__file__).parent / "data" / "aai_log.csv"
CSV_COLUMNS = [
    "timestamp",
    "filename",
    "label",
    "rms",
    "band_energy",
    "bi",
    "activity_ratio",
    "aai_score",
]

FEATURE_WEIGHTS = {
    "rms": 0.30,
    "band_energy": 0.30,
    "bi": 0.25,
    "activity_ratio": 0.15,
}

BEE_BAND_LOW = 200   # Hz
BEE_BAND_HIGH = 600  # Hz


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_rms(signal: np.ndarray) -> float:
    """Root mean square energy of the raw signal."""
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_band_energy(signal: np.ndarray, sr: int,
                        low: int = BEE_BAND_LOW,
                        high: int = BEE_BAND_HIGH) -> float:
    """RMS energy after bandpass-filtering to the bee-relevant frequency range."""
    nyquist = sr / 2.0
    low_norm = low / nyquist
    high_norm = high / nyquist

    if high_norm >= 1.0:
        high_norm = 0.99
    if low_norm <= 0.0:
        low_norm = 0.01

    b, a = butter(4, [low_norm, high_norm], btype="band")
    filtered = filtfilt(b, a, signal)
    return float(np.sqrt(np.mean(filtered ** 2)))


def compute_bioacoustic_index(signal: np.ndarray, sr: int,
                              low: int = BEE_BAND_LOW,
                              high: int = BEE_BAND_HIGH) -> float:
    """Sum of mean spectral power in the bee-relevant frequency bins (STFT)."""
    S = np.abs(librosa.stft(signal))
    freqs = librosa.fft_frequencies(sr=sr)
    band_mask = (freqs >= low) & (freqs <= high)
    band_power = S[band_mask, :]
    return float(np.sum(np.mean(band_power, axis=1)))


def compute_activity_ratio(signal: np.ndarray) -> float:
    """Fraction of RMS frames exceeding a noise-floor threshold (20th percentile)."""
    rms_frames = librosa.feature.rms(y=signal)[0]
    if len(rms_frames) == 0:
        return 0.0
    threshold = np.percentile(rms_frames, 20)
    active_frames = np.sum(rms_frames > threshold)
    return float(active_frames / len(rms_frames))


def extract_features(filepath: str) -> dict:
    """Load an audio file and return all acoustic features."""
    signal, sr = librosa.load(filepath, sr=None)

    if len(signal) == 0:
        raise ValueError(f"Audio file is empty: {filepath}")

    return {
        "rms": compute_rms(signal),
        "band_energy": compute_band_energy(signal, sr),
        "bi": compute_bioacoustic_index(signal, sr),
        "activity_ratio": compute_activity_ratio(signal),
    }


# ---------------------------------------------------------------------------
# Composite AAI score
# ---------------------------------------------------------------------------

def _load_history() -> pd.DataFrame:
    """Load existing CSV log, returning an empty DataFrame if none exists."""
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame(columns=CSV_COLUMNS)


def compute_aai_score(features: dict, history: pd.DataFrame) -> float:
    """
    Compute a composite AAI in [0, 1] by min-max normalizing each feature
    against historical values and taking a weighted average.

    When there is no prior history, raw features are scaled so the first
    reading lands at 0.5 (midpoint).
    """
    feature_keys = ["rms", "band_energy", "bi", "activity_ratio"]
    normalized = {}

    for key in feature_keys:
        val = features[key]
        if len(history) > 0 and key in history.columns:
            col = history[key]
            hist_min = col.min()
            hist_max = col.max()
            all_min = min(hist_min, val)
            all_max = max(hist_max, val)
        else:
            all_min = val
            all_max = val

        if all_max - all_min > 1e-12:
            normalized[key] = (val - all_min) / (all_max - all_min)
        else:
            normalized[key] = 0.5

    score = sum(FEATURE_WEIGHTS[k] * normalized[k] for k in feature_keys)
    return round(float(score), 4)


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------

def append_to_csv(row: dict) -> None:
    """Append a single result row to the CSV log."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    df.to_csv(CSV_PATH, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# CLI: analyze
# ---------------------------------------------------------------------------

def resolve_audio_files(path_arg: str) -> list[str]:
    """Return a list of audio file paths from a file or directory argument."""
    p = Path(path_arg)
    if p.is_file():
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            return [str(p)]
        else:
            print(f"Warning: '{p}' is not a recognized audio format. Skipping.")
            return []
    elif p.is_dir():
        files = sorted(
            str(f) for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not files:
            print(f"No audio files found in '{p}'.")
        return files
    else:
        print(f"Path not found: '{p}'")
        return []


def cmd_analyze(args: argparse.Namespace) -> None:
    """Process audio file(s) and append AAI results to the CSV."""
    files = resolve_audio_files(args.path)
    if not files:
        sys.exit(1)

    history = _load_history()

    for filepath in files:
        fname = os.path.basename(filepath)
        print(f"\nProcessing: {fname}")

        try:
            features = extract_features(filepath)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        aai_score = compute_aai_score(features, history)

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": fname,
            "label": args.label if args.label else fname,
            "rms": round(features["rms"], 6),
            "band_energy": round(features["band_energy"], 6),
            "bi": round(features["bi"], 4),
            "activity_ratio": round(features["activity_ratio"], 4),
            "aai_score": aai_score,
        }

        append_to_csv(row)

        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

        print(f"  RMS Energy      : {row['rms']}")
        print(f"  Band Energy     : {row['band_energy']}")
        print(f"  Bioacoustic Idx : {row['bi']}")
        print(f"  Activity Ratio  : {row['activity_ratio']}")
        print(f"  AAI Score       : {row['aai_score']}")

    print(f"\nResults saved to {CSV_PATH}")


# ---------------------------------------------------------------------------
# CLI: chart
# ---------------------------------------------------------------------------

def cmd_chart(args: argparse.Namespace) -> None:
    """Generate a trend chart from the AAI log."""
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        print("No data yet. Run 'analyze' on some audio files first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("CSV log is empty.")
        sys.exit(1)

    scores = df["aai_score"].values
    labels = df["label"].values
    indices = np.arange(len(scores))

    mean_score = np.mean(scores)
    std_score = np.std(scores) if len(scores) > 1 else mean_score * 0.15

    fig, ax = plt.subplots(figsize=(12, 5))

    # Threshold bands
    ax.axhspan(mean_score + std_score, 1.05, alpha=0.10, color="green", label="Increasing")
    ax.axhspan(mean_score - std_score, mean_score + std_score, alpha=0.10, color="blue", label="Stable")
    ax.axhspan(mean_score - 2 * std_score, mean_score - std_score, alpha=0.10, color="orange", label="Declining")
    ax.axhspan(-0.05, mean_score - 2 * std_score, alpha=0.15, color="red", label="Critical")

    ax.plot(indices, scores, "o-", color="#1f77b4", linewidth=2, markersize=6, label="AAI Score")

    # Rolling average
    if len(scores) >= 3:
        rolling = pd.Series(scores).rolling(window=3, min_periods=1).mean().values
        ax.plot(indices, rolling, "--", color="#ff7f0e", linewidth=2, label="Rolling Avg (3)")

    ax.axhline(y=mean_score, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"Mean ({mean_score:.2f})")

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Recording")
    ax.set_ylabel("AAI Score")
    ax.set_title("Bee Acoustic Activity Index — Trend")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Chart saved to {args.output}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bee Acoustic Activity Index (AAI) — track colony activity from audio"
    )
    subparsers = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Compute AAI for audio file(s)")
    p_analyze.add_argument("path", help="Path to an audio file or directory of audio files")
    p_analyze.add_argument("--label", default=None, help="Optional label for this recording (e.g. 'Day 1')")

    # chart
    p_chart = subparsers.add_parser("chart", help="Show AAI trend chart")
    p_chart.add_argument("--output", default=None, help="Save chart to file instead of displaying")

    args = parser.parse_args()

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "chart":
        cmd_chart(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
