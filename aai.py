"""
Bee Band Energy Monitor CLI Tool

Records RMS energy and bee-frequency Band Energy (200–600 Hz) from hive
audio recordings and tracks the Band Energy trend over time.
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
]

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


def extract_features(filepath: str) -> dict:
    """Load an audio file and return RMS energy and bee-band energy."""
    signal, sr = librosa.load(filepath, sr=None)

    if len(signal) == 0:
        raise ValueError(f"Audio file is empty: {filepath}")

    return {
        "rms": compute_rms(signal),
        "band_energy": compute_band_energy(signal, sr),
    }


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
    """Process audio file(s) and append band energy results to the CSV."""
    files = resolve_audio_files(args.path)
    if not files:
        sys.exit(1)

    for filepath in files:
        fname = os.path.basename(filepath)
        print(f"\nProcessing: {fname}")

        try:
            features = extract_features(filepath)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": fname,
            "label": args.label if args.label else fname,
            "rms": round(features["rms"], 6),
            "band_energy": round(features["band_energy"], 6),
        }

        append_to_csv(row)

        print(f"  RMS Energy  (overall hive sound)      : {row['rms']}")
        print(f"  Band Energy (bee-frequency 200-600 Hz): {row['band_energy']}")

    print(f"\nResults saved to {CSV_PATH}")


# ---------------------------------------------------------------------------
# CLI: chart
# ---------------------------------------------------------------------------

def cmd_chart(args: argparse.Namespace) -> None:
    """Generate a Band Energy trend chart from the log."""
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        print("No data yet. Run 'analyze' on some audio files first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("CSV log is empty.")
        sys.exit(1)

    band_vals = df["band_energy"].values
    rms_vals = df["rms"].values
    labels = df["label"].values
    indices = np.arange(len(band_vals))

    # Two-pass baseline: first pass uses all recordings to get an initial threshold,
    # second pass excludes readings below that threshold so outlier drops don't
    # pull the baseline down and mask genuine alerts.
    first_pass_baseline = np.mean(band_vals)
    first_pass_threshold = first_pass_baseline * 0.70
    healthy_vals = band_vals[band_vals >= first_pass_threshold]
    baseline = np.mean(healthy_vals) if len(healthy_vals) > 0 else first_pass_baseline
    drop_threshold = baseline * 0.70

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(indices, band_vals, "o-", color="#1f77b4", linewidth=2, markersize=7, label="Band Energy (200–600 Hz)")
    ax.plot(indices, rms_vals, "s--", color="#aec7e8", linewidth=1.5, markersize=5, alpha=0.7, label="RMS Energy (overall)")

    ax.axhline(y=baseline, color="green", linestyle="-.", linewidth=1.5,
               label=f"Baseline ({baseline:.4f})")
    ax.axhline(y=drop_threshold, color="red", linestyle=":", linewidth=1.5,
               label=f"Alert threshold — 70% of baseline ({drop_threshold:.4f})")

    # Annotate points that fall below the alert threshold
    for i, val in enumerate(band_vals):
        if val < drop_threshold:
            ax.annotate("⚠ Low", (i, val), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=8, color="red")

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Recording")
    ax.set_ylabel("Band Energy")
    ax.set_title("Hive Bee-Frequency Band Energy — Trend Over Time")
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
        description="Bee Band Energy Monitor — track hive bee-frequency activity from audio"
    )
    subparsers = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Compute band energy for audio file(s)")
    p_analyze.add_argument("path", help="Path to an audio file or directory of audio files")
    p_analyze.add_argument("--label", default=None, help="Optional label for this recording (e.g. 'Day 1')")

    # chart
    p_chart = subparsers.add_parser("chart", help="Show band energy trend chart")
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
