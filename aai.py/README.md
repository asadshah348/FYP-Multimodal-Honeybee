# Acoustic Activity Indicator (AAI) — CLI Tool

Part of the **Multimodal Honeybee Hive Monitoring System** FYP.

`aai.py` is a lightweight, standalone command-line tool for tracking **bee-frequency acoustic energy** in hive audio recordings over time. It does not require a GPU or a trained model — it uses classical digital signal processing (DSP) to extract two energy metrics from each recording and logs them to a CSV file. A built-in charting command visualises trends and automatically flags recordings with abnormally low activity.

---

## Table of Contents

- [Purpose & Motivation](#purpose--motivation)
- [How It Works](#how-it-works)
- [Features Extracted](#features-extracted)
- [CSV Log Schema](#csv-log-schema)
- [Trend Chart & Alert Threshold](#trend-chart--alert-threshold)
- [Installation](#installation)
- [Usage](#usage)
- [Supported Audio Formats](#supported-audio-formats)
- [Project Structure](#project-structure)

---

## Purpose & Motivation

The BeeCNN model (`audio-cnn/`) requires a trained model file to run. `aai.py` was built as a complementary tool that:

- Works **without any model** — pure signal processing only.
- Is ideal for **longitudinal studies**: record hive audio at regular intervals, run `aai.py analyze`, and build up a dataset over days or weeks.
- Provides an **early-warning indicator** when band energy drops significantly below the established baseline, suggesting possible colony decline, queen loss, or other anomalies.
- Can be run on any machine with Python — no Jetson, CUDA, or Roboflow account needed.

---

## How It Works

```
Audio file (WAV / MP3 / FLAC / OGG / M4A)
        │
        ▼
  librosa.load()  ←─── loads at native sample rate
        │
        ├─── RMS Energy ─────────────────────────────────────────────┐
        │    √( mean(signal²) )                                      │
        │                                                            ▼
        └─── Band Energy (200–600 Hz)                           Append row
             Butterworth bandpass filter (order 4)            to data/aai_log.csv
             → filtfilt (zero-phase)
             → RMS of filtered signal
```

All results are appended to `data/aai_log.csv` with a timestamp, filename, and optional user-supplied label.

---

## Features Extracted

### RMS Energy (overall)

```
RMS = √( (1/N) × Σ x[n]² )
```

The root mean square amplitude of the raw audio signal. Measures the overall loudness of the hive environment, including background noise and low-frequency vibrations.

### Band Energy (bee-frequency, 200–600 Hz)

```
Band Energy = RMS( bandpass_filter(signal, 200 Hz, 600 Hz) )
```

Honeybee wing-beat frequencies and colony buzz lie predominantly in the **200–600 Hz** range. By bandpass-filtering the signal to this range before computing RMS, `aai.py` isolates bee-specific acoustic activity and suppresses environmental noise outside this window.

The filter used is a **4th-order Butterworth bandpass filter** applied with `scipy.signal.filtfilt` (zero-phase, no group delay distortion).

Band Energy is the **primary health indicator**. RMS is logged alongside it for context.

---

## CSV Log Schema

Results are appended to `data/aai_log.csv` (created automatically on first run):

| Column | Type | Description |
|---|---|---|
| `timestamp` | `YYYY-MM-DD HH:MM:SS` | Date and time the recording was analysed |
| `filename` | string | Base filename of the audio file |
| `label` | string | User-supplied label (`--label`) or filename if not provided |
| `rms` | float (6 dp) | Overall RMS energy of the raw signal |
| `band_energy` | float (6 dp) | RMS energy in the 200–600 Hz bee-frequency band |

Example:

```csv
timestamp,filename,label,rms,band_energy
2026-04-20 09:00:00,hive_day1.wav,Day 1,0.042318,0.031452
2026-04-21 09:00:00,hive_day2.wav,Day 2,0.039871,0.029983
2026-04-22 09:00:00,hive_day3.wav,Day 3,0.011204,0.008712
```

---

## Trend Chart & Alert Threshold

Running `python aai.py chart` generates a **Band Energy trend chart** over all logged recordings.

### Baseline calculation (two-pass)

A naïve single-pass mean would be pulled downward by any anomalous low-energy readings, making future alerts harder to trigger. `aai.py` uses a two-pass approach:

1. **First pass:** compute a preliminary mean across all recordings. Recordings below 70% of this preliminary mean are considered outliers.
2. **Second pass:** exclude outlier readings and compute the final **baseline** from the remaining healthy readings.

```
first_pass_baseline   = mean(all band_energy values)
first_pass_threshold  = first_pass_baseline × 0.70
healthy_vals          = band_energy values ≥ first_pass_threshold
baseline              = mean(healthy_vals)
```

### Alert threshold

```
drop_threshold = baseline × 0.70
```

Any recording whose band energy falls below this threshold is flagged on the chart with a **⚠ Low** annotation in red.

### Chart elements

| Element | Description |
|---|---|
| Blue solid line `o-` | Band Energy (200–600 Hz) — primary indicator |
| Light blue dashed line `s--` | RMS Energy (overall) — context only |
| Green dash-dot line | Baseline (mean of healthy readings) |
| Red dotted line | Alert threshold (70% of baseline) |
| ⚠ Low annotation | Recordings that triggered the alert |

---

## Installation

```bash
pip install -r requirements.txt
```

The `requirements.txt` at the project root covers all `aai.py` dependencies:

| Package | Purpose |
|---|---|
| `librosa` | Audio loading and analysis |
| `scipy` | Butterworth filter (`butter`, `filtfilt`) |
| `numpy` | Numerical computation |
| `pandas` | CSV read/write |
| `matplotlib` | Chart generation |
| `soundfile` | Backend audio decoder for librosa |

---

## Usage

### Analyze — extract features and log results

**Single file:**

```bash
python aai.py analyze audio/hive_day1.wav --label "Day 1"
```

**All audio files in a directory:**

```bash
python aai.py analyze audio/
```

The `--label` flag is optional. If omitted, the filename is used as the label.

**Sample output:**

```
Processing: hive_day1.wav
  RMS Energy  (overall hive sound)      : 0.042318
  Band Energy (bee-frequency 200-600 Hz): 0.031452

Results saved to data/aai_log.csv
```

---

### Chart — visualise the trend

**Display in a window:**

```bash
python aai.py chart
```

**Save to a file:**

```bash
python aai.py chart --output hive_trend.png
```

The chart is saved at 150 DPI. Supported output formats: PNG, PDF, SVG (any format supported by matplotlib).

---

### Help

```bash
python aai.py --help
python aai.py analyze --help
python aai.py chart --help
```

---

## Supported Audio Formats

`aai.py` accepts any format supported by `librosa` + `soundfile`:

| Extension | Format |
|---|---|
| `.wav` | Waveform Audio |
| `.mp3` | MPEG Audio Layer III |
| `.flac` | Free Lossless Audio Codec |
| `.ogg` | Ogg Vorbis |
| `.m4a` | MPEG-4 Audio |

---

## Project Structure

```
FYP-Multimodal-Honeybee/
├── aai.py                   # This CLI tool
├── requirements.txt         # Python dependencies
│
├── audio/                   # Place your WAV recordings here
│   ├── tryaud1.wav
│   ├── tryaud2.wav
│   └── tryaud3.wav
│
└── data/
    └── aai_log.csv          # Auto-created on first run
```

The `data/` directory is created automatically by `aai.py` if it does not exist. The CSV file is opened in **append mode**, so re-running the tool on the same file adds a new row rather than overwriting existing data. If you want a fresh log, delete `data/aai_log.csv` before running.
