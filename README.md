# Bee Band Energy Monitor CLI Tool

Track bee hive acoustic activity over time by measuring bee-frequency sound energy from audio recordings.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Analyze audio files

Process a single file:

```bash
python aai.py analyze audio/recording_day1.wav --label "Day 1"
```

Process all audio files in a folder:

```bash
python aai.py analyze audio/
```

Results are appended to `data/aai_log.csv`.

### View trend chart

Display the Band Energy trend chart:

```bash
python aai.py chart
```

Save the chart to a file:

```bash
python aai.py chart --output chart.png
```

## How It Works

The tool extracts two acoustic features from each recording:

- **RMS Energy** — overall sound amplitude of the entire recording
- **Band Energy** — sound energy filtered to the 200–600 Hz range where bee wing vibrations and buzzing occur; the primary activity indicator

### Trend Chart

The chart plots Band Energy (and RMS for context) over all recordings with:

- **Baseline** — mean Band Energy across all healthy recordings (outlier drops below the alert threshold are excluded so they don't distort the reference level)
- **Alert threshold** — 70% of the baseline; any recording falling below this line is flagged with a ⚠ Low annotation, indicating unusually low bee activity

## Project Structure

```
FYP/
  audio/              # place bee audio files here
  data/
    aai_log.csv       # auto-created, stores all results
  aai.py              # main CLI script
  requirements.txt    # Python dependencies
  README.md           # this file
```
