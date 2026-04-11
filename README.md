# Bee Acoustic Activity Index (AAI) CLI Tool

Estimate relative bee population trends from audio recordings using acoustic feature analysis.

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

Display the AAI trend chart:

```bash
python aai.py chart
```

Save the chart to a file:

```bash
python aai.py chart --output chart.png
```

## How It Works

The tool extracts four acoustic features from each recording:

- **RMS Energy** — overall signal amplitude
- **Band Energy** — energy in the 200–600 Hz bee-relevant frequency band
- **Bioacoustic Index (BI)** — mean spectral power in the bee band
- **Activity Ratio** — fraction of audio frames with signal above the noise floor

These are combined into a single **composite AAI score** (0–1 scale) that tracks relative colony activity over time. The trend chart shows this score with a rolling average and threshold bands for stable / declining / critical states.

## Project Structure

```
FYP/
  audio/              # place bee audio files here
  data/
    aai_log.csv       # auto-created, stores all AAI results
  aai.py              # main CLI script
  requirements.txt    # Python dependencies
  README.md           # this file
```
