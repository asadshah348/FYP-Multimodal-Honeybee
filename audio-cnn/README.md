# Audio CNN Component — BeeCNN Hive Activity Classifier

Part of the **Multimodal Honeybee Hive Monitoring System** FYP.

This component uses a custom convolutional neural network (**BeeCNN**) to classify short hive audio recordings into three activity levels. The model operates on **mel spectrograms** extracted from 3-second WAV clips and outputs a rich set of hive health indicators. It is integrated into the Flask dashboard (`final_jetson.py`) and runs on the **NVIDIA Jetson Orin Nano**.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Model Architecture — BeeCNN](#model-architecture--beecnn)
- [Input Processing](#input-processing)
- [Output & Classification](#output--classification)
- [Fallback Mode](#fallback-mode)
- [Audio Capture on Jetson](#audio-capture-on-jetson)
- [API Endpoints](#api-endpoints)
- [Alert Logic](#alert-logic)
- [Data Persistence](#data-persistence)
- [Model File](#model-file)

---

## How It Works

```
Audio clip (WAV, 16 kHz, mono)
        │
        ▼
  Pad / truncate to exactly 3 seconds (48 000 samples)
        │
        ▼
  librosa.feature.melspectrogram
  (128 mel bins, default hop length)
        │
        ▼
  Convert to dB scale (power_to_db, ref=max)
        │
        ▼
  Z-score normalise  →  shape: (1, 1, 128, T)
        │
        ▼
  BeeCNN forward pass  (GPU via CUDA, or CPU)
        │
        ▼
  Softmax → [p_low, p_med, p_high]
        │
        ▼
  Weighted score → 6-level label + estimated count + swarm risk
```

---

## Model Architecture — BeeCNN

```python
BeeCNN(
  conv: Sequential(
    Conv2d(1 → 16, 3×3, pad=1) → ReLU → MaxPool2d(2),
    Conv2d(16 → 32, 3×3, pad=1) → ReLU → MaxPool2d(2),
    Conv2d(32 → 64, 3×3, pad=1) → ReLU → MaxPool2d(2),
  ),
  pool: AdaptiveAvgPool2d(4×4),
  fc: Sequential(
    Linear(64×4×4 = 1024 → 128) → ReLU → Dropout(0.3),
    Linear(128 → 3)
  )
)
```

| Property | Value |
|---|---|
| Input channels | 1 (grayscale spectrogram) |
| Output classes | 3 (Low / Medium / High activity) |
| Trainable parameters | ~145 000 |
| Inference device | CUDA (Jetson GPU) or CPU |
| Model file | `bee_audio_model.pth` (PyTorch state dict) |

---

## Input Processing

| Step | Detail |
|---|---|
| Sample rate | 16 000 Hz |
| Duration | 3 seconds (48 000 samples) |
| Padding | Zero-pad if shorter than 3 s |
| Truncation | Cut to first 3 s if longer |
| Mel bins | 128 |
| Scale | Log (power_to_dB, reference = max value) |
| Normalisation | Z-score per clip: `(mel_db − mean) / (std + 1e-6)` |
| Tensor shape fed to model | `(1, 1, 128, T)` |

---

## Output & Classification

### Raw probabilities

The model outputs a 3-element probability vector via softmax:

| Index | Class | Meaning |
|---|---|---|
| 0 | Low | Quiet / small colony |
| 1 | Medium | Normal active colony |
| 2 | High | Highly active / stressed / swarming |

### 6-Level Activity Score

A weighted score `S = p_low × 1 + p_med × 2 + p_high × 3` maps the probabilities to a human-readable label and estimated bee-count range:

| Score range | Level | Estimated count |
|---|---|---|
| S < 1.3 | **Very Low** | 0 – 30 |
| 1.3 ≤ S < 1.7 | **Low** | 30 – 100 |
| 1.7 ≤ S < 2.1 | **Medium** | 100 – 300 |
| 2.1 ≤ S < 2.4 | **High** | 300 – 600 |
| 2.4 ≤ S < 2.7 | **Very High** | 600 – 1 000 |
| S ≥ 2.7 | **Extreme Swarm** | 1 000+ |

### Full prediction output

```json
{
  "level": "High",
  "bee_range": "300 - 600",
  "low_prob": 0.08,
  "med_prob": 0.31,
  "high_prob": 0.61,
  "activity_intensity": 92,
  "stress_level": "High",
  "swarming_probability": 61,
  "anomaly_detected": true,
  "estimated_count": 487,
  "frequency_data": [82, 75, 68, ...],
  "model_loaded": true
}
```

| Field | Description |
|---|---|
| `level` | Human-readable 6-level label |
| `bee_range` | Count range corresponding to the level |
| `low_prob` / `med_prob` / `high_prob` | Raw softmax probabilities |
| `activity_intensity` | `(p_med × 0.5 + p_high × 1.0) × 100` — overall activity % |
| `stress_level` | `"High"` if p_high > 0.5, `"Moderate"` if p_med > 0.5, else `"Low"` |
| `swarming_probability` | `p_high × 100` (%) |
| `anomaly_detected` | `True` if p_high > 0.6 |
| `estimated_count` | Deterministic interpolation within `bee_range` using p_high |
| `frequency_data` | 20-point pseudo-frequency array for dashboard chart |
| `model_loaded` | `True` when `bee_audio_model.pth` was found and loaded |

---

## Fallback Mode

If `bee_audio_model.pth` is not present at startup, the system switches to a **deterministic hash-based fallback**. The fallback derives all output values from an MD5 hash of the filename, ensuring:

- The same audio file always produces the same result (reproducible demos).
- No model, GPU, or PyTorch computation is required.
- Results are clearly marked with `"model_loaded": false`.

The fallback should only be used for development or demonstration purposes — it does not perform real acoustic analysis.

---

## Audio Capture on Jetson

For live hive monitoring, audio is captured directly on the Jetson and analysed immediately.

### PyAudio (primary)

```
Microphone → PyAudio stream → 1 024-sample chunks → WAV file → BeeCNN
```

Settings:

| Parameter | Value |
|---|---|
| Format | 16-bit PCM (`paInt16`) |
| Channels | 1 (mono) |
| Sample rate | 16 000 Hz |
| Duration | 3 seconds |
| Buffer | 1 024 samples per chunk |

### `arecord` (Jetson Linux fallback)

If PyAudio is unavailable, the system calls `arecord` via subprocess:

```bash
arecord -D plughw:1,0 -d 3 -r 16000 -c 1 -f S16_LE output.wav
```

### Synthetic tone (last resort fallback)

If neither PyAudio nor `arecord` is available (e.g. pure PC testing with no mic), a synthetic 200 Hz tone with harmonics and Gaussian noise is generated in memory and written as a WAV file.

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `POST` | `/upload-audio` | Analyse a user-uploaded WAV/audio file |
| `POST` | `/record-audio` | Trigger a live microphone recording on Jetson |
| `POST` | `/analyze-recorded-audio` | Run BeeCNN on the most recently recorded clip |
| `GET` | `/api/audio-history` | Last N audio analysis records from S3 |

### `/upload-audio` request

Send a multipart form with the file field named `audio`. Supported formats: WAV, MP3, FLAC, OGG, M4A.

### `/record-audio` request

```json
{ "duration": 3 }
```

Triggers an on-device recording and returns the path of the saved WAV file.

---

## Alert Logic

Audio-based alerts are raised automatically and written to S3 (`data/alerts.json`):

| Condition | Alert title | Severity |
|---|---|---|
| `anomaly_detected == True` | "Audio Anomaly" | `critical` |
| `swarming_probability > 50` | "Swarm Risk" | `critical` |
| Level is "Extreme Swarm" or "Very High" | "Extreme Population" | `warning` |

---

## Data Persistence

Each analysis result is appended to `data/audio_history.json` on AWS S3:

```json
{
  "timestamp": "2026-04-22T14:35:00.654321",
  "estimated_count": 487,
  "level": "High",
  "swarming_probability": 61,
  "anomaly_detected": true
}
```

The last 1 000 records are retained on a rolling basis.

---

## Model File

The trained BeeCNN weights must be placed at the project root:

```
FYP-Multimodal-Honeybee/
└── bee_audio_model.pth
```

The file is loaded at Flask startup:

```python
audio_model = BeeCNN().to(device)
audio_model.load_state_dict(torch.load("bee_audio_model.pth", map_location=device))
audio_model.eval()
```

If the file is missing, the system prints a warning and activates [Fallback Mode](#fallback-mode).
