# Computer Vision Component — Bee Detection & Counting

Part of the **Multimodal Honeybee Hive Monitoring System** FYP.

This component uses **YOLOv11** (via the Roboflow Inference SDK) to detect and count honeybees in images, video files, and live camera feeds. It is fully integrated into the Flask dashboard (`final_jetson.py`) and designed to run on the **NVIDIA Jetson Orin Nano** with either a CSI or USB camera.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Detection Modes](#detection-modes)
- [Camera Setup (Jetson)](#camera-setup-jetson)
- [API Endpoints](#api-endpoints)
- [Alert Logic](#alert-logic)
- [Data Persistence](#data-persistence)
- [Key Configuration](#key-configuration)

---

## How It Works

```
Input (image / frame / video)
        │
        ▼
  Base64 encode
        │
        ▼
  Roboflow Inference API  ←─── YOLOv11 model (cloud-hosted)
        │
        ▼
  Parse predictions (bounding boxes + class labels)
        │
        ▼
  Draw boxes on frame (OpenCV)
        │
        ▼
  Count bees, log to S3, return annotated image to dashboard
```

Inference is performed remotely on Roboflow's serverless GPU infrastructure, keeping the Jetson's CPU/GPU free for camera capture, audio analysis, and serving the web dashboard simultaneously.

---

## Model Details

| Property | Value |
|---|---|
| **Architecture** | YOLOv11 |
| **Hosting** | Roboflow Serverless (`https://serverless.roboflow.com`) |
| **Task** | Object detection — class: `bee` |
| **Input resolution** | Variable (frames resized by Roboflow) |

Two Roboflow API clients are used with separate API keys and workspaces:

| Client variable | Workspace purpose |
|---|---|
| `client` | Static image uploads (`/upload-image`) |
| `realtime_client` | Live webcam polling, video-frame workflow (`/capture-and-detect`, `/upload-video`) |

---

## Detection Modes

### 1. Image Upload
- **Route:** `POST /upload-image`
- User uploads a JPEG/PNG image via the dashboard.
- The image is saved locally, base64-encoded, and sent to the Roboflow image-inference endpoint.
- The annotated result (bounding boxes drawn with OpenCV) is uploaded to AWS S3 and returned as a base64 preview.
- Detection history is appended to S3 (`data/detection_history.json`).

### 2. Video Upload
- **Route:** `POST /upload-video`
- User uploads an MP4/AVI/MOV video.
- The video is sanitised via `ffmpeg` (re-encoded to clean H.264/yuv420p) to avoid codec artefacts.
- **5 evenly-spaced frames** are extracted using OpenCV.
- Each frame is run through the Roboflow `run_workflow` endpoint (same YOLOv11 workflow as live detection).
- Annotated frames are uploaded to S3; a per-frame gallery is returned to the dashboard.

### 3. Live Camera Detection
- **Routes:** `POST /start-live-detection`, `POST /stop-live-detection`, `GET /live-feed`, `GET /live-count`
- A background thread continuously captures frames from the camera and polls Roboflow for detections.
- The annotated frame stream is served as an **MJPEG stream** (`/live-feed`).
- The latest bee count is polled by the frontend via `/live-count` (JSON).

### 4. Single Capture & Detect
- **Route:** `POST /capture-and-detect`
- Captures one frame from the live camera, runs inference, saves the annotated result, and returns it immediately.
- Increments the `captures` dashboard counter on S3.

### 5. Burst Capture
- **Route:** `POST /capture-burst`
- Captures multiple frames in quick succession and runs inference on each.
- Useful for getting an averaged or peak bee count over a short window.

### 6. Live Preview (no inference)
- **Route:** `GET /video_feed`
- Raw MJPEG stream from the camera with no inference overhead — used for the "preview" panel in the dashboard before starting detection.

---

## Camera Setup (Jetson)

The system supports two camera types, selected at initialisation (`POST /init-camera`):

### CSI Camera (primary on Jetson)
Uses a **GStreamer pipeline** with NVIDIA hardware acceleration:

```
nvarguscamerasrc → NV12 NVMM → nvvidconv → BGRx → videoconvert → BGR → appsink
```

| Parameter | Default |
|---|---|
| Resolution | 640 × 480 |
| Frame rate | 30 fps |
| Flip method | 0 (no flip) |

If the CSI pipeline fails to open, the system automatically falls back to a USB camera.

### USB Camera / PC Webcam (fallback)
Standard OpenCV `VideoCapture(device_id)` at 640 × 480.

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `POST` | `/init-camera` | Initialise camera (`type`: `csi` or `usb`, `device_id`) |
| `GET` | `/video_feed` | Raw MJPEG preview stream |
| `POST` | `/start-live-detection` | Start background detection thread |
| `POST` | `/stop-live-detection` | Stop detection thread & release camera |
| `GET` | `/live-feed` | Annotated MJPEG stream from live detection |
| `GET` | `/live-count` | `{"bee_count": N}` — latest count from live session |
| `POST` | `/capture-and-detect` | Capture one frame, run inference, return result |
| `POST` | `/capture-burst` | Capture burst of frames and analyse each |
| `POST` | `/upload-image` | Analyse a user-uploaded image |
| `POST` | `/upload-video` | Analyse a user-uploaded video (5-frame sampling) |
| `GET` | `/api/cv-history` | Last N CV detection records from S3 |

---

## Alert Logic

Alerts are automatically raised and persisted to S3 (`data/alerts.json`) when:

| Condition | Alert title | Severity |
|---|---|---|
| Bee count > 250 | "High Bee Activity" | `warning` |
| Bee count == 0 | "No Bees Detected" | `info` |

Alerts appear in the dashboard alert feed in real time.

---

## Data Persistence

Each detection is appended to `data/detection_history.json` on AWS S3:

```json
{
  "timestamp": "2026-04-22T14:30:00.123456",
  "bee_count": 147,
  "source": "camera"
}
```

The `source` field records where the detection came from: `"camera"`, `"image"`, `"video"`, or `"burst"`.

The last 1 000 records are retained on a rolling basis.

---

## Key Configuration

All camera and capture settings are at the top of `final_jetson.py`:

```python
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAPTURE_FPS   = 30

NUM_VIDEO_FRAMES = 5   # frames sampled per uploaded video
```

Output folders (created automatically at startup):

| Folder | Contents |
|---|---|
| `uploads/` | Uploaded images |
| `results/` | Annotated result images |
| `video_uploads/` | Uploaded videos (+ sanitised copies) |
| `captures/` | Single-capture and burst JPEG frames |
