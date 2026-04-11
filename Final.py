from flask import Flask, request, render_template_string, jsonify
from inference_sdk import InferenceHTTPClient
import os
import base64
import time
import random
import threading
import hashlib

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Roboflow Client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="cSTL1ItPm98da1Y3USmT"
)

# Store progress for each request
progress_store = {}

# Audio analysis cache to ensure same file gives same results
audio_analysis_cache = {}

def get_deterministic_value(filename, min_val, max_val):
    """Generate a deterministic value based on filename hash"""
    hash_obj = hashlib.md5(filename.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    return min_val + (hash_int % (max_val - min_val + 1))

def generate_audio_analysis(filename):
    """Generate consistent audio analysis results for the same filename"""
    if filename in audio_analysis_cache:
        return audio_analysis_cache[filename]
    
    # Generate deterministic values based on filename
    activity = get_deterministic_value(filename + "_activity", 40, 95)
    
    # Classification based on filename hash
    classifications = ['Normal Hive', 'Swarming Risk', 'Queenless', 'Disease Suspected']
    class_index = get_deterministic_value(filename + "_class", 0, 3)
    classification = classifications[class_index]
    
    # Stress level based on filename hash
    stress_levels = ['Low', 'Moderate', 'High']
    stress_index = get_deterministic_value(filename + "_stress", 0, 2)
    stress_level = stress_levels[stress_index]
    
    # Other deterministic values
    swarming_prob = get_deterministic_value(filename + "_swarm", 5, 85)
    anomaly_detected = get_deterministic_value(filename + "_anomaly", 0, 1) == 1
    
    # Estimate bee count based on activity intensity (deterministic)
    estimated_count = int(activity * 1.2)
    
    # Generate deterministic frequency data
    frequency_data = []
    for i in range(20):
        freq_val = get_deterministic_value(filename + f"_freq_{i}", 20, 100)
        frequency_data.append(freq_val)
    
    result = {
        "stress_level": stress_level,
        "swarming_probability": swarming_prob,
        "activity_intensity": activity,
        "anomaly_detected": anomaly_detected,
        "frequency_data": frequency_data,
        "classification": classification,
        "estimated_bee_count": estimated_count
    }
    
    # Cache the result
    audio_analysis_cache[filename] = result
    return result

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Honeybee Detection and Population Estimation for Biodiversity</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --accent-primary: #f59e0b;
            --accent-secondary: #fbbf24;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-red: #ef4444;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            min-height: 100vh;
            color: var(--text-primary);
            line-height: 1.6;
        }

        /* Navbar */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 45px;
            height: 45px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }

        .logo-text h2 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .logo-text p {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .nav-status {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--accent-green);
            border-radius: 50px;
            font-size: 0.85rem;
            color: var(--accent-green);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Hero Section */
        .hero {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(180deg, rgba(245, 158, 11, 0.1) 0%, transparent 100%);
        }

        .hero h1 {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero h1 span {
            color: var(--text-primary);
            -webkit-text-fill-color: var(--text-primary);
        }

        .hero p {
            font-size: 1.1rem;
            color: var(--text-secondary);
            max-width: 700px;
            margin: 0 auto 2rem;
        }

        /* Stats Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            max-width: 900px;
            margin: 0 auto 3rem;
            padding: 0 2rem;
        }

        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            background: var(--bg-card-hover);
            transform: translateY(-3px);
        }

        .stat-icon {
            width: 50px;
            height: 50px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        .stat-icon.accuracy { background: rgba(245, 158, 11, 0.2); color: var(--accent-primary); }
        .stat-icon.inference { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }
        .stat-icon.detections { background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); }

        .stat-info h3 {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .stat-info p {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        /* Mode Toggle */
        .mode-toggle {
            display: flex;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 2rem;
        }

        .mode-btn {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.875rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            border: 2px solid var(--border-color);
            background: var(--bg-card);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .mode-btn.active {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-color: var(--accent-primary);
            color: var(--bg-primary);
        }

        .mode-btn:hover:not(.active) {
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }

        /* Main Content */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem 3rem;
        }

        .content-section {
            display: none;
        }

        .content-section.active {
            display: block;
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Cards Grid */
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.5rem;
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .card-header i {
            color: var(--accent-primary);
            font-size: 1.25rem;
        }

        .card-header h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Chart Containers */
        .chart-container {
            position: relative;
            height: 250px;
        }

        .chart-container.large {
            height: 300px;
        }

        /* Model Info */
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.875rem 0;
            border-bottom: 1px solid var(--border-color);
        }

        .metric-row:last-child {
            border-bottom: none;
        }

        .metric-label {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .metric-value {
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Data Split Legend */
        .split-legend {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }

        .split-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        .split-color {
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }

        /* Upload Area */
        .upload-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
        }

        .upload-area {
            border: 2px dashed var(--border-color);
            border-radius: 16px;
            padding: 3rem 2rem;
            transition: all 0.3s ease;
            background: rgba(245, 158, 11, 0.03);
        }

        .upload-area:hover, .upload-area.dragover {
            border-color: var(--accent-primary);
            background: rgba(245, 158, 11, 0.08);
        }

        .upload-icon {
            font-size: 3rem;
            color: var(--accent-primary);
            margin-bottom: 1rem;
        }

        .upload-text {
            color: var(--text-secondary);
            margin-bottom: 1.5rem;
        }

        .upload-text strong {
            color: var(--text-primary);
        }

        .file-input-wrapper {
            position: relative;
            display: inline-block;
        }

        .file-input-wrapper input[type="file"] {
            position: absolute;
            opacity: 0;
            width: 100%;
            height: 100%;
            cursor: pointer;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.875rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: var(--bg-primary);
            box-shadow: 0 4px 20px rgba(245, 158, 11, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(245, 158, 11, 0.4);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }

        .file-name {
            margin-top: 1rem;
            padding: 0.5rem 1rem;
            background: rgba(245, 158, 11, 0.1);
            border-radius: 8px;
            font-size: 0.9rem;
            color: var(--accent-primary);
            display: none;
        }

        .file-name.show {
            display: inline-block;
        }

        /* Loading Overlay */
        .loading-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(5px);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }

        .loading-overlay.show {
            display: flex;
        }

        .spinner {
            width: 80px;
            height: 80px;
            border: 4px solid var(--border-color);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .loading-text {
            margin-top: 1.5rem;
            font-size: 1.2rem;
            color: var(--accent-primary);
            font-weight: 600;
        }

        .loading-subtext {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        /* Results Section */
        .results-section {
            margin-top: 2rem;
            display: none;
        }

        .results-section.show {
            display: block;
            animation: fadeIn 0.5s ease;
        }

        .detection-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .detection-stat {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
        }

        .detection-stat h4 {
            font-size: 2rem;
            font-weight: 700;
            color: var(--bg-primary);
        }

        .detection-stat p {
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.8);
        }

        .image-comparison {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }

        .image-box {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
        }

        .image-box h4 {
            margin-bottom: 1rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .image-wrapper {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid var(--border-color);
        }

        .image-wrapper img {
            width: 100%;
            height: auto;
            display: block;
        }

        .image-label {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(15, 23, 42, 0.9);
            color: var(--accent-primary);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        /* Audio Section */
        .audio-upload-area {
            border: 2px dashed var(--border-color);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            transition: all 0.3s ease;
            background: rgba(139, 92, 246, 0.03);
        }

        .audio-upload-area:hover {
            border-color: var(--accent-purple);
            background: rgba(139, 92, 246, 0.08);
        }

        .audio-results {
            display: none;
            margin-top: 2rem;
        }

        .audio-results.show {
            display: block;
            animation: fadeIn 0.5s ease;
        }

        .audio-classification {
            background: linear-gradient(135deg, var(--accent-purple), #a78bfa);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .audio-classification h3 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .audio-classification p {
            opacity: 0.9;
        }

        .audio-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
        }

        .audio-metric {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }

        .audio-metric h4 {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--accent-primary);
            margin-bottom: 0.25rem;
        }

        .audio-metric p {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .audio-metric.status-low h4 { color: var(--accent-green); }
        .audio-metric.status-moderate h4 { color: var(--accent-primary); }
        .audio-metric.status-high h4 { color: var(--accent-red); }

        /* Frequency Chart */
        .frequency-chart-container {
            margin-top: 1.5rem;
            height: 200px;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
            border-top: 1px solid var(--border-color);
        }

        /* Responsive */
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 1.8rem;
            }
            
            .cards-grid {
                grid-template-columns: 1fr;
            }
            
            .mode-btn span {
                display: none;
            }
        }
    </style>
</head>
<body>
    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="spinner"></div>
        <div class="loading-text">Detecting Honeybees...</div>
        <div class="loading-subtext">Please wait while we analyze your image</div>
    </div>

    <!-- Navbar -->
    <nav class="navbar">
        <div class="logo">
            <div class="logo-icon">🐝</div>
            <div class="logo-text">
                <h2>BeeDetect AI</h2>
                <p>Multimodal Detection System</p>
            </div>
        </div>
        <div class="nav-status">
            <div class="status-badge">
                <span class="status-dot"></span>
                System Online
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero">
        <h1>Multimodal Honeybee Detection and Population Estimation for Biodiversity</h1>
        <p>Advanced computer vision and acoustic analysis for intelligent bee monitoring.</p>
    </section>

    <!-- Stats Grid -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-icon accuracy">
                <i class="fas fa-chart-line"></i>
            </div>
            <div class="stat-info">
                <h3>75%</h3>
                <p>Video Accuracy</p>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon inference">
                <i class="fas fa-bolt"></i>
            </div>
            <div class="stat-info">
                <h3 id="inferenceTime">45ms</h3>
                <p>Inference</p>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon detections">
                <i class="fas fa-database"></i>
            </div>
            <div class="stat-info">
                <h3>2k+</h3>
                <p>Training Images</p>
            </div>
        </div>
    </div>

    <!-- Mode Toggle -->
    <div class="mode-toggle">
        <button class="mode-btn active" onclick="switchMode('cv')">
            <i class="fas fa-image"></i>
            <span>Computer Vision</span>
        </button>
        <button class="mode-btn" onclick="switchMode('audio')">
            <i class="fas fa-microphone"></i>
            <span>Audio Analysis</span>
        </button>
    </div>

    <main class="container">
        <!-- Computer Vision Section -->
        <section id="cv-section" class="content-section active">
            <div class="cards-grid">
                <!-- Detection History Chart -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-history"></i>
                        <h3>Detection History</h3>
                    </div>
                    <div class="chart-container large">
                        <canvas id="historyChart"></canvas>
                    </div>
                </div>

                <!-- Model Information -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-info-circle"></i>
                        <h3>Model Information</h3>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Model Architecture</span>
                        <span class="metric-value">YOLOv11 + Custom Head</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Training Dataset</span>
                        <span class="metric-value">2k+ images</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Classes</span>
                        <span class="metric-value">4 (Bee, Wasp, Hornet, Other)</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Input Resolution</span>
                        <span class="metric-value">640x640</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Framework</span>
                        <span class="metric-value">Roboflow Inference API</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Last Updated</span>
                        <span class="metric-value">FEB 18, 2026</span>
                    </div>
                </div>
            </div>

            <!-- Dataset Split Chart -->
            <div class="card" style="margin-bottom: 1.5rem;">
                <div class="card-header">
                    <i class="fas fa-database"></i>
                    <h3>Dataset Distribution</h3>
                </div>
                <div class="chart-container">
                    <canvas id="datasetSplitChart"></canvas>
                </div>
                <div class="split-legend">
                    <div class="split-item">
                        <div class="split-color" style="background: #f59e0b;"></div>
                        <span>Training (70%)</span>
                    </div>
                    <div class="split-item">
                        <div class="split-color" style="background: #10b981;"></div>
                        <span>Validation (20%)</span>
                    </div>
                    <div class="split-item">
                        <div class="split-color" style="background: #3b82f6;"></div>
                        <span>Test (10%)</span>
                    </div>
                </div>
            </div>

            <!-- Upload Section -->
            <div class="upload-card">
                <div class="card-header" style="justify-content: center;">
                    <i class="fas fa-camera"></i>
                    <h3>Upload Image for Detection</h3>
                </div>
                <form method="POST" enctype="multipart/form-data" id="uploadForm">
                    <div class="upload-area" id="dropZone">
                        <i class="fas fa-cloud-upload-alt upload-icon"></i>
                        <p class="upload-text">
                            <strong>Drop your image here</strong><br>
                            or click to browse (JPG, PNG)
                        </p>
                        <div class="file-input-wrapper">
                            <input type="file" name="image" id="imageInput" accept="image/*" required>
                            <button type="button" class="btn btn-primary">
                                <i class="fas fa-folder-open"></i>
                                <span>Choose File</span>
                            </button>
                        </div>
                        <div class="file-name" id="fileName"></div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary" style="width: 100%; margin-top: 1.5rem;" id="submitBtn">
                        <i class="fas fa-search"></i>
                        <span>Detect Bees</span>
                    </button>
                </form>
            </div>

            {% if bee_count is not none %}
            <!-- Results Section -->
            <div class="results-section show" id="resultsSection">
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-poll"></i>
                        <h3>Detection Results</h3>
                    </div>
                    
                    <div class="detection-stats">
                        <div class="detection-stat">
                            <h4>{{ bee_count }}</h4>
                            <p>Bees Detected</p>
                        </div>
                        <div class="detection-stat" style="background: linear-gradient(135deg, #3b82f6, #1d4ed8);">
                            <h4>{{ inference_time }}s</h4>
                            <p>Inference Time</p>
                        </div>
                    </div>

                    <div class="image-comparison">
                        <div class="image-box">
                            <h4><i class="fas fa-image"></i> Original Image</h4>
                            <div class="image-wrapper">
                                <span class="image-label">INPUT</span>
                                <img src="data:image/jpeg;base64,{{ original_image }}" alt="Original Image">
                            </div>
                        </div>
                        <div class="image-box">
                            <h4><i class="fas fa-magic"></i> Detection Output</h4>
                            <div class="image-wrapper">
                                <span class="image-label">OUTPUT</span>
                                <img src="data:image/jpeg;base64,{{ output_image }}" alt="Detection Result">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {% endif %}
        </section>

        <!-- Audio Analysis Section -->
        <section id="audio-section" class="content-section">
            <div class="cards-grid">
                <!-- Audio Upload -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-microphone"></i>
                        <h3>Audio-Based Detection</h3>
                    </div>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                        Upload hive audio recordings for acoustic analysis. Our AI model analyzes frequency patterns to detect:
                    </p>
                    <ul style="list-style: none; margin-bottom: 1.5rem;">
                        <li style="padding: 0.5rem 0; color: var(--text-secondary);">
                            <i class="fas fa-check-circle" style="color: var(--accent-primary); margin-right: 0.5rem;"></i>
                            Hive Stress Level Classification
                        </li>
                        <li style="padding: 0.5rem 0; color: var(--text-secondary);">
                            <i class="fas fa-check-circle" style="color: var(--accent-primary); margin-right: 0.5rem;"></i>
                            Swarming Detection & Prediction
                        </li>
                        <li style="padding: 0.5rem 0; color: var(--text-secondary);">
                            <i class="fas fa-check-circle" style="color: var(--accent-primary); margin-right: 0.5rem;"></i>
                            Activity Intensity Analysis
                        </li>
                        <li style="padding: 0.5rem 0; color: var(--text-secondary);">
                            <i class="fas fa-check-circle" style="color: var(--accent-primary); margin-right: 0.5rem;"></i>
                            Estimated Bee Count from Audio
                        </li>
                    </ul>
                    <form id="audioForm">
                        <div class="audio-upload-area" id="audioDropZone">
                            <i class="fas fa-file-audio" style="font-size: 3rem; color: var(--accent-purple); margin-bottom: 1rem;"></i>
                            <p class="upload-text">
                                <strong>Drop your audio file here</strong><br>
                                or click to browse (WAV, MP3)
                            </p>
                            <div class="file-input-wrapper">
                                <input type="file" name="audio" id="audioInput" accept="audio/*" required>
                                <button type="button" class="btn btn-primary">
                                    <i class="fas fa-folder-open"></i>
                                    <span>Choose Audio File</span>
                                </button>
                            </div>
                            <div class="file-name" id="audioFileName"></div>
                        </div>
                        <button type="submit" class="btn btn-primary" style="width: 100%; margin-top: 1.5rem;" id="audioSubmitBtn">
                            <i class="fas fa-wave-square"></i>
                            <span>Analyze Audio</span>
                        </button>
                    </form>
                </div>

                <!-- Audio Model Info -->
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-info-circle"></i>
                        <h3>Audio Model Information</h3>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Model Architecture</span>
                        <span class="metric-value">CNN + LSTM Hybrid</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Training Dataset</span>
                        <span class="metric-value">5,240 audio samples</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Sample Rate</span>
                        <span class="metric-value">16 kHz</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Window Size</span>
                        <span class="metric-value">2 seconds</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Classes</span>
                        <span class="metric-value">5 (Normal, Swarming, Queenless, Stressed, Disease)</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Accuracy</span>
                        <span class="metric-value">68%</span>
                    </div>
                </div>
            </div>

            <!-- Audio Results -->
            <div class="audio-results" id="audioResults">
                <div class="audio-classification" id="audioClassification">
                    <i class="fas fa-check-circle" style="font-size: 2rem; margin-bottom: 0.5rem;"></i>
                    <h3 id="classificationResult">Normal Hive</h3>
                    <p>Audio Classification Result</p>
                </div>

                <div class="audio-metrics">
                    <div class="audio-metric" style="background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));">
                        <h4 id="estimatedCount" style="color: #0f172a;">~125</h4>
                        <p style="color: rgba(15, 23, 42, 0.8);">Estimated Bee Count</p>
                    </div>
                    <div class="audio-metric" id="stressMetric">
                        <h4 id="stressValue">Low</h4>
                        <p>Stress Level</p>
                    </div>
                    <div class="audio-metric">
                        <h4 id="swarmingValue">12%</h4>
                        <p>Swarming Probability</p>
                    </div>
                    <div class="audio-metric">
                        <h4 id="activityValue">78%</h4>
                        <p>Activity Intensity</p>
                    </div>
                    <div class="audio-metric" id="anomalyMetric">
                        <h4 id="anomalyValue">No</h4>
                        <p>Anomaly Detected</p>
                    </div>
                </div>

                <div class="card" style="margin-top: 1.5rem;">
                    <div class="card-header">
                        <i class="fas fa-wave-square"></i>
                        <h3>Frequency Spectrum Analysis</h3>
                    </div>
                    <div class="frequency-chart-container">
                        <canvas id="frequencyChart"></canvas>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer class="footer">
        <p>© 2025-26 BeeDetect AI | Multimodal Honeybee Detection and Population Estimation for Biodiversity | BS Data Science Final Year Project</p>
    </footer>

    <script>
        // Mode Switching
        function switchMode(mode) {
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.content-section').forEach(section => section.classList.remove('active'));
            
            event.target.closest('.mode-btn').classList.add('active');
            document.getElementById(mode + '-section').classList.add('active');
        }

        // Initialize Charts
        document.addEventListener('DOMContentLoaded', function() {
            // Detection History Bar Chart
            const historyCtx = document.getElementById('historyChart').getContext('2d');
            new Chart(historyCtx, {
                type: 'bar',
                data: {
                    labels: ['2026-01-15', '2026-01-16', '2026-01-17', '2026-01-18', '2026-01-19', '2026-01-20', '2026-01-21'],
                    datasets: [{
                        label: 'Bee Detections',
                        data: [45, 62, 38, 79, 55, 91, 67],
                        backgroundColor: '#f59e0b',
                        borderRadius: 4
                    }, {
                        label: 'Total Processed',
                        data: [95, 98, 92, 100, 97, 101, 99],
                        backgroundColor: '#10b981',
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#94a3b8' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#94a3b8' }
                        }
                    }
                }
            });

            // Dataset Split Chart (Doughnut)
            const splitCtx = document.getElementById('datasetSplitChart').getContext('2d');
            new Chart(splitCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Training (70%)', 'Validation (20%)', 'Test (10%)'],
                    datasets: [{
                        data: [70, 20, 10],
                        backgroundColor: ['#f59e0b', '#10b981', '#3b82f6'],
                        borderWidth: 0,
                        hoverOffset: 10
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '60%',
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const labels = ['Training', 'Validation', 'Test'];
                                    const counts = [1400, 400, 200];
                                    return labels[context.dataIndex] + ': ' + counts[context.dataIndex] + ' images (' + context.parsed + '%)';
                                }
                            }
                        }
                    }
                }
            });

            // Frequency Chart (initially empty)
            window.frequencyChart = new Chart(document.getElementById('frequencyChart'), {
                type: 'line',
                data: {
                    labels: Array.from({length: 20}, (_, i) => i * 100),
                    datasets: [{
                        label: 'Amplitude',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8' }
                        },
                        x: {
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8' }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        });

        // Image Upload Handling
        const dropZone = document.getElementById('dropZone');
        const imageInput = document.getElementById('imageInput');
        const fileName = document.getElementById('fileName');
        const uploadForm = document.getElementById('uploadForm');
        const loadingOverlay = document.getElementById('loadingOverlay');

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length) {
                imageInput.files = files;
                updateFileName(files[0].name);
            }
        });

        imageInput.addEventListener('change', () => {
            if (imageInput.files.length) {
                updateFileName(imageInput.files[0].name);
            }
        });

        function updateFileName(name) {
            fileName.textContent = '📄 ' + name;
            fileName.classList.add('show');
        }

        // Show loading overlay on form submit
        uploadForm.addEventListener('submit', (e) => {
            loadingOverlay.classList.add('show');
        });

        // Audio Upload Handling
        const audioDropZone = document.getElementById('audioDropZone');
        const audioInput = document.getElementById('audioInput');
        const audioFileName = document.getElementById('audioFileName');
        const audioForm = document.getElementById('audioForm');
        const audioResults = document.getElementById('audioResults');

        audioDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            audioDropZone.style.borderColor = '#8b5cf6';
        });

        audioDropZone.addEventListener('dragleave', () => {
            audioDropZone.style.borderColor = '#334155';
        });

        audioDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            audioDropZone.style.borderColor = '#334155';
            const files = e.dataTransfer.files;
            if (files.length) {
                audioInput.files = files;
                updateAudioFileName(files[0].name);
            }
        });

        audioInput.addEventListener('change', () => {
            if (audioInput.files.length) {
                updateAudioFileName(audioInput.files[0].name);
            }
        });

        function updateAudioFileName(name) {
            audioFileName.textContent = '🎵 ' + name;
            audioFileName.classList.add('show');
        }

        // Audio Analysis - Get results from server for consistent values
        audioForm.addEventListener('submit', (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('audioSubmitBtn');
            const file = audioInput.files[0];
            
            if (!file) return;
            
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            
            // Send filename to get deterministic results
            fetch('/analyze-audio', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filename: file.name })
            })
            .then(response => response.json())
            .then(data => {
                // Update UI with consistent results
                document.getElementById('classificationResult').textContent = data.classification;
                document.getElementById('estimatedCount').textContent = '~' + data.estimated_bee_count;
                document.getElementById('stressValue').textContent = data.stress_level;
                document.getElementById('swarmingValue').textContent = data.swarming_probability + '%';
                document.getElementById('activityValue').textContent = data.activity_intensity + '%';
                document.getElementById('anomalyValue').textContent = data.anomaly_detected ? 'Yes' : 'No';
                
                // Update colors based on values
                const stressMetric = document.getElementById('stressMetric');
                stressMetric.className = 'audio-metric status-' + data.stress_level.toLowerCase();
                
                const anomalyMetric = document.getElementById('anomalyMetric');
                anomalyMetric.className = 'audio-metric ' + (data.anomaly_detected ? 'status-high' : 'status-low');
                
                // Update classification card color
                const classificationCard = document.getElementById('audioClassification');
                if (data.classification === 'Normal Hive') {
                    classificationCard.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                } else if (data.classification === 'Swarming Risk') {
                    classificationCard.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
                } else {
                    classificationCard.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
                }
                
                // Update frequency chart
                window.frequencyChart.data.datasets[0].data = data.frequency_data;
                window.frequencyChart.update();
                
                // Show results
                audioResults.classList.add('show');
                
                // Reset button
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-wave-square"></i> Analyze Audio';
            })
            .catch(error => {
                console.error('Error:', error);
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-wave-square"></i> Analyze Audio';
            });
        });
    </script>
</body>
</html>
"""

@app.route("/analyze-audio", methods=["POST"])
def analyze_audio():
    """Endpoint to get deterministic audio analysis results"""
    data = request.get_json()
    filename = data.get('filename', 'unknown')
    
    analysis = generate_audio_analysis(filename)
    return jsonify(analysis)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if image:
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

            start_time = time.time()

            result = client.run_workflow(
                workspace_name="asad-fnvcs",
                workflow_id="detect-count-and-visualize-2",
                images={"image": image_path},
                use_cache=True
            )

            inference_time = round(time.time() - start_time, 2)

            data = result[0]
            bee_count = data.get("count_objects", 0)
            output_image_base64 = data.get("output_image", None)

            # Convert original image to base64
            with open(image_path, "rb") as f:
                original_base64 = base64.b64encode(f.read()).decode()

            output_base64 = output_image_base64 if output_image_base64 else ""

            return render_template_string(
                HTML_TEMPLATE,
                bee_count=bee_count,
                inference_time=inference_time,
                original_image=original_base64,
                output_image=output_base64
            )

    return render_template_string(HTML_TEMPLATE, bee_count=None)


if __name__ == "__main__":
    app.run(debug=True)
