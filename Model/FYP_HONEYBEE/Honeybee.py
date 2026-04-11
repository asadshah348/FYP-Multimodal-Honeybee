from flask import Flask, request, render_template_string
from inference_sdk import InferenceHTTPClient
import os
import base64
import time

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Roboflow Client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="cSTL1ItPm98da1Y3USmT"   # 🔴 Replace with your API key
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multimodal Honey Bee Detection - FYP</title>
    <style>
        body {
            font-family: Arial;
            background: #f4f6f9;
            text-align: center;
        }
        .container {
            width: 90%;
            margin: auto;
        }
        .card {
            background: white;
            padding: 20px;
            margin: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
        button {
            padding: 10px 20px;
            background: #ffb703;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        img {
            margin-top: 10px;
            border-radius: 8px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
        }
    </style>
</head>
<body>

<h1>🐝 Multimodal Honey Bee Detection System</h1>
<p><b>Final Year Project – BS Data Science</b></p>
<p>Computer Vision + Audio-Based Analysis</p>

<div class="container grid">

    <!-- MODEL INFO -->
    <div class="card">
        <h2>📊 Model Information</h2>
        <p><b>CV Model:</b> YOLOv11 (Roboflow)</p>
        <p><b>Framework:</b> Serverless API</p>
        <p><b>Accuracy (mAP):</b> 89%</p>
        <p><b>Precision:</b> 91%</p>
        <p><b>Recall:</b> 87%</p>
        <p><b>Audio Model:</b> 🔄 Under Development</p>
    </div>

    <!-- IMAGE UPLOAD -->
    <div class="card">
        <h2>📷 Upload Image</h2>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" required>
            <br><br>
            <button type="submit">Detect Bees</button>
        </form>
    </div>

</div>

{% if bee_count is not none %}
<div class="container">
    <div class="card">
        <h2>🐝 Detection Results</h2>
        <p><b>Total Bees Detected:</b> {{ bee_count }}</p>
        <p><b>Inference Time:</b> {{ inference_time }} seconds</p>

        <h3>Original Image</h3>
        <img src="data:image/jpeg;base64,{{ original_image }}" width="350">

        <h3>Detected Output</h3>
        <img src="data:image/jpeg;base64,{{ output_image }}" width="350">
    </div>
</div>
{% endif %}

<div class="container">
    <div class="card">
        <h2>🎤 Audio-Based Detection</h2>
        <p>This module will classify:</p>
        <p>• Hive Stress Level</p>
        <p>• Swarming Detection</p>
        <p>• Activity Intensity</p>
        <br>
        <button disabled>Upload Audio (Coming Soon)</button>
    </div>
</div>

</body>
</html>
"""

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
