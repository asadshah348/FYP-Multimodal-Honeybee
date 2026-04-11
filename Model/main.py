


from tkinter import Tk, filedialog
import cv2
from inference_sdk import InferenceHTTPClient

# Hide root window
Tk().withdraw()

# Open file picker
image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    raise SystemExit("No image selected")

# Verify image
img = cv2.imread(image_path)
assert img is not None, "Failed to load image"

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="81cDGyRa50ZzPDIPD8Bk",
)

# Inference
result = CLIENT.infer(
    img,  # pass numpy image (safe)
    model_id="bees-model/1"
)
predictions = result["predictions"]

total_bees = len(predictions)
avg_conf = sum(p["confidence"] for p in predictions) / total_bees

print("🐝 Bee Detection Summary")
print(f"Total Bees Detected : {total_bees}")
print(f"Average Confidence  : {avg_conf:.2f}")


