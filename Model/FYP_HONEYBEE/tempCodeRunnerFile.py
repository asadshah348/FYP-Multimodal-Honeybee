# 1. Import libraries
from inference_sdk import InferenceHTTPClient
import tkinter as tk
from tkinter import filedialog
import os
import base64

# 2. Open file dialog to select image
root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Honeybee Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

# Check if image selected
if not image_path:
    print("❌ No image selected!")
    exit()

if not os.path.exists(image_path):
    print("❌ File not found!")
    exit()

print(f"\n📂 Selected Image: {image_path}")

# 3. Connect to Roboflow Server
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="cSTL1ItPm98da1Y3USmT"  # Replace with your API key
)

# 4. Run workflow
print("\n⏳ Processing image...\n")

result = client.run_workflow(
    workspace_name="asad-fnvcs",
    workflow_id="detect-count-and-visualize-2",
    images={"image": image_path},
    use_cache=True
)

# 5. Clean Output Formatting
if result and isinstance(result, list):
    data = result[0]

    bee_count = data.get("count_objects", 0)
    output_image_base64 = data.get("output_image", None)

    print("========== 🐝 HONEYBEE DETECTION RESULT ==========")
    print(f"🐝 Total Bees Detected : {bee_count}")
    print("===================================================")

    # 6. Save output image instead of printing Base64
    if output_image_base64:
        output_image_path = "output_detected_bees.jpg"

        with open(output_image_path, "wb") as f:
            f.write(base64.b64decode(output_image_base64))

        print(f"\n🖼 Output image saved as: {output_image_path}")

else:
    print("❌ Unexpected response format:", result)


