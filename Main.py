from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # nano model (fast, FYP friendly)

# Train the model
model.train(
    data="bee.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="bee_detection"
)
