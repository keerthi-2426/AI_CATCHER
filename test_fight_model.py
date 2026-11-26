import sys
from ultralytics import YOLO
try:
    model = YOLO("fight_yolov8.pt")
    print("Model type:", type(model))
    print("Classes:", model.names)
    print("Model loaded successfully")
except Exception as e:
    print("Error:", e)
