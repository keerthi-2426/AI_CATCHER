import sys
from ultralytics import YOLO
import numpy as np

# load pose model
try:
    model = YOLO("yolov8n-pose.pt")
except Exception as e:
    print("Error loading model:", e)
    sys.exit(1)

# create dummy image with a square
img = np.zeros((640, 640, 3), dtype=np.uint8)
img[200:400, 200:400] = 255

results = model.track(img, persist=True)

if results and len(results) > 0:
    res = results[0]
    print("Boxes:", res.boxes is not None)
    
    if hasattr(res, 'keypoints') and res.keypoints is not None:
        print("Keypoints exist!")
        if hasattr(res.keypoints, 'data') and res.keypoints.data is not None:
            print("Data shape:", res.keypoints.data.shape)
        else:
            print("No data attribute in keypoints")
    else:
        print("No keypoints in results[0]")
else:
    print("No results")
