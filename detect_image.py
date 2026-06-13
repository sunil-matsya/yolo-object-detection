"""
detect_image.py
---------------
YOLO Object Detection on a Static Image

What happens here:
  1. Load a pre-trained YOLOv8 model (already knows 80 object types)
  2. Feed it an image
  3. It returns bounding boxes, class names, and confidence scores
  4. We draw those boxes and save the result
"""

# ultralytics is the library that wraps all of YOLOv8's complexity
from ultralytics import YOLO

# opencv handles image reading, drawing, and displaying
import cv2

import sys

# ---------------------------------------------------------------
# STEP 1 — Load the model
# ---------------------------------------------------------------
# yolov8n.pt = "nano" version — smallest and fastest
# .pt = PyTorch model file (auto-downloaded on first run, ~6 MB)
# It was pre-trained on the COCO dataset:
#   80 classes including person, car, dog, cat, phone, chair, etc.
model = YOLO("yolov8n.pt")

# ---------------------------------------------------------------
# STEP 2 — Pick an image to run detection on
# ---------------------------------------------------------------
# If you pass an image path when running the script, use that.
# Otherwise fall back to a built-in sample image from Ultralytics.
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    from ultralytics.utils import ASSETS
    image_path = str(ASSETS / "bus.jpg")
    print(f"No image provided — using sample: {image_path}")

print(f"\nRunning YOLO on: {image_path}")

# ---------------------------------------------------------------
# STEP 3 — Run inference (the actual detection)
# ---------------------------------------------------------------
# This single line does ALL of the following:
#   - Resizes the image to 640x640 (YOLO's expected input size)
#   - Normalises pixel values from 0-255 to 0.0-1.0
#   - Passes the image through the neural network (one forward pass)
#   - Applies Non-Maximum Suppression to remove duplicate boxes
#   - Returns results with boxes, scores, and class labels
#
# conf=0.5 means: only keep detections with >= 50% confidence
results = model(image_path, conf=0.5)

# ---------------------------------------------------------------
# STEP 4 — Read and print the results
# ---------------------------------------------------------------
result = results[0]  # we passed one image, so take the first result

print(f"\n=== Detection Results ===")
print(f"Total objects detected: {len(result.boxes)}\n")

for box in result.boxes:
    class_id   = int(box.cls[0])          # e.g. 0, 2, 5
    class_name = result.names[class_id]   # e.g. "person", "car"
    confidence = float(box.conf[0])       # e.g. 0.87
    coords     = box.xyxy[0].tolist()     # [x1, y1, x2, y2] in pixels

    print(f"  • {class_name:<15} confidence: {confidence:.0%}  "
          f"box: [{coords[0]:.0f}, {coords[1]:.0f}, "
          f"{coords[2]:.0f}, {coords[3]:.0f}]")

# ---------------------------------------------------------------
# STEP 5 — Draw boxes and save the output image
# ---------------------------------------------------------------
# .plot() draws all bounding boxes + labels onto the image
# and returns it as a NumPy array (a grid of pixel values)
annotated = result.plot()

output_path = "output_detected.jpg"
cv2.imwrite(output_path, annotated)
print(f"\nSaved result → {output_path}")

# Show the image in a pop-up window (press any key to close)
cv2.imshow("YOLO Detection Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()