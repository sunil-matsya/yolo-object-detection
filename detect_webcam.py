"""
detect_webcam.py
----------------
Real-Time YOLO Object Detection — Live Webcam

This is where YOLO's speed really shines.
Because it processes an image in ONE forward pass,
it can handle 30+ frames per second in real time.

Press 'q' to quit.
"""

from ultralytics import YOLO
import cv2

# ---------------------------------------------------------------
# STEP 1 — Load the model
# ---------------------------------------------------------------
# Same model as before — already downloaded so loads instantly
model = YOLO("yolov8n.pt")

# ---------------------------------------------------------------
# STEP 2 — Open your webcam
# ---------------------------------------------------------------
# 0 = default built-in webcam
# 1 or 2 = external webcam if you have one
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened! Point it at objects around you.")
print("Press 'q' to quit.\n")

# ---------------------------------------------------------------
# STEP 3 — The real-time detection loop
# ---------------------------------------------------------------
# This loop runs continuously:
#   READ one frame → RUN YOLO → DRAW boxes → SHOW → REPEAT
#
# Each iteration = one frame of video
# YOLO processes each frame completely independently
while True:

    # Read one frame from the webcam
    # ret = True if frame was captured successfully
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Run YOLO on this single frame
    # verbose=False stops it printing results for every frame
    results = model(frame, verbose=False, conf=0.4)

    # Draw bounding boxes on the frame
    for result in results:
        annotated_frame = result.plot()

        # Show how many objects are detected in top-left corner
        count = len(result.boxes)
        cv2.putText(
            annotated_frame,
            f"Objects detected: {count}",
            (10, 30),                   # position on screen
            cv2.FONT_HERSHEY_SIMPLEX,   # font style
            0.9,                        # font size
            (0, 255, 0),                # green colour (BGR format)
            2                           # thickness
        )

    # Show the annotated frame in a window
    cv2.imshow("YOLO Real-Time Detection  |  Press q to quit", annotated_frame)

    # ---------------------------------------------------------------
    # STEP 4 — Check if user pressed 'q' to quit
    # ---------------------------------------------------------------
    # waitKey(1) waits 1ms between frames
    # 0xFF is needed for compatibility on 64-bit systems
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quitting...")
        break

# ---------------------------------------------------------------
# STEP 5 — Clean up
# ---------------------------------------------------------------
# Always release the camera and close windows — good practice
cap.release()
cv2.destroyAllWindows()
print("Done!")