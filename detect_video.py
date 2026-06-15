"""
detect_video.py
---------------
YOLO Video Analysis — Unique Object Detection

What this module does:
  1. Reads a video file frame by frame
  2. Runs YOLO on each frame
  3. Tracks every unique object class seen in the video
  4. Saves the BEST quality cropped photo for each unique class
  5. Generates a summary report
  6. Returns results to the UI for display

Key concept — why we track by CLASS not by individual object:
  If 3 different cars appear in a video, they all count as "car".
  We keep the single best (highest confidence) crop across all frames.
"""

import cv2
import os
import time
from ultralytics import YOLO

# ================================================================
# CONFIGURATION
# ================================================================
CONF_THRESHOLD = 0.45      # Minimum confidence to count a detection
SKIP_FRAMES    = 2         # Process every Nth frame (1 = every frame, 2 = every other)
                           # Higher = faster but might miss quick objects
OUTPUT_FOLDER  = "detections"   # Folder to save crops and summary


# ================================================================
# HELPER — convert frame number to timestamp string
# ================================================================
def frame_to_time(frame_num, fps):
    """
    Convert a frame number to a human-readable timestamp.
    Example: frame 150 at 30fps → "0:05"
    """
    total_seconds = int(frame_num / fps)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


# ================================================================
# HELPER — crop an object from a frame using its bounding box
# ================================================================
def crop_object(frame, box, padding=10):
    """
    Extract just the detected object from the full frame.
    
    box.xyxy gives us [x1, y1, x2, y2] — the pixel coordinates
    of the top-left and bottom-right corners of the bounding box.
    
    We add a small padding around the box so the crop
    doesn't cut the object too tightly.
    """
    h, w = frame.shape[:2]
    coords = box.xyxy[0].tolist()
    
    x1 = max(0, int(coords[0]) - padding)
    y1 = max(0, int(coords[1]) - padding)
    x2 = min(w, int(coords[2]) + padding)
    y2 = min(h, int(coords[3]) + padding)
    
    return frame[y1:y2, x1:x2]   # NumPy array slicing = crop


# ================================================================
# MAIN FUNCTION — process a video file
# ================================================================
def process_video(video_path, model, progress_callback=None, frame_callback=None):
    """
    Process a video file with YOLO and return detection results.
    
    Args:
        video_path      : path to the .mp4 file
        model           : loaded YOLO model
        progress_callback(pct, msg) : called each frame to update UI progress bar
        frame_callback(frame)       : called each frame to show preview in UI
    
    Returns:
        dict of {class_name: {
            "crop"        : best cropped image (NumPy array),
            "count"       : number of frames this class appeared in,
            "first_frame" : frame number when first seen,
            "first_time"  : timestamp string "0:03",
            "best_conf"   : highest confidence score seen,
            "crop_path"   : path to saved crop image
        }}
    """

    # ── Open the video file ──────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration     = total_frames / fps

    print(f"\n=== Video Analysis Started ===")
    print(f"File        : {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}")
    print(f"FPS         : {fps:.1f}")
    print(f"Duration    : {frame_to_time(total_frames, fps)}")
    print(f"Processing every {SKIP_FRAMES} frame(s)\n")

    # ── Create output folder ─────────────────────────────────────
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ── Detection tracker ────────────────────────────────────────
    # This dictionary holds everything we know about each unique class
    # Key   = class name (e.g. "person")
    # Value = dict with crop, count, timestamps, confidence
    detections = {}

    frame_num = 0

    # ── Frame-by-frame processing loop ──────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break   # End of video

        frame_num += 1

        # Skip frames for speed (still tracks everything)
        if frame_num % SKIP_FRAMES != 0:
            continue

        # ── Run YOLO on this frame ───────────────────────────────
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        result  = results[0]

        # ── Send frame preview to UI ─────────────────────────────
        if frame_callback and frame_num % 10 == 0:
            annotated = result.plot()
            frame_callback(annotated)

        # ── Update progress bar ──────────────────────────────────
        if progress_callback:
            pct = (frame_num / total_frames) * 100
            progress_callback(
                pct,
                f"Frame {frame_num}/{total_frames}  —  "
                f"{len(detections)} unique objects found so far"
            )

        # ── Process each detection in this frame ─────────────────
        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            if class_name not in detections:
                # First time we see this class — initialise its record
                detections[class_name] = {
                    "count"       : 1,
                    "first_frame" : frame_num,
                    "first_time"  : frame_to_time(frame_num, fps),
                    "best_conf"   : confidence,
                    "crop"        : crop_object(frame, box),
                    "crop_path"   : None
                }
            else:
                # We've seen this class before — update its record
                detections[class_name]["count"] += 1

                # If this detection has higher confidence, save a better crop
                if confidence > detections[class_name]["best_conf"]:
                    detections[class_name]["best_conf"] = confidence
                    detections[class_name]["crop"]      = crop_object(frame, box)

    cap.release()

    # ── Save crops and generate summary ─────────────────────────
    _save_results(detections, video_path, fps, total_frames)

    print(f"\n=== Analysis Complete ===")
    print(f"Unique objects found: {len(detections)}")
    for name, data in sorted(detections.items(),
                              key=lambda x: x[1]["count"], reverse=True):
        print(f"  • {name:<15} {data['count']:>5} frames  "
              f"first at {data['first_time']}  "
              f"conf: {data['best_conf']:.0%}")

    return detections


# ================================================================
# SAVE CROPS + SUMMARY TEXT FILE
# ================================================================
def _save_results(detections, video_path, fps, total_frames):
    """Save each object's best crop image and write a summary.txt"""

    summary_lines = [
        "=" * 50,
        "YOLO VIDEO ANALYSIS REPORT",
        "=" * 50,
        f"Video    : {os.path.basename(video_path)}",
        f"Duration : {frame_to_time(total_frames, fps)}",
        f"Unique objects detected: {len(detections)}",
        "",
        "DETECTIONS (sorted by frequency):",
        "-" * 50,
    ]

    # Sort by most-seen first
    sorted_detections = sorted(
        detections.items(), key=lambda x: x[1]["count"], reverse=True)

    for class_name, data in sorted_detections:
        # Save the crop image
        crop_filename = f"{class_name}.jpg"
        crop_path     = os.path.join(OUTPUT_FOLDER, crop_filename)

        if data["crop"] is not None and data["crop"].size > 0:
            cv2.imwrite(crop_path, data["crop"])
            data["crop_path"] = crop_path

        # Add to summary
        summary_lines += [
            f"Object    : {class_name}",
            f"Frames    : {data['count']}",
            f"First seen: {data['first_time']}",
            f"Confidence: {data['best_conf']:.0%}",
            f"Crop saved: {crop_path}",
            "",
        ]

    # Write summary.txt
    summary_path = os.path.join(OUTPUT_FOLDER, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\nResults saved to: {OUTPUT_FOLDER}/")
    print(f"Summary    : {summary_path}")