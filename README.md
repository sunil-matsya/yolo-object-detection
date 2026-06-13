# ⬡ YOLO Object Detection

A real-time object detection desktop app built with **YOLOv8**, **OpenCV**, and **Tkinter**.  
Detect objects in uploaded images or live from your webcam — with a clean dark UI showing labels, confidence scores, and inference time.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-cyan?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## What is YOLO?

**YOLO (You Only Look Once)** is a real-time object detection algorithm that processes an entire image in a **single forward pass** through a neural network.

Unlike older detection methods that scan an image multiple times, YOLO:
- Divides the image into an **S × S grid**
- Each cell predicts **bounding boxes + confidence scores + class probabilities**
- Applies **Non-Maximum Suppression (NMS)** to remove duplicate detections
- Returns final results in **~30–250ms** depending on hardware

This makes it fast enough for **live video detection** — something older models couldn't do.

---

## Features

- 📁 **Image Detection** — upload any `.jpg`, `.png`, or `.webp` and detect objects instantly
- 📷 **Webcam Detection** — live real-time detection with FPS counter
- 📊 **Stats Panel** — shows object count, inference time, and detected labels
- 🎨 **Dark UI** — clean desktop interface built with Tkinter (no browser needed)
- 🧠 **YOLOv8 Nano** — fast and lightweight, runs on CPU without a GPU

---

## Project Structure

```
yolo-object-detection/
├── app.py               ← Main UI app (image + webcam detection)
├── detect_image.py      ← Command-line image detection script
├── detect_webcam.py     ← Command-line webcam detection script
├── requirements.txt     ← Python dependencies
└── README.md            ← You are here
```

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- A webcam (for live detection)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/yolo-object-detection.git
cd yolo-object-detection
```

### 2. Create a virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On first run, YOLOv8 will automatically download the model weights file (`yolov8n.pt`, ~6 MB).

---

## Usage

### Desktop UI (recommended)

```bash
python app.py
```

- Click **📁 Upload Image** to detect objects in any photo
- Click **📷 Start Webcam** for live real-time detection
- Click **✕ Exit** to close cleanly

### Command line — image detection

```bash
# Use built-in sample image
python detect_image.py

# Use your own image
python detect_image.py path/to/your/photo.jpg
```

### Command line — webcam detection

```bash
python detect_webcam.py
# Press 'q' to quit
```

---

## How It Works

```
Input (image or webcam frame)
        ↓
Resize to 640 × 640
        ↓
YOLOv8 Neural Network (single forward pass)
        ↓
Grid cells predict: bounding box + confidence + class
        ↓
Non-Maximum Suppression (remove duplicate boxes)
        ↓
Final detections with labels and confidence scores
        ↓
Draw on frame → display in UI
```

---

## Model Options

You can change the model in `app.py` by editing this line:

```python
MODEL_NAME = "yolov8n.pt"   # Change this
```

| Model | File | Size | Speed | Accuracy |
|---|---|---|---|---|
| Nano | `yolov8n.pt` | 6 MB | ⚡ Fastest | Good |
| Small | `yolov8s.pt` | 22 MB | Fast | Better |
| Medium | `yolov8m.pt` | 52 MB | Medium | Great |
| Large | `yolov8l.pt` | 87 MB | Slow | Excellent |
| Extra | `yolov8x.pt` | 136 MB | Slowest | Best |

> Nano is recommended for CPU. Use Medium or above with a GPU.

---

## COCO Classes

The pre-trained model detects **80 object classes** from the COCO dataset including:

`person` · `car` · `truck` · `bus` · `motorcycle` · `bicycle` · `dog` · `cat` · `chair` · `bottle` · `laptop` · `phone` · `book` · `clock` · `cup` · and 65 more.

---

## Tech Stack

| Library | Purpose |
|---|---|
| [Ultralytics YOLOv8](https://docs.ultralytics.com) | Object detection model |
| [OpenCV](https://opencv.org) | Image processing & webcam access |
| [Tkinter](https://docs.python.org/3/library/tkinter.html) | Desktop GUI (built into Python) |
| [Pillow](https://python-pillow.org) | Convert OpenCV frames for Tkinter display |

---

## Troubleshooting

**Webcam not opening?**  
Change `VideoCapture(0)` to `VideoCapture(1)` in `app.py`

**Low accuracy?**  
Switch from `yolov8n.pt` to `yolov8s.pt` in `app.py`

**ModuleNotFoundError?**  
Make sure your virtual environment is active — you should see `(venv)` in your terminal

**App window too large/small?**  
Edit `win_w` and `win_h` at the bottom of `app.py`

---

## Future Improvements

- [ ] Train on a custom dataset (e.g. detect specific products or defects)
- [ ] Export detections to CSV or JSON
- [ ] Add object tracking across video frames
- [ ] Deploy as a web app using Streamlit
- [ ] Run on Raspberry Pi or Jetson Nano for edge deployment

---

## Real-World Applications of YOLO

| Field | Use Case |
|---|---|
| 🚗 Autonomous Vehicles | Detect pedestrians, signs, and other vehicles in real time |
| 🏭 Manufacturing | Defect detection on assembly lines |
| 🏥 Medical Imaging | Detect tumours and lesions in X-rays |
| 🌾 Agriculture | Identify crop diseases via drone footage |
| 🔒 Security | Intrusion detection and crowd monitoring |
| 🛒 Retail | Shelf stock monitoring and checkout automation |

---

## Learning Resources

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [Original YOLO Paper (2016)](https://arxiv.org/abs/1506.02640)
- [COCO Dataset](https://cocodataset.org)
- [Roboflow — label your own dataset](https://roboflow.com)

---

## License

MIT License — free to use, modify, and distribute.

---

*Built as a learning project to understand real-time object detection with YOLOv8.*