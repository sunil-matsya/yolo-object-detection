"""
app.py
------
YOLO Object Detection — Desktop UI
 
A clean desktop application with two modes:
  1. Image Detection  — upload any image and detect objects
  2. Webcam Detection — live real-time detection from your camera
 
Built with Tkinter (Python's built-in GUI library — no extra install needed)
+ Ultralytics YOLOv8 for detection
+ OpenCV for image processing
+ PIL (Pillow) for displaying images inside Tkinter
"""
 
# --- Standard library ---
import threading          # Run webcam loop in background so UI stays responsive
import time               # For FPS calculation
 
# --- GUI ---
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk   # Pillow: converts OpenCV images → Tkinter-displayable images
 
# --- Computer Vision ---
import cv2
from ultralytics import YOLO
 
# ================================================================
# CONFIGURATION
# ================================================================
MODEL_NAME   = "yolov8n.pt"   # Change to yolov8s.pt for better accuracy
CONF_IMAGE   = 0.45            # Confidence threshold for image detection
CONF_WEBCAM  = 0.40            # Confidence threshold for webcam detection
DISPLAY_W    = 800             # Max display width in the UI
DISPLAY_H    = 500             # Max display height in the UI
 
# Colour palette — dark tech theme fitting for a vision AI app
BG_DARK      = "#0f1117"       # Main background
BG_PANEL     = "#1a1d27"       # Panel / card background
BG_CARD      = "#22263a"       # Slightly lighter card
ACCENT       = "#00d4ff"       # Cyan accent — the "AI eye" colour
ACCENT_DIM   = "#007a99"       # Dimmed accent for hover states
TEXT_PRIMARY = "#e8eaf6"       # Main text
TEXT_MUTED   = "#6b7280"       # Secondary / label text
TEXT_SUCCESS = "#22c55e"       # Green for detected labels
BTN_DANGER   = "#ef4444"       # Red for stop/exit buttons
BTN_DARK     = "#2d3148"       # Button background
 
FONT_TITLE   = ("Segoe UI", 22, "bold")
FONT_SUB     = ("Segoe UI", 11)
FONT_LABEL   = ("Segoe UI", 9)
FONT_MONO    = ("Consolas", 10)
FONT_BTN     = ("Segoe UI", 10, "bold")
FONT_COUNT   = ("Segoe UI", 32, "bold")
 
 
# ================================================================
# HELPER — resize an image to fit the display area
# ================================================================
def fit_image(cv_img, max_w=DISPLAY_W, max_h=DISPLAY_H):
    """Scale a CV2 image (NumPy array) to fit within max_w × max_h,
    preserving aspect ratio."""
    h, w = cv_img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv_img
 
 
# ================================================================
# HELPER — convert a CV2 BGR image to a Tkinter PhotoImage
# ================================================================
def cv2_to_tk(cv_img):
    """OpenCV uses BGR colour order; Tkinter needs RGB.
    PIL acts as the bridge between them."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)
 
 
# ================================================================
# MAIN APPLICATION CLASS
# ================================================================
class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)
 
        # ── State ──
        self.model         = None          # Loaded YOLO model
        self.webcam_active = False         # Is webcam loop running?
        self.webcam_thread = None          # Background thread for webcam
        self.cap           = None          # cv2.VideoCapture object
        self._tk_image     = None          # Keep reference to avoid GC
 
        # ── Build UI ──
        self._build_ui()
 
        # ── Load model in background so UI appears instantly ──
        threading.Thread(target=self._load_model, daemon=True).start()
 
        # ── Handle window close ──
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)
 
    # ------------------------------------------------------------
    # MODEL LOADING
    # ------------------------------------------------------------
    def _load_model(self):
        """Load YOLOv8 in a background thread.
        Updating the status label lets the user know it's working."""
        self._set_status("Loading YOLOv8 model...", ACCENT)
        try:
            self.model = YOLO(MODEL_NAME)
            self._set_status(f"Model ready — {MODEL_NAME}", TEXT_SUCCESS)
            self._enable_buttons()
        except Exception as e:
            self._set_status(f"Model load failed: {e}", BTN_DANGER)
 
    # ------------------------------------------------------------
    # UI CONSTRUCTION
    # ------------------------------------------------------------
    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────
        header = tk.Frame(self.root, bg=BG_DARK, pady=18)
        header.pack(fill="x", padx=30)
 
        tk.Label(header, text="⬡  YOLO", font=FONT_TITLE,
                 bg=BG_DARK, fg=ACCENT).pack(side="left")
        tk.Label(header, text="Object Detection", font=FONT_TITLE,
                 bg=BG_DARK, fg=TEXT_PRIMARY).pack(side="left", padx=(8, 0))
 
        # Status badge on the right
        self.status_label = tk.Label(
            header, text="Initialising...", font=FONT_LABEL,
            bg=BG_DARK, fg=TEXT_MUTED)
        self.status_label.pack(side="right", padx=4)
 
        # Thin accent divider
        tk.Frame(self.root, bg=ACCENT, height=1).pack(fill="x", padx=30)
 
        # ── Control panel ────────────────────────────────────────
        ctrl = tk.Frame(self.root, bg=BG_DARK, pady=14)
        ctrl.pack(fill="x", padx=30)
 
        self.btn_image = self._make_button(
            ctrl, "📁  Upload Image", self._pick_image, ACCENT)
        self.btn_image.pack(side="left", padx=(0, 10))
 
        self.btn_webcam = self._make_button(
            ctrl, "📷  Start Webcam", self._toggle_webcam, BTN_DARK)
        self.btn_webcam.pack(side="left", padx=(0, 10))
 
        self.btn_exit = self._make_button(
            ctrl, "✕  Exit", self._on_exit, BTN_DANGER)
        self.btn_exit.pack(side="right")
 
        # Disable until model loads
        self.btn_image.config(state="disabled")
        self.btn_webcam.config(state="disabled")
 
        # ── Display canvas ───────────────────────────────────────
        canvas_frame = tk.Frame(self.root, bg=BG_PANEL,
                                bd=0, relief="flat")
        canvas_frame.pack(padx=30, pady=(0, 14))
 
        self.canvas = tk.Canvas(
            canvas_frame,
            width=DISPLAY_W, height=DISPLAY_H,
            bg=BG_CARD, bd=0, highlightthickness=0)
        self.canvas.pack()
 
        # Placeholder text on canvas
        self.canvas_placeholder = self.canvas.create_text(
            DISPLAY_W // 2, DISPLAY_H // 2,
            text="Upload an image or start the webcam",
            font=("Segoe UI", 13), fill=TEXT_MUTED)
 
        # ── Stats bar ────────────────────────────────────────────
        stats = tk.Frame(self.root, bg=BG_PANEL, pady=12)
        stats.pack(fill="x", padx=30, pady=(0, 20))
 
        # Objects detected count
        left = tk.Frame(stats, bg=BG_PANEL)
        left.pack(side="left", padx=24)
        tk.Label(left, text="OBJECTS DETECTED", font=FONT_LABEL,
                 bg=BG_PANEL, fg=TEXT_MUTED).pack()
        self.lbl_count = tk.Label(left, text="—", font=FONT_COUNT,
                                   bg=BG_PANEL, fg=ACCENT)
        self.lbl_count.pack()
 
        # Vertical divider
        tk.Frame(stats, bg=BG_CARD, width=1).pack(
            side="left", fill="y", padx=10, pady=6)
 
        # Inference time
        mid = tk.Frame(stats, bg=BG_PANEL)
        mid.pack(side="left", padx=24)
        tk.Label(mid, text="INFERENCE TIME", font=FONT_LABEL,
                 bg=BG_PANEL, fg=TEXT_MUTED).pack()
        self.lbl_time = tk.Label(mid, text="—", font=FONT_COUNT,
                                  bg=BG_PANEL, fg=TEXT_PRIMARY)
        self.lbl_time.pack()
 
        # Vertical divider
        tk.Frame(stats, bg=BG_CARD, width=1).pack(
            side="left", fill="y", padx=10, pady=6)
 
        # Detected labels list
        right = tk.Frame(stats, bg=BG_PANEL)
        right.pack(side="left", padx=24, fill="both", expand=True)
        tk.Label(right, text="DETECTED LABELS", font=FONT_LABEL,
                 bg=BG_PANEL, fg=TEXT_MUTED).pack(anchor="w")
        self.lbl_objects = tk.Label(
            right, text="—", font=FONT_MONO,
            bg=BG_PANEL, fg=TEXT_SUCCESS,
            wraplength=380, justify="left")
        self.lbl_objects.pack(anchor="w")
 
    # ------------------------------------------------------------
    # BUTTON FACTORY
    # ------------------------------------------------------------
    def _make_button(self, parent, text, command, bg_color):
        btn = tk.Button(
            parent, text=text, command=command,
            font=FONT_BTN, bg=bg_color, fg=TEXT_PRIMARY,
            activebackground=ACCENT_DIM, activeforeground=TEXT_PRIMARY,
            relief="flat", padx=18, pady=8, cursor="hand2", bd=0)
        return btn
 
    # ------------------------------------------------------------
    # STATUS LABEL
    # ------------------------------------------------------------
    def _set_status(self, msg, color=TEXT_MUTED):
        self.root.after(0, lambda: self.status_label.config(
            text=msg, fg=color))
 
    def _enable_buttons(self):
        self.root.after(0, lambda: (
            self.btn_image.config(state="normal"),
            self.btn_webcam.config(state="normal")
        ))
 
    # ------------------------------------------------------------
    # IMAGE DETECTION
    # ------------------------------------------------------------
    def _pick_image(self):
        """Open a file picker, run YOLO on the chosen image."""
        # Stop webcam if it's running
        if self.webcam_active:
            self._stop_webcam()
 
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
 
        if not path:
            return  # User cancelled
 
        self._set_status("Running detection...", ACCENT)
 
        # Run in background so UI doesn't freeze
        threading.Thread(
            target=self._run_image_detection, args=(path,), daemon=True
        ).start()
 
    def _run_image_detection(self, path):
        try:
            t0 = time.perf_counter()
            results = self.model(path, conf=CONF_IMAGE, verbose=False)
            elapsed = (time.perf_counter() - t0) * 1000  # ms
 
            result = results[0]
            annotated = result.plot()                # Draw bounding boxes
            annotated = fit_image(annotated)         # Resize to fit display
 
            # Gather detection info
            count = len(result.boxes)
            labels = []
            for box in result.boxes:
                cid  = int(box.cls[0])
                name = result.names[cid]
                conf = float(box.conf[0])
                labels.append(f"{name} {conf:.0%}")
 
            # Update UI on the main thread
            self.root.after(0, lambda: self._show_image_result(
                annotated, count, elapsed, labels))
 
        except Exception as e:
            self._set_status(f"Error: {e}", BTN_DANGER)
 
    def _show_image_result(self, cv_img, count, elapsed_ms, labels):
        self._display_frame(cv_img)
        self.lbl_count.config(text=str(count))
        self.lbl_time.config(text=f"{elapsed_ms:.0f} ms")
        self.lbl_objects.config(
            text=",  ".join(labels) if labels else "nothing detected")
        self._set_status("Detection complete", TEXT_SUCCESS)
 
    # ------------------------------------------------------------
    # WEBCAM DETECTION
    # ------------------------------------------------------------
    def _toggle_webcam(self):
        if self.webcam_active:
            self._stop_webcam()
        else:
            self._start_webcam()
 
    def _start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Webcam Error",
                "Could not open webcam.\n"
                "Try changing VideoCapture(0) to VideoCapture(1).")
            return
 
        self.webcam_active = True
        self.btn_webcam.config(text="⏹  Stop Webcam", bg=BTN_DANGER)
        self._set_status("Webcam running — detecting in real time", ACCENT)
 
        # Run the detection loop in a background thread
        # This is important — if we ran it on the main thread,
        # the UI would freeze because the loop never ends
        self.webcam_thread = threading.Thread(
            target=self._webcam_loop, daemon=True)
        self.webcam_thread.start()
 
    def _webcam_loop(self):
        """
        The core real-time loop.
        Runs in a background thread continuously until stopped.
        Each iteration:
          1. Read a frame from the webcam
          2. Run YOLO on it
          3. Draw boxes on the frame
          4. Send the annotated frame to the UI
        """
        fps_counter = 0
        fps_start   = time.perf_counter()
        fps_display = 0.0
 
        while self.webcam_active:
            ret, frame = self.cap.read()
            if not ret:
                break
 
            # Run YOLO — this is the single forward pass
            t0 = time.perf_counter()
            results = self.model(frame, conf=CONF_WEBCAM, verbose=False)
            elapsed = (time.perf_counter() - t0) * 1000
 
            result      = results[0]
            annotated   = result.plot()
            count       = len(result.boxes)
 
            # FPS calculation
            fps_counter += 1
            if time.perf_counter() - fps_start >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_start   = time.perf_counter()
 
            # Draw FPS overlay on the frame
            cv2.putText(annotated, f"FPS: {fps_display}",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 212, 255), 2)
 
            # Gather labels
            labels = []
            for box in result.boxes:
                cid  = int(box.cls[0])
                name = result.names[cid]
                conf = float(box.conf[0])
                labels.append(f"{name} {conf:.0%}")
 
            # Resize and send to UI (must update UI from main thread)
            annotated_small = fit_image(annotated)
            self.root.after(0, lambda f=annotated_small, c=count,
                            e=elapsed, l=labels:
                            self._show_webcam_frame(f, c, e, l))
 
    def _show_webcam_frame(self, cv_img, count, elapsed_ms, labels):
        self._display_frame(cv_img)
        self.lbl_count.config(text=str(count))
        self.lbl_time.config(text=f"{elapsed_ms:.0f} ms")
        self.lbl_objects.config(
            text=",  ".join(labels) if labels else "nothing detected")
 
    def _stop_webcam(self):
        self.webcam_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_webcam.config(text="📷  Start Webcam", bg=BTN_DARK)
        self._set_status("Webcam stopped", TEXT_MUTED)
 
    # ------------------------------------------------------------
    # DISPLAY A FRAME ON THE CANVAS
    # ------------------------------------------------------------
    def _display_frame(self, cv_img):
        """Convert a CV2 image and draw it on the Tkinter canvas."""
        tk_img = cv2_to_tk(cv_img)
        self._tk_image = tk_img   # Keep reference! Python GC will delete it otherwise
 
        # Centre the image on the canvas
        x = DISPLAY_W // 2
        y = DISPLAY_H // 2
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor="center", image=tk_img)
 
    # ------------------------------------------------------------
    # EXIT
    # ------------------------------------------------------------
    def _on_exit(self):
        """Clean up before closing."""
        self._stop_webcam()
        self.root.destroy()
 
 
# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    # Install Pillow if missing
    try:
        from PIL import Image, ImageTk
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image, ImageTk
 
    root = tk.Tk()
 
    # Centre window on screen
    win_w, win_h = 900, 780
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - win_w) // 2
    y = (screen_h - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")
 
    app = YOLOApp(root)
    root.mainloop()
 