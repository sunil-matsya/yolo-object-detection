"""
app.py
------
YOLO Object Detection — Desktop UI (v1.1.0)

Three detection modes:
  1. Image Detection  — upload any image, detect objects instantly
  2. Webcam Detection — live real-time detection from your camera
  3. Video Analysis   — upload .mp4, get unique objects with crop photos,
                        frame counts, timestamps, and saved report
"""

import threading
import time
import os

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

import cv2
from ultralytics import YOLO
from detect_video import process_video, OUTPUT_FOLDER

# ================================================================
# CONFIGURATION
# ================================================================
MODEL_NAME   = "yolov8n.pt"
CONF_IMAGE   = 0.45
CONF_WEBCAM  = 0.40
DISPLAY_W    = 820
DISPLAY_H    = 460

BG_DARK      = "#0f1117"
BG_PANEL     = "#1a1d27"
BG_CARD      = "#22263a"
ACCENT       = "#00d4ff"
ACCENT_DIM   = "#007a99"
TEXT_PRIMARY = "#e8eaf6"
TEXT_MUTED   = "#6b7280"
TEXT_SUCCESS = "#22c55e"
BTN_DANGER   = "#ef4444"
BTN_DARK     = "#2d3148"
BTN_VIDEO    = "#7c3aed"

FONT_TITLE   = ("Segoe UI", 20, "bold")
FONT_SUB     = ("Segoe UI", 11)
FONT_LABEL   = ("Segoe UI", 9)
FONT_MONO    = ("Consolas", 10)
FONT_BTN     = ("Segoe UI", 10, "bold")
FONT_COUNT   = ("Segoe UI", 28, "bold")
FONT_SMALL   = ("Segoe UI", 8)


# ================================================================
# HELPERS
# ================================================================
def fit_image(cv_img, max_w=DISPLAY_W, max_h=DISPLAY_H):
    h, w = cv_img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        cv_img = cv2.resize(cv_img, (int(w*scale), int(h*scale)),
                            interpolation=cv2.INTER_AREA)
    return cv_img


def cv2_to_tk(cv_img):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def fit_crop(cv_img, size=80):
    """Resize a crop image to a thumbnail square."""
    h, w = cv_img.shape[:2]
    scale = size / max(h, w)
    return cv2.resize(cv_img, (int(w*scale), int(h*scale)),
                      interpolation=cv2.INTER_AREA)


# ================================================================
# MAIN APP
# ================================================================
class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Object Detection  v1.1.0")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        self.model          = None
        self.webcam_active  = False
        self.webcam_thread  = None
        self.cap            = None
        self._tk_image      = None
        self._crop_images   = []   # Keep references to crop thumbnails

        self._build_ui()
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_exit)

    # ── MODEL ────────────────────────────────────────────────────
    def _load_model(self):
        self._set_status("Loading YOLOv8...", ACCENT)
        try:
            self.model = YOLO(MODEL_NAME)
            self._set_status(f"Ready — {MODEL_NAME}", TEXT_SUCCESS)
            self._enable_buttons()
        except Exception as e:
            self._set_status(f"Model error: {e}", BTN_DANGER)

    # ── UI ───────────────────────────────────────────────────────
    def _build_ui(self):

        # Header
        header = tk.Frame(self.root, bg=BG_DARK, pady=14)
        header.pack(fill="x", padx=28)
        tk.Label(header, text="⬡  YOLO", font=FONT_TITLE,
                 bg=BG_DARK, fg=ACCENT).pack(side="left")
        tk.Label(header, text="Object Detection", font=FONT_TITLE,
                 bg=BG_DARK, fg=TEXT_PRIMARY).pack(side="left", padx=(8,0))
        self.status_label = tk.Label(header, text="Initialising...",
                                      font=FONT_LABEL, bg=BG_DARK, fg=TEXT_MUTED)
        self.status_label.pack(side="right")

        tk.Frame(self.root, bg=ACCENT, height=1).pack(fill="x", padx=28)

        # Control buttons
        ctrl = tk.Frame(self.root, bg=BG_DARK, pady=12)
        ctrl.pack(fill="x", padx=28)

        self.btn_image  = self._btn(ctrl, "📁  Upload Image",
                                    self._pick_image, ACCENT)
        self.btn_webcam = self._btn(ctrl, "📷  Start Webcam",
                                    self._toggle_webcam, BTN_DARK)
        self.btn_video  = self._btn(ctrl, "📹  Analyse Video",
                                    self._pick_video, BTN_VIDEO)
        self.btn_exit   = self._btn(ctrl, "✕  Exit",
                                    self._on_exit, BTN_DANGER)

        self.btn_image.pack(side="left", padx=(0,8))
        self.btn_webcam.pack(side="left", padx=(0,8))
        self.btn_video.pack(side="left", padx=(0,8))
        self.btn_exit.pack(side="right")

        self.btn_image.config(state="disabled")
        self.btn_webcam.config(state="disabled")
        self.btn_video.config(state="disabled")

        # Notebook — two tabs: Preview | Video Results
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                         background=BG_DARK, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                         background=BG_CARD, foreground=TEXT_MUTED,
                         padding=[14, 6], font=("Segoe UI", 9, "bold"))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG_PANEL)],
                  foreground=[("selected", ACCENT)])

        self.notebook = ttk.Notebook(self.root, style="Dark.TNotebook")
        self.notebook.pack(fill="both", padx=28, pady=(4,0))

        # Tab 1 — Preview canvas
        tab_preview = tk.Frame(self.notebook, bg=BG_CARD)
        self.notebook.add(tab_preview, text="  Preview  ")

        self.canvas = tk.Canvas(tab_preview, width=DISPLAY_W, height=DISPLAY_H,
                                bg=BG_CARD, bd=0, highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_text(DISPLAY_W//2, DISPLAY_H//2,
                                text="Upload an image, start the webcam,\nor analyse a video",
                                font=("Segoe UI", 13), fill=TEXT_MUTED, justify="center")

        # Tab 2 — Video results
        tab_results = tk.Frame(self.notebook, bg=BG_PANEL)
        self.notebook.add(tab_results, text="  Video Results  ")
        self._build_results_tab(tab_results)

        # Progress bar (hidden until video processing)
        self.progress_frame = tk.Frame(self.root, bg=BG_DARK)
        self.progress_frame.pack(fill="x", padx=28, pady=(6,0))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, variable=self.progress_var,
            maximum=100, length=DISPLAY_W)
        self.progress_msg = tk.Label(
            self.progress_frame, text="", font=FONT_LABEL,
            bg=BG_DARK, fg=TEXT_MUTED)
        # Don't pack yet — shown only during video processing

        # Stats bar
        stats = tk.Frame(self.root, bg=BG_PANEL, pady=10)
        stats.pack(fill="x", padx=28, pady=(6, 16))

        left = tk.Frame(stats, bg=BG_PANEL)
        left.pack(side="left", padx=20)
        tk.Label(left, text="OBJECTS DETECTED", font=FONT_LABEL,
                 bg=BG_PANEL, fg=TEXT_MUTED).pack()
        self.lbl_count = tk.Label(left, text="—", font=FONT_COUNT,
                                   bg=BG_PANEL, fg=ACCENT)
        self.lbl_count.pack()

        tk.Frame(stats, bg=BG_CARD, width=1).pack(
            side="left", fill="y", padx=8, pady=4)

        mid = tk.Frame(stats, bg=BG_PANEL)
        mid.pack(side="left", padx=20)
        tk.Label(mid, text="INFERENCE TIME", font=FONT_LABEL,
                 bg=BG_PANEL, fg=TEXT_MUTED).pack()
        self.lbl_time = tk.Label(mid, text="—", font=FONT_COUNT,
                                  bg=BG_PANEL, fg=TEXT_PRIMARY)
        self.lbl_time.pack()

        tk.Frame(stats, bg=BG_CARD, width=1).pack(
            side="left", fill="y", padx=8, pady=4)

        right = tk.Frame(stats, bg=BG_PANEL)
        right.pack(side="left", padx=20, fill="both", expand=True)
        tk.Label(right, text="DETECTED LABELS", font=FONT_LABEL,
                 bg=BG_PANEL, fg=TEXT_MUTED).pack(anchor="w")
        self.lbl_objects = tk.Label(right, text="—", font=FONT_MONO,
                                     bg=BG_PANEL, fg=TEXT_SUCCESS,
                                     wraplength=400, justify="left")
        self.lbl_objects.pack(anchor="w")

    def _build_results_tab(self, parent):
        """Scrollable results panel for video analysis."""
        # Scrollable canvas
        self.results_canvas = tk.Canvas(parent, bg=BG_PANEL,
                                         highlightthickness=0,
                                         height=DISPLAY_H)
        scrollbar = ttk.Scrollbar(parent, orient="vertical",
                                   command=self.results_canvas.yview)
        self.results_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.results_canvas.pack(side="left", fill="both", expand=True)

        self.results_inner = tk.Frame(self.results_canvas, bg=BG_PANEL)
        self.results_canvas.create_window(
            (0, 0), window=self.results_inner, anchor="nw")
        self.results_inner.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")))

        # Placeholder
        self.results_placeholder = tk.Label(
            self.results_inner,
            text="Analyse a video to see results here",
            font=("Segoe UI", 12), bg=BG_PANEL, fg=TEXT_MUTED)
        self.results_placeholder.pack(pady=80)

    # ── BUTTON FACTORY ───────────────────────────────────────────
    def _btn(self, parent, text, cmd, bg):
        return tk.Button(parent, text=text, command=cmd,
                         font=FONT_BTN, bg=bg, fg=TEXT_PRIMARY,
                         activebackground=ACCENT_DIM,
                         activeforeground=TEXT_PRIMARY,
                         relief="flat", padx=16, pady=8,
                         cursor="hand2", bd=0)

    def _set_status(self, msg, color=TEXT_MUTED):
        self.root.after(0, lambda: self.status_label.config(
            text=msg, fg=color))

    def _enable_buttons(self):
        self.root.after(0, lambda: [
            self.btn_image.config(state="normal"),
            self.btn_webcam.config(state="normal"),
            self.btn_video.config(state="normal")])

    # ── IMAGE DETECTION ──────────────────────────────────────────
    def _pick_image(self):
        if self.webcam_active:
            self._stop_webcam()
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        self._set_status("Running detection...", ACCENT)
        threading.Thread(target=self._run_image,
                         args=(path,), daemon=True).start()

    def _run_image(self, path):
        try:
            t0 = time.perf_counter()
            results = self.model(path, conf=CONF_IMAGE, verbose=False)
            elapsed = (time.perf_counter() - t0) * 1000
            result  = results[0]
            ann     = fit_image(result.plot())
            count   = len(result.boxes)
            labels  = [f"{result.names[int(b.cls[0])]} {float(b.conf[0]):.0%}"
                       for b in result.boxes]
            self.root.after(0, lambda: self._show_image_result(
                ann, count, elapsed, labels))
        except Exception as e:
            self._set_status(f"Error: {e}", BTN_DANGER)

    def _show_image_result(self, cv_img, count, ms, labels):
        self._display_frame(cv_img)
        self.notebook.select(0)
        self.lbl_count.config(text=str(count))
        self.lbl_time.config(text=f"{ms:.0f} ms")
        self.lbl_objects.config(
            text=",  ".join(labels) or "nothing detected")
        self._set_status("Detection complete", TEXT_SUCCESS)

    # ── WEBCAM ───────────────────────────────────────────────────
    def _toggle_webcam(self):
        if self.webcam_active:
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not open webcam.")
            return
        self.webcam_active = True
        self.btn_webcam.config(text="⏹  Stop Webcam", bg=BTN_DANGER)
        self._set_status("Webcam running", ACCENT)
        self.notebook.select(0)
        self.webcam_thread = threading.Thread(
            target=self._webcam_loop, daemon=True)
        self.webcam_thread.start()

    def _webcam_loop(self):
        fps_c, fps_t, fps_d = 0, time.perf_counter(), 0
        while self.webcam_active:
            ret, frame = self.cap.read()
            if not ret:
                break
            t0      = time.perf_counter()
            results = self.model(frame, conf=CONF_WEBCAM, verbose=False)
            elapsed = (time.perf_counter() - t0) * 1000
            result  = results[0]
            ann     = result.plot()
            count   = len(result.boxes)
            labels  = [f"{result.names[int(b.cls[0])]} {float(b.conf[0]):.0%}"
                       for b in result.boxes]
            fps_c  += 1
            if time.perf_counter() - fps_t >= 1.0:
                fps_d = fps_c; fps_c = 0; fps_t = time.perf_counter()
            cv2.putText(ann, f"FPS: {fps_d}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,212,255), 2)
            small = fit_image(ann)
            self.root.after(0, lambda f=small, c=count, e=elapsed, l=labels:
                            self._show_webcam_frame(f, c, e, l))

    def _show_webcam_frame(self, cv_img, count, ms, labels):
        self._display_frame(cv_img)
        self.lbl_count.config(text=str(count))
        self.lbl_time.config(text=f"{ms:.0f} ms")
        self.lbl_objects.config(text=",  ".join(labels) or "nothing detected")

    def _stop_webcam(self):
        self.webcam_active = False
        if self.cap:
            self.cap.release(); self.cap = None
        self.btn_webcam.config(text="📷  Start Webcam", bg=BTN_DARK)
        self._set_status("Webcam stopped", TEXT_MUTED)

    # ── VIDEO ANALYSIS ───────────────────────────────────────────
    def _pick_video(self):
        if self.webcam_active:
            self._stop_webcam()
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self._start_video_analysis(path)

    def _start_video_analysis(self, path):
        # Show progress bar
        self.progress_bar.pack(fill="x", pady=(4,0))
        self.progress_msg.pack(anchor="w")
        self.progress_var.set(0)
        self._set_status("Analysing video...", ACCENT)
        self.btn_video.config(state="disabled")
        self.notebook.select(0)

        # Clear previous results
        for w in self.results_inner.winfo_children():
            w.destroy()
        self._crop_images.clear()

        threading.Thread(
            target=self._run_video, args=(path,), daemon=True).start()

    def _run_video(self, path):
        try:
            detections = process_video(
                path,
                self.model,
                progress_callback=self._video_progress,
                frame_callback=self._video_frame_preview
            )
            self.root.after(0, lambda: self._show_video_results(detections))
        except Exception as e:
            self._set_status(f"Error: {e}", BTN_DANGER)
            self.root.after(0, lambda: self.btn_video.config(state="normal"))

    def _video_progress(self, pct, msg):
        self.root.after(0, lambda: (
            self.progress_var.set(pct),
            self.progress_msg.config(text=msg)
        ))

    def _video_frame_preview(self, frame):
        small = fit_image(frame)
        self.root.after(0, lambda f=small: self._display_frame(f))

    def _show_video_results(self, detections):
        """Build the results tab with crop thumbnails and details."""

        # Hide progress bar
        self.progress_bar.pack_forget()
        self.progress_msg.pack_forget()

        # Sort by frame count descending
        sorted_items = sorted(
            detections.items(), key=lambda x: x[1]["count"], reverse=True)

        # Update stats bar
        unique_count = len(detections)
        all_labels   = [f"{n} ({d['best_conf']:.0%})"
                        for n, d in sorted_items]
        self.lbl_count.config(text=str(unique_count))
        self.lbl_time.config(text=f"{unique_count} types")
        self.lbl_objects.config(text=",  ".join(all_labels))

        # Header in results tab
        tk.Label(self.results_inner,
                 text=f"Found {unique_count} unique object type(s)",
                 font=("Segoe UI", 11, "bold"),
                 bg=BG_PANEL, fg=ACCENT).pack(
                     anchor="w", padx=16, pady=(12, 4))

        tk.Label(self.results_inner,
                 text=f"Crops saved to:  {os.path.abspath(OUTPUT_FOLDER)}",
                 font=FONT_LABEL, bg=BG_PANEL, fg=TEXT_MUTED).pack(
                     anchor="w", padx=16, pady=(0, 10))

        tk.Frame(self.results_inner, bg=BG_CARD, height=1).pack(
            fill="x", padx=16, pady=(0, 8))

        # One row per unique object
        for class_name, data in sorted_items:
            self._results_row(class_name, data)

        # Open folder button
        tk.Button(
            self.results_inner,
            text="📂  Open detections folder",
            command=lambda: os.startfile(os.path.abspath(OUTPUT_FOLDER)),
            font=FONT_BTN, bg=BTN_DARK, fg=TEXT_PRIMARY,
            relief="flat", padx=14, pady=7, cursor="hand2", bd=0
        ).pack(anchor="w", padx=16, pady=16)

        # Switch to results tab and re-enable button
        self.notebook.select(1)
        self.btn_video.config(state="normal")
        self._set_status(
            f"Analysis complete — {unique_count} unique objects found",
            TEXT_SUCCESS)

    def _results_row(self, class_name, data):
        """Build one result row: [thumbnail] | name | stats"""
        row = tk.Frame(self.results_inner, bg=BG_CARD,
                       pady=10, padx=12)
        row.pack(fill="x", padx=16, pady=4)

        # Thumbnail
        thumb_label = tk.Label(row, bg=BG_CARD)
        thumb_label.pack(side="left", padx=(0, 14))

        if data.get("crop") is not None and data["crop"].size > 0:
            try:
                thumb_cv  = fit_crop(data["crop"], size=90)
                tk_thumb  = cv2_to_tk(thumb_cv)
                self._crop_images.append(tk_thumb)   # prevent GC
                thumb_label.config(image=tk_thumb)
            except Exception:
                thumb_label.config(text="[no img]", fg=TEXT_MUTED,
                                   font=FONT_SMALL, width=8)
        else:
            thumb_label.config(text="[no img]", fg=TEXT_MUTED,
                               font=FONT_SMALL, width=8)

        # Info
        info = tk.Frame(row, bg=BG_CARD)
        info.pack(side="left", fill="both", expand=True)

        tk.Label(info, text=class_name.upper(),
                 font=("Segoe UI", 13, "bold"),
                 bg=BG_CARD, fg=TEXT_PRIMARY).pack(anchor="w")

        details = (
            f"Appeared in  {data['count']}  frames    •    "
            f"First seen at  {data['first_time']}    •    "
            f"Best confidence  {data['best_conf']:.0%}"
        )
        tk.Label(info, text=details, font=FONT_MONO,
                 bg=BG_CARD, fg=TEXT_MUTED).pack(anchor="w", pady=(3,0))

        if data.get("crop_path"):
            tk.Label(info,
                     text=f"Saved → {data['crop_path']}",
                     font=FONT_SMALL, bg=BG_CARD, fg=TEXT_SUCCESS).pack(
                         anchor="w", pady=(2,0))

    # ── CANVAS DISPLAY ───────────────────────────────────────────
    def _display_frame(self, cv_img):
        tk_img = cv2_to_tk(cv_img)
        self._tk_image = tk_img
        self.canvas.delete("all")
        self.canvas.create_image(
            DISPLAY_W//2, DISPLAY_H//2, anchor="center", image=tk_img)

    # ── EXIT ─────────────────────────────────────────────────────
    def _on_exit(self):
        self._stop_webcam()
        self.root.destroy()


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image, ImageTk

    root = tk.Tk()
    win_w, win_h = 920, 820
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{win_w}x{win_h}+{(sw-win_w)//2}+{(sh-win_h)//2}")
    YOLOApp(root)
    root.mainloop()