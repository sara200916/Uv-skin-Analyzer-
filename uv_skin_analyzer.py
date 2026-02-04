# Sara Mhisen بسم الله
"""
UV Skin Analyzer - Upload + Forecast + Trend Tracker (Full)
Dependencies: opencv-python, numpy, PyQt5, matplotlib

Run:
pip install opencv-python numpy pyqt5 matplotlib
python uv_skin_analyzer_pro.py
"""

import sys
import os
import time
import math
from datetime import datetime, timedelta

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
    QGridLayout, QFrame, QFileDialog, QDialog, QListWidget, QListWidgetItem,
    QFormLayout, QLineEdit, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ---------------- Helper: compute metrics & masks ----------------
def compute_metrics(frame):
    """
    Compute simulated skin metrics and create masks.
    Returns metrics dict (float 0-100) and masks (oil_mask, dry_mask, pigment_mask) normalized 0..1 resized to frame.
    """
    if frame is None:
        return None
    proc = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)

    # Brightness used for hydration inference (higher brightness under UV -> more fluorescence -> hydrated)
    brightness = np.clip((gray.astype(np.float32).mean() / 255.0) * 100, 0, 100)

    # Local variance (texture)
    kernel = (7, 7)
    local_mean = cv2.blur(gray.astype(np.float32), kernel)
    local_sq_mean = cv2.blur((gray.astype(np.float32) ** 2), kernel)
    local_var = np.maximum(0, local_sq_mean - (local_mean ** 2))
    texture_score = np.clip((local_var.mean() / (255.0 ** 2)) * 1000, 0, 100)

    # HSV channels for detection
    v = hsv[:, :, 2].astype(np.uint8)
    h = hsv[:, :, 0].astype(np.uint8)
    s = hsv[:, :, 1].astype(np.uint8)

    # Oil detection: specular highlights (very bright small regions)
    _, oil_mask = cv2.threshold(v, 220, 255, cv2.THRESH_BINARY)
    oil_score = np.clip(np.sum(oil_mask > 0) / (oil_mask.size) * 500, 0, 100)

    # Pigmentation: darker brown-ish pixels (low V, moderate S, H in red-yellow ranges)
    pigment_mask = ((v < 100) & (s > 50) & ((h < 25) | (h > 160))).astype(np.uint8) * 255
    pigment_score = np.clip(np.sum(pigment_mask > 0) / pigment_mask.size * 500, 0, 100)

    # Dead cells / horny layer: bright white spots in gray
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    dead_cells_score = np.clip(np.sum(white_mask > 0) / white_mask.size * 400, 0, 100)

    # Hydration proxy: brightness & blue prominence
    blue = proc[:, :, 0].astype(np.float32).mean()
    green = proc[:, :, 1].astype(np.float32).mean()
    red = proc[:, :, 2].astype(np.float32).mean()
    blue_ratio = np.clip((blue / (red + green + 1.0)) * 100, 0, 100)
    hydrated_score = np.clip(brightness * 0.6 + blue_ratio * 0.4, 0, 100)

    # Thin / dehydrated proxies
    thin_score = np.clip(max(0, 50 - brightness * 0.6 + texture_score * 0.2), 0, 100)
    dehydrated_score = np.clip((texture_score * 0.8 + (50 - brightness) * 0.4), 0, 100)

    # Thick corneum
    thick_score = np.clip(brightness * 0.6 + texture_score * 0.5, 0, 100)

    # Overall health composite
    overall = (
            0.5 * hydrated_score +
            0.2 * (100 - pigment_score) +
            0.15 * (100 - oil_score) +
            0.15 * (100 - texture_score)
    )
    overall = np.clip(overall, 0, 100)

    # Damage %
    damage = np.clip((pigment_score * 0.5 + texture_score * 0.4 + (100 - hydrated_score) * 0.6) / 2.0, 0, 100)

    # Normalize masks to 0..1 and resize to original frame size
    heat_oil = (oil_mask.astype(np.float32) / 255.0)
    heat_pigment = (pigment_mask.astype(np.float32) / 255.0)
    heat_dead = (white_mask.astype(np.float32) / 255.0)

    # Dry areas: where brightness is low and texture is high (approx)
    dry_mask = ((gray < 90) & (local_var > (np.percentile(local_var, 60)))).astype(np.uint8)
    heat_dry = (dry_mask.astype(np.float32))

    # Resize masks to match input frame size
    h_full, w_full = frame.shape[:2]
    oil_full = cv2.resize(heat_oil, (w_full, h_full))
    pig_full = cv2.resize(heat_pigment, (w_full, h_full))
    dead_full = cv2.resize(heat_dead, (w_full, h_full))
    dry_full = cv2.resize(heat_dry, (w_full, h_full))

    # Clamp to 0..1
    oil_full = np.clip(oil_full, 0.0, 1.0)
    pig_full = np.clip(pig_full, 0.0, 1.0)
    dead_full = np.clip(dead_full, 0.0, 1.0)
    dry_full = np.clip(dry_full, 0.0, 1.0)

    metrics = {
        "overall": float(overall),
        "hydrated": float(hydrated_score),
        "thin_skin": float(thin_score),
        "dehydrated": float(dehydrated_score),
        "oily": float(oil_score),
        "pigmentation": float(pigment_score),
        "dead_cells": float(dead_cells_score),
        "thick_corneum": float(thick_score),
        "damage": float(damage),
        "heat_oil": oil_full,
        "heat_dry": dry_full,
        "heat_pigment": pig_full,
        "heat_dead": dead_full
    }
    return metrics


# ---------------- GUI helper components ----------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=3.5, height=2.0, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        fig.tight_layout()


class SkinCard(QFrame):
    def __init__(self, title, color):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("border-radius:8px; background-color: #f5f6f7;")
        layout = QVBoxLayout()
        self.title_lbl = QLabel(title)
        self.title_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        self.value_lbl = QLabel("0%")
        self.value_lbl.setFont(QFont("Arial", 16))
        self.color_bar = QLabel()
        self.color_bar.setFixedHeight(12)
        self.color_bar.setStyleSheet(f"background: {color}; border-radius:6px;")
        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.color_bar)
        self.setLayout(layout)

    def set_value(self, v):
        self.value_lbl.setText(f"{v:.1f}%")


# ---------------- Main App ----------------
class UVApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SkinHealth Scanner")
        self.setGeometry(120, 80, 1200, 700)
        self.init_ui()

        # capture/video state
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.history = []  # list of overall health numbers for quick plotting
        self.last_history_time = time.time()
        self.captured_snapshots = []  # list of tuples (filename, metrics, timestamp)
        self.current_frame = None
        self.video_file = None

    def init_ui(self):
        main_layout = QHBoxLayout()

        # Left: video/display area
        left_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("background-color: #222; border-radius:6px;")
        left_layout.addWidget(self.video_label)

        controls = QHBoxLayout()
        self.start_cam_btn = QPushButton("Start Camera")
        self.start_cam_btn.clicked.connect(self.start_webcam)
        self.load_file_btn = QPushButton("Upload Image/Video")
        self.load_file_btn.clicked.connect(self.load_file)
        self.capture_btn = QPushButton("Capture Snapshot")
        self.capture_btn.clicked.connect(self.capture)
        self.predict_btn = QPushButton("Skin Forecast")
        self.predict_btn.clicked.connect(self.show_forecast)
        self.trend_btn = QPushButton("Trend Tracker")
        self.trend_btn.clicked.connect(self.open_trend_tracker)
        self.quit_btn = QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close)

        controls.addWidget(self.start_cam_btn)
        controls.addWidget(self.load_file_btn)
        controls.addWidget(self.capture_btn)
        controls.addWidget(self.predict_btn)
        controls.addWidget(self.trend_btn)
        controls.addWidget(self.quit_btn)
        left_layout.addLayout(controls)

        # Right: dashboard
        right_layout = QVBoxLayout()
        header = QLabel("SkinHealth Scanner")
        header.setFont(QFont("Arial", 18, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(header)

        grid = QGridLayout()
        self.card_overall = SkinCard("Overall Skin Health", "#a8d0ff")
        self.card_hyd = SkinCard("Hydrated Skin", "#80d0ff")
        self.card_thin = SkinCard("Thin / Low Moisture", "#c08bff")
        self.card_dehyd = SkinCard("Dehydrated Skin", "#d6b3ff")
        self.card_oil = SkinCard("Oily Areas", "#ffd166")
        self.card_pig = SkinCard("Pigmentation / Dark Spots", "#f3c25f")
        self.card_dead = SkinCard("Dead Cells / Horny Layer", "#ffffff")
        self.card_thick = SkinCard("Thick Corneum Layer", "#ffffff")
        self.card_damage = SkinCard("Damage %", "#ff6b6b")

        grid.addWidget(self.card_overall, 0, 0, 1, 2)
        grid.addWidget(self.card_damage, 0, 2, 1, 1)
        grid.addWidget(self.card_hyd, 1, 0)
        grid.addWidget(self.card_thin, 1, 1)
        grid.addWidget(self.card_dehyd, 1, 2)
        grid.addWidget(self.card_oil, 2, 0)
        grid.addWidget(self.card_pig, 2, 1)
        grid.addWidget(self.card_dead, 2, 2)
        grid.addWidget(self.card_thick, 3, 0)

        right_layout.addLayout(grid)

        # Small chart for history
        self.canvas = MplCanvas(width=4.0, height=2.0)
        right_layout.addWidget(self.canvas)

        # small controls under chart
        sub_controls = QHBoxLayout()
        self.overlay_alpha = 0.5  # default overlay transparency
        self.alpha_increase = QPushButton("Overlay +")
        self.alpha_inc_label = QLabel(f"{self.overlay_alpha:.2f}")
        self.alpha_decrease = QPushButton("Overlay -")
        self.alpha_increase.clicked.connect(self.increase_alpha)
        self.alpha_decrease.clicked.connect(self.decrease_alpha)
        sub_controls.addWidget(self.alpha_decrease)
        sub_controls.addWidget(self.alpha_inc_label)
        sub_controls.addWidget(self.alpha_increase)
        right_layout.addLayout(sub_controls)

        right_layout.addStretch()

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 2)
        self.setLayout(main_layout)

    # ---------------- Overlay alpha ----------------
    def refresh_overlay(self):
        if self.current_frame is not None:
            self.update_frame()

    def increase_alpha(self):
        self.overlay_alpha = min(0.95, self.overlay_alpha + 0.1)
        self.alpha_inc_label.setText(f"{self.overlay_alpha:.2f}")
        self.refresh_overlay()

    def decrease_alpha(self):
        self.overlay_alpha = max(0.0, self.overlay_alpha - 0.1)
        self.alpha_inc_label.setText(f"{self.overlay_alpha:.2f}")
        self.refresh_overlay()

    # ---------------- Webcam / Load -----------------
    def start_webcam(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(0)
        self.video_file = None
        self.timer.start(30)

    def load_file(self):
        dlg = QFileDialog()
        file_path, _ = dlg.getOpenFileName(self, "Select Image or Video", "",
                                           "Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi *.mov)")
        if not file_path:
            return
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.warning(self, "Load error", "Could not read image file.")
                return
            self.current_frame = img.copy()
            # stop any active capture
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.video_file = None
            # update display immediately
            self.update_frame()
        else:
            # video
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = cv2.VideoCapture(file_path)
            self.video_file = file_path
            self.timer.start(30)

    # ---------------- Frame update & overlay ----------------
    def update_frame(self):
        # acquire frame (from camera, video, or loaded image)
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                # if video ended and file mode, stop timer
                if self.video_file is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                    self.timer.stop()
                    return
                else:
                    return
            self.current_frame = frame.copy()
        else:
            if self.current_frame is None:
                return
            frame = self.current_frame.copy()

        # compute metrics
        metrics = compute_metrics(frame)

        # update cards
        self.card_overall.set_value(metrics["overall"])
        self.card_hyd.set_value(metrics["hydrated"])
        self.card_thin.set_value(metrics["thin_skin"])
        self.card_dehyd.set_value(metrics["dehydrated"])
        self.card_oil.set_value(metrics["oily"])
        self.card_pig.set_value(metrics["pigmentation"])
        self.card_dead.set_value(metrics["dead_cells"])
        self.card_thick.set_value(metrics["thick_corneum"])
        self.card_damage.set_value(metrics["damage"])

        # history (update once per second)
        now = time.time()
        if now - self.last_history_time >= 1.0:
            self.history.append(metrics["overall"])
            if len(self.history) > 120:
                self.history.pop(0)
            self.draw_history()
            self.last_history_time = now

        # build AR overlay heatmap
        hmap_oil = metrics["heat_oil"]
        hmap_dry = metrics["heat_dry"]
        hmap_pig = metrics["heat_pigment"]
        hmap_dead = metrics["heat_dead"]

        # scale masks and create color layers
        overlay = frame.copy().astype(np.float32) / 255.0
        color_layer = np.zeros_like(overlay)

        # red channel = oil
        color_layer[:, :, 2] += hmap_oil * 1.0  # red
        # blue channel = dry (make it prominent in blue)
        color_layer[:, :, 0] += hmap_dry * 1.0  # blue
        # yellow = pigment (red + green)
        color_layer[:, :, 2] += hmap_pig * 0.8  # red
        color_layer[:, :, 1] += hmap_pig * 0.9  # green
        # white-ish overlay for dead cells
        color_layer[:, :, :] += np.expand_dims(hmap_dead * 0.9, axis=2)

        # Clip color_layer
        color_layer = np.clip(color_layer, 0.0, 1.0)

        alpha = self.overlay_alpha
        combined = (1.0 - alpha) * overlay + alpha * color_layer
        combined = np.clip(combined * 255.0, 0, 255).astype(np.uint8)

        # Draw a legend box
        cv2.rectangle(combined, (10, 10), (440, 100), (20, 20, 20), -1)
        cv2.putText(combined, "Legend:", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        cv2.putText(combined, "Red = Oily    Blue = Dry    Yellow = Pigment    White = Dead cells", (18, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        # Convert and show
        rgb_image = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

        # store last_metrics for snapshot/report
        self.last_metrics = metrics

    # ---------------- plotting history ----------------
    def draw_history(self):
        self.canvas.axes.cla()
        self.canvas.axes.set_title("Overall Health History")
        self.canvas.axes.set_ylim(0, 100)
        self.canvas.axes.set_xlabel("Frames")
        self.canvas.axes.set_ylabel("% Health")
        if len(self.history) > 0:
            self.canvas.axes.plot(self.history, color="#2a9df4", linewidth=2)
        self.canvas.draw()

    # ---------------- snapshot & saving ----------------
    def capture(self):
        if self.current_frame is None:
            QMessageBox.information(self, "No frame", "No frame available to capture.")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"snapshot_{timestamp}.png"
        cv2.imwrite(fname, self.current_frame)
        metrics = compute_metrics(self.current_frame)
        t = datetime.now()
        self.captured_snapshots.append((fname, metrics, t))
        self.card_overall.title_lbl.setText("Overall Skin Health")
        QMessageBox.information(self, "Saved", f"Saved snapshot {fname}")

    # ---------------- Forecast / Prediction ----------------
    def show_forecast(self):
        """
        Predict short-term risks based on current metrics and historical trend.
        - Acne risk from oil buildup
        - Sun damage risk from pigmentation and damage
        Also performs a simple trend projection for 'overall health' for the next 14/30 days.
        """
        if not hasattr(self, "last_metrics"):
            QMessageBox.information(self, "No data", "No scan data available. Capture or load an image first.")
            return
        m = self.last_metrics

        # Heuristic textual risk predictions
        risks = []
        acne_risk_score = m["oily"]  # 0..100
        if acne_risk_score > 60:
            risks.append(("High acne risk",
                          "Oil buildup suggests acne risk in next 2-3 days. Use oil-control and gentle cleansing."))
        elif acne_risk_score > 30:
            risks.append(("Moderate acne risk", "Some oil buildup detected — keep skin clean and monitor."))
        else:
            risks.append(("Low acne risk", "Oil levels low — acne unlikely from oil alone."))

        sun_risk_score = m["pigmentation"]
        if sun_risk_score > 50 or m["damage"] > 45:
            risks.append(("Sun damage risk",
                          "High pigmentation/damage detected — sunscreen and specialist consult recommended."))
        elif sun_risk_score > 25:
            risks.append(("Moderate sun risk", "Some pigmentation detected — use sunscreen regularly."))
        else:
            risks.append(("Low sun risk", "Pigmentation low."))

        # Simple trend projection for overall health using saved snapshots (if any), else use last metric
        # We'll convert snapshot timestamps to days (float) and fit linear model (1st degree). If not enough points, use last value.
        x = []
        y = []
        for (_, metrics, ts) in self.captured_snapshots:
            x.append(ts.timestamp() / 86400.0)  # days
            y.append(metrics["overall"])
        # include current moment (if not captured in snapshots)
        now_days = datetime.now().timestamp() / 86400.0
        if len(x) == 0:
            # no history: use current overall as constant
            proj_vals = [m["overall"]] * 7
            days_future = [1, 2, 3, 7, 14, 21, 30]
            projection_text = f"Projection (no history): overall ~ {m['overall']:.1f}% (stable)\n"
        else:
            # fit linear regression
            # ensure arrays
            x_arr = np.array(x)
            y_arr = np.array(y)
            # fit degree 1 polynomial
            coeffs = np.polyfit(x_arr, y_arr, 1)
            slope, intercept = coeffs[0], coeffs[1]
            days_future = [1, 2, 3, 7, 14, 21, 30]
            proj_vals = []
            for d in days_future:
                future_day = now_days + d
                val = slope * future_day + intercept
                proj_vals.append(float(np.clip(val, 0, 100)))
            projection_text = f"Projection (linear fit): slope={slope:.4f} overall/day\n"

        # Build forecast dialog text
        text_lines = []
        text_lines.append("Skin Forecast Summary\n")
        for title, advice in risks:
            text_lines.append(f"{title}: {advice}")
        text_lines.append("\nFuture overall health projection (next days):")
        for d, v in zip(days_future, proj_vals):
            text_lines.append(f"In {d} day(s): {v:.1f}% overall health")
        # Make a small dialog with forecast and plot
        dlg = QDialog(self)
        dlg.setWindowTitle("Skin Forecast & Projection")
        dlg.resize(600, 500)
        layout = QVBoxLayout()

        # Text area
        info_label = QLabel("\n".join(text_lines))
        info_label.setFont(QFont("Arial", 10))
        layout.addWidget(info_label)

        # Plot projection
        fig_canvas = MplCanvas(width=4.5, height=2.2)
        fig_canvas.axes.plot(list(range(-len(y), 0)), y if len(y) > 0 else [], label="History (last snapshots)")
        fig_canvas.axes.plot(days_future, proj_vals, marker='o', label="Projection")
        fig_canvas.axes.set_ylim(0, 100)
        fig_canvas.axes.set_xlabel("days (relative)")
        fig_canvas.axes.set_ylabel("Overall health (%)")
        fig_canvas.axes.legend()
        fig_canvas.draw()
        layout.addWidget(fig_canvas)

        dlg.setLayout(layout)
        dlg.exec_()

    # ---------------- Trend tracker window ----------------
    def open_trend_tracker(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Trend Tracker - Past Scans & Projection")
        dlg.resize(900, 600)
        layout = QHBoxLayout()

        # Left: list of saved snapshots
        left = QVBoxLayout()
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.MultiSelection)
        for (fname, metrics, ts) in self.captured_snapshots:
            item = QListWidgetItem(
                f"{os.path.basename(fname)}  |  {ts.strftime('%Y-%m-%d %H:%M:%S')}  |  Overall: {metrics['overall']:.1f}%")
            item.setData(Qt.UserRole, (fname, metrics, ts))
            list_widget.addItem(item)
        left.addWidget(list_widget)
        # Buttons for actions
        btns = QHBoxLayout()
        view_btn = QPushButton("View Selected")
        compare_btn = QPushButton("Compare Selected")
        export_btn = QPushButton("Export Report for Selected")
        btns.addWidget(view_btn)
        btns.addWidget(compare_btn)
        btns.addWidget(export_btn)
        left.addLayout(btns)

        layout.addLayout(left, 2)

        # Right: preview area and chart
        right = QVBoxLayout()
        preview_label = QLabel()
        preview_label.setFixedSize(500, 420)
        preview_label.setStyleSheet("background-color:#111; border-radius:6px;")
        right.addWidget(preview_label)
        chart_canvas = MplCanvas(width=5, height=2.5)
        right.addWidget(chart_canvas)
        layout.addLayout(right, 3)

        def view_selected():
            item = list_widget.currentItem()
            if not item:
                QMessageBox.information(dlg, "No selection", "Select a saved snapshot from the list.")
                return
            fname, metrics, ts = item.data(Qt.UserRole)
            img = cv2.imread(fname)
            if img is None:
                QMessageBox.warning(dlg, "File missing", f"{fname} not found.")
                return
            # show in preview
            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = display.shape
            qimg = QImage(display.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(preview_label.width(), preview_label.height(), Qt.KeepAspectRatio)
            preview_label.setPixmap(pix)

        def compare_selected():
            sel = list_widget.selectedItems()
            if len(sel) < 2:
                QMessageBox.information(dlg, "Select two", "Please select two snapshots (Ctrl+click) to compare.")
                return
            (f1, m1, t1) = sel[0].data(Qt.UserRole)
            (f2, m2, t2) = sel[1].data(Qt.UserRole)

            # --- Generate overlayed images for comparison ---
            def overlay_image(fname, metrics):
                img = cv2.imread(fname)
                if img is None:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = img.copy()
                hmap_oil = metrics["heat_oil"]
                hmap_dry = metrics["heat_dry"]
                hmap_pig = metrics["heat_pigment"]
                hmap_dead = metrics["heat_dead"]
                overlay = frame.astype(np.float32) / 255.0
                color_layer = np.zeros_like(overlay)
                color_layer[:, :, 2] += hmap_oil * 1.0  # red = oil
                color_layer[:, :, 0] += hmap_dry * 1.0  # blue = dry
                color_layer[:, :, 2] += hmap_pig * 0.8  # yellow (red+green)
                color_layer[:, :, 1] += hmap_pig * 0.9
                color_layer[:, :, :] += np.expand_dims(hmap_dead * 0.9, axis=2)
                color_layer = np.clip(color_layer, 0.0, 1.0)
                alpha = 0.5
                combined = (1.0 - alpha) * overlay + alpha * color_layer
                combined = np.clip(combined * 255.0, 0, 255).astype(np.uint8)
                return combined

            i1_overlay = overlay_image(f1, m1)
            i2_overlay = overlay_image(f2, m2)

            # --- Side-by-side composite ---
            h_target = 360
            i1r = cv2.resize(i1_overlay, (int(i1_overlay.shape[1] * h_target / i1_overlay.shape[0]), h_target))
            i2r = cv2.resize(i2_overlay, (int(i2_overlay.shape[1] * h_target / i2_overlay.shape[0]), h_target))
            composite = np.ones((h_target, i1r.shape[1] + i2r.shape[1], 3), dtype=np.uint8) * 245
            composite[:, :i1r.shape[1], :] = i1r
            composite[:, i1r.shape[1]:, :] = i2r

            display = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            h, w, ch = display.shape
            qimg = QImage(display.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(preview_label.width(), preview_label.height(), Qt.KeepAspectRatio)
            preview_label.setPixmap(pix)

            # --- Chart of overall health + 30-day projection ---
            chart_canvas.axes.cla()
            chart_canvas.axes.bar([0, 1], [m1["overall"], m2["overall"]], color=["#2a9df4", "#f39c12"])
            chart_canvas.axes.set_ylim(0, 100)
            chart_canvas.axes.set_xticks([0, 1])
            chart_canvas.axes.set_xticklabels([t1.strftime("%Y-%m-%d"), t2.strftime("%Y-%m-%d")])
            # Linear projection from m1 -> m2 to future 30 days
            proj_val = np.clip(m2["overall"] + (m2["overall"] - m1["overall"]), 0, 100)
            chart_canvas.axes.plot([0, 1, 2], [m1["overall"], m2["overall"], proj_val], marker='o', color="#2ca02c",
                                   label="Projection")
            chart_canvas.axes.set_ylabel("Overall Health (%)")
            chart_canvas.axes.legend()
            chart_canvas.draw()

        def export_selected():
            item = list_widget.currentItem()
            if not item:
                QMessageBox.information(dlg, "No selection", "Select a saved snapshot.")
                return
            fname, metrics, ts = item.data(Qt.UserRole)
            report_img = self.compose_report_image(fname, metrics)
            outname = f"export_report_{ts.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(outname, report_img)
            QMessageBox.information(dlg, "Exported", f"Report exported as {outname}")

        view_btn.clicked.connect(view_selected)
        compare_btn.clicked.connect(compare_selected)
        export_btn.clicked.connect(export_selected)

        dlg.setLayout(layout)
        dlg.exec_()

    # ---------------- Report composition & saving ----------------
    def compose_report_image(self, image_path, metrics):
        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        img_resized = cv2.resize(img, (640, 480))
        h, w, _ = img_resized.shape
        canvas_h = max(520, h + 140)
        canvas_w = 1000
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245
        canvas[20:20 + h, 20:20 + w, :] = img_resized
        x0 = 680
        y = 40
        cv2.putText(canvas, "Scan Report", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2)
        y += 40
        cv2.putText(canvas, f"Overall Health: {metrics['overall']:.1f}%", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (60, 60, 60), 2)
        y += 35
        cv2.putText(canvas, f"Damage %: {metrics['damage']:.1f}%", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (180, 30, 30), 2)
        y += 35
        cv2.putText(canvas, f"Hydrated: {metrics['hydrated']:.1f}%", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (30, 120, 200), 1)
        y += 30
        cv2.putText(canvas, f"Oily: {metrics['oily']:.1f}%", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 160, 30), 1)
        y += 30
        cv2.putText(canvas, f"Pigmentation: {metrics['pigmentation']:.1f}%", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (120, 80, 40), 1)
        y += 30
        cv2.putText(canvas, f"Dead cells: {metrics['dead_cells']:.1f}%", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (60, 60, 60), 1)
        y += 40
        cv2.putText(canvas, "Recommendations:", (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        y += 28
        recos = self.generate_recommendations(metrics)
        for line in recos:
            # wrap text if too long (simple)
            wrapped = [line[i:i + 40] for i in range(0, len(line), 40)]
            for wline in wrapped:
                cv2.putText(canvas, "- " + wline, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
                y += 20
        # small history chart embed
        # create chart file temporarily
        chart_file = "temp_history_small.png"
        self.save_history_chart(chart_file, width=4, height=1.6)
        if os.path.exists(chart_file):
            ch_img = cv2.imread(chart_file)
            if ch_img is not None:
                ch, cw, _ = ch_img.shape
                # paste chart at bottom right of the report
                canvas[canvas_h - ch - 20:canvas_h - 20, x0:x0 + cw, :] = ch_img
        return canvas

    def save_history_chart(self, outname, width=3, height=1.6):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(width, height), dpi=100)
        if len(self.history) > 0:
            plt.plot(self.history[-120:], color="#2a9df4")
        plt.ylim(0, 100)
        plt.title("Overall Health (recent)")
        plt.xlabel("Frames")
        plt.tight_layout()
        plt.savefig(outname)
        plt.close()

    def generate_recommendations(self, metrics):
        recos = []

        if metrics["hydrated"] < 40:
            recos.append("Increase moisturizing (use humectants like glycerin).")
        else:
            recos.append("Hydration looks OK for this area.")
        if metrics["oily"] > 40:
            recos.append("Use gentle cleanser and blotting for oily zones.")
        if metrics["pigmentation"] > 30:
            recos.append("Use sunscreen and consider dermatology review for pigment spots.")
        if metrics["dead_cells"] > 30 or metrics["thick_corneum"] > 40:
            recos.append("Consider gentle exfoliation to remove dead cells.")
        if metrics["damage"] > 40:
            recos.append("High damage detected — avoid sun exposure and consult specialist if needed.")
        if len(recos) == 0:
            recos.append("No immediate actions; maintain healthy routine.")
        return recos


# ------------------ Main ------------------
def main():
    app = QApplication(sys.argv)
    win = UVApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
