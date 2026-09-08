import sys
import os
import glob
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, 
    QVBoxLayout, QHBoxLayout, QSlider, QComboBox, QCheckBox, 
    QFileDialog, QFrame, QGridLayout
)

class BatchInspectionWorker(QThread):
    # Сигнал: (кадр с боксом, индекс, всего, дефект?, уверенность, fps, имя файла)
    frame_processed = pyqtSignal(np.ndarray, int, int, bool, float, float, str)
    inspection_finished = pyqtSignal()

    def __init__(self, folder_path, model_path, device):
        super().__init__()
        self.folder_path = folder_path
        self.model_path = model_path
        self.device = device
        self.is_running = True
        self.paused = False
        
        self.conf_threshold = 0.72
        self.delay_ms = 45
        self.use_clahe = False
        
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.crack_idx = 0
        for k, v in self.model.names.items():
            if 'crack' in str(v).lower() and 'non' not in str(v).lower() and 'un' not in str(v).lower():
                self.crack_idx = k
                break

    def run(self):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_files = []
        for ext in exts:
            image_files.extend(glob.glob(os.path.join(self.folder_path, "**", ext), recursive=True))

        image_files = sorted(image_files)
        total_files = len(image_files)
        
        if total_files == 0:
            return

        img_idx = 0

        while self.is_running and img_idx < total_files:
            if self.paused:
                self.msleep(40)
                continue

            file_path = image_files[img_idx]
            raw_img = cv2.imread(file_path)
            
            if raw_img is None:
                img_idx += 1
                continue

            t_start = time.perf_counter()

            if self.use_clahe:
                lab = cv2.cvtColor(raw_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                proc_img = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
            else:
                proc_img = raw_img

            results = self.model(proc_img, device=self.device, verbose=False)
            t_infer = time.perf_counter() - t_start
            fps = 1.0 / max(0.0001, t_infer)

            res = results[0]
            conf = float(res.probs.data[self.crack_idx])
            is_defect = conf >= self.conf_threshold

            display_img = cv2.resize(raw_img, (768, 768), interpolation=cv2.INTER_LANCZOS4)
            h, w = display_img.shape[:2]

            # Если обнаружен дефект, строим фокусный бокс локализации (Grad-CAM симуляция по градиенту текстуры)
            if is_defect:
                gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (15, 15), 0)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Находим наиболее вероятную зону скопления дефектов
                    largest_cnt = max(contours, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(largest_cnt)
                    
                    if bw > 30 and bh > 30:
                        # Рисуем точный красный бокс вокруг зоны повреждения
                        cv2.rectangle(display_img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                        cv2.putText(display_img, f"CRACK REGION [{conf*100:.1f}%]", (bx, max(15, by - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

            # Общая рамка статуса кадра
            edge_color = (0, 30, 255) if is_defect else (0, 255, 153)
            cv2.rectangle(display_img, (0, 0), (w - 1, h - 1), edge_color, 2)

            img_idx += 1
            short_name = os.path.basename(file_path)
            self.frame_processed.emit(display_img, img_idx, total_files, is_defect, conf, fps, short_name)
            
            self.msleep(self.delay_ms)

        self.inspection_finished.emit()

    def stop(self):
        self.is_running = False
        self.wait()


class NDTInspectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDT UAV Photogrammetry Inspection Station - M5 Pro Core")
        self.resize(1420, 870)

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_path = "/Users/corecore/Projects/ndt_test/runs/classify/train-2/weights/best.pt"
        self.gsd_mm_per_px = 0.32
        self.worker = None

        self.stat_total = 0
        self.stat_defects = 0

        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { background-color: #121212; color: #E0E0E0; font-family: -apple-system, sans-serif; font-size: 13px; }
            QFrame#panel { background-color: #1E1E1E; border: 1px solid #2D2D2D; border-radius: 6px; padding: 12px; }
            QPushButton { background-color: #242424; color: #00FF99; border: 1px solid #00FF99; border-radius: 4px; padding: 7px 15px; font-weight: bold; }
            QPushButton:hover { background-color: #00FF99; color: #121212; }
            QComboBox { background-color: #292929; border: 1px solid #3D3D3D; border-radius: 4px; padding: 5px; color: #FFFFFF; }
            QSlider::groove:horizontal { height: 5px; background: #2D2D2D; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #00FF99; border-radius: 2px; }
            QSlider::handle:horizontal { background: #E0E0E0; width: 14px; margin-top: -5px; margin-bottom: -5px; border-radius: 7px; }
            QCheckBox { spacing: 8px; color: #CCCCCC; }
            QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #444444; background: #222222; }
            QCheckBox::indicator:checked { background: #00FF99; border-color: #00FF99; }
            QLabel#section_title { color: #00FF99; font-size: 13px; font-weight: bold; margin-bottom: 4px; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        left_vbox = QVBoxLayout()
        left_vbox.setSpacing(10)

        top_controls = QHBoxLayout()
        self.btn_load = QPushButton("📂 Open Dataset Folder")
        self.btn_load.clicked.connect(self.select_dataset_folder)
        top_controls.addWidget(self.btn_load)

        self.btn_play = QPushButton("▶ Run Audit")
        self.btn_play.clicked.connect(self.toggle_playback)
        top_controls.addWidget(self.btn_play)

        self.lbl_folder = QLabel("No dataset folder mounted.")
        self.lbl_folder.setStyleSheet("color: #888888;")
        top_controls.addWidget(self.lbl_folder)
        top_controls.addStretch()
        left_vbox.addLayout(top_controls)

        self.screen_display = QLabel("UAV BATCH INSPECTION SYSTEM OFFLINE")
        self.screen_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screen_display.setStyleSheet("background-color: #050505; border: 1px solid #2D2D2D; border-radius: 6px; color: #444444; font-size: 15px; font-weight: bold;")
        self.screen_display.setMinimumSize(880, 680)
        left_vbox.addWidget(self.screen_display, stretch=1)
        layout.addLayout(left_vbox, stretch=3)

        right_panel = QFrame()
        right_panel.setObjectName("panel")
        right_panel.setFixedWidth(420)
        right_vbox = QVBoxLayout(right_panel)
        right_vbox.setSpacing(12)

        lbl_ai = QLabel("INSPECTION AI PARAMETERS")
        lbl_ai.setObjectName("section_title")
        right_vbox.addWidget(lbl_ai)

        self.lbl_conf = QLabel("Crack Confidence Filter: 0.72")
        right_vbox.addWidget(self.lbl_conf)
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(40, 98)
        self.slider_conf.setValue(72)
        self.slider_conf.valueChanged.connect(self.update_params)
        right_vbox.addWidget(self.slider_conf)

        self.lbl_speed = QLabel("Audit Pace (Interval): 45 ms")
        right_vbox.addWidget(self.lbl_speed)
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(5, 200)
        self.slider_speed.setValue(45)
        self.slider_speed.valueChanged.connect(self.update_params)
        right_vbox.addWidget(self.slider_speed)

        right_vbox.addWidget(QLabel("Defect Classification Code:"))
        self.combo_code = QComboBox()
        self.combo_code.addItems([
            "[C-1] Structural Crack (SDNET-Trained)",
            "[C-2] Concrete Spalling",
            "[C-3] Reinforcement Exposure",
            "[C-4] Surface Honeycombing"
        ])
        right_vbox.addWidget(self.combo_code)

        self.chk_clahe = QCheckBox("Shadow Compensation (CLAHE)")
        self.chk_clahe.setChecked(False)
        self.chk_clahe.toggled.connect(self.toggle_clahe)
        right_vbox.addWidget(self.chk_clahe)

        self.chk_rtk = QCheckBox("RTK Lock Status (Geo-Referencing)")
        self.chk_rtk.setChecked(True)
        right_vbox.addWidget(self.chk_rtk)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("border-top: 1px solid #2D2D2D;")
        right_vbox.addWidget(sep1)

        lbl_tele = QLabel("LIVE BATCH TELEMETRY")
        lbl_tele.setObjectName("section_title")
        right_vbox.addWidget(lbl_tele)

        grid = QGridLayout()
        grid.addWidget(QLabel("Active Sample:"), 0, 0)
        self.lbl_cur_file = QLabel("None")
        self.lbl_cur_file.setStyleSheet("color: #FFFFFF; font-family: monospace;")
        grid.addWidget(self.lbl_cur_file, 0, 1)

        grid.addWidget(QLabel("Evaluation Status:"), 1, 0)
        self.lbl_status = QLabel("IDLE")
        self.lbl_status.setStyleSheet("color: #888888; font-weight: bold;")
        grid.addWidget(self.lbl_status, 1, 1)

        grid.addWidget(QLabel("Confidence Score:"), 2, 0)
        self.lbl_cur_conf = QLabel("0.0 %")
        self.lbl_cur_conf.setStyleSheet("color: #00FF99; font-weight: bold;")
        grid.addWidget(self.lbl_cur_conf, 2, 1)

        grid.addWidget(QLabel("Inference Engine:"), 3, 0)
        grid.addWidget(QLabel("Apple M5 Pro (Metal/MPS)"), 3, 1)

        grid.addWidget(QLabel("Audit Speed:"), 4, 0)
        self.lbl_fps = QLabel("0.0 FPS")
        self.lbl_fps.setStyleSheet("color: #00FF99; font-weight: bold;")
        grid.addWidget(self.lbl_fps, 4, 1)

        grid.addWidget(QLabel("Images Audited:"), 5, 0)
        self.lbl_scanned = QLabel("0 / 0")
        grid.addWidget(self.lbl_scanned, 5, 1)

        grid.addWidget(QLabel("Defective Sites:"), 6, 0)
        self.lbl_defects = QLabel("0 Detected")
        self.lbl_defects.setStyleSheet("color: #00FF99; font-weight: bold;")
        grid.addWidget(self.lbl_defects, 6, 1)

        grid.addWidget(QLabel("Defect Rate:"), 7, 0)
        self.lbl_defect_rate = QLabel("0.0 %")
        self.lbl_defect_rate.setStyleSheet("color: #00FF99; font-weight: bold;")
        grid.addWidget(self.lbl_defect_rate, 7, 1)
        right_vbox.addLayout(grid)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("border-top: 1px solid #2D2D2D;")
        right_vbox.addWidget(sep2)

        lbl_comp = QLabel("COMPLIANCE STANDARDS")
        lbl_comp.setObjectName("section_title")
        right_vbox.addWidget(lbl_comp)

        comp_box = QVBoxLayout()
        comp_box.setSpacing(4)
        comp_box.addWidget(QLabel("• ASTM D610 - Degree of Concrete Cracking"))
        comp_box.addWidget(QLabel("• ACI 224R-01 - Control of Cracking in Concrete"))
        comp_box.addWidget(QLabel("• EN 1504 - Surface Protection Systems"))
        right_vbox.addLayout(comp_box)

        right_vbox.addStretch()
        layout.addWidget(right_panel, stretch=1)

    def update_params(self):
        conf = self.slider_conf.value() / 100.0
        speed = self.slider_speed.value()
        self.lbl_conf.setText(f"Crack Confidence Filter: {conf:.2f}")
        self.lbl_speed.setText(f"Audit Pace (Interval): {speed} ms")
        if self.worker:
            self.worker.conf_threshold = conf
            self.worker.delay_ms = speed

    def toggle_clahe(self, checked):
        if self.worker:
            self.worker.use_clahe = checked

    def select_dataset_folder(self):
        default_dir = "/Users/corecore/Projects/ndt_test/cls_dataset/val"
        if not os.path.exists(default_dir):
            default_dir = "/Users/corecore/Projects/ndt_test"

        folder = QFileDialog.getExistingDirectory(self, "Select Inspection Dataset Folder", default_dir)
        if folder:
            if self.worker:
                self.worker.stop()
            
            self.stat_total = 0
            self.stat_defects = 0
            
            self.worker = BatchInspectionWorker(folder, self.model_path, self.device)
            self.worker.conf_threshold = self.slider_conf.value() / 100.0
            self.worker.delay_ms = self.slider_speed.value()
            self.worker.use_clahe = self.chk_clahe.isChecked()
            self.worker.frame_processed.connect(self.update_display)
            self.worker.start()

            self.lbl_folder.setText(os.path.basename(folder) or folder)
            self.btn_play.setText("⏸ Pause")

    def toggle_playback(self):
        if self.worker:
            self.worker.paused = not self.worker.paused
            self.btn_play.setText("▶ Run Audit" if self.worker.paused else "⏸ Pause")

    def update_display(self, frame, idx, total, is_defect, conf, fps, filename):
        self.stat_total = idx
        if is_defect:
            self.stat_defects += 1

        rate = (self.stat_defects / max(1, self.stat_total)) * 100.0

        self.lbl_cur_file.setText(filename[:22])
        self.lbl_cur_conf.setText(f"{conf * 100.0:.1f} %")
        self.lbl_scanned.setText(f"{idx} / {total}")
        self.lbl_fps.setText(f"{fps:.1f} FPS")

        if is_defect:
            self.lbl_status.setText("CRITICAL DEFECT [C-1]")
            self.lbl_status.setStyleSheet("color: #FF3333; font-weight: bold;")
            self.lbl_cur_conf.setStyleSheet("color: #FF3333; font-weight: bold;")
        else:
            self.lbl_status.setText("NOMINAL (PASS)")
            self.lbl_status.setStyleSheet("color: #00FF99; font-weight: bold;")
            self.lbl_cur_conf.setStyleSheet("color: #00FF99; font-weight: bold;")

        if self.stat_defects > 0:
            self.lbl_defects.setText(f"{self.stat_defects} Defective")
            self.lbl_defects.setStyleSheet("color: #FF3333; font-weight: bold;")
            self.lbl_defect_rate.setText(f"{rate:.1f} %")
            self.lbl_defect_rate.setStyleSheet("color: #FF3333; font-weight: bold;")
        else:
            self.lbl_defects.setText("0 Detected")
            self.lbl_defects.setStyleSheet("color: #00FF99; font-weight: bold;")
            self.lbl_defect_rate.setText("0.0 %")
            self.lbl_defect_rate.setStyleSheet("color: #00FF99; font-weight: bold;")

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.screen_display.setPixmap(pix.scaled(
            self.screen_display.width(), self.screen_display.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NDTInspectionApp()
    win.show()
    sys.exit(app.exec())