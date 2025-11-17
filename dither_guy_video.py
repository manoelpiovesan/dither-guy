import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QFileDialog, QColorDialog,
    QScrollArea, QGroupBox, QMessageBox, QSplitter, QToolBar, QCheckBox,
    QProgressBar
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QMutex
from PySide6.QtGui import QPixmap, QImage, QAction, QIcon


class DitherWorker(QThread):
    """Worker thread for dithering images/video frames"""
    finished = Signal(object)
    progress = Signal(str)
    
    def __init__(self, img, pixel_size, threshold, replace_color, method, 
                 brightness, contrast, blur, sharpen):
        super().__init__()
        self.img = img
        self.pixel_size = pixel_size
        self.threshold = threshold
        self.replace_color = replace_color
        self.method = method
        self.brightness = brightness
        self.contrast = contrast
        self.blur = blur
        self.sharpen = sharpen
        self._is_running = True
        self.mutex = QMutex()
    
    def run(self):
        try:
            self.progress.emit("Processing...")
            dithered_img = apply_dither_optimized(
                self.img, self.pixel_size, self.threshold,
                self.replace_color, self.method,
                self.brightness, self.contrast, self.blur, self.sharpen
            )
            self.mutex.lock()
            if self._is_running:
                self.finished.emit(dithered_img)
            self.mutex.unlock()
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
    
    def stop(self):
        self.mutex.lock()
        self._is_running = False
        self.mutex.unlock()


class VideoProcessWorker(QThread):
    """Worker thread for processing and saving video"""
    frame_processed = Signal(object)
    progress = Signal(int, int)
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, video_path, save_path, pixel_size, threshold, replace_color, 
                 method, brightness, contrast, blur, sharpen):
        super().__init__()
        self.video_path = video_path
        self.save_path = save_path
        self.pixel_size = pixel_size
        self.threshold = threshold
        self.replace_color = replace_color
        self.method = method
        self.brightness = brightness
        self.contrast = contrast
        self.blur = blur
        self.sharpen = sharpen
        self._is_running = True
        self.mutex = QMutex()
    
    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("Failed to open video")
                return
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.save_path, fourcc, fps, (width, height))
            
            frame_count = 0
            
            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Apply dithering
                dithered_img = apply_dither_optimized(
                    img, self.pixel_size, self.threshold,
                    self.replace_color, self.method,
                    self.brightness, self.contrast, self.blur, self.sharpen
                )
                
                # Convert back to BGR for video writer
                dithered_array = np.array(dithered_img)
                dithered_bgr = cv2.cvtColor(dithered_array, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(dithered_bgr)
                
                frame_count += 1
                self.progress.emit(frame_count, total_frames)
                
                # Emit preview frame occasionally
                if frame_count % 5 == 0:
                    self.frame_processed.emit(dithered_img)
            
            cap.release()
            out.release()
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self.mutex.lock()
        self._is_running = False
        self.mutex.unlock()


def apply_image_adjustments(img, brightness, contrast, blur, sharpen):
    """Apply image adjustments before dithering"""
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    
    if sharpen > 0:
        for _ in range(int(sharpen)):
            img = img.filter(ImageFilter.SHARPEN)
    
    return img


def apply_dither_optimized(img, pixel_size, threshold, replace_color, method,
                          brightness=1.0, contrast=1.0, blur=0, sharpen=0):
    """Optimized dithering with various algorithms"""
    img = apply_image_adjustments(img, brightness, contrast, blur, sharpen)
    img = img.convert('L')
    
    new_size = (max(1, img.width // pixel_size), max(1, img.height // pixel_size))
    img = img.resize(new_size, Image.NEAREST)
    img_array = np.array(img, dtype=np.float32)

    if method == "Bayer":
        bayer_matrix = np.array([
            [0, 128, 32, 160],
            [192, 64, 224, 96],
            [48, 176, 16, 144],
            [240, 112, 208, 80]
        ], dtype=np.float32)
        
        bayer_matrix = (bayer_matrix / 255.0) * 255
        h, w = img_array.shape
        tile_h, tile_w = (h + 3) // 4, (w + 3) // 4
        tiled = np.tile(bayer_matrix, (tile_h, tile_w))[:h, :w]
        img_array = np.where(img_array + tiled / 255.0 * (255 - threshold) > threshold, 255, 0).astype(np.uint8)

    elif method == "Clustered-Dot":
        cluster_matrix = np.array([
            [12, 5, 6, 13],
            [4, 0, 1, 7],
            [11, 3, 2, 8],
            [15, 10, 9, 14]
        ], dtype=np.float32)
        
        cluster_matrix = (cluster_matrix / 16.0) * 255
        h, w = img_array.shape
        tile_h, tile_w = (h + 3) // 4, (w + 3) // 4
        tiled = np.tile(cluster_matrix, (tile_h, tile_w))[:h, :w]
        img_array = np.where(img_array + (tiled - 128) * (1 - threshold / 255.0) > threshold, 255, 0).astype(np.uint8)

    elif method == "Floyd-Steinberg":
        img_array = img_array.astype(np.float32)
        h, w = img_array.shape
        for y in range(h):
            for x in range(w):
                old_pixel = img_array[y, x]
                new_pixel = 255.0 if old_pixel > threshold else 3.0
                img_array[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                if x + 1 < w:
                    img_array[y, x + 1] += error * 0.4375
                if y + 1 < h:
                    if x > 0:
                        img_array[y + 1, x - 1] += error * 0.1875
                    img_array[y + 1, x] += error * 0.3125
                    if x + 1 < w:
                        img_array[y + 1, x + 1] += error * 0.0625
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    elif method == "Random":
        random_threshold = np.random.randint(0, 256, img_array.shape, dtype=np.float32)
        effective_threshold = threshold * 0.7 + random_threshold * 0.3
        img_array = np.where(img_array > effective_threshold, 255, 0).astype(np.uint8)

    # Convert back to PIL Image
    img = Image.fromarray(img_array, mode='L')
    img = img.resize((new_size[0] * pixel_size, new_size[1] * pixel_size), Image.NEAREST)
    
    # Convert to RGB and replace white with chosen color
    img = img.convert("RGB")
    img_data = np.array(img)
    mask = (img_data[:, :, 0] == 255) & (img_data[:, :, 1] == 255) & (img_data[:, :, 2] == 255)
    img_data[mask] = replace_color
    
    return Image.fromarray(img_data)


class ZoomableImageLabel(QLabel):
    """Optimized zoomable image label"""
    def __init__(self):
        super().__init__()
        self.zoom_level = 1.0
        self.original_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setScaledContents(False)
    
    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.zoom_level = 0
        self.update_display()
    
    def update_display(self):
        if not self.original_pixmap:
            return
        
        if self.zoom_level == 0:
            scaled = self.original_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.setPixmap(scaled)
        else:
            width = int(self.original_pixmap.width() * self.zoom_level)
            height = int(self.original_pixmap.height() * self.zoom_level)
            scaled = self.original_pixmap.scaled(
                width, height,
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.setPixmap(scaled)
    
    def zoom_in(self):
        if self.zoom_level == 0:
            self.zoom_level = 1.0
        self.zoom_level = min(self.zoom_level * 1.25, 5.0)
        self.update_display()
    
    def zoom_out(self):
        if self.zoom_level == 0:
            self.zoom_level = 1.0
        self.zoom_level = max(self.zoom_level * 0.8, 0.1)
        self.update_display()
    
    def fit_to_window(self):
        self.zoom_level = 0
        self.update_display()
    
    def actual_size(self):
        self.zoom_level = 1.0
        self.update_display()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.zoom_level == 0 and self.original_pixmap:
            self.update_display()


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_cap = None
        self.video_path = None
        self.current_frame = None
        self.is_playing = False
        self.worker = None
        self.video_worker = None
        self.current_color = (101, 138, 0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_next_frame)
        
        self.init_ui()
        self.apply_dark_theme()
        icon_path = Path("app_icon.png")
        icon_pixmap = QPixmap(icon_path)
        icon = QIcon(icon_pixmap)
        self.setWindowIcon(icon)
    
    def init_ui(self):
        self.setWindowTitle("Dither Guy - Video Edition")
        self.setMinimumSize(1200, 800)
        
        self.create_toolbar()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        self.create_video_panel(splitter)
        self.create_control_panel(splitter)
        
        splitter.setSizes([900, 300])
        
        self.statusBar().showMessage("Ready - Load a video or image")
    
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
        fit_action = QAction("Fit", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.fit_to_window)
        toolbar.addAction(fit_action)
        
        toolbar.addSeparator()
        
        self.zoom_label = QLabel("Fit")
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        toolbar.addWidget(self.zoom_label)
    
    def create_video_panel(self, parent):
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        
        self.info_label = QLabel("No video loaded")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 11px; padding: 6px; background: #252525;")
        video_layout.addWidget(self.info_label)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.video_label = ZoomableImageLabel()
        self.video_label.setText("Load a video or image\n\nSupported: MP4, AVI, PNG, JPG")
        self.video_label.setStyleSheet("font-size: 14px; color: #666;")
        
        scroll_area.setWidget(self.video_label)
        video_layout.addWidget(scroll_area)
        
        # Progress bar for video processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        video_layout.addWidget(self.progress_bar)
        
        parent.addWidget(video_widget)
    
    def create_control_panel(self, parent):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(8)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(6)
        
        btn_open = QPushButton("Open Video/Image")
        btn_open.clicked.connect(self.open_file)
        btn_open.setMinimumHeight(32)
        file_layout.addWidget(btn_open)
        
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setMinimumHeight(32)
        self.btn_play.setEnabled(False)
        file_layout.addWidget(self.btn_play)
        
        btn_save = QPushButton("Save Video")
        btn_save.clicked.connect(self.save_video)
        btn_save.setMinimumHeight(32)
        file_layout.addWidget(btn_save)
        
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        # Dithering method
        method_group = QGroupBox("Dithering Algorithm")
        method_layout = QVBoxLayout()
        
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Bayer",
            "Clustered-Dot",
            "Floyd-Steinberg",
            "Random"
        ])
        self.method_combo.setMinimumHeight(28)
        method_layout.addWidget(self.method_combo)
        
        method_group.setLayout(method_layout)
        control_layout.addWidget(method_group)
        
        # Parameters
        params_group = QGroupBox("Dither Parameters")
        params_layout = QVBoxLayout()
        params_layout.setSpacing(4)
        
        self.pixel_label = QLabel("Pixel Size: 4")
        self.pixel_slider = QSlider(Qt.Horizontal)
        self.pixel_slider.setMinimum(1)
        self.pixel_slider.setMaximum(20)
        self.pixel_slider.setValue(4)
        self.pixel_slider.valueChanged.connect(self.on_pixel_changed)
        params_layout.addWidget(self.pixel_label)
        params_layout.addWidget(self.pixel_slider)
        
        self.threshold_label = QLabel("Threshold: 128")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        params_layout.addWidget(self.threshold_label)
        params_layout.addWidget(self.threshold_slider)
        
        params_group.setLayout(params_layout)
        control_layout.addWidget(params_group)
        
        # Image adjustments
        adjust_group = QGroupBox("Image Adjustments")
        adjust_layout = QVBoxLayout()
        adjust_layout.setSpacing(4)
        
        self.brightness_label = QLabel("Brightness: 1.0")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(0)
        self.brightness_slider.setMaximum(200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        adjust_layout.addWidget(self.brightness_label)
        adjust_layout.addWidget(self.brightness_slider)
        
        self.contrast_label = QLabel("Contrast: 1.0")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(0)
        self.contrast_slider.setMaximum(200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        adjust_layout.addWidget(self.contrast_label)
        adjust_layout.addWidget(self.contrast_slider)
        
        self.blur_label = QLabel("Blur: 0")
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setMinimum(0)
        self.blur_slider.setMaximum(5)
        self.blur_slider.setValue(0)
        self.blur_slider.valueChanged.connect(self.on_blur_changed)
        adjust_layout.addWidget(self.blur_label)
        adjust_layout.addWidget(self.blur_slider)
        
        self.sharpen_label = QLabel("Sharpen: 0")
        self.sharpen_slider = QSlider(Qt.Horizontal)
        self.sharpen_slider.setMinimum(0)
        self.sharpen_slider.setMaximum(3)
        self.sharpen_slider.setValue(0)
        self.sharpen_slider.valueChanged.connect(self.on_sharpen_changed)
        adjust_layout.addWidget(self.sharpen_label)
        adjust_layout.addWidget(self.sharpen_slider)
        
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self.reset_adjustments)
        btn_reset.setMinimumHeight(28)
        adjust_layout.addWidget(btn_reset)
        
        adjust_group.setLayout(adjust_layout)
        control_layout.addWidget(adjust_group)
        
        # Color settings
        color_group = QGroupBox("Color Settings")
        color_layout = QVBoxLayout()
        color_layout.setSpacing(6)
        
        self.color_preview = QLabel()
        self.color_preview.setFixedHeight(35)
        self.color_preview.setStyleSheet(
            f"background-color: rgb{self.current_color}; border: 2px solid #555; border-radius: 4px;"
        )
        color_layout.addWidget(self.color_preview)
        
        btn_color = QPushButton("Choose Color")
        btn_color.clicked.connect(self.choose_color)
        btn_color.setMinimumHeight(32)
        color_layout.addWidget(btn_color)
        
        color_group.setLayout(color_layout)
        control_layout.addWidget(color_group)
        
        control_layout.addStretch()
        
        about_label = QLabel("Dither Guy v3.0\nVideo Edition")
        about_label.setAlignment(Qt.AlignCenter)
        about_label.setStyleSheet("font-size: 10px; color: #555; padding: 8px;")
        control_layout.addWidget(about_label)
        
        parent.addWidget(control_widget)
    
    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
            QToolBar {
                background-color: #252525;
                border-bottom: 1px solid #3a3a3a;
                padding: 3px;
            }
            QToolBar QLabel {
                color: #658a00;
                font-weight: bold;
                padding: 0 6px;
            }
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 6px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
            QPushButton:pressed {
                background-color: #252525;
            }
            QPushButton:disabled {
                background-color: #1e1e1e;
                color: #555;
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e0e0e0;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                selection-background-color: #658a00;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #2d2d2d;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #658a00;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #7aa800;
            }
            QScrollArea {
                border: 1px solid #3a3a3a;
                background-color: #1a1a1a;
            }
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #658a00;
                border-radius: 3px;
            }
            QStatusBar {
                background-color: #252525;
                color: #888;
                font-size: 11px;
            }
        """)
    
    def on_pixel_changed(self, value):
        self.pixel_label.setText(f"Pixel Size: {value}")
    
    def on_threshold_changed(self, value):
        self.threshold_label.setText(f"Threshold: {value}")
    
    def on_brightness_changed(self, value):
        brightness = value / 100.0
        self.brightness_label.setText(f"Brightness: {brightness:.1f}")
    
    def on_contrast_changed(self, value):
        contrast = value / 100.0
        self.contrast_label.setText(f"Contrast: {contrast:.1f}")
    
    def on_blur_changed(self, value):
        self.blur_label.setText(f"Blur: {value}")
    
    def on_sharpen_changed(self, value):
        self.sharpen_label.setText(f"Sharpen: {value}")
    
    def reset_adjustments(self):
        self.brightness_slider.setValue(100)
        self.contrast_slider.setValue(100)
        self.blur_slider.setValue(0)
        self.sharpen_slider.setValue(0)
    
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video or Image",
            "",
            "Media Files (*.mp4 *.avi *.mov *.png *.jpg *.jpeg *.bmp);;All Files (*.*)"
        )
        
        if file_path:
            ext = Path(file_path).suffix.lower()
            
            if ext in ['.mp4', '.avi', '.mov']:
                # Load video
                if self.video_cap is not None:
                    self.video_cap.release()
                
                self.video_cap = cv2.VideoCapture(file_path)
                self.video_path = file_path
                
                if self.video_cap.isOpened():
                    fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    self.info_label.setText(
                        f"{Path(file_path).name} - {total_frames} frames @ {fps:.1f} fps"
                    )
                    self.btn_play.setEnabled(True)
                    self.statusBar().showMessage(f"Loaded: {Path(file_path).name}")
                    self.show_next_frame()
                else:
                    QMessageBox.critical(self, "Error", "Failed to open video")
            
            else:
                # Load image
                try:
                    img = Image.open(file_path)
                    self.current_frame = img
                    self.info_label.setText(
                        f"{Path(file_path).name} - {img.width}x{img.height}px"
                    )
                    self.process_current_frame()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to open image:\n{str(e)}")
    
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.is_playing = True
            self.btn_play.setText("Pause")
            if self.video_cap is not None:
                fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                self.timer.start(int(1000 / fps))
    
    def show_next_frame(self):
        if self.video_cap is None or not self.video_cap.isOpened():
            return
        
        ret, frame = self.video_cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = Image.fromarray(frame_rgb)
            self.process_current_frame()
        else:
            # Loop video
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.is_playing:
                self.show_next_frame()
    
    def process_current_frame(self):
        if self.current_frame is None:
            return
        
        brightness = self.brightness_slider.value() / 100.0
        contrast = self.contrast_slider.value() / 100.0
        blur = self.blur_slider.value()
        sharpen = self.sharpen_slider.value()
        
        dithered_img = apply_dither_optimized(
            self.current_frame,
            self.pixel_slider.value(),
            self.threshold_slider.value(),
            self.current_color,
            self.method_combo.currentText(),
            brightness,
            contrast,
            blur,
            sharpen
        )
        
        # Convert to QPixmap and display
        img_data = dithered_img.tobytes("raw", "RGB")
        qimage = QImage(
            img_data,
            dithered_img.width,
            dithered_img.height,
            dithered_img.width * 3,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        self.video_label.set_image(pixmap)
    
    def save_video(self):
        if self.video_cap is None:
            QMessageBox.warning(self, "Warning", "No video loaded!")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video",
            "dithered_video.mp4",
            "MP4 Files (*.mp4)"
        )
        
        if save_path:
            # Stop playback
            if self.is_playing:
                self.toggle_play()
            
            # Reset video to start
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            brightness = self.brightness_slider.value() / 100.0
            contrast = self.contrast_slider.value() / 100.0
            blur = self.blur_slider.value()
            sharpen = self.sharpen_slider.value()
            
            # Create worker thread
            self.video_worker = VideoProcessWorker(
                self.video_path,
                save_path,
                self.pixel_slider.value(),
                self.threshold_slider.value(),
                self.current_color,
                self.method_combo.currentText(),
                brightness,
                contrast,
                blur,
                sharpen
            )
            
            self.video_worker.frame_processed.connect(self.on_preview_frame)
            self.video_worker.progress.connect(self.on_save_progress)
            self.video_worker.finished.connect(self.on_save_finished)
            self.video_worker.error.connect(self.on_save_error)
            
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage("Processing video...")
            
            self.video_worker.start()
    
    def on_preview_frame(self, dithered_img):
        img_data = dithered_img.tobytes("raw", "RGB")
        qimage = QImage(
            img_data,
            dithered_img.width,
            dithered_img.height,
            dithered_img.width * 3,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        self.video_label.set_image(pixmap)
    
    def on_save_progress(self, current, total):
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(f"Processing: {current}/{total} frames ({progress}%)")
    
    def on_save_finished(self):
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Video saved successfully!")
        QMessageBox.information(self, "Success", "Video saved successfully!")
        
        # Reset video to start
        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def on_save_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Error saving video")
        QMessageBox.critical(self, "Error", f"Failed to save video:\n{error_msg}")
    
    def choose_color(self):
        from PySide6.QtGui import QColor
        color = QColorDialog.getColor(
            QColor(*self.current_color),
            self,
            "Choose Replacement Color"
        )
        
        if color.isValid():
            self.current_color = (color.red(), color.green(), color.blue())
            self.color_preview.setStyleSheet(
                f"background-color: rgb{self.current_color}; border: 2px solid #555; border-radius: 4px;"
            )
    
    def zoom_in(self):
        self.video_label.zoom_in()
        self.update_zoom_label()
    
    def zoom_out(self):
        self.video_label.zoom_out()
        self.update_zoom_label()
    
    def fit_to_window(self):
        self.video_label.fit_to_window()
        self.zoom_label.setText("Fit")
    
    def update_zoom_label(self):
        if self.video_label.zoom_level == 0:
            self.zoom_label.setText("Fit")
        else:
            self.zoom_label.setText(f"{int(self.video_label.zoom_level * 100)}%")
    
    def closeEvent(self, event):
        if self.video_cap is not None:
            self.video_cap.release()
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Dither Guy - Video Edition")
    
    window = VideoPlayer()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
