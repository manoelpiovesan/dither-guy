import sys
from pathlib import Path
import numpy as np
import random
from PIL import Image, ImageFilter, ImageEnhance
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QFileDialog, QColorDialog,
    QScrollArea, QGroupBox, QMessageBox, QSplitter, QToolBar, QCheckBox
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QMutex
from PySide6.QtGui import QPixmap, QImage, QAction, QIcon


class DitherWorker(QThread):
    """Optimized worker thread for dithering"""
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


def apply_image_adjustments(img, brightness, contrast, blur, sharpen):
    """Apply image adjustments before dithering"""
    # Apply brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    
    # Apply contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    
    # Apply blur
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    
    # Apply sharpen
    if sharpen > 0:
        for _ in range(int(sharpen)):
            img = img.filter(ImageFilter.SHARPEN)
    
    return img


def apply_dither_optimized(img, pixel_size, threshold, replace_color, method,
                          brightness=1.0, contrast=1.0, blur=0, sharpen=0):
    """Optimized dithering with various artistic algorithms"""
    # Apply image adjustments first
    img = apply_image_adjustments(img, brightness, contrast, blur, sharpen)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize down
    new_size = (max(1, img.width // pixel_size), max(1, img.height // pixel_size))
    img = img.resize(new_size, Image.NEAREST)
    img_array = np.array(img, dtype=np.float32)

    if method == "Bayer":
        # Standard Bayer matrix dithering
        bayer_matrix = np.array([
            [0, 128, 32, 160],
            [192, 64, 224, 96],
            [48, 176, 16, 144],
            [240, 112, 208, 80]
        ], dtype=np.float32)
        
        # Normalize and apply threshold
        bayer_matrix = (bayer_matrix / 255.0) * 255
        h, w = img_array.shape
        tile_h, tile_w = (h + 3) // 4, (w + 3) // 4
        tiled = np.tile(bayer_matrix, (tile_h, tile_w))[:h, :w]
        
        # Apply threshold comparison
        img_array = np.where(img_array + tiled / 255.0 * (255 - threshold) > threshold, 255, 0).astype(np.uint8)

    elif method == "Clustered-Dot":
        # Clustered dot dithering
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
        
        # Apply threshold
        img_array = np.where(img_array + (tiled - 128) * (1 - threshold / 255.0) > threshold, 255, 0).astype(np.uint8)

    elif method == "Halftone":
        # Halftone pattern dithering
        halftone_matrix = np.array([
            [7, 13, 11, 4],
            [12, 16, 14, 8],
            [10, 15, 6, 2],
            [5, 9, 3, 1]
        ], dtype=np.float32)
        
        halftone_matrix = (halftone_matrix / 17.0) * 255
        h, w = img_array.shape
        tile_h, tile_w = (h + 3) // 4, (w + 3) // 4
        tiled = np.tile(halftone_matrix, (tile_h, tile_w))[:h, :w]
        
        # Apply threshold
        img_array = np.where(img_array + (tiled - 128) * (1 - threshold / 255.0) > threshold, 255, 0).astype(np.uint8)

    elif method == "Blue-Noise":
        # Blue noise dithering
        np.random.seed(42)
        blue_noise = np.random.rand(*img_array.shape) * 255
        
        # Apply threshold-sensitive comparison
        adjusted_threshold = threshold + (blue_noise - 128) * 0.5
        img_array = np.where(img_array > adjusted_threshold, 255, 0).astype(np.uint8)

    elif method == "Random":
        # Random dithering with threshold
        random_threshold = np.random.randint(0, 256, img_array.shape, dtype=np.float32)
        # Mix threshold with random
        effective_threshold = threshold * 0.7 + random_threshold * 0.3
        img_array = np.where(img_array > effective_threshold, 255, 0).astype(np.uint8)

    elif method == "Pattern":
        # Pattern dithering with threshold
        pattern = np.zeros((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                pattern[i, j] = ((i + j) % 8) * 32
        
        h, w = img_array.shape
        tile_h, tile_w = (h + 7) // 8, (w + 7) // 8
        tiled = np.tile(pattern, (tile_h, tile_w))[:h, :w]
        
        # Apply threshold
        img_array = np.where(img_array + (tiled - 128) * (1 - threshold / 255.0) > threshold, 255, 0).astype(np.uint8)

    elif method == "Crosshatch":
        # Crosshatch pattern with threshold
        h, w = img_array.shape
        crosshatch = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                val = (np.sin(x * 0.5) + np.sin(y * 0.5)) * 64 + 128
                crosshatch[y, x] = val
        
        # Apply threshold
        img_array = np.where(img_array + (crosshatch - 128) * (1 - threshold / 255.0) > threshold, 255, 0).astype(np.uint8)

    elif method == "Riemersma":
        # Riemersma dithering with threshold
        img_array = img_array.astype(np.float32)
        h, w = img_array.shape
        error_buffer = [0] * 16
        
        for y in range(h):
            if y % 2 == 0:
                x_range = range(w)
            else:
                x_range = range(w - 1, -1, -1)
            
            for x in x_range:
                old_pixel = img_array[y, x] + error_buffer[0]
                new_pixel = 255.0 if old_pixel > threshold else 0.0
                img_array[y, x] = new_pixel
                error = old_pixel - new_pixel
                error_buffer = error_buffer[1:] + [error * 0.0625]
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    elif method == "Variable-Error":
        # Variable error diffusion with threshold
        img_array = img_array.astype(np.float32)
        h, w = img_array.shape
        
        for y in range(h - 1):
            for x in range(1, w - 1):
                old_pixel = img_array[y, x]
                new_pixel = 255.0 if old_pixel > threshold else 0.0
                img_array[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                intensity_factor = old_pixel / 255.0
                w1 = 7 * intensity_factor / 16
                w2 = 3 * (1 - intensity_factor) / 16
                w3 = 5 / 16
                w4 = 1 / 16
                
                img_array[y, x + 1] = np.clip(img_array[y, x + 1] + error * w1, 0, 255)
                img_array[y + 1, x - 1] = np.clip(img_array[y + 1, x - 1] + error * w2, 0, 255)
                img_array[y + 1, x] = np.clip(img_array[y + 1, x] + error * w3, 0, 255)
                img_array[y + 1, x + 1] = np.clip(img_array[y + 1, x + 1] + error * w4, 0, 255)
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    elif method == "Dot-Diffusion":
        # Dot diffusion with threshold
        class_matrix = np.array([
            [0, 2, 4, 6],
            [1, 3, 5, 7],
            [0, 2, 4, 6],
            [1, 3, 5, 7]
        ])
        
        img_array = img_array.astype(np.float32)
        h, w = img_array.shape
        
        for y in range(h):
            for x in range(w):
                cm = class_matrix[y % 4, x % 4]
                old_pixel = img_array[y, x]
                new_pixel = 255.0 if old_pixel > threshold else 0.0
                img_array[y, x] = new_pixel
                error = (old_pixel - new_pixel) / (cm + 1)
                
                if x + 1 < w:
                    img_array[y, x + 1] = np.clip(img_array[y, x + 1] + error, 0, 255)
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    elif method == "Floyd-Steinberg":
        img_array = img_array.astype(np.float32)
        h, w = img_array.shape
        for y in range(h):
            for x in range(w):
                old_pixel = img_array[y, x]
                new_pixel = 255.0 if old_pixel > threshold else 0.0
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

    elif method == "Atkinson":
        img_array = img_array.astype(np.float32)
        h, w = img_array.shape
        for y in range(h):
            for x in range(w):
                old_pixel = img_array[y, x]
                new_pixel = 255.0 if old_pixel > threshold else 0.0
                img_array[y, x] = new_pixel
                error = (old_pixel - new_pixel) * 0.125
                
                if x + 1 < w:
                    img_array[y, x + 1] += error
                if x + 2 < w:
                    img_array[y, x + 2] += error
                if y + 1 < h:
                    if x > 0:
                        img_array[y + 1, x - 1] += error
                    img_array[y + 1, x] += error
                    if x + 1 < w:
                        img_array[y + 1, x + 1] += error
                if y + 2 < h:
                    img_array[y + 2, x] += error
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    elif method == "Jarvis-Judice-Ninke":
        img_array = img_array.astype(np.float32)
        h, w = img_array.shape
        divisor = 48.0
        for y in range(h):
            for x in range(w):
                old_pixel = img_array[y, x]
                new_pixel = 255.0 if old_pixel > threshold else 0.0
                img_array[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                if x + 1 < w:
                    img_array[y, x + 1] += error * 7 / divisor
                if x + 2 < w:
                    img_array[y, x + 2] += error * 5 / divisor
                
                if y + 1 < h:
                    if x > 1:
                        img_array[y + 1, x - 2] += error * 3 / divisor
                    if x > 0:
                        img_array[y + 1, x - 1] += error * 5 / divisor
                    img_array[y + 1, x] += error * 7 / divisor
                    if x + 1 < w:
                        img_array[y + 1, x + 1] += error * 5 / divisor
                    if x + 2 < w:
                        img_array[y + 1, x + 2] += error * 3 / divisor
                
                if y + 2 < h:
                    if x > 1:
                        img_array[y + 2, x - 2] += error * 1 / divisor
                    if x > 0:
                        img_array[y + 2, x - 1] += error * 3 / divisor
                    img_array[y + 2, x] += error * 5 / divisor
                    if x + 1 < w:
                        img_array[y + 2, x + 1] += error * 3 / divisor
                    if x + 2 < w:
                        img_array[y + 2, x + 2] += error * 1 / divisor
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    elif method == "Stucki":
        img_array = img_array.astype(np.float32)
        h, w = img_array.shape
        divisor = 42.0
        for y in range(h):
            for x in range(w):
                old_pixel = img_array[y, x]
                new_pixel = 255.0 if old_pixel > threshold else 0.0
                img_array[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                if x + 1 < w:
                    img_array[y, x + 1] += error * 8 / divisor
                if x + 2 < w:
                    img_array[y, x + 2] += error * 4 / divisor
                
                if y + 1 < h:
                    if x > 1:
                        img_array[y + 1, x - 2] += error * 2 / divisor
                    if x > 0:
                        img_array[y + 1, x - 1] += error * 4 / divisor
                    img_array[y + 1, x] += error * 8 / divisor
                    if x + 1 < w:
                        img_array[y + 1, x + 1] += error * 4 / divisor
                    if x + 2 < w:
                        img_array[y + 1, x + 2] += error * 2 / divisor
                
                if y + 2 < h:
                    if x > 1:
                        img_array[y + 2, x - 2] += error * 1 / divisor
                    if x > 0:
                        img_array[y + 2, x - 1] += error * 2 / divisor
                    img_array[y + 2, x] += error * 4 / divisor
                    if x + 1 < w:
                        img_array[y + 2, x + 1] += error * 2 / divisor
                    if x + 2 < w:
                        img_array[y + 2, x + 2] += error * 1 / divisor
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    img = Image.fromarray(img_array, mode='L')
    
    # Resize up with NEAREST for pixelated effect
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
        self.cached_pixmap = None
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
            self.cached_pixmap = scaled
        else:
            width = int(self.original_pixmap.width() * self.zoom_level)
            height = int(self.original_pixmap.height() * self.zoom_level)
            
            scaled = self.original_pixmap.scaled(
                width, height,
                Qt.KeepAspectRatio,
                Qt.FastTransformation
            )
            self.setPixmap(scaled)
            self.cached_pixmap = scaled
    
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


class DitherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_img = None
        self.dithered_img = None
        self.current_color = (101, 138, 0)
        self.worker = None
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._process_update)
        self.auto_update = True
        
        self.init_ui()
        self.apply_dark_theme()
        icon_path = Path("app_icon.png")
        icon_pixmap = QPixmap(icon_path)
        icon = QIcon(icon_pixmap)
        self.setWindowIcon(icon)
    
    def init_ui(self):
        self.setWindowTitle("Dither Guy - Professional Image Dithering")
        self.setMinimumSize(1200, 800)
        
        self.create_toolbar()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        self.create_image_panel(splitter)
        self.create_control_panel(splitter)
        
        splitter.setSizes([900, 300])
        
        self.statusBar().showMessage("Ready - Load an image to begin")
    
    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize() * 0.8)
        self.addToolBar(toolbar)
        
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
        fit_action = QAction("Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.fit_to_window)
        toolbar.addAction(fit_action)
        
        actual_action = QAction("Actual Size", self)
        actual_action.setShortcut("Ctrl+1")
        actual_action.triggered.connect(self.actual_size)
        toolbar.addAction(actual_action)
        
        toolbar.addSeparator()
        
        self.zoom_label = QLabel("Fit")
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        toolbar.addWidget(self.zoom_label)
    
    def create_image_panel(self, parent):
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)
        
        self.info_label = QLabel("No image loaded")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 11px; padding: 6px; background: #252525;")
        image_layout.addWidget(self.info_label)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.img_label = ZoomableImageLabel()
        self.img_label.setText("Load an image to start dithering\n\nSupported: PNG, JPG, BMP, GIF, TIFF")
        self.img_label.setStyleSheet("font-size: 14px; color: #666;")
        
        scroll_area.setWidget(self.img_label)
        image_layout.addWidget(scroll_area)
        
        parent.addWidget(image_widget)
    
    def create_control_panel(self, parent):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(8)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(6)
        
        btn_open = QPushButton("Open Image")
        btn_open.clicked.connect(self.open_file)
        btn_open.setMinimumHeight(32)
        file_layout.addWidget(btn_open)
        
        btn_save = QPushButton("Save Image")
        btn_save.clicked.connect(self.save_file)
        btn_save.setMinimumHeight(32)
        file_layout.addWidget(btn_save)
        
        file_group.setLayout(file_layout)
        control_layout.addWidget(file_group)
        
        # Dithering method
        method_group = QGroupBox("Dithering Algorithm")
        method_layout = QVBoxLayout()
        method_layout.setSpacing(6)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Bayer",
            "Clustered-Dot",
            "Halftone",
            "Blue-Noise",
            "Pattern",
            "Crosshatch",
            "Riemersma",
            "Variable-Error",
            "Dot-Diffusion",
            "Floyd-Steinberg",
            "Atkinson",
            "Jarvis-Judice-Ninke",
            "Stucki",
            "Random"
        ])
        self.method_combo.currentTextChanged.connect(self.schedule_update)
        self.method_combo.setMinimumHeight(28)
        method_layout.addWidget(self.method_combo)
        
        self.auto_checkbox = QCheckBox("Auto Update")
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.stateChanged.connect(self.toggle_auto_update)
        method_layout.addWidget(self.auto_checkbox)
        
        self.btn_apply = QPushButton("Apply Dither")
        self.btn_apply.clicked.connect(self.update_image)
        self.btn_apply.setMinimumHeight(32)
        self.btn_apply.setVisible(False)
        method_layout.addWidget(self.btn_apply)
        
        method_group.setLayout(method_layout)
        control_layout.addWidget(method_group)
        
        # Dither parameters
        dither_params = QGroupBox("Dither Parameters")
        dither_layout = QVBoxLayout()
        dither_layout.setSpacing(4)
        
        self.pixel_label = QLabel("Pixel Size: 4")
        self.pixel_slider = QSlider(Qt.Horizontal)
        self.pixel_slider.setMinimum(1)
        self.pixel_slider.setMaximum(20)
        self.pixel_slider.setValue(4)
        self.pixel_slider.valueChanged.connect(self.on_pixel_changed)
        dither_layout.addWidget(self.pixel_label)
        dither_layout.addWidget(self.pixel_slider)
        
        self.threshold_label = QLabel("Threshold: 128")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        dither_layout.addWidget(self.threshold_label)
        dither_layout.addWidget(self.threshold_slider)
        
        dither_params.setLayout(dither_layout)
        control_layout.addWidget(dither_params)
        
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
        self.blur_slider.setMaximum(10)
        self.blur_slider.setValue(0)
        self.blur_slider.valueChanged.connect(self.on_blur_changed)
        adjust_layout.addWidget(self.blur_label)
        adjust_layout.addWidget(self.blur_slider)
        
        self.sharpen_label = QLabel("Sharpen: 0")
        self.sharpen_slider = QSlider(Qt.Horizontal)
        self.sharpen_slider.setMinimum(0)
        self.sharpen_slider.setMaximum(5)
        self.sharpen_slider.setValue(0)
        self.sharpen_slider.valueChanged.connect(self.on_sharpen_changed)
        adjust_layout.addWidget(self.sharpen_label)
        adjust_layout.addWidget(self.sharpen_slider)
        
        # Reset button
        btn_reset = QPushButton("Reset Adjustments")
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
        
        btn_invert = QPushButton("Invert Colors")
        btn_invert.clicked.connect(self.invert_colors)
        btn_invert.setMinimumHeight(32)
        color_layout.addWidget(btn_invert)
        
        color_group.setLayout(color_layout)
        control_layout.addWidget(color_group)
        
        control_layout.addStretch()
        
        about_label = QLabel("Dither Guy v2.3\nProfessional Edition")
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
                spacing: 6px;
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
                border: 1px solid #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #252525;
            }
            QComboBox {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px;
            }
            QComboBox:hover {
                border: 1px solid #4a4a4a;
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
            QCheckBox {
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #3a3a3a;
                background: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background: #658a00;
                border: 1px solid #658a00;
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
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #2d2d2d;
                width: 10px;
                height: 10px;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #4a4a4a;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
                background: #5a5a5a;
            }
            QStatusBar {
                background-color: #252525;
                color: #888;
                font-size: 11px;
            }
        """)
    
    def toggle_auto_update(self, state):
        self.auto_update = state == Qt.Checked
        self.btn_apply.setVisible(not self.auto_update)
    
    def on_pixel_changed(self, value):
        self.pixel_label.setText(f"Pixel Size: {value}")
        if self.auto_update:
            self.schedule_update()
    
    def on_threshold_changed(self, value):
        self.threshold_label.setText(f"Threshold: {value}")
        if self.auto_update:
            self.schedule_update()
    
    def on_brightness_changed(self, value):
        brightness = value / 100.0
        self.brightness_label.setText(f"Brightness: {brightness:.1f}")
        if self.auto_update:
            self.schedule_update()
    
    def on_contrast_changed(self, value):
        contrast = value / 100.0
        self.contrast_label.setText(f"Contrast: {contrast:.1f}")
        if self.auto_update:
            self.schedule_update()
    
    def on_blur_changed(self, value):
        self.blur_label.setText(f"Blur: {value}")
        if self.auto_update:
            self.schedule_update()
    
    def on_sharpen_changed(self, value):
        self.sharpen_label.setText(f"Sharpen: {value}")
        if self.auto_update:
            self.schedule_update()
    
    def reset_adjustments(self):
        self.brightness_slider.setValue(100)
        self.contrast_slider.setValue(100)
        self.blur_slider.setValue(0)
        self.sharpen_slider.setValue(0)
        if self.auto_update:
            self.schedule_update()
    
    def schedule_update(self):
        if not self.auto_update:
            return
        self.update_timer.stop()
        self.update_timer.start(500)
    
    def _process_update(self):
        self.update_image()
    
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.original_img = Image.open(file_path)
                self.statusBar().showMessage(f"Loaded: {Path(file_path).name}")
                self.info_label.setText(
                    f"{Path(file_path).name} - {self.original_img.width}x{self.original_img.height}px"
                )
                self.update_image()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open image:\n{str(e)}")
    
    def save_file(self):
        if self.dithered_img is None:
            QMessageBox.warning(self, "Warning", "No image to save!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "dithered_image.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.dithered_img.save(file_path)
                self.statusBar().showMessage(f"Saved: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image:\n{str(e)}")
    
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
            if self.auto_update:
                self.schedule_update()
    
    def invert_colors(self):
        if self.original_img is None:
            QMessageBox.warning(self, "Warning", "No image loaded!")
            return
        
        try:
            img_data = np.array(self.original_img.convert("RGB"))
            img_data = 255 - img_data
            self.original_img = Image.fromarray(img_data)
            self.statusBar().showMessage("Colors inverted")
            self.update_image()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to invert colors:\n{str(e)}")
    
    def update_image(self):
        if self.original_img is None:
            return
        
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.quit()
            self.worker.wait(100)
        
        self.statusBar().showMessage("Processing...")
        
        brightness = self.brightness_slider.value() / 100.0
        contrast = self.contrast_slider.value() / 100.0
        blur = self.blur_slider.value()
        sharpen = self.sharpen_slider.value()
        
        self.worker = DitherWorker(
            self.original_img,
            self.pixel_slider.value(),
            self.threshold_slider.value(),
            self.current_color,
            self.method_combo.currentText(),
            brightness,
            contrast,
            blur,
            sharpen
        )
        self.worker.finished.connect(self.on_dither_complete)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.start()
    
    def on_dither_complete(self, dithered_img):
        self.dithered_img = dithered_img
        
        img_data = dithered_img.tobytes("raw", "RGB")
        qimage = QImage(
            img_data,
            dithered_img.width,
            dithered_img.height,
            dithered_img.width * 3,
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        
        self.img_label.set_image(pixmap)
        self.img_label.setStyleSheet("")
        self.update_zoom_label()
        self.statusBar().showMessage("Ready")
    
    def zoom_in(self):
        self.img_label.zoom_in()
        self.update_zoom_label()
    
    def zoom_out(self):
        self.img_label.zoom_out()
        self.update_zoom_label()
    
    def fit_to_window(self):
        self.img_label.fit_to_window()
        self.zoom_label.setText("Fit")
    
    def actual_size(self):
        self.img_label.actual_size()
        self.update_zoom_label()
    
    def update_zoom_label(self):
        if self.img_label.zoom_level == 0:
            self.zoom_label.setText("Fit")
        else:
            self.zoom_label.setText(f"{int(self.img_label.zoom_level * 100)}%")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Dither Guy")
    
    window = DitherApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
