"""Main application window."""

import os
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QStatusBar,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QGroupBox,
    QSlider,
    QCheckBox,
    QComboBox,
    QFrame,
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject
from PySide6.QtGui import QImage, QPixmap

import cv2
import numpy as np

from ..core.video_loader import YouTubeDownloader, VideoReader
from ..core.pipeline import ProcessingPipeline, PipelineConfig
from ..rendering.silhouette import SilhouetteConfig
from ..rendering.glove import GloveConfig
from ..rendering.ribbon import RibbonConfig
from ..rendering.compositor import BackgroundConfig


class DownloadWorker(QObject):
    """Worker for downloading YouTube videos."""

    finished = Signal(str)  # video_path
    error = Signal(str)
    progress = Signal(float)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.downloader = YouTubeDownloader()

    def run(self):
        """Execute download."""
        try:
            path = self.downloader.download(self.url, self.progress.emit)
            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))


class ProcessingWorker(QObject):
    """Worker for video processing."""

    finished = Signal(bool, str)  # success, message
    progress = Signal(float, int, int)  # percent, current, total
    frame_preview = Signal(np.ndarray)  # preview frame

    def __init__(self, input_path: str, output_path: str, config: PipelineConfig):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self._cancelled = False

    def run(self):
        """Execute processing."""
        try:
            pipeline = ProcessingPipeline(self.config)

            def on_progress(percent, current, total):
                if self._cancelled:
                    raise InterruptedError("Processing cancelled")
                self.progress.emit(percent, current, total)

            success, message = pipeline.process_video(
                self.input_path,
                self.output_path,
                on_progress,
            )
            pipeline.close()
            self.finished.emit(success, message)

        except InterruptedError:
            self.finished.emit(False, "Processing cancelled")
        except Exception as e:
            self.finished.emit(False, str(e))

    def cancel(self):
        """Request cancellation."""
        self._cancelled = True


class VideoPreviewWidget(QLabel):
    """Widget for displaying video preview."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        self.setText("Load a video to preview")

    def display_frame(self, frame: np.ndarray):
        """Display a BGR frame."""
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = QImage(
            rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888
        )

        # Scale to fit widget
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)


class SettingsPanel(QWidget):
    """Panel for adjusting visual settings."""

    settings_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setFixedWidth(300)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # Title
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Silhouette settings
        silhouette_group = QGroupBox("Silhouette")
        silhouette_layout = QVBoxLayout(silhouette_group)

        self.silhouette_thickness = QSlider(Qt.Horizontal)
        self.silhouette_thickness.setRange(15, 50)
        self.silhouette_thickness.setValue(28)
        silhouette_layout.addWidget(QLabel("Thickness"))
        silhouette_layout.addWidget(self.silhouette_thickness)

        layout.addWidget(silhouette_group)

        # Glove settings
        glove_group = QGroupBox("Glove Effect")
        glove_layout = QVBoxLayout(glove_group)

        self.glove_glow = QSlider(Qt.Horizontal)
        self.glove_glow.setRange(0, 100)
        self.glove_glow.setValue(70)
        glove_layout.addWidget(QLabel("Glow Intensity"))
        glove_layout.addWidget(self.glove_glow)

        self.glove_pulse = QCheckBox("Animate Pulse")
        self.glove_pulse.setChecked(True)
        glove_layout.addWidget(self.glove_pulse)

        layout.addWidget(glove_group)

        # Ribbon settings
        ribbon_group = QGroupBox("Motion Ribbon")
        ribbon_layout = QVBoxLayout(ribbon_group)

        self.ribbon_enabled = QCheckBox("Enable Preview Ribbon")
        self.ribbon_enabled.setChecked(True)
        ribbon_layout.addWidget(self.ribbon_enabled)

        self.ribbon_length = QSlider(Qt.Horizontal)
        self.ribbon_length.setRange(15, 90)
        self.ribbon_length.setValue(45)
        ribbon_layout.addWidget(QLabel("Preview Length (frames)"))
        ribbon_layout.addWidget(self.ribbon_length)

        self.ribbon_opacity = QSlider(Qt.Horizontal)
        self.ribbon_opacity.setRange(20, 100)
        self.ribbon_opacity.setValue(60)
        ribbon_layout.addWidget(QLabel("Opacity"))
        ribbon_layout.addWidget(self.ribbon_opacity)

        layout.addWidget(ribbon_group)

        # Background settings
        bg_group = QGroupBox("Background")
        bg_layout = QVBoxLayout(bg_group)

        self.bg_style = QComboBox()
        self.bg_style.addItems(["Gradient", "Solid Color", "Dimmed Original"])
        bg_layout.addWidget(self.bg_style)

        layout.addWidget(bg_group)

        layout.addStretch()

        # Connect signals
        self._connect_signals()

    def _connect_signals(self):
        """Connect all settings to emit settings_changed."""
        self.silhouette_thickness.valueChanged.connect(self.settings_changed.emit)
        self.glove_glow.valueChanged.connect(self.settings_changed.emit)
        self.glove_pulse.stateChanged.connect(self.settings_changed.emit)
        self.ribbon_enabled.stateChanged.connect(self.settings_changed.emit)
        self.ribbon_length.valueChanged.connect(self.settings_changed.emit)
        self.ribbon_opacity.valueChanged.connect(self.settings_changed.emit)
        self.bg_style.currentIndexChanged.connect(self.settings_changed.emit)

    def get_config(self) -> PipelineConfig:
        """Get current settings as PipelineConfig."""
        bg_styles = ["gradient", "solid", "original"]

        return PipelineConfig(
            silhouette=SilhouetteConfig(
                body_thickness=self.silhouette_thickness.value(),
            ),
            glove=GloveConfig(
                glow_intensity=self.glove_glow.value() / 100.0,
                animate_pulse=self.glove_pulse.isChecked(),
            ),
            ribbon=RibbonConfig(
                trail_length=self.ribbon_length.value() if self.ribbon_enabled.isChecked() else 0,
                base_opacity=self.ribbon_opacity.value() / 100.0,
            ),
            background=BackgroundConfig(
                style=bg_styles[self.bg_style.currentIndex()],
            ),
        )


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Just Dance Video Processor")
        self.setMinimumSize(1100, 700)

        self.current_video_path: Optional[str] = None
        self.download_thread: Optional[QThread] = None
        self.processing_thread: Optional[QThread] = None

        self._setup_ui()
        self._setup_connections()

    def _setup_ui(self):
        """Initialize UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # URL input section
        url_frame = QFrame()
        url_frame.setStyleSheet(
            "QFrame { background-color: #2d2d44; border-radius: 8px; padding: 10px; }"
        )
        url_layout = QHBoxLayout(url_frame)

        url_label = QLabel("YouTube URL:")
        url_label.setStyleSheet("color: white;")
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=...")
        self.url_input.setStyleSheet(
            "QLineEdit { padding: 8px; border-radius: 4px; background: #3d3d5c; color: white; border: none; }"
        )

        self.load_button = QPushButton("Load Video")
        self.load_button.setStyleSheet(
            "QPushButton { padding: 8px 20px; background: #6c5ce7; color: white; border-radius: 4px; }"
            "QPushButton:hover { background: #5b4cdb; }"
            "QPushButton:disabled { background: #4a4a6a; }"
        )

        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input, stretch=1)
        url_layout.addWidget(self.load_button)

        main_layout.addWidget(url_frame)

        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)

        # Left side: Video preview
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.video_preview = VideoPreviewWidget()
        preview_layout.addWidget(self.video_preview, stretch=1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border-radius: 4px; background: #2d2d44; text-align: center; }"
            "QProgressBar::chunk { background: #6c5ce7; border-radius: 4px; }"
        )
        preview_layout.addWidget(self.progress_bar)

        # Action buttons
        button_layout = QHBoxLayout()

        self.process_button = QPushButton("Process Video")
        self.process_button.setEnabled(False)
        self.process_button.setStyleSheet(
            "QPushButton { padding: 12px 30px; background: #00b894; color: white; border-radius: 6px; font-size: 14px; }"
            "QPushButton:hover { background: #00a383; }"
            "QPushButton:disabled { background: #4a4a6a; }"
        )

        self.export_button = QPushButton("Export...")
        self.export_button.setEnabled(False)
        self.export_button.setStyleSheet(
            "QPushButton { padding: 12px 30px; background: #0984e3; color: white; border-radius: 6px; font-size: 14px; }"
            "QPushButton:hover { background: #0773c7; }"
            "QPushButton:disabled { background: #4a4a6a; }"
        )

        button_layout.addStretch()
        button_layout.addWidget(self.process_button)
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()

        preview_layout.addLayout(button_layout)

        content_splitter.addWidget(preview_container)

        # Right side: Settings panel
        self.settings_panel = SettingsPanel()
        content_splitter.addWidget(self.settings_panel)

        content_splitter.setSizes([750, 300])

        main_layout.addWidget(content_splitter, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, stretch=1)

        # Set dark theme
        self.setStyleSheet(
            """
            QMainWindow { background-color: #1a1a2e; }
            QLabel { color: #e0e0e0; }
            QGroupBox {
                color: #e0e0e0;
                border: 1px solid #3d3d5c;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3d3d5c;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6c5ce7;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QCheckBox { color: #e0e0e0; }
            QComboBox {
                padding: 6px;
                background: #3d3d5c;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QStatusBar { background: #2d2d44; color: #e0e0e0; }
            """
        )

    def _setup_connections(self):
        """Connect signals and slots."""
        self.load_button.clicked.connect(self._on_load_video)
        self.process_button.clicked.connect(self._on_process_video)
        self.export_button.clicked.connect(self._on_export_video)
        self.url_input.returnPressed.connect(self._on_load_video)

    @Slot()
    def _on_load_video(self):
        """Handle loading video from URL or file."""
        url = self.url_input.text().strip()

        if not url:
            # Open file dialog
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Video",
                "",
                "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
            )
            if path:
                self._load_local_video(path)
            return

        # Download from YouTube
        self.status_label.setText("Downloading video...")
        self.load_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.download_worker = DownloadWorker(url)
        self.download_thread = QThread()
        self.download_worker.moveToThread(self.download_thread)

        self.download_worker.finished.connect(self._on_download_complete)
        self.download_worker.error.connect(self._on_download_error)
        self.download_worker.progress.connect(
            lambda p: self.progress_bar.setValue(int(p))
        )

        self.download_thread.started.connect(self.download_worker.run)
        self.download_thread.start()

    def _load_local_video(self, path: str):
        """Load a local video file."""
        try:
            reader = VideoReader(path)
            # Show first frame
            _, frame = next(iter(reader))
            self.video_preview.display_frame(frame)
            reader.release()

            self.current_video_path = path
            self.process_button.setEnabled(True)
            self.status_label.setText(f"Loaded: {Path(path).name}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load video: {e}")

    @Slot(str)
    def _on_download_complete(self, video_path: str):
        """Handle successful video download."""
        self._load_local_video(video_path)
        self.load_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.download_thread.quit()

    @Slot(str)
    def _on_download_error(self, error: str):
        """Handle download error."""
        QMessageBox.warning(self, "Download Error", error)
        self.load_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        self.download_thread.quit()

    @Slot()
    def _on_process_video(self):
        """Start video processing."""
        if not self.current_video_path:
            return

        # Get output path
        default_name = Path(self.current_video_path).stem + "_justdance.mp4"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Video",
            default_name,
            "MP4 Video (*.mp4)",
        )

        if not output_path:
            return

        # Get config from settings
        config = self.settings_panel.get_config()

        # Start processing
        self.status_label.setText("Processing video...")
        self.process_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.processing_worker = ProcessingWorker(
            self.current_video_path, output_path, config
        )
        self.processing_thread = QThread()
        self.processing_worker.moveToThread(self.processing_thread)

        self.processing_worker.finished.connect(self._on_processing_complete)
        self.processing_worker.progress.connect(self._on_processing_progress)

        self.processing_thread.started.connect(self.processing_worker.run)
        self.processing_thread.start()

    @Slot(float, int, int)
    def _on_processing_progress(self, percent: float, current: int, total: int):
        """Update processing progress."""
        self.progress_bar.setValue(int(percent))
        self.status_label.setText(f"Processing frame {current}/{total}")

    @Slot(bool, str)
    def _on_processing_complete(self, success: bool, message: str):
        """Handle processing completion."""
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.processing_thread.quit()

        if success:
            self.status_label.setText("Processing complete!")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_label.setText("Processing failed")
            QMessageBox.warning(self, "Error", message)

    @Slot()
    def _on_export_video(self):
        """Export with different settings."""
        # For now, same as process
        self._on_process_video()

    def closeEvent(self, event):
        """Clean up on close."""
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.quit()
            self.download_thread.wait()

        if self.processing_thread and self.processing_thread.isRunning():
            if hasattr(self, "processing_worker"):
                self.processing_worker.cancel()
            self.processing_thread.quit()
            self.processing_thread.wait()

        event.accept()
