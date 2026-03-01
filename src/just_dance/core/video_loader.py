"""Video loading and YouTube download functionality."""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Callable, Generator, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
import yt_dlp


def sanitize_filename(filename: str) -> str:
    """Remove special characters from filename to ensure compatibility."""
    # Replace special unicode characters with ASCII equivalents or remove them
    # Keep only alphanumeric, spaces, hyphens, underscores
    sanitized = re.sub(r'[^\w\s\-.]', '', filename)
    # Replace multiple spaces/underscores with single
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Ensure not empty
    if not sanitized:
        sanitized = "video"
    return sanitized


@dataclass
class DownloadConfig:
    """YouTube download configuration."""

    # Use a format that's already muxed (no separate video+audio merge needed)
    # This ensures OpenCV compatibility
    format: str = "best[height<=1080][ext=mp4]/best[height<=720][ext=mp4]/best[ext=mp4]/best"
    output_format: str = "mp4"
    temp_dir: Optional[str] = None


class YouTubeDownloader:
    """Downloads videos from YouTube using yt-dlp."""

    def __init__(self, config: Optional[DownloadConfig] = None):
        self.config = config or DownloadConfig()

    def download(
        self,
        url: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        """
        Download video from YouTube URL.

        Args:
            url: YouTube video URL
            progress_callback: Optional callback for download progress (0-100)

        Returns:
            Path to downloaded video file
        """
        temp_dir = self.config.temp_dir or tempfile.mkdtemp()
        # Use video ID for filename to avoid special character issues
        output_template = str(Path(temp_dir) / "%(id)s.%(ext)s")

        def progress_hook(d):
            if d["status"] == "downloading" and progress_callback:
                try:
                    percent = d.get("_percent_str", "0%")
                    value = float(percent.replace("%", "").strip())
                    progress_callback(value)
                except (ValueError, KeyError):
                    pass

        ydl_opts = {
            "format": self.config.format,
            "outtmpl": output_template,
            "progress_hooks": [progress_hook],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

            # Handle format conversion - check various possible extensions
            if not Path(filename).exists():
                # Try with .mp4 extension
                base = Path(filename).stem
                mp4_path = Path(temp_dir) / f"{base}.mp4"
                if mp4_path.exists():
                    filename = str(mp4_path)
                else:
                    # List files in temp dir to find the downloaded file
                    files = list(Path(temp_dir).glob(f"{base}.*"))
                    if files:
                        filename = str(files[0])

        # Verify file exists
        if not Path(filename).exists():
            raise FileNotFoundError(f"Downloaded file not found: {filename}")

        # Convert to h264 for OpenCV compatibility
        filename = self._ensure_opencv_compatible(filename)

        return filename

    def _ensure_opencv_compatible(self, input_path: str) -> str:
        """Convert video to h264 format for OpenCV compatibility."""
        import sys

        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_converted.mp4"

        # Try to open with OpenCV first
        cap = cv2.VideoCapture(input_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                # File is already compatible
                return input_path

        print("    Converting video for OpenCV compatibility...", file=sys.stderr)

        # Convert using ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            print("    Conversion complete.", file=sys.stderr)
            # Remove original and return converted
            os.remove(input_path)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"    Conversion failed: {e.stderr.decode()}", file=sys.stderr)
            # If conversion fails, return original and hope for the best
            return input_path

    def get_video_info(self, url: str) -> dict:
        """Get video metadata without downloading."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        return {
            "title": info.get("title"),
            "duration": info.get("duration"),
            "thumbnail": info.get("thumbnail"),
            "width": info.get("width"),
            "height": info.get("height"),
        }


class VideoReader:
    """Reads video frames from a file."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height) tuple."""
        return (self.width, self.height)

    def __iter__(self) -> Generator[Tuple[float, np.ndarray], None, None]:
        """Iterate over frames, yielding (timestamp, frame) tuples."""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_idx / self.fps if self.fps > 0 else 0
            yield timestamp, frame
            frame_idx += 1

    def read_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Read a specific frame by number."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()
