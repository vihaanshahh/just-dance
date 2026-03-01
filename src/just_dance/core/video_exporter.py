"""Video export functionality using FFmpeg."""

import subprocess
import shutil
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class ExportConfig:
    """Video export configuration."""

    codec: str = "h264"
    crf: int = 23
    preset: str = "ultrafast"
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"


class VideoWriter:
    """Writes video frames and muxes with audio."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        resolution: Tuple[int, int],
        audio_source: Optional[str] = None,
        config: Optional[ExportConfig] = None,
    ):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution  # (width, height)
        self.audio_source = audio_source
        self.config = config or ExportConfig()

        # Create temp file for video without audio
        self.temp_dir = tempfile.mkdtemp()
        self.temp_video = os.path.join(self.temp_dir, "temp_video.mp4")

        # Use FFmpeg pipe for reliable video writing
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        width, height = resolution

        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",  # Read from stdin
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            self.temp_video,
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.frame_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def write(self, frame: np.ndarray) -> None:
        """Write a frame."""
        # Ensure frame is BGR (no alpha)
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Ensure contiguous memory layout for FFmpeg
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        # Write raw bytes to FFmpeg stdin
        self.process.stdin.write(frame.tobytes())
        self.frame_count += 1

    def release(self) -> Tuple[bool, str]:
        """Finish writing and mux with audio."""
        # Close FFmpeg stdin and wait for encoding to finish
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()

        if self.frame_count == 0:
            self._cleanup()
            return False, "No frames written"

        # Check if temp video was created
        if not os.path.exists(self.temp_video):
            self._cleanup()
            return False, "FFmpeg failed to create temp video"

        temp_size = os.path.getsize(self.temp_video)
        if temp_size == 0:
            self._cleanup()
            return False, "FFmpeg created empty temp video"

        # Mux video with audio using FFmpeg
        success = self._mux_audio()

        self._cleanup()

        if success:
            return True, "Export complete"
        else:
            return False, "FFmpeg muxing failed"

    def _mux_audio(self) -> bool:
        """Combine temp video with audio from source."""
        print("\n    Finalizing...", end="", flush=True)

        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

        if self.audio_source and Path(self.audio_source).exists():
            # Mux video + audio
            cmd = [
                ffmpeg, "-y",
                "-i", self.temp_video,
                "-i", self.audio_source,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", self.config.audio_codec,
                "-b:a", self.config.audio_bitrate,
                "-shortest",
                self.output_path,
            ]
        else:
            # No audio, just copy video
            cmd = [
                ffmpeg, "-y",
                "-i", self.temp_video,
                "-c:v", "copy",
                self.output_path,
            ]

        try:
            print(f"\n    Output: {self.output_path}", flush=True)
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(f" FAILED: {result.stderr.decode()}", flush=True)
                return False
            if not Path(self.output_path).exists():
                print(f" ERROR: File not created!", flush=True)
                return False
            print(f" done! ({Path(self.output_path).stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
            return True
        except Exception as e:
            print(f" error: {e}", flush=True)
            return False

    def _cleanup(self):
        """Remove temp files."""
        try:
            if os.path.exists(self.temp_video):
                os.remove(self.temp_video)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception:
            pass
