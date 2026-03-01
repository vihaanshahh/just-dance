"""Main processing pipeline with parallel processing."""

from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import sys

import numpy as np
import cv2

# Use spawn on macOS for MediaPipe compatibility
if sys.platform == 'darwin':
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        pass  # Already set

from ..pose.detector import PoseDetector
from ..pose.smoother import MotionSmoother
from ..rendering.silhouette import SilhouetteRenderer, SilhouetteConfig
from ..rendering.glove import GloveRenderer, GloveConfig
from ..rendering.compositor import Compositor, BackgroundConfig
from .video_loader import VideoReader
from .video_exporter import VideoWriter, ExportConfig


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline."""

    # Performance
    model_complexity: int = 0  # Lite model
    detection_scale: float = 0.5  # 50% resolution for detection (need decent seg mask)
    frame_skip: int = 1  # Detect every frame (segmentation mask needs it)
    num_workers: int = 0  # 0 = auto (cpu_count)

    # Smoothing
    process_noise: float = 0.01
    measurement_noise: float = 0.08

    # Rendering
    silhouette: SilhouetteConfig = field(default_factory=SilhouetteConfig)
    glove: GloveConfig = field(default_factory=GloveConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)

    # Export
    export: ExportConfig = field(default_factory=ExportConfig)


def _process_frame_batch(args):
    """Process a batch of frames (runs in separate process)."""
    frames, frame_indices, config_dict, frame_size = args

    # Create components for this worker
    detector = PoseDetector(
        model_complexity=config_dict['model_complexity'],
        detection_scale=config_dict['detection_scale'],
        enable_segmentation=True,
    )
    smoother = MotionSmoother(
        process_noise=config_dict['process_noise'],
        measurement_noise=config_dict['measurement_noise'],
    )

    silhouette_renderer = SilhouetteRenderer()
    glove_renderer = GloveRenderer()
    compositor = Compositor()

    results = []
    frame_skip = config_dict['frame_skip']
    last_keypoints = None
    last_seg_mask = None

    for i, (frame_idx, frame) in enumerate(zip(frame_indices, frames)):
        # Detect pose (with frame skipping)
        if i % frame_skip == 0 or last_keypoints is None:
            keypoints, seg_mask = detector.detect(frame)
            if keypoints is not None:
                last_keypoints = keypoints
                last_seg_mask = seg_mask
        else:
            keypoints = last_keypoints
            seg_mask = last_seg_mask

        if keypoints is None:
            rendered = compositor.create_background(frame_size, frame)
            results.append((frame_idx, rendered))
            continue

        # Smooth keypoints (for glove positioning)
        smoothed = smoother.smooth(keypoints)

        # Render
        background = compositor.create_background(frame_size, frame)
        layers = []

        layers.append(silhouette_renderer.render(smoothed, frame_size, seg_mask))
        layers.append(glove_renderer.render(smoothed, frame_size, frame_idx / 30.0))

        rendered = compositor.composite(background, layers)
        results.append((frame_idx, rendered))

    detector.close()
    return results


class ProcessingPipeline:
    """Orchestrates parallel video processing."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        if self.config.num_workers == 0:
            self.config.num_workers = max(1, mp.cpu_count() - 1)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
    ) -> Tuple[bool, str]:
        """Process video with parallel frame processing."""

        try:
            # Read all frames first
            print("    Loading frames...", end="", flush=True)
            with VideoReader(input_path) as reader:
                frames = []
                for _, frame in reader:
                    frames.append(frame)
                fps = reader.fps
                resolution = reader.resolution
                frame_size = (reader.height, reader.width)

            total_frames = len(frames)
            print(f" {total_frames} frames", flush=True)

            # Split into batches for parallel processing
            num_workers = min(self.config.num_workers, max(1, total_frames // 100))
            batch_size = (total_frames + num_workers - 1) // num_workers

            batches = []
            for i in range(0, total_frames, batch_size):
                end = min(i + batch_size, total_frames)
                batch_frames = frames[i:end]
                batch_indices = list(range(i, end))
                batches.append((batch_frames, batch_indices, {
                    'model_complexity': self.config.model_complexity,
                    'detection_scale': self.config.detection_scale,
                    'frame_skip': self.config.frame_skip,
                    'process_noise': self.config.process_noise,
                    'measurement_noise': self.config.measurement_noise,
                }, frame_size))

            # Process batches in parallel
            print(f"    Processing with {num_workers} workers...", flush=True)
            all_results = {}
            completed = 0

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_frame_batch, batch): i for i, batch in enumerate(batches)}

                for future in as_completed(futures):
                    results = future.result()
                    for frame_idx, rendered in results:
                        all_results[frame_idx] = rendered
                        completed += 1

                        if progress_callback and completed % 50 == 0:
                            progress_callback(completed / total_frames * 100, completed, total_frames)

            if progress_callback:
                progress_callback(100, total_frames, total_frames)

            # Write frames in order
            print("\n    Writing video...", end="", flush=True)
            with VideoWriter(output_path, fps, resolution, input_path, self.config.export) as writer:
                for i in range(total_frames):
                    writer.write(all_results[i])

            return True, f"Processed {total_frames} frames"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, str(e)

    def close(self):
        """Release resources."""
        pass
