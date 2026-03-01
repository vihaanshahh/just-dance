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
from ..pose.predictor import MotionPredictor
from ..rendering.silhouette import SilhouetteRenderer, SilhouetteConfig
from ..rendering.glove import GloveRenderer, GloveConfig
from ..rendering.ribbon import RibbonRenderer, RibbonConfig
from ..rendering.compositor import Compositor, BackgroundConfig
from .video_loader import VideoReader
from .video_exporter import VideoWriter, ExportConfig


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline."""

    # Performance (aggressive defaults for speed)
    model_complexity: int = 0  # Lite model
    detection_scale: float = 0.4  # 40% resolution for detection
    frame_skip: int = 3  # Detect every 3rd frame
    num_workers: int = 0  # 0 = auto (cpu_count)

    # Smoothing
    process_noise: float = 0.01
    measurement_noise: float = 0.08

    # Prediction
    prediction_frames: int = 30

    # Rendering
    silhouette: SilhouetteConfig = field(default_factory=SilhouetteConfig)
    glove: GloveConfig = field(default_factory=GloveConfig)
    ribbon: RibbonConfig = field(default_factory=RibbonConfig)
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
    )
    smoother = MotionSmoother(
        process_noise=config_dict['process_noise'],
        measurement_noise=config_dict['measurement_noise'],
    )
    predictor = MotionPredictor(prediction_frames=config_dict['prediction_frames'])

    silhouette_renderer = SilhouetteRenderer()
    glove_renderer = GloveRenderer()
    ribbon_renderer = RibbonRenderer()
    compositor = Compositor()

    results = []
    frame_skip = config_dict['frame_skip']
    last_keypoints = None

    for i, (frame_idx, frame) in enumerate(zip(frame_indices, frames)):
        # Detect pose (with frame skipping)
        if i % frame_skip == 0 or last_keypoints is None:
            keypoints = detector.detect(frame)
            if keypoints is not None:
                last_keypoints = keypoints
        else:
            keypoints = last_keypoints

        if keypoints is None:
            # No pose - just background
            rendered = compositor.create_background(frame_size, frame)
            results.append((frame_idx, rendered))
            continue

        # Smooth
        smoothed = smoother.smooth(keypoints)
        predictor.update(smoothed)
        predicted = predictor.predict() if i >= 5 else None

        # Render
        height, width = frame_size
        background = compositor.create_background(frame_size, frame)
        layers = []

        if predicted:
            layers.append(ribbon_renderer.render(smoothed, predicted, frame_size))
        layers.append(silhouette_renderer.render(smoothed, frame_size))
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
                    'prediction_frames': self.config.prediction_frames,
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
