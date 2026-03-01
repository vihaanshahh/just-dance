#!/usr/bin/env python3
"""Test script to verify the pipeline works with a local video or webcam."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import cv2
import numpy as np
from just_dance.core.pipeline import ProcessingPipeline, PipelineConfig


def test_with_webcam():
    """Test pipeline with webcam feed."""
    print("Testing with webcam...")
    print("Press 'q' to quit")

    pipeline = ProcessingPipeline()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        timestamp = frame_count / 30.0  # Assume 30fps
        result = pipeline.process_single_frame(frame, timestamp)

        # Display
        cv2.imshow("Just Dance Preview", result)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pipeline.close()
    print("Done!")


def test_with_video(video_path: str, output_path: str = None):
    """Test pipeline with video file."""
    print(f"Testing with video: {video_path}")

    if output_path is None:
        output_path = str(Path(video_path).stem) + "_justdance.mp4"

    config = PipelineConfig()
    pipeline = ProcessingPipeline(config)

    def progress_callback(percent, current, total):
        print(f"\rProcessing: {percent:.1f}% ({current}/{total})", end="", flush=True)

    success, message = pipeline.process_video(video_path, output_path, progress_callback)
    print()  # New line after progress

    if success:
        print(f"Success! Output saved to: {output_path}")
    else:
        print(f"Failed: {message}")

    pipeline.close()


def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        test_with_video(video_path, output_path)
    else:
        print("Usage:")
        print("  python test_pipeline.py                    # Test with webcam")
        print("  python test_pipeline.py video.mp4          # Process video")
        print("  python test_pipeline.py video.mp4 out.mp4  # Process with custom output")
        print()
        test_with_webcam()


if __name__ == "__main__":
    main()
