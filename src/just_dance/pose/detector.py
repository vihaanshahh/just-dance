"""MediaPipe pose detection wrapper."""

from typing import Optional
import numpy as np
import cv2
import mediapipe as mp


class PoseDetector:
    """Detects body pose keypoints using MediaPipe."""

    # MediaPipe pose landmark names (33 total)
    LANDMARK_NAMES = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
        detection_scale: float = 1.0,
    ):
        """
        Initialize pose detector.

        Args:
            model_complexity: 0=lite, 1=full, 2=heavy (accuracy vs speed)
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            enable_segmentation: Whether to enable body segmentation mask
            detection_scale: Scale factor for detection (0.5 = half resolution, faster)
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=enable_segmentation,
        )
        self.enable_segmentation = enable_segmentation
        self.detection_scale = detection_scale

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose in a frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Array of shape (33, 3) with [x, y, visibility] per keypoint,
            or None if no pose detected. Coordinates are normalized [0, 1].
        """
        # Downscale for faster processing if needed
        if self.detection_scale < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.detection_scale)
            new_h = int(h * self.detection_scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks is None:
            return None

        # Extract keypoints (coordinates are already normalized 0-1)
        keypoints = np.zeros((33, 3), dtype=np.float32)

        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y, landmark.visibility]

        return keypoints

    def detect_with_segmentation(
        self, frame: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect pose and return segmentation mask.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Tuple of (keypoints, segmentation_mask) or (None, None)
        """
        if not self.enable_segmentation:
            raise ValueError("Segmentation not enabled. Set enable_segmentation=True")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks is None:
            return None, None

        # Extract keypoints
        keypoints = np.zeros((33, 3), dtype=np.float32)
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = [landmark.x, landmark.y, landmark.visibility]

        # Get segmentation mask
        seg_mask = None
        if results.segmentation_mask is not None:
            seg_mask = results.segmentation_mask

        return keypoints, seg_mask

    def get_landmark_index(self, name: str) -> int:
        """Get index of a landmark by name."""
        return self.LANDMARK_NAMES.index(name)

    def close(self):
        """Release resources."""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
