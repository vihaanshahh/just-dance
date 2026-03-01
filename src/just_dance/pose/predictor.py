"""Motion prediction for upcoming poses."""

import numpy as np
from typing import List, Optional
from collections import deque


class MotionPredictor:
    """Predicts future pose keypoints using velocity extrapolation."""

    def __init__(
        self,
        prediction_frames: int = 45,
        velocity_smoothing: int = 5,
        acceleration_decay: float = 0.95,
    ):
        """
        Initialize motion predictor.

        Args:
            prediction_frames: Number of frames to predict ahead (~1.5 sec at 30fps)
            velocity_smoothing: Number of frames to average velocity over
            acceleration_decay: How quickly acceleration decays (0-1)
        """
        self.prediction_frames = prediction_frames
        self.velocity_smoothing = velocity_smoothing
        self.acceleration_decay = acceleration_decay

        # History buffers
        self.position_history: deque = deque(maxlen=velocity_smoothing + 5)
        self.velocity_history: deque = deque(maxlen=velocity_smoothing)

    def update(self, keypoints: np.ndarray) -> None:
        """
        Update predictor with new keypoint observation.

        Args:
            keypoints: Current frame keypoints (33, 3)
        """
        self.position_history.append(keypoints[:, :2].copy())

        # Compute velocity if we have enough history
        if len(self.position_history) >= 2:
            velocity = self.position_history[-1] - self.position_history[-2]
            self.velocity_history.append(velocity)

    def predict(self) -> List[np.ndarray]:
        """
        Predict future keypoint positions.

        Returns:
            List of predicted keypoint arrays (each 33x3)
        """
        if len(self.position_history) < 2:
            # Not enough history - return copies of last position
            if len(self.position_history) > 0:
                return [
                    self._make_keypoints(self.position_history[-1], 0.3)
                    for _ in range(self.prediction_frames)
                ]
            return []

        # Get average velocity
        if len(self.velocity_history) > 0:
            avg_velocity = np.mean(list(self.velocity_history), axis=0)
        else:
            avg_velocity = np.zeros((33, 2))

        # Compute acceleration if possible
        if len(self.velocity_history) >= 3:
            recent_velocities = list(self.velocity_history)[-3:]
            accelerations = [
                recent_velocities[i + 1] - recent_velocities[i]
                for i in range(len(recent_velocities) - 1)
            ]
            avg_acceleration = np.mean(accelerations, axis=0) * 0.5  # Dampened
        else:
            avg_acceleration = np.zeros((33, 2))

        # Project forward
        result = []
        current_pos = self.position_history[-1].copy()
        current_vel = avg_velocity.copy()
        current_acc = avg_acceleration.copy()

        for t in range(self.prediction_frames):
            # Update position with velocity
            current_pos = current_pos + current_vel

            # Update velocity with acceleration (with damping)
            current_vel = current_vel * 0.98 + current_acc * 0.5
            current_acc *= self.acceleration_decay

            # Clamp positions to valid range [0, 1]
            current_pos = np.clip(current_pos, 0.0, 1.0)

            # Create keypoint array with decreasing confidence
            confidence = max(0.1, 0.7 - t * 0.015)
            result.append(self._make_keypoints(current_pos, confidence))

        return result

    def _make_keypoints(self, positions: np.ndarray, confidence: float) -> np.ndarray:
        """Create keypoint array from positions."""
        keypoints = np.zeros((33, 3))
        keypoints[:, :2] = positions
        keypoints[:, 2] = confidence
        return keypoints

    def reset(self):
        """Reset prediction state."""
        self.position_history.clear()
        self.velocity_history.clear()


class SplineMotionPredictor:
    """Motion predictor using spline interpolation for smoother curves."""

    def __init__(
        self,
        prediction_frames: int = 45,
        history_frames: int = 15,
    ):
        """
        Initialize spline-based motion predictor.

        Args:
            prediction_frames: Number of frames to predict ahead
            history_frames: Number of past frames to use for spline fitting
        """
        self.prediction_frames = prediction_frames
        self.history_frames = history_frames
        self.position_history: deque = deque(maxlen=history_frames)

    def update(self, keypoints: np.ndarray) -> None:
        """Update with new keypoint observation."""
        self.position_history.append(keypoints[:, :2].copy())

    def predict(self) -> List[np.ndarray]:
        """Predict future positions using polynomial extrapolation."""
        if len(self.position_history) < 4:
            # Not enough for good extrapolation
            if len(self.position_history) > 0:
                return [
                    self._make_keypoints(self.position_history[-1], 0.3)
                    for _ in range(self.prediction_frames)
                ]
            return []

        # Use polynomial fitting for each keypoint
        history = np.array(list(self.position_history))  # (T, 33, 2)
        T = len(history)

        # Fit polynomials (degree 2 for smoothness)
        t_past = np.arange(T)
        t_future = np.arange(T, T + self.prediction_frames)

        result = []

        for future_t in t_future:
            future_pos = np.zeros((33, 2))

            for kp_idx in range(33):
                for coord in range(2):  # x and y
                    values = history[:, kp_idx, coord]

                    # Fit polynomial
                    try:
                        coeffs = np.polyfit(t_past, values, deg=2)
                        predicted = np.polyval(coeffs, future_t)
                        future_pos[kp_idx, coord] = np.clip(predicted, 0, 1)
                    except np.linalg.LinAlgError:
                        # Fallback to last known value
                        future_pos[kp_idx, coord] = values[-1]

            confidence = max(0.1, 0.7 - (future_t - T) * 0.015)
            result.append(self._make_keypoints(future_pos, confidence))

        return result

    def _make_keypoints(self, positions: np.ndarray, confidence: float) -> np.ndarray:
        """Create keypoint array from positions."""
        keypoints = np.zeros((33, 3))
        keypoints[:, :2] = positions
        keypoints[:, 2] = confidence
        return keypoints

    def reset(self):
        """Reset prediction state."""
        self.position_history.clear()
