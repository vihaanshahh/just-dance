"""Motion smoothing using Kalman filters."""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Optional


class KeypointKalmanFilter:
    """Kalman filter for a single 2D keypoint with velocity estimation."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        """
        Initialize Kalman filter for a single keypoint.

        Args:
            process_noise: Process noise covariance (lower = smoother)
            measurement_noise: Measurement noise covariance (higher = more smoothing)
        """
        # State: [x, y, vx, vy] (position and velocity)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (constant velocity model)
        dt = 1.0  # Normalized time step
        self.kf.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        # Measurement matrix (we only observe position)
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        # Measurement noise covariance
        self.kf.R = np.eye(2) * measurement_noise

        # Process noise covariance
        self.kf.Q = (
            np.array(
                [
                    [dt**4 / 4, 0, dt**3 / 2, 0],
                    [0, dt**4 / 4, 0, dt**3 / 2],
                    [dt**3 / 2, 0, dt**2, 0],
                    [0, dt**3 / 2, 0, dt**2],
                ]
            )
            * process_noise
        )

        # Initial covariance
        self.kf.P *= 10

        self.initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement and return smoothed position."""
        if not self.initialized:
            # Initialize state with first measurement
            self.kf.x = np.array([[measurement[0]], [measurement[1]], [0], [0]])
            self.initialized = True
            return measurement

        # Predict next state
        self.kf.predict()

        # Update with measurement
        self.kf.update(measurement.reshape(2, 1))

        # Return smoothed position
        return self.kf.x[:2].flatten()

    def predict_ahead(self, steps: int = 1) -> np.ndarray:
        """Predict position n steps ahead without updating state."""
        # Clone current state
        x = self.kf.x.copy()
        F = self.kf.F

        # Apply transition matrix multiple times
        for _ in range(steps):
            x = F @ x

        return x[:2].flatten()

    def get_velocity(self) -> np.ndarray:
        """Get current estimated velocity."""
        return self.kf.x[2:4].flatten()

    def reset(self):
        """Reset filter state."""
        self.initialized = False
        self.kf.x = np.zeros((4, 1))
        self.kf.P = np.eye(4) * 10


class MotionSmoother:
    """Smooths all 33 MediaPipe pose keypoints using Kalman filters."""

    def __init__(
        self,
        num_keypoints: int = 33,
        process_noise: float = 0.005,
        measurement_noise: float = 0.05,
        visibility_threshold: float = 0.5,
    ):
        """
        Initialize motion smoother.

        Args:
            num_keypoints: Number of keypoints to track
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
            visibility_threshold: Minimum visibility to trust detection
        """
        self.num_keypoints = num_keypoints
        self.visibility_threshold = visibility_threshold

        # Create filter for each keypoint
        self.filters = [
            KeypointKalmanFilter(process_noise, measurement_noise)
            for _ in range(num_keypoints)
        ]

        # Track consecutive missing detections per keypoint
        self.missing_counts = np.zeros(num_keypoints)
        self.max_missing = 10  # Reset filter after this many missing frames

        # Store last valid positions for interpolation
        self.last_valid = np.zeros((num_keypoints, 2))

    def smooth(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Smooth a frame's keypoints.

        Args:
            keypoints: Array of shape (33, 3) with [x, y, visibility] per keypoint

        Returns:
            Smoothed keypoints array of shape (33, 3)
        """
        smoothed = np.zeros_like(keypoints)

        for i in range(self.num_keypoints):
            x, y, visibility = keypoints[i]

            if visibility >= self.visibility_threshold:
                # Valid detection - update filter
                measurement = np.array([x, y])
                smoothed_pos = self.filters[i].update(measurement)
                smoothed[i] = [smoothed_pos[0], smoothed_pos[1], visibility]

                self.last_valid[i] = smoothed_pos
                self.missing_counts[i] = 0

            else:
                # Missing detection - use prediction or last valid
                self.missing_counts[i] += 1

                if self.missing_counts[i] <= self.max_missing:
                    if self.filters[i].initialized:
                        # Use Kalman prediction
                        predicted = self.filters[i].predict_ahead(1)
                        smoothed[i] = [predicted[0], predicted[1], 0.3]
                    else:
                        # Use last valid position
                        smoothed[i] = [
                            self.last_valid[i][0],
                            self.last_valid[i][1],
                            0.1,
                        ]
                else:
                    # Too many missing - reset filter
                    self.filters[i].reset()
                    smoothed[i] = [
                        self.last_valid[i][0],
                        self.last_valid[i][1],
                        0.0,
                    ]

        return smoothed

    def get_velocities(self) -> np.ndarray:
        """Get estimated velocities for all keypoints."""
        velocities = np.zeros((self.num_keypoints, 2))
        for i, f in enumerate(self.filters):
            if f.initialized:
                velocities[i] = f.get_velocity()
        return velocities

    def reset(self):
        """Reset all filters."""
        for f in self.filters:
            f.reset()
        self.missing_counts = np.zeros(self.num_keypoints)


class AdaptiveMotionSmoother(MotionSmoother):
    """Motion smoother with adaptive noise based on motion intensity."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.velocity_history = []
        self.history_length = 10
        self.base_measurement_noise = kwargs.get("measurement_noise", 0.05)

    def smooth(self, keypoints: np.ndarray) -> np.ndarray:
        """Smooth keypoints with adaptive noise adjustment."""
        # First do regular smoothing
        smoothed = super().smooth(keypoints)

        # Compute current frame's average velocity
        velocities = self.get_velocities()
        avg_velocity = np.linalg.norm(velocities, axis=1).mean()

        self.velocity_history.append(avg_velocity)
        if len(self.velocity_history) > self.history_length:
            self.velocity_history.pop(0)

        # Adapt measurement noise based on motion intensity
        if len(self.velocity_history) >= 3:
            recent_avg = np.mean(self.velocity_history[-3:])

            # Scale noise inversely with velocity
            # Fast motion: trust measurements more (lower noise)
            # Slow motion: smooth more aggressively (higher noise)
            noise_scale = max(0.5, min(2.0, 1.0 / (1.0 + recent_avg * 10)))

            for f in self.filters:
                f.kf.R = np.eye(2) * (self.base_measurement_noise * noise_scale)

        return smoothed
