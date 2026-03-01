"""Iconic Just Dance glove effect renderer."""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GloveConfig:
    """Configuration for glove effect."""

    color: Tuple[int, int, int] = (255, 50, 50)  # Red (RGB) - Just Dance classic
    glow_intensity: float = 0.7
    glow_radius: int = 25
    glove_radius: int = 30
    outline_thickness: int = 3
    outline_color: Tuple[int, int, int] = (255, 255, 255)
    animate_pulse: bool = True
    pulse_speed: float = 2.0  # Pulses per second


class GloveRenderer:
    """Renders iconic Just Dance glove effect on hands."""

    # MediaPipe hand landmark indices
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_THUMB = 21
    RIGHT_THUMB = 22

    def __init__(self, config: Optional[GloveConfig] = None):
        self.config = config or GloveConfig()
        self.frame_count = 0

    def render(
        self,
        keypoints: np.ndarray,
        frame_size: Tuple[int, int],
        time_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Render glove effects on both hands.

        Args:
            keypoints: Pose keypoints (33, 3)
            frame_size: (height, width)
            time_offset: Time in seconds for pulse animation

        Returns:
            BGRA layer with glove effects
        """
        height, width = frame_size
        layer = np.zeros((height, width, 4), dtype=np.uint8)

        self.frame_count += 1

        # Calculate pulse factor for animation
        pulse_factor = 1.0
        if self.config.animate_pulse:
            pulse_phase = (time_offset * self.config.pulse_speed) % 1.0
            pulse_factor = 0.85 + 0.15 * np.sin(pulse_phase * 2 * np.pi)

        # Render right hand glove (primary in Just Dance)
        self._render_hand_glove(
            layer,
            keypoints,
            width,
            height,
            self.RIGHT_WRIST,
            self.RIGHT_INDEX,
            self.RIGHT_PINKY,
            self.RIGHT_THUMB,
            pulse_factor,
        )

        # Render left hand glove
        self._render_hand_glove(
            layer,
            keypoints,
            width,
            height,
            self.LEFT_WRIST,
            self.LEFT_INDEX,
            self.LEFT_PINKY,
            self.LEFT_THUMB,
            pulse_factor,
        )

        return layer

    def _render_hand_glove(
        self,
        layer: np.ndarray,
        keypoints: np.ndarray,
        width: int,
        height: int,
        wrist_idx: int,
        index_idx: int,
        pinky_idx: int,
        thumb_idx: int,
        pulse_factor: float,
    ):
        """Render glove effect for a single hand."""
        # Check if hand keypoints are visible
        wrist = keypoints[wrist_idx]
        index_finger = keypoints[index_idx]

        if wrist[2] < 0.3 and index_finger[2] < 0.3:
            return  # Hand not visible

        # Calculate hand center (between wrist and fingers)
        visible_points = []
        for idx in [wrist_idx, index_idx, pinky_idx, thumb_idx]:
            if keypoints[idx, 2] > 0.3:
                visible_points.append(keypoints[idx, :2])

        if not visible_points:
            return

        center = np.mean(visible_points, axis=0)
        center_px = (int(center[0] * width), int(center[1] * height))

        # Calculate hand size from visible landmarks
        if len(visible_points) >= 2:
            distances = [np.linalg.norm(p - center) for p in visible_points]
            hand_radius = int(max(distances) * min(width, height) * 1.3)
            hand_radius = max(hand_radius, self.config.glove_radius)
        else:
            hand_radius = self.config.glove_radius

        # Apply pulse
        current_radius = int(hand_radius * pulse_factor)
        glow_radius = int(self.config.glow_radius * pulse_factor)

        # Render outer glow first (behind main shape)
        self._render_glow(
            layer,
            center_px,
            current_radius + glow_radius,
            self.config.color,
            self.config.glow_intensity * pulse_factor,
        )

        # Render main glove shape
        self._render_glove_shape(
            layer,
            keypoints,
            width,
            height,
            wrist_idx,
            index_idx,
            pinky_idx,
            thumb_idx,
            current_radius,
        )

    def _render_glow(
        self,
        layer: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
        intensity: float,
    ):
        """Render radial glow effect."""
        # Create temporary layer for glow
        height, width = layer.shape[:2]
        glow_layer = np.zeros((height, width, 4), dtype=np.uint8)

        # Draw multiple concentric circles with decreasing opacity
        num_circles = max(8, radius // 4)

        for i in range(num_circles, 0, -1):
            circle_radius = int(radius * (i / num_circles))
            # Exponential falloff for more natural glow
            falloff = ((num_circles - i + 1) / num_circles) ** 1.5
            opacity = int(255 * intensity * falloff * 0.5)

            cv2.circle(
                glow_layer,
                center,
                circle_radius,
                (color[2], color[1], color[0], opacity),  # BGRA
                -1,
                cv2.LINE_AA,
            )

        # Apply gaussian blur for smooth glow
        glow_layer = cv2.GaussianBlur(glow_layer, (31, 31), 0)

        # Blend with main layer using alpha
        alpha = glow_layer[:, :, 3:4].astype(float) / 255.0
        layer[:, :, :3] = np.clip(
            layer[:, :, :3].astype(float) + glow_layer[:, :, :3].astype(float) * alpha,
            0,
            255,
        ).astype(np.uint8)
        layer[:, :, 3] = np.clip(
            layer[:, :, 3].astype(float) + glow_layer[:, :, 3].astype(float) * 0.7,
            0,
            255,
        ).astype(np.uint8)

    def _render_glove_shape(
        self,
        layer: np.ndarray,
        keypoints: np.ndarray,
        width: int,
        height: int,
        wrist_idx: int,
        index_idx: int,
        pinky_idx: int,
        thumb_idx: int,
        radius: int,
    ):
        """Render the solid glove shape."""
        # Collect visible hand points
        points = []
        for idx in [wrist_idx, thumb_idx, index_idx, pinky_idx]:
            if keypoints[idx, 2] > 0.3:
                pt = (
                    int(keypoints[idx, 0] * width),
                    int(keypoints[idx, 1] * height),
                )
                points.append(pt)

        if len(points) < 2:
            # Fallback to simple circle
            if keypoints[index_idx, 2] > 0.3:
                center = (
                    int(keypoints[index_idx, 0] * width),
                    int(keypoints[index_idx, 1] * height),
                )
                cv2.circle(
                    layer,
                    center,
                    radius,
                    (self.config.color[2], self.config.color[1], self.config.color[0], 255),
                    -1,
                    cv2.LINE_AA,
                )
                # Add outline
                cv2.circle(
                    layer,
                    center,
                    radius,
                    (
                        self.config.outline_color[2],
                        self.config.outline_color[1],
                        self.config.outline_color[0],
                        255,
                    ),
                    self.config.outline_thickness,
                    cv2.LINE_AA,
                )
            return

        # Create convex hull of hand points
        points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points)

        # Expand hull slightly for better coverage
        center = np.mean(points, axis=0)
        expanded_hull = []
        for pt in hull:
            direction = pt[0] - center
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            expanded_pt = pt[0] + direction * (radius * 0.3)
            expanded_hull.append(expanded_pt.astype(np.int32))
        expanded_hull = np.array(expanded_hull)

        # Draw filled hull
        cv2.fillConvexPoly(
            layer,
            expanded_hull,
            (self.config.color[2], self.config.color[1], self.config.color[0], 255),
            cv2.LINE_AA,
        )

        # Draw outline
        cv2.polylines(
            layer,
            [expanded_hull],
            True,
            (
                self.config.outline_color[2],
                self.config.outline_color[1],
                self.config.outline_color[0],
                255,
            ),
            self.config.outline_thickness,
            cv2.LINE_AA,
        )
