"""Ghost preview renderer showing upcoming poses."""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class RibbonConfig:
    """Configuration for ghost preview rendering."""

    num_ghosts: int = 2  # Number of ghost frames to show
    ghost_interval: int = 15  # Frames between each ghost (0.5 sec at 30fps)
    ghost_opacity: float = 0.25  # Opacity of ghosts
    ghost_color: Tuple[int, int, int] = (200, 220, 255)  # Light blue-white
    line_thickness: int = 12


class RibbonRenderer:
    """Renders ghost preview silhouettes showing upcoming poses."""

    # Body segments for ghost (simplified)
    BODY_SEGMENTS = [
        (11, 12),  # shoulders
        (11, 23), (12, 24),  # torso
        (23, 24),  # hips
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]

    def __init__(self, config: Optional[RibbonConfig] = None):
        self.config = config or RibbonConfig()

    def render(
        self,
        current_keypoints: np.ndarray,
        predicted_keypoints: List[np.ndarray],
        frame_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Render ghost silhouettes showing future poses.

        Args:
            current_keypoints: Current frame keypoints (33, 3)
            predicted_keypoints: List of predicted future keypoint frames
            frame_size: (height, width) of output frame

        Returns:
            BGRA image with ghost overlay
        """
        height, width = frame_size
        layer = np.zeros((height, width, 4), dtype=np.uint8)

        if not predicted_keypoints:
            return layer

        # Draw ghosts at intervals
        for i in range(self.config.num_ghosts):
            frame_idx = (i + 1) * self.config.ghost_interval
            if frame_idx >= len(predicted_keypoints):
                break

            ghost_kp = predicted_keypoints[frame_idx]
            # Fade opacity for further ghosts
            opacity = self.config.ghost_opacity * (1 - i * 0.3)
            self._draw_ghost(layer, ghost_kp, width, height, opacity)

        return layer

    def _draw_ghost(
        self,
        layer: np.ndarray,
        keypoints: np.ndarray,
        width: int,
        height: int,
        opacity: float,
    ):
        """Draw a single ghost silhouette."""
        color = self.config.ghost_color
        alpha = int(opacity * 255)
        color_bgra = (color[2], color[1], color[0], alpha)

        # Draw body segments
        for start_idx, end_idx in self.BODY_SEGMENTS:
            if keypoints[start_idx, 2] < 0.2 or keypoints[end_idx, 2] < 0.2:
                continue

            pt1 = self._to_pixel(keypoints[start_idx], width, height)
            pt2 = self._to_pixel(keypoints[end_idx], width, height)

            cv2.line(layer, pt1, pt2, color_bgra, self.config.line_thickness, cv2.LINE_AA)

        # Draw head
        if keypoints[0, 2] > 0.2:
            head_pt = self._to_pixel(keypoints[0], width, height)
            cv2.circle(layer, head_pt, 20, color_bgra, -1, cv2.LINE_AA)

    def _to_pixel(self, kp: np.ndarray, w: int, h: int) -> Tuple[int, int]:
        return (int(np.clip(kp[0], 0, 1) * w), int(np.clip(kp[1], 0, 1) * h))
