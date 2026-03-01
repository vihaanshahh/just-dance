"""Clean white body silhouette renderer with shadow."""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SilhouetteConfig:
    """Configuration for silhouette rendering."""

    color: Tuple[int, int, int] = (255, 255, 255)  # White
    shadow_color: Tuple[int, int, int] = (40, 40, 50)  # Dark shadow
    shadow_offset: Tuple[int, int] = (8, 8)  # Shadow offset (x, y)
    shadow_blur: int = 15
    body_thickness: int = 22
    joint_radius: int = 11
    edge_blur: int = 2


class SilhouetteRenderer:
    """Renders clean white body silhouette with shadow."""

    BODY_CONNECTIONS = [
        (11, 12),  # shoulders
        (11, 23), (12, 24),  # torso sides
        (23, 24),  # hips
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]

    JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    THICKNESS = {
        (11, 12): 1.4, (23, 24): 1.3,  # shoulders/hips wider
        (11, 23): 1.2, (12, 24): 1.2,  # torso
        (11, 13): 0.9, (12, 14): 0.9,  # upper arms
        (13, 15): 0.75, (14, 16): 0.75,  # forearms
        (23, 25): 1.0, (24, 26): 1.0,  # thighs
        (25, 27): 0.8, (26, 28): 0.8,  # shins
    }

    def __init__(self, config: Optional[SilhouetteConfig] = None):
        self.config = config or SilhouetteConfig()

    def render(self, keypoints: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
        """Render silhouette with shadow."""
        height, width = frame_size
        layer = np.zeros((height, width, 4), dtype=np.uint8)

        # Draw shadow first (offset and blurred)
        shadow_layer = np.zeros((height, width, 4), dtype=np.uint8)
        self._draw_body(shadow_layer, keypoints, width, height, is_shadow=True)

        # Blur the shadow
        if self.config.shadow_blur > 0:
            k = self.config.shadow_blur * 2 + 1
            shadow_layer = cv2.GaussianBlur(shadow_layer, (k, k), 0)

        # Composite shadow onto layer
        self._composite(layer, shadow_layer)

        # Draw main body on top
        self._draw_body(layer, keypoints, width, height, is_shadow=False)

        # Slight edge softening
        if self.config.edge_blur > 0:
            k = self.config.edge_blur * 2 + 1
            alpha = layer[:, :, 3]
            layer[:, :, 3] = cv2.GaussianBlur(alpha, (k, k), 0)

        return layer

    def _draw_body(
        self,
        layer: np.ndarray,
        keypoints: np.ndarray,
        width: int,
        height: int,
        is_shadow: bool,
    ):
        """Draw the body segments and joints."""
        if is_shadow:
            color = self.config.shadow_color
            offset = self.config.shadow_offset
            alpha = 180
        else:
            color = self.config.color
            offset = (0, 0)
            alpha = 255

        color_bgra = (color[2], color[1], color[0], alpha)

        # Draw segments
        for start_idx, end_idx in self.BODY_CONNECTIONS:
            if keypoints[start_idx, 2] < 0.3 or keypoints[end_idx, 2] < 0.3:
                continue

            pt1 = self._to_pixel(keypoints[start_idx], width, height, offset)
            pt2 = self._to_pixel(keypoints[end_idx], width, height, offset)

            thickness = int(self.config.body_thickness * self.THICKNESS.get((start_idx, end_idx), 0.8))
            cv2.line(layer, pt1, pt2, color_bgra, thickness, cv2.LINE_AA)

        # Draw joints
        for idx in self.JOINTS:
            if keypoints[idx, 2] < 0.3:
                continue
            pt = self._to_pixel(keypoints[idx], width, height, offset)
            cv2.circle(layer, pt, self.config.joint_radius, color_bgra, -1, cv2.LINE_AA)

        # Draw head
        if keypoints[0, 2] > 0.3:
            head_pt = self._to_pixel(keypoints[0], width, height, offset)

            # Estimate head size from ears
            if keypoints[7, 2] > 0.3 and keypoints[8, 2] > 0.3:
                ear_l = self._to_pixel(keypoints[7], width, height, (0, 0))
                ear_r = self._to_pixel(keypoints[8], width, height, (0, 0))
                ear_dist = np.sqrt((ear_l[0] - ear_r[0])**2 + (ear_l[1] - ear_r[1])**2)
                head_w = int(ear_dist * 1.3)
                head_h = int(ear_dist * 1.6)
            else:
                head_w = self.config.body_thickness * 2
                head_h = int(self.config.body_thickness * 2.5)

            cv2.ellipse(layer, head_pt, (head_w // 2, head_h // 2), 0, 0, 360, color_bgra, -1, cv2.LINE_AA)

    def _to_pixel(self, kp: np.ndarray, w: int, h: int, offset: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
        x = int(kp[0] * w) + offset[0]
        y = int(kp[1] * h) + offset[1]
        return (x, y)

    def _composite(self, base: np.ndarray, overlay: np.ndarray):
        """Composite overlay onto base using alpha blending."""
        alpha = overlay[:, :, 3:4].astype(float) / 255.0
        base[:, :, :3] = (base[:, :, :3].astype(float) * (1 - alpha) + overlay[:, :, :3].astype(float) * alpha).astype(np.uint8)
        base[:, :, 3] = np.clip(base[:, :, 3].astype(float) + overlay[:, :, 3].astype(float) * 0.5, 0, 255).astype(np.uint8)
