"""Clean body silhouette renderer using segmentation mask."""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SilhouetteConfig:
    """Configuration for silhouette rendering."""

    color: Tuple[int, int, int] = (255, 255, 255)  # White
    shadow_color: Tuple[int, int, int] = (40, 40, 50)  # Dark shadow
    shadow_offset: Tuple[int, int] = (8, 8)
    shadow_blur: int = 21
    edge_blur: int = 3
    mask_threshold: float = 0.5
    # Morphological cleanup
    morph_close_size: int = 9
    morph_open_size: int = 5
    # Glow
    glow_size: int = 25
    glow_intensity: float = 0.4
    glow_color: Tuple[int, int, int] = (180, 200, 255)  # Soft blue-white


class SilhouetteRenderer:
    """Renders clean body silhouette from segmentation mask."""

    def __init__(self, config: Optional[SilhouetteConfig] = None):
        self.config = config or SilhouetteConfig()

    def render(
        self,
        keypoints: np.ndarray,
        frame_size: Tuple[int, int],
        seg_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render silhouette from segmentation mask with shadow and glow."""
        height, width = frame_size
        layer = np.zeros((height, width, 4), dtype=np.uint8)

        if seg_mask is None:
            return layer

        # Clean up the mask
        mask = self._clean_mask(seg_mask, width, height)

        # Draw shadow (offset + blurred copy)
        self._draw_shadow(layer, mask, width, height)

        # Draw outer glow
        self._draw_glow(layer, mask)

        # Draw the main body
        self._draw_body(layer, mask)

        return layer

    def _clean_mask(self, seg_mask: np.ndarray, width: int, height: int) -> np.ndarray:
        """Threshold and morphologically clean the segmentation mask."""
        # Resize if needed
        if seg_mask.shape[:2] != (height, width):
            seg_mask = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_LINEAR)

        # Threshold to binary
        mask = (seg_mask > self.config.mask_threshold).astype(np.uint8) * 255

        # Close small gaps (fills holes in the body)
        k = self.config.morph_close_size
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # Open to remove small noise blobs
        k = self.config.morph_open_size
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # Smooth edges
        if self.config.edge_blur > 0:
            kk = self.config.edge_blur * 2 + 1
            mask = cv2.GaussianBlur(mask, (kk, kk), 0)

        return mask

    def _draw_shadow(self, layer: np.ndarray, mask: np.ndarray, width: int, height: int):
        """Draw offset blurred shadow."""
        ox, oy = self.config.shadow_offset
        # Shift mask for shadow
        M = np.float32([[1, 0, ox], [0, 1, oy]])
        shadow_mask = cv2.warpAffine(mask, M, (width, height))

        # Heavy blur for soft shadow
        k = self.config.shadow_blur * 2 + 1
        shadow_mask = cv2.GaussianBlur(shadow_mask, (k, k), 0)

        alpha = shadow_mask.astype(np.float32) / 255.0
        sc = self.config.shadow_color
        # Blend shadow into layer
        for c, val in enumerate([sc[2], sc[1], sc[0]]):  # BGR
            layer[:, :, c] = np.clip(
                layer[:, :, c].astype(np.float32) + val * alpha * 0.7, 0, 255
            ).astype(np.uint8)
        layer[:, :, 3] = np.clip(
            layer[:, :, 3].astype(np.float32) + alpha * 180, 0, 255
        ).astype(np.uint8)

    def _draw_glow(self, layer: np.ndarray, mask: np.ndarray):
        """Draw soft outer glow around the body."""
        k = self.config.glow_size * 2 + 1
        glow = cv2.GaussianBlur(mask, (k, k), 0)

        alpha = glow.astype(np.float32) / 255.0 * self.config.glow_intensity
        gc = self.config.glow_color
        for c, val in enumerate([gc[2], gc[1], gc[0]]):  # BGR
            layer[:, :, c] = np.clip(
                layer[:, :, c].astype(np.float32) + val * alpha, 0, 255
            ).astype(np.uint8)
        layer[:, :, 3] = np.clip(
            layer[:, :, 3].astype(np.float32) + alpha * 200, 0, 255
        ).astype(np.uint8)

    def _draw_body(self, layer: np.ndarray, mask: np.ndarray):
        """Draw the solid white body from the clean mask."""
        alpha = mask.astype(np.float32) / 255.0
        bc = self.config.color
        # Overwrite with body color where mask is active
        for c, val in enumerate([bc[2], bc[1], bc[0]]):  # BGR
            layer[:, :, c] = np.clip(
                layer[:, :, c].astype(np.float32) * (1 - alpha) + val * alpha, 0, 255
            ).astype(np.uint8)
        layer[:, :, 3] = np.clip(
            layer[:, :, 3].astype(np.float32) * (1 - alpha) + 255 * alpha, 0, 255
        ).astype(np.uint8)
