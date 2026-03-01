"""Layer compositor and background generator."""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class BackgroundConfig:
    """Configuration for background rendering."""

    style: str = "gradient"  # "gradient", "solid", "original"
    color: Tuple[int, int, int] = (20, 15, 40)  # Dark purple (RGB)
    gradient_top: Tuple[int, int, int] = (40, 20, 60)  # Purple
    gradient_bottom: Tuple[int, int, int] = (10, 10, 30)  # Dark
    original_opacity: float = 0.2  # If using original as background


class Compositor:
    """Composites multiple BGRA layers into final frame."""

    def __init__(self, background_config: Optional[BackgroundConfig] = None):
        self.bg_config = background_config or BackgroundConfig()
        self._gradient_cache = {}

    def create_background(
        self,
        frame_size: Tuple[int, int],
        original_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create background layer.

        Args:
            frame_size: (height, width)
            original_frame: Optional original video frame

        Returns:
            BGR image (no alpha needed for background)
        """
        height, width = frame_size

        if self.bg_config.style == "solid":
            return self._create_solid_background(width, height)
        elif self.bg_config.style == "original" and original_frame is not None:
            return self._create_original_background(original_frame)
        else:  # gradient (default)
            return self._create_gradient_background(width, height)

    def _create_solid_background(self, width: int, height: int) -> np.ndarray:
        """Create solid color background."""
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        # Convert RGB to BGR
        bg[:, :] = (
            self.bg_config.color[2],
            self.bg_config.color[1],
            self.bg_config.color[0],
        )
        return bg

    def _create_gradient_background(self, width: int, height: int) -> np.ndarray:
        """Create vertical gradient background."""
        cache_key = (width, height)

        if cache_key in self._gradient_cache:
            return self._gradient_cache[cache_key].copy()

        bg = np.zeros((height, width, 3), dtype=np.uint8)

        # Create vertical gradient
        for y in range(height):
            ratio = y / height
            r = int(
                self.bg_config.gradient_top[0] * (1 - ratio)
                + self.bg_config.gradient_bottom[0] * ratio
            )
            g = int(
                self.bg_config.gradient_top[1] * (1 - ratio)
                + self.bg_config.gradient_bottom[1] * ratio
            )
            b = int(
                self.bg_config.gradient_top[2] * (1 - ratio)
                + self.bg_config.gradient_bottom[2] * ratio
            )
            # BGR order for OpenCV
            bg[y, :] = (b, g, r)

        self._gradient_cache[cache_key] = bg.copy()
        return bg

    def _create_original_background(self, original_frame: np.ndarray) -> np.ndarray:
        """Create darkened original frame as background."""
        # Darken the original frame
        bg = (original_frame.astype(float) * self.bg_config.original_opacity).astype(
            np.uint8
        )
        return bg

    def composite(
        self,
        background: np.ndarray,
        layers: List[np.ndarray],
    ) -> np.ndarray:
        """
        Composite multiple BGRA layers onto background.

        Args:
            background: BGR background image
            layers: List of BGRA layers to composite (bottom to top)

        Returns:
            BGR composited image
        """
        result = background.copy().astype(float)

        for layer in layers:
            if layer is None:
                continue

            # Extract alpha channel and normalize
            alpha = layer[:, :, 3:4].astype(float) / 255.0

            # Blend layer onto result
            layer_rgb = layer[:, :, :3].astype(float)
            result = result * (1 - alpha) + layer_rgb * alpha

        return np.clip(result, 0, 255).astype(np.uint8)

    def composite_with_alpha(
        self,
        background: np.ndarray,
        layers: List[np.ndarray],
    ) -> np.ndarray:
        """
        Composite layers and return BGRA result with alpha.

        Args:
            background: BGR background image
            layers: List of BGRA layers

        Returns:
            BGRA composited image
        """
        height, width = background.shape[:2]
        result = np.zeros((height, width, 4), dtype=np.float32)

        # Start with background (full opacity)
        result[:, :, :3] = background.astype(float)
        result[:, :, 3] = 255.0

        for layer in layers:
            if layer is None:
                continue

            layer_alpha = layer[:, :, 3:4].astype(float) / 255.0
            layer_rgb = layer[:, :, :3].astype(float)

            # Porter-Duff over operation
            result[:, :, :3] = result[:, :, :3] * (1 - layer_alpha) + layer_rgb * layer_alpha
            result[:, :, 3:4] = result[:, :, 3:4] + layer[:, :, 3:4].astype(float) * (
                1 - result[:, :, 3:4] / 255.0
            )

        return np.clip(result, 0, 255).astype(np.uint8)


def blend_additive(
    base: np.ndarray,
    overlay: np.ndarray,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Blend overlay onto base using additive blending.

    Args:
        base: Base BGR image
        overlay: BGRA overlay image
        intensity: Blend intensity (0-1)

    Returns:
        Blended BGR image
    """
    if overlay is None:
        return base

    alpha = overlay[:, :, 3:4].astype(float) / 255.0 * intensity
    overlay_rgb = overlay[:, :, :3].astype(float)

    result = base.astype(float) + overlay_rgb * alpha
    return np.clip(result, 0, 255).astype(np.uint8)
