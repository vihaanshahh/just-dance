"""3D wooden art mannequin renderer driven by pose keypoints."""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class MannequinConfig:
    """Configuration for 3D mannequin rendering."""

    # Wood-tone colors (BGR for OpenCV)
    skin_color: Tuple[int, int, int] = (125, 170, 205)  # Warm wood BGR
    highlight_color: Tuple[int, int, int] = (175, 210, 235)  # Light wood BGR
    shadow_color: Tuple[int, int, int] = (60, 100, 130)  # Dark wood BGR
    joint_color: Tuple[int, int, int] = (100, 145, 175)  # Slightly darker wood BGR
    outline_color: Tuple[int, int, int] = (30, 45, 60)  # Dark outline BGR

    # Lighting
    light_dir: Tuple[float, float] = (-0.6, -0.8)  # Top-left light (x, y)

    # Body proportions (relative to shoulder width)
    head_radius_factor: float = 0.35
    neck_width_factor: float = 0.12
    upper_arm_width_factor: float = 0.10
    forearm_width_factor: float = 0.08
    upper_leg_width_factor: float = 0.12
    lower_leg_width_factor: float = 0.09
    joint_radius_factor: float = 0.06
    torso_taper: float = 0.85  # Hip width relative to shoulder width

    # Drop shadow
    shadow_enabled: bool = True
    shadow_offset: Tuple[int, int] = (6, 6)
    shadow_blur: int = 15
    shadow_alpha: float = 0.5

    # Outline
    outline_thickness: int = 2

    # Visibility
    visibility_threshold: float = 0.2


# MediaPipe landmark indices
_NOSE = 0
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28

# Limb definitions: (start_idx, end_idx, width_factor_attr)
_LIMBS = [
    (_LEFT_SHOULDER, _LEFT_ELBOW, 'upper_arm_width_factor'),
    (_LEFT_ELBOW, _LEFT_WRIST, 'forearm_width_factor'),
    (_RIGHT_SHOULDER, _RIGHT_ELBOW, 'upper_arm_width_factor'),
    (_RIGHT_ELBOW, _RIGHT_WRIST, 'forearm_width_factor'),
    (_LEFT_HIP, _LEFT_KNEE, 'upper_leg_width_factor'),
    (_LEFT_KNEE, _LEFT_ANKLE, 'lower_leg_width_factor'),
    (_RIGHT_HIP, _RIGHT_KNEE, 'upper_leg_width_factor'),
    (_RIGHT_KNEE, _RIGHT_ANKLE, 'lower_leg_width_factor'),
]

# Joint indices that get sphere rendering
_JOINT_INDICES = [
    _LEFT_SHOULDER, _RIGHT_SHOULDER,
    _LEFT_ELBOW, _RIGHT_ELBOW,
    _LEFT_WRIST, _RIGHT_WRIST,
    _LEFT_HIP, _RIGHT_HIP,
    _LEFT_KNEE, _RIGHT_KNEE,
    _LEFT_ANKLE, _RIGHT_ANKLE,
]


class MannequinRenderer:
    """Renders a 3D-looking wooden art mannequin from pose keypoints."""

    def __init__(self, config: Optional[MannequinConfig] = None):
        self.config = config or MannequinConfig()
        self._sphere_cache: Dict[int, np.ndarray] = {}
        # Normalize light direction
        lx, ly = self.config.light_dir
        mag = max(np.sqrt(lx * lx + ly * ly), 1e-6)
        self._light_x = lx / mag
        self._light_y = ly / mag

    def render(
        self,
        keypoints: np.ndarray,
        frame_size: Tuple[int, int],
        seg_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render mannequin figure. Same signature as SilhouetteRenderer.render().

        Args:
            keypoints: (N, 4) array [x, y, z, visibility]
            frame_size: (height, width)
            seg_mask: Ignored (mannequin doesn't use segmentation)

        Returns:
            BGRA image layer
        """
        height, width = frame_size
        layer = np.zeros((height, width, 4), dtype=np.uint8)

        if keypoints is None or len(keypoints) < 29:
            return layer

        # Convert normalized keypoints to pixel coordinates
        kp = keypoints.copy()
        px = kp[:, 0] * width
        py = kp[:, 1] * height
        vis = kp[:, 3] if kp.shape[1] > 3 else np.ones(len(kp))

        # Compute reference scale from shoulder width
        if not self._visible(vis, _LEFT_SHOULDER) or not self._visible(vis, _RIGHT_SHOULDER):
            return layer

        shoulder_width = np.sqrt(
            (px[_LEFT_SHOULDER] - px[_RIGHT_SHOULDER]) ** 2
            + (py[_LEFT_SHOULDER] - py[_RIGHT_SHOULDER]) ** 2
        )
        if shoulder_width < 5:
            return layer

        # Determine facing direction for depth ordering
        facing_right = px[_LEFT_SHOULDER] < px[_RIGHT_SHOULDER]

        # Build draw order based on facing
        draw_calls = self._build_draw_order(px, py, vis, shoulder_width, facing_right)

        # Draw drop shadow first
        if self.config.shadow_enabled:
            self._draw_drop_shadow(layer, draw_calls, px, py, vis, shoulder_width)

        # Draw all parts in depth order
        for call in draw_calls:
            call(layer, px, py, vis, shoulder_width)

        return layer

    def _visible(self, vis: np.ndarray, idx: int) -> bool:
        return vis[idx] > self.config.visibility_threshold

    def _build_draw_order(
        self, px, py, vis, sw, facing_right: bool
    ) -> list:
        """Build list of draw callables in back-to-front order."""
        calls = []

        if facing_right:
            # Back: right arm, right leg
            back_arm = [(_RIGHT_SHOULDER, _RIGHT_ELBOW, 'upper_arm_width_factor'),
                        (_RIGHT_ELBOW, _RIGHT_WRIST, 'forearm_width_factor')]
            back_leg = [(_RIGHT_HIP, _RIGHT_KNEE, 'upper_leg_width_factor'),
                        (_RIGHT_KNEE, _RIGHT_ANKLE, 'lower_leg_width_factor')]
            back_arm_joints = [_RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST]
            back_leg_joints = [_RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE]
            front_arm = [(_LEFT_SHOULDER, _LEFT_ELBOW, 'upper_arm_width_factor'),
                         (_LEFT_ELBOW, _LEFT_WRIST, 'forearm_width_factor')]
            front_leg = [(_LEFT_HIP, _LEFT_KNEE, 'upper_leg_width_factor'),
                         (_LEFT_KNEE, _LEFT_ANKLE, 'lower_leg_width_factor')]
            front_arm_joints = [_LEFT_SHOULDER, _LEFT_ELBOW, _LEFT_WRIST]
            front_leg_joints = [_LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE]
        else:
            back_arm = [(_LEFT_SHOULDER, _LEFT_ELBOW, 'upper_arm_width_factor'),
                        (_LEFT_ELBOW, _LEFT_WRIST, 'forearm_width_factor')]
            back_leg = [(_LEFT_HIP, _LEFT_KNEE, 'upper_leg_width_factor'),
                        (_LEFT_KNEE, _LEFT_ANKLE, 'lower_leg_width_factor')]
            back_arm_joints = [_LEFT_SHOULDER, _LEFT_ELBOW, _LEFT_WRIST]
            back_leg_joints = [_LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE]
            front_arm = [(_RIGHT_SHOULDER, _RIGHT_ELBOW, 'upper_arm_width_factor'),
                         (_RIGHT_ELBOW, _RIGHT_WRIST, 'forearm_width_factor')]
            front_leg = [(_RIGHT_HIP, _RIGHT_KNEE, 'upper_leg_width_factor'),
                         (_RIGHT_KNEE, _RIGHT_ANKLE, 'lower_leg_width_factor')]
            front_arm_joints = [_RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST]
            front_leg_joints = [_RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE]

        # Draw order: back arm -> back leg -> torso -> neck/head -> front leg -> front arm
        calls.append(lambda l, px, py, v, sw, limbs=back_arm, joints=back_arm_joints:
                     self._draw_limb_group(l, px, py, v, sw, limbs, joints))
        calls.append(lambda l, px, py, v, sw, limbs=back_leg, joints=back_leg_joints:
                     self._draw_limb_group(l, px, py, v, sw, limbs, joints))
        calls.append(lambda l, px, py, v, sw: self._draw_torso(l, px, py, v, sw))
        calls.append(lambda l, px, py, v, sw: self._draw_neck_head(l, px, py, v, sw))
        calls.append(lambda l, px, py, v, sw, limbs=front_leg, joints=front_leg_joints:
                     self._draw_limb_group(l, px, py, v, sw, limbs, joints))
        calls.append(lambda l, px, py, v, sw, limbs=front_arm, joints=front_arm_joints:
                     self._draw_limb_group(l, px, py, v, sw, limbs, joints))

        return calls

    def _draw_limb_group(self, layer, px, py, vis, sw, limbs, joints):
        """Draw a group of limbs and their joints."""
        for start_idx, end_idx, width_attr in limbs:
            if self._visible(vis, start_idx) and self._visible(vis, end_idx):
                width = getattr(self.config, width_attr) * sw
                self._draw_shaded_capsule(
                    layer,
                    (px[start_idx], py[start_idx]),
                    (px[end_idx], py[end_idx]),
                    width,
                )
        for j in joints:
            if self._visible(vis, j):
                r = int(self.config.joint_radius_factor * sw)
                self._draw_shaded_sphere(layer, int(px[j]), int(py[j]), max(r, 3))

    def _draw_torso(self, layer, px, py, vis, sw):
        """Draw torso as a shaded trapezoid."""
        if not (self._visible(vis, _LEFT_SHOULDER) and self._visible(vis, _RIGHT_SHOULDER)
                and self._visible(vis, _LEFT_HIP) and self._visible(vis, _RIGHT_HIP)):
            return

        ls = np.array([px[_LEFT_SHOULDER], py[_LEFT_SHOULDER]])
        rs = np.array([px[_RIGHT_SHOULDER], py[_RIGHT_SHOULDER]])
        lh = np.array([px[_LEFT_HIP], py[_LEFT_HIP]])
        rh = np.array([px[_RIGHT_HIP], py[_RIGHT_HIP]])

        # Slightly expand shoulders and taper hips
        s_center = (ls + rs) / 2
        h_center = (lh + rh) / 2
        ls_exp = s_center + (ls - s_center) * 1.1
        rs_exp = s_center + (rs - s_center) * 1.1
        lh_tap = h_center + (lh - h_center) * self.config.torso_taper
        rh_tap = h_center + (rh - h_center) * self.config.torso_taper

        # Build polygon points
        pts = np.array([ls_exp, rs_exp, rh_tap, lh_tap], dtype=np.float32)

        # Draw with horizontal strip shading
        n_strips = 8
        for i in range(n_strips):
            t0 = i / n_strips
            t1 = (i + 1) / n_strips

            # Interpolate corners
            tl = ls_exp * (1 - t0) + lh_tap * t0
            tr = rs_exp * (1 - t0) + rh_tap * t0
            bl = ls_exp * (1 - t1) + lh_tap * t1
            br = rs_exp * (1 - t1) + rh_tap * t1

            strip_pts = np.array([tl, tr, br, bl], dtype=np.int32)

            # Shade: lighter at top, darker at bottom
            shade = 1.0 - t0 * 0.4
            # Also apply left-right lighting
            light_factor = 0.5 + 0.5 * (-self._light_y)  # Top light
            shade *= (0.7 + 0.3 * light_factor)
            color = self._shade_color(shade)

            cv2.fillConvexPoly(layer, strip_pts, (*color, 255))

        # Draw outline
        outline_pts = np.array([ls_exp, rs_exp, rh_tap, lh_tap], dtype=np.int32)
        cv2.polylines(layer, [outline_pts], True, (*self.config.outline_color, 255),
                      self.config.outline_thickness, cv2.LINE_AA)

        # Subtle center seam
        top_mid = ((ls_exp + rs_exp) / 2).astype(int)
        bot_mid = ((lh_tap + rh_tap) / 2).astype(int)
        seam_color = tuple(max(0, c - 20) for c in self.config.skin_color) + (120,)
        cv2.line(layer, tuple(top_mid), tuple(bot_mid), seam_color, 1, cv2.LINE_AA)

    def _draw_neck_head(self, layer, px, py, vis, sw):
        """Draw neck capsule and head sphere."""
        if not (self._visible(vis, _LEFT_SHOULDER) and self._visible(vis, _RIGHT_SHOULDER)):
            return

        mid_shoulder_x = (px[_LEFT_SHOULDER] + px[_RIGHT_SHOULDER]) / 2
        mid_shoulder_y = (py[_LEFT_SHOULDER] + py[_RIGHT_SHOULDER]) / 2

        # Head position: use nose if available, otherwise estimate above shoulders
        if self._visible(vis, _NOSE):
            head_x, head_y = px[_NOSE], py[_NOSE]
        else:
            head_x = mid_shoulder_x
            head_y = mid_shoulder_y - sw * 0.6

        # Neck: from mid-shoulders to head base
        head_radius = int(self.config.head_radius_factor * sw)
        neck_base_y = head_y + head_radius * 0.7
        neck_width = self.config.neck_width_factor * sw

        self._draw_shaded_capsule(
            layer,
            (mid_shoulder_x, mid_shoulder_y),
            (head_x, neck_base_y),
            neck_width,
        )

        # Head sphere
        self._draw_shaded_sphere(layer, int(head_x), int(head_y), max(head_radius, 5))

    def _draw_shaded_capsule(self, layer, p1, p2, width):
        """Draw a shaded capsule (limb) between two points."""
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])

        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 1:
            return

        # Perpendicular direction
        nx = -dy / length
        ny = dx / length

        half_w = width / 2
        n_strips = 7

        for i in range(n_strips):
            # Offset from center: -1 to +1
            t0 = -1.0 + 2.0 * i / n_strips
            t1 = -1.0 + 2.0 * (i + 1) / n_strips

            off0 = t0 * half_w
            off1 = t1 * half_w

            # Four corners of the strip
            pts = np.array([
                [x1 + nx * off0, y1 + ny * off0],
                [x1 + nx * off1, y1 + ny * off1],
                [x2 + nx * off1, y2 + ny * off1],
                [x2 + nx * off0, y2 + ny * off0],
            ], dtype=np.int32)

            # Compute surface normal for this strip (points outward)
            strip_center = (t0 + t1) / 2
            surf_nx = nx * strip_center
            surf_ny = ny * strip_center

            # Dot product with light direction
            dot = surf_nx * self._light_x + surf_ny * self._light_y
            # Map from [-1,1] to shade factor [0.3, 1.2]
            shade = 0.5 + 0.5 * dot
            shade = max(0.3, min(1.2, shade * 1.2))

            color = self._shade_color(shade)
            cv2.fillConvexPoly(layer, pts, (*color, 255), cv2.LINE_AA)

        # Outline edges
        edge1_start = (int(x1 + nx * half_w), int(y1 + ny * half_w))
        edge1_end = (int(x2 + nx * half_w), int(y2 + ny * half_w))
        edge2_start = (int(x1 - nx * half_w), int(y1 - ny * half_w))
        edge2_end = (int(x2 - nx * half_w), int(y2 - ny * half_w))

        cv2.line(layer, edge1_start, edge1_end,
                 (*self.config.outline_color, 200), self.config.outline_thickness, cv2.LINE_AA)
        cv2.line(layer, edge2_start, edge2_end,
                 (*self.config.outline_color, 200), self.config.outline_thickness, cv2.LINE_AA)

    def _draw_shaded_sphere(self, layer, cx: int, cy: int, radius: int):
        """Draw a shaded sphere (joint) with lighting."""
        if radius < 2:
            return

        # Check cache
        cache_key = radius
        if cache_key not in self._sphere_cache:
            self._sphere_cache[cache_key] = self._create_sphere_template(radius)

        template = self._sphere_cache[cache_key]
        t_h, t_w = template.shape[:2]

        # Compute placement bounds
        y1 = cy - t_h // 2
        x1 = cx - t_w // 2
        y2 = y1 + t_h
        x2 = x1 + t_w

        h, w = layer.shape[:2]

        # Clip to layer bounds
        src_y1 = max(0, -y1)
        src_x1 = max(0, -x1)
        dst_y1 = max(0, y1)
        dst_x1 = max(0, x1)
        dst_y2 = min(h, y2)
        dst_x2 = min(w, x2)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        src_x2 = src_x1 + (dst_x2 - dst_x1)

        if dst_y2 <= dst_y1 or dst_x2 <= dst_x1:
            return

        # Alpha-blend the template onto the layer
        src = template[src_y1:src_y2, src_x1:src_x2]
        dst = layer[dst_y1:dst_y2, dst_x1:dst_x2]

        src_alpha = src[:, :, 3:4].astype(np.float32) / 255.0
        dst_alpha = dst[:, :, 3:4].astype(np.float32) / 255.0

        out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
        safe = out_alpha > 0
        out_rgb = np.where(
            safe,
            (src[:, :, :3].astype(np.float32) * src_alpha
             + dst[:, :, :3].astype(np.float32) * dst_alpha * (1.0 - src_alpha))
            / np.maximum(out_alpha, 1e-6),
            0,
        )

        layer[dst_y1:dst_y2, dst_x1:dst_x2, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        layer[dst_y1:dst_y2, dst_x1:dst_x2, 3] = np.clip(out_alpha * 255, 0, 255).astype(np.uint8).squeeze(-1)

    def _create_sphere_template(self, radius: int) -> np.ndarray:
        """Create a pre-rendered shaded sphere template."""
        size = radius * 2 + 2
        template = np.zeros((size, size, 4), dtype=np.uint8)

        center = size / 2.0
        # Shift highlight toward light source
        highlight_cx = center + self._light_x * radius * 0.3
        highlight_cy = center + self._light_y * radius * 0.3

        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                dist = np.sqrt(dx * dx + dy * dy)

                if dist > radius:
                    continue

                # Surface normal at this point (pointing outward on sphere)
                nz = np.sqrt(max(0, 1.0 - (dist / radius) ** 2))
                norm_x = dx / radius
                norm_y = dy / radius

                # Diffuse lighting
                dot = norm_x * self._light_x + norm_y * self._light_y + nz * 0.3
                shade = 0.4 + 0.6 * max(0, dot)

                # Specular highlight
                dist_to_hl = np.sqrt((x - highlight_cx) ** 2 + (y - highlight_cy) ** 2)
                specular = max(0, 1.0 - dist_to_hl / (radius * 0.6)) ** 3 * 0.4

                shade = min(1.3, shade + specular)

                color = self._shade_color(shade)

                # Anti-aliased edge
                edge_alpha = min(1.0, (radius - dist) * 2)
                template[y, x] = (*color, int(edge_alpha * 255))

        # Draw outline circle
        cv2.circle(template, (int(center), int(center)), radius,
                   (*self.config.outline_color, 220), 1, cv2.LINE_AA)

        return template

    def _shade_color(self, factor: float) -> Tuple[int, int, int]:
        """Interpolate between shadow, base, and highlight colors."""
        if factor <= 0.5:
            # Shadow to base
            t = factor / 0.5
            b = int(self.config.shadow_color[0] * (1 - t) + self.config.skin_color[0] * t)
            g = int(self.config.shadow_color[1] * (1 - t) + self.config.skin_color[1] * t)
            r = int(self.config.shadow_color[2] * (1 - t) + self.config.skin_color[2] * t)
        else:
            # Base to highlight
            t = (factor - 0.5) / 0.5
            t = min(t, 1.0)
            b = int(self.config.skin_color[0] * (1 - t) + self.config.highlight_color[0] * t)
            g = int(self.config.skin_color[1] * (1 - t) + self.config.highlight_color[1] * t)
            r = int(self.config.skin_color[2] * (1 - t) + self.config.highlight_color[2] * t)

        return (max(0, min(255, b)), max(0, min(255, g)), max(0, min(255, r)))

    def _draw_drop_shadow(self, layer, draw_calls, px, py, vis, sw):
        """Draw an offset blurred shadow of the entire figure."""
        h, w = layer.shape[:2]
        shadow_layer = np.zeros((h, w, 4), dtype=np.uint8)

        # Draw the figure onto shadow layer
        for call in draw_calls:
            call(shadow_layer, px, py, vis, sw)

        # Extract alpha as shadow mask
        shadow_mask = shadow_layer[:, :, 3].copy()

        # Offset
        ox, oy = self.config.shadow_offset
        M = np.float32([[1, 0, ox], [0, 1, oy]])
        shadow_mask = cv2.warpAffine(shadow_mask, M, (w, h))

        # Blur
        k = self.config.shadow_blur * 2 + 1
        shadow_mask = cv2.GaussianBlur(shadow_mask, (k, k), 0)

        # Apply shadow to main layer
        alpha = shadow_mask.astype(np.float32) / 255.0 * self.config.shadow_alpha
        shadow_bgr = self.config.outline_color
        for c in range(3):
            layer[:, :, c] = np.clip(
                layer[:, :, c].astype(np.float32) + shadow_bgr[c] * alpha, 0, 255
            ).astype(np.uint8)
        layer[:, :, 3] = np.clip(
            layer[:, :, 3].astype(np.float32) + alpha * 180, 0, 255
        ).astype(np.uint8)
