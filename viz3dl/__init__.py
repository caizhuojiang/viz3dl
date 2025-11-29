"""viz3dl: lightweight 3D layout visualization helpers."""

from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

Color = Tuple[int, int, int]


@dataclass
class SceneObject:
    """Minimal scene primitive described by a box footprint."""

    name: str
    location: Sequence[float]
    dimensions: Sequence[float]
    yaw: float = 0.0
    color: Optional[Color] = None

    def __post_init__(self) -> None:
        self.location = list(map(float, self.location))
        self.dimensions = list(map(float, self.dimensions))
        self.yaw = float(self.yaw)

    @classmethod
    def from_rotation(
        cls,
        name: str,
        location: Sequence[float],
        dimensions: Sequence[float],
        rotation: Sequence[float],
        yaw_index: int = -1,
        color: Optional[Color] = None,
    ) -> "SceneObject":
        yaw = float(rotation[yaw_index]) if rotation else 0.0
        return cls(name=name, location=location, dimensions=dimensions, yaw=yaw, color=color)

    def bounding_box(self) -> np.ndarray:
        """Return the eight vertices of the oriented box in world space."""
        w, h, d = map(abs, self.dimensions)
        cx, cy, cz = self.location
        vertices = np.array(
            [
                [-w, -h, -d],
                [w, -h, -d],
                [w, -h, d],
                [-w, -h, d],
                [-w, h, -d],
                [w, h, -d],
                [w, h, d],
                [-w, h, d],
            ],
            dtype=np.float32,
        )
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        rotation = np.array(
            [
                [cos_yaw, 0.0, sin_yaw],
                [0.0, 1.0, 0.0],
                [-sin_yaw, 0.0, cos_yaw],
            ],
            dtype=np.float32,
        )
        vertices = (rotation @ vertices.T).T
        vertices[:, 0] += cx
        vertices[:, 1] += cy
        vertices[:, 2] += cz
        return vertices


class SceneLayout:
    """Collection of objects plus helper transforms."""

    def __init__(self, objects: Iterable[SceneObject]):
        self.objects: List[SceneObject] = list(objects)
        self._calculate_bounds()

    def _calculate_bounds(self) -> None:
        if not self.objects:
            self.min_bounds = np.zeros(3, dtype=np.float32)
            self.max_bounds = np.zeros(3, dtype=np.float32)
            self.center = np.zeros(3, dtype=np.float32)
            return
        all_vertices = np.vstack([obj.bounding_box() for obj in self.objects])
        self.min_bounds = np.min(all_vertices, axis=0)
        self.max_bounds = np.max(all_vertices, axis=0)
        self.center = (self.min_bounds + self.max_bounds) / 2.0

    def get_scene_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.min_bounds.copy(), self.max_bounds.copy()

    def get_scene_center(self) -> np.ndarray:
        return self.center.copy()

    def get_scene_size(self) -> np.ndarray:
        return self.max_bounds - self.min_bounds

    def normalize_to_origin(self, align_floor: bool = False) -> None:
        if not self.objects:
            return
        offset = -self.center
        if align_floor:
            floor = min(np.min(obj.bounding_box()[:, 1]) for obj in self.objects)
            offset[1] = -floor
        for obj in self.objects:
            obj.location = [obj.location[i] + offset[i] for i in range(3)]
        self._calculate_bounds()

    def scale_to_unit_cube(self, target_extent: float = 5.0) -> None:
        if not self.objects:
            return
        self.normalize_to_origin()
        size = self.get_scene_size()
        max_dim = float(np.max(size))
        if max_dim <= 1e-6:
            return
        scale = target_extent / max_dim
        for obj in self.objects:
            obj.location = [coord * scale for coord in obj.location]
            obj.dimensions = [dim * scale for dim in obj.dimensions]
        self._calculate_bounds()

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)


@dataclass
class CameraConfig:
    width: int = 1100
    height: int = 700
    focal_length: int = 1300
    background: Color = (255, 255, 255)


class _Camera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.matrix = np.array(
            [
                [config.focal_length, 0, config.width // 2],
                [0, config.focal_length, config.height // 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    def project(self, points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 3)
        projected, _ = cv2.projectPoints(points, rvec, tvec, self.matrix, self.dist_coeffs)
        return projected.reshape(-1, 2)


class CategoryColorizer:
    """Lazy color palette seeded for reproducibility."""

    def __init__(self, preset: Optional[Mapping[str, Sequence[int]]] = None, seed: int = 13):
        self.mapping: Dict[str, Color] = {
            k: tuple(int(v) for v in value)  # type: ignore[arg-type]
            for k, value in (preset or {}).items()
        }
        self._rng = np.random.default_rng(seed)

    def __call__(self, label: str, fallback: Optional[Sequence[int]] = None) -> Color:
        if fallback is not None:
            return tuple(int(c) for c in fallback)
        if label not in self.mapping:
            self.mapping[label] = tuple(int(v) for v in self._rng.integers(64, 240, size=3))
        return self.mapping[label]


class TextRasterizer:
    """Caches vectorized glyph silhouettes for OpenCV text rendering."""

    def __init__(self) -> None:
        self.cache: Dict[Tuple[str, int, float, int], np.ndarray] = {}

    def glyph_points(
        self,
        text: str,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        scale: float = 1.0,
        thickness: int = 1,
        stride: int = 2,
    ) -> np.ndarray:
        key = (text, font, scale, thickness)
        if key not in self.cache:
            (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
            canvas = np.zeros((h + baseline + 2, w + 2), dtype=np.uint8)
            cv2.putText(canvas, text, (0, h), font, scale, 255, thickness, cv2.LINE_AA)
            points = np.column_stack(np.where(canvas > 0)).astype(np.float32)
            if len(points) == 0:
                points = np.zeros((0, 2), dtype=np.float32)
            points[:, 0], points[:, 1] = points[:, 1], canvas.shape[0] - points[:, 0]
            self.cache[key] = points
        points = self.cache[key]
        return points[:: max(1, stride)]


class LayoutVisualizer:
    """High-level helper that renders layouts into NumPy frames or GIFs."""

    def __init__(
        self,
        layout: SceneLayout,
        camera: Optional[CameraConfig] = None,
        colorizer: Optional[Callable[[SceneObject], Color]] = None,
        text_stride: int = 2,
    ) -> None:
        self.layout = layout
        self.camera = _Camera(camera or CameraConfig())
        self.text_stride = max(1, text_stride)
        if colorizer is None:
            palette = CategoryColorizer()

            def default_color(obj: SceneObject) -> Color:
                return palette(obj.name, obj.color)

            self.colorizer = default_color
        else:
            self.colorizer = colorizer
        self._text = TextRasterizer()

    def _camera_pose(
        self,
        radius: float,
        elevation: float,
        azimuth: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = self.layout.get_scene_center()
        elev = np.deg2rad(elevation)
        azim = np.deg2rad(azimuth)
        position = center + np.array(
            [
                -radius * np.cos(elev) * np.sin(azim),
                -radius * np.sin(elev),
                -radius * np.cos(elev) * np.cos(azim),
            ]
        )
        forward = (center - position)
        forward /= np.linalg.norm(forward) + 1e-8
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-8
        rotation = np.stack([right, up, -forward]).astype(np.float32)
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = -(rotation @ position.reshape(-1, 1)).reshape(-1)
        return rvec, tvec, rotation

    def _draw_box(
        self,
        frame: np.ndarray,
        vertices: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        color: Color,
    ) -> None:
        points_2d = self.camera.project(vertices, rvec, tvec).astype(np.int32)
        for i in range(4):
            self._draw_line(frame, points_2d[i], points_2d[(i + 1) % 4], color)
            self._draw_line(frame, points_2d[i + 4], points_2d[(i + 1) % 4 + 4], color)
            self._draw_line(frame, points_2d[i], points_2d[i + 4], color)
        top_front = ((points_2d[6] + points_2d[7]) / 2).astype(np.int32)
        top_back = ((points_2d[4] + points_2d[5]) / 2).astype(np.int32)
        tip = (top_front + 0.2 * (top_back - top_front)).astype(np.int32)
        self._draw_line(frame, top_front, tip, color, thickness=2)

    def _draw_line(self, frame: np.ndarray, pt1: np.ndarray, pt2: np.ndarray, color: Color, thickness: int = 1) -> None:
        h, w = self.camera.config.height, self.camera.config.width
        if (0 <= pt1[0] <= w and 0 <= pt1[1] <= h) or (0 <= pt2[0] <= w and 0 <= pt2[1] <= h):
            cv2.line(frame, tuple(pt1), tuple(pt2), color, thickness, cv2.LINE_AA)

    def _draw_label(
        self,
        frame: np.ndarray,
        obj: SceneObject,
        vertices: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        color: Color,
    ) -> None:
        if max(obj.dimensions) <= 0.2:
            return
        glyph = self._text.glyph_points(obj.name, stride=self.text_stride)
        if len(glyph) == 0:
            return
        centroid = np.mean(vertices, axis=0)
        right = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        glyph_centered = glyph - np.mean(glyph, axis=0)
        points_3d = centroid + (glyph_centered[:, 0:1] * right + glyph_centered[:, 1:2] * up) * 0.005
        projected = self.camera.project(points_3d, rvec, tvec).astype(np.int32)
        for pt in projected:
            cv2.circle(frame, tuple(pt), 1, color, -1)

    def render_view(
        self,
        radius: float = 10.0,
        elevation: float = -10.0,
        azimuth: float = 0.0,
        draw_labels: bool = True,
    ) -> np.ndarray:
        frame = np.full(
            (self.camera.config.height, self.camera.config.width, 3),
            self.camera.config.background,
            dtype=np.uint8,
        )
        rvec, tvec, rotation = self._camera_pose(radius, elevation, azimuth)
        camera_offset = tvec.reshape(3, 1)
        queue: List[Tuple[float, SceneObject, np.ndarray]] = []
        for obj in self.layout:
            vertices = obj.bounding_box()
            depth_vertices = (rotation @ vertices.T + camera_offset).T
            depth = float(np.mean(depth_vertices[:, 2]))
            queue.append((depth, obj, vertices))
        queue.sort(key=lambda item: item[0], reverse=True)
        for _, obj, vertices in queue:
            color = self.colorizer(obj)
            self._draw_box(frame, vertices, rvec, tvec, color)
            if draw_labels:
                self._draw_label(frame, obj, vertices, rvec, tvec, color)
        return frame

    def render_rotation(
        self,
        output_path: str,
        radius: float = 10.0,
        elevation: float = -10.0,
        start_azimuth: float = -40.0,
        end_azimuth: float = 40.0,
        step: float = 4.0,
        frame_duration_ms: int = 100,
    ) -> None:
        frames: List[Image.Image] = []
        iterator = np.arange(start_azimuth, end_azimuth, step)
        iterator_back = np.arange(end_azimuth, start_azimuth, -step)
        for azim in iterator:
            frame = self.render_view(radius=radius, elevation=elevation, azimuth=float(azim))
            frames.append(Image.fromarray(frame))
        for azim in iterator_back:
            frame = self.render_view(radius=radius, elevation=elevation, azimuth=float(azim))
            frames.append(Image.fromarray(frame))
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        frames[0].save(
            output_path,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=frame_duration_ms,
            loop=0,
        )


def load_layouts(
    records: Iterable[Mapping[str, object]],
    builder: Callable[[Mapping[str, object]], Tuple[str, Iterable[SceneObject]]],
) -> Dict[str, SceneLayout]:
    """Convert iterable records into named layouts using a builder callback."""
    layouts: Dict[str, SceneLayout] = {}
    for record in records:
        scene_name, objects = builder(record)
        layouts[scene_name] = SceneLayout(objects)
    return layouts


__all__ = [
    "SceneObject",
    "SceneLayout",
    "CameraConfig",
    "LayoutVisualizer",
    "CategoryColorizer",
    "TextRasterizer",
    "load_layouts",
]
