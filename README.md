# viz3dl

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Renderer](https://img.shields.io/badge/Renderer-3D%20Layout-orange)
![Status](https://img.shields.io/badge/Status-Toolkit-success)

<p align="center">
    <img src="assets/1cc0fd55-1c5a-4885-a2de-f30c6d387353_DiningRoom-33249.gif" alt="Dining room preview" width="32%" />
    <img src="assets/6d3ebe90-a07a-4c02-93c6-a997c9890cc3_SecondBedroom-72827.gif" alt="Bedroom preview" width="32%" />
    <img src="assets/95f3ed13-768e-4d9e-9c55-47c0bbc213fd_LivingDiningRoom-2943.gif" alt="Living dining preview" width="32%" />
</p>

viz3dl is a tiny toolkit for rendering 3D layout into clean 2D previews or GIF turntables.

## Installation

```bash
pip install git+https://github.com/caizhuojiang/viz3dl.git
```

## Quickstart

```python
from viz3dl import SceneObject, SceneLayout, LayoutVisualizer

# Describe your scene using half extents (width/height/depth) and yaw in radians
table = SceneObject(
    name="table",
    location=(0.0, 0.4, 0.0),
    dimensions=(0.6, 0.4, 0.4),
    yaw=1.57,
)
chair = SceneObject(
    name="chair",
    location=(0.8, 0.35, 0.0),
    dimensions=(0.35, 0.35, 0.35),
    yaw=-1.57,
)

layout = SceneLayout([table, chair])
layout.normalize_to_origin(align_floor=True)
layout.scale_to_unit_cube()

viz = LayoutVisualizer(layout)
frame = viz.render_view(radius=10, elevation=-20, azimuth=0)

from PIL import Image
Image.fromarray(frame).save("scene.png")
```

Generate a simple turntable GIF:

```python
viz.render_rotation(
    output_path="scene.gif",
    radius=10.0,
    elevation=-15,
    start_azimuth=-45,
    end_azimuth=45,
    step=3,
    frame_duration_ms=80,
)
```

## Conventions

- Coordinates follow an **X-right, Y-up, Z-forward** convention.
- `dimensions` are half extents in meters (i.e., width/height/depth from center to box face). Divide by two if you store full sizes.
- `yaw` values are in **radians** and rotate around the Y axis.
- Label rendering uses OpenCV fonts; for non-Latin text, adjust `TextRasterizer` to use your glyph rasterizer of choice.

## License

This project is distributed under the MIT License.
