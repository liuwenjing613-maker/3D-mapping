"""Read the fixed Replica protocol and lift valid depth pixels into 3D.

This module does not associate instances or fuse frames. Mask IDs remain local
to their source frame, and every projected point retains its source pixel.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_png_units_per_meter: float

    @property
    def intrinsic(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Frame:
    frame_id: int
    rgb: np.ndarray  # uint8, H x W x 3
    depth_m: np.ndarray  # float32, H x W, zero indicates invalid depth
    mask_local: np.ndarray  # uint8 or uint16, H x W; IDs have frame-local meaning
    camera_to_world: np.ndarray  # float64, 4 x 4
    camera: Camera


@dataclass(frozen=True)
class ProjectedFrame:
    frame_id: int
    pixel_uv: np.ndarray  # int32, N x 2, (u, v)
    camera_xyz_m: np.ndarray  # float32, N x 3
    world_xyz_m: np.ndarray  # float32, N x 3
    rgb: np.ndarray  # uint8, N x 3
    mask_local: np.ndarray  # uint8 or uint16, N


class ReplicaFrameSource:
    def __init__(self, config_path: str | Path):
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if config["dataset"] != "Replica":
            raise ValueError("This source only supports the fixed Replica contract")
        self.config = config
        self.scene_root = Path(config["source"]["scene_root"])
        self.mask_root = Path(config["source"]["mask_root"])
        selection = config["frame_selection"]
        self.frame_ids = tuple(range(selection["start"], selection["stop_exclusive"], selection["stride"]))
        self._frame_set = frozenset(self.frame_ids)
        camera = config["camera"]
        self.camera = Camera(
            width=camera["width"],
            height=camera["height"],
            fx=camera["fx"],
            fy=camera["fy"],
            cx=camera["cx"],
            cy=camera["cy"],
            depth_png_units_per_meter=camera["depth_png_units_per_meter"],
        )
        self._poses = np.loadtxt(self.scene_root / config["source"]["trajectory"], dtype=np.float64).reshape(-1, 4, 4)
        if len(self._poses) < selection["stop_exclusive"]:
            raise ValueError("Trajectory is shorter than the frame selection")
        if not np.allclose(self._poses[self.frame_ids, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError("Trajectory contains a non-homogeneous camera pose")

    def load_frame(self, frame_id: int) -> Frame:
        if frame_id not in self._frame_set:
            raise ValueError(f"Frame {frame_id} is outside the frozen input protocol")
        source = self.config["source"]
        rgb_path = self.scene_root / source["rgb_pattern"].format(frame=frame_id)
        depth_path = self.scene_root / source["depth_pattern"].format(frame=frame_id)
        mask_path = self.mask_root / source["mask_pattern"].format(frame=frame_id)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(depth_path) as image:
            raw_depth = np.asarray(image)
        with Image.open(mask_path) as image:
            mask = np.asarray(image)

        shape = (self.camera.height, self.camera.width)
        if rgb.shape != (*shape, 3) or raw_depth.shape != shape or mask.shape != shape:
            raise ValueError(f"Frame {frame_id} image shapes do not match {shape}")
        if raw_depth.dtype != np.uint16:
            raise ValueError(f"Frame {frame_id} depth is {raw_depth.dtype}, expected uint16")
        if mask.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"Frame {frame_id} mask is {mask.dtype}, expected uint8 or uint16")
        depth_m = raw_depth.astype(np.float32) / self.camera.depth_png_units_per_meter
        pose = self._poses[frame_id].copy()
        for array in (rgb, depth_m, mask, pose):
            array.setflags(write=False)
        return Frame(frame_id, rgb, depth_m, mask, pose, self.camera)

    def iter_frames(self) -> Iterator[Frame]:
        for frame_id in self.frame_ids:
            yield self.load_frame(frame_id)


def unproject_frame(frame: Frame, pixel_stride: int = 1) -> ProjectedFrame:
    """Project all sampled valid depth pixels; instance mask does not gate geometry."""
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be >= 1")
    camera = frame.camera
    ys = np.arange(0, camera.height, pixel_stride, dtype=np.int32)
    xs = np.arange(0, camera.width, pixel_stride, dtype=np.int32)
    u, v = np.meshgrid(xs, ys)
    depth = frame.depth_m[::pixel_stride, ::pixel_stride]
    valid = np.isfinite(depth) & (depth > 0)
    u = u[valid]
    v = v[valid]
    z = depth[valid].astype(np.float64)
    x = (u - camera.cx) * z / camera.fx
    y = (v - camera.cy) * z / camera.fy
    camera_xyz = np.column_stack((x, y, z))
    pose = frame.camera_to_world
    world_xyz = camera_xyz @ pose[:3, :3].T + pose[:3, 3]
    return ProjectedFrame(
        frame_id=frame.frame_id,
        pixel_uv=np.column_stack((u, v)).astype(np.int32),
        camera_xyz_m=camera_xyz.astype(np.float32),
        world_xyz_m=world_xyz.astype(np.float32),
        rgb=frame.rgb[::pixel_stride, ::pixel_stride][valid].copy(),
        mask_local=frame.mask_local[::pixel_stride, ::pixel_stride][valid].copy(),
    )
