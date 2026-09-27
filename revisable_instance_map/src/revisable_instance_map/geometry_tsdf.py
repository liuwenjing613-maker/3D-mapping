"""Sparse shared TSDF geometry, independent of instance masks and identities."""

import time

import numpy as np
import open3d as o3d

from .frame_io import Frame


class SparseTSDFGeometry:
    def __init__(
        self,
        voxel_size_m: float = 0.01,
        block_resolution: int = 8,
        block_count: int = 100000,
        depth_max_m: float = 10.0,
        trunc_voxel_multiplier: float = 8.0,
    ):
        self.voxel_size_m = voxel_size_m
        self.depth_max_m = depth_max_m
        self.trunc_voxel_multiplier = trunc_voxel_multiplier
        self.integrated_frames = set()
        self.grid = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=voxel_size_m,
            block_resolution=block_resolution,
            block_count=block_count,
            device=o3d.core.Device("CPU:0"),
        )

    @staticmethod
    def _camera_tensors(frame: Frame):
        intrinsic = o3d.core.Tensor(frame.camera.intrinsic, dtype=o3d.core.float64)
        world_to_camera = o3d.core.Tensor(
            np.linalg.inv(frame.camera_to_world), dtype=o3d.core.float64
        )
        return intrinsic, world_to_camera

    @staticmethod
    def _images(frame: Frame):
        depth = o3d.t.geometry.Image(
            o3d.core.Tensor(np.ascontiguousarray(frame.depth_m))
        )
        color = o3d.t.geometry.Image(
            o3d.core.Tensor(np.ascontiguousarray(frame.rgb.astype(np.float32) / 255.0))
        )
        return depth, color

    def integrate(self, frame: Frame) -> dict:
        if frame.frame_id in self.integrated_frames:
            raise ValueError(f"Frame {frame.frame_id} already integrated")
        depth, color = self._images(frame)
        intrinsic, extrinsic = self._camera_tensors(frame)
        start = time.perf_counter()
        blocks = self.grid.compute_unique_block_coordinates(
            depth,
            intrinsic,
            extrinsic,
            depth_scale=1.0,
            depth_max=self.depth_max_m,
            trunc_voxel_multiplier=self.trunc_voxel_multiplier,
        )
        self.grid.integrate(
            blocks,
            depth,
            color,
            intrinsic,
            intrinsic,
            extrinsic,
            depth_scale=1.0,
            depth_max=self.depth_max_m,
            trunc_voxel_multiplier=self.trunc_voxel_multiplier,
        )
        self.integrated_frames.add(frame.frame_id)
        return {
            "frame_id": frame.frame_id,
            "frustum_blocks": int(blocks.shape[0]),
            "active_blocks": int(self.grid.hashmap().size()),
            "seconds": time.perf_counter() - start,
        }

    def extract_point_cloud(self, weight_threshold: float = 1.0):
        return self.grid.extract_point_cloud(weight_threshold=weight_threshold)

    def raycast_depth(self, frame: Frame, weight_threshold: float = 1.0):
        depth, _ = self._images(frame)
        intrinsic, extrinsic = self._camera_tensors(frame)
        blocks = self.grid.compute_unique_block_coordinates(
            depth,
            intrinsic,
            extrinsic,
            depth_scale=1.0,
            depth_max=self.depth_max_m,
            trunc_voxel_multiplier=self.trunc_voxel_multiplier,
        )
        rendered = self.grid.ray_cast(
            blocks,
            intrinsic,
            extrinsic,
            frame.camera.width,
            frame.camera.height,
            render_attributes=["depth"],
            depth_scale=1.0,
            depth_max=self.depth_max_m,
            weight_threshold=weight_threshold,
            trunc_voxel_multiplier=self.trunc_voxel_multiplier,
        )
        return rendered["depth"].numpy()

    def save(self, path):
        self.grid.save(str(path))

