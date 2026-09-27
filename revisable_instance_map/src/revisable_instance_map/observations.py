"""Immutable per-frame mask proposals with exact source-pixel provenance.

Each observation is a proposal, not a persistent object or an indivisible unit.
The read-only source mask remains the authority for its complete pixel support.
"""

from dataclasses import dataclass
import numpy as np

from .frame_io import Frame


@dataclass(frozen=True)
class RawInstanceObservation:
    observation_id: str
    frame_id: int
    mask_local_id: int
    source_mask_sha256: str
    pixel_count: int
    projectable_pixel_count: int
    bbox_xyxy_exclusive: tuple[int, int, int, int]
    world_centroid_m: tuple[float, float, float] | None
    world_aabb_min_m: tuple[float, float, float] | None
    world_aabb_max_m: tuple[float, float, float] | None


def source_pixel_indices(frame: Frame, observation: RawInstanceObservation) -> np.ndarray:
    """Recover every source pixel as a flat row-major index, including invalid depth."""
    if frame.frame_id != observation.frame_id:
        raise ValueError("Observation and frame IDs do not match")
    if observation.mask_local_id <= 0:
        raise ValueError("Observation mask ID must be positive")
    indices = np.flatnonzero(frame.mask_local.ravel() == observation.mask_local_id)
    if len(indices) != observation.pixel_count:
        raise ValueError("Source mask support differs from the recorded observation")
    return indices.astype(np.int32, copy=False)


def extract_frame_observations(
    frame: Frame, dataset: str, scene: str, source_mask_sha256: str,
    depth_max_m: float = 10.0, observation_namespace: str = "",
) -> tuple[RawInstanceObservation, ...]:
    """Describe all nonzero raw mask IDs without assigning cross-frame identities."""
    if not dataset or not scene:
        raise ValueError("Dataset and scene are required for stable observation IDs")
    if len(source_mask_sha256) != 64:
        raise ValueError("A SHA-256 source mask digest is required")
    if depth_max_m <= 0:
        raise ValueError("depth_max_m must be positive")
    if observation_namespace and ("/" in observation_namespace or observation_namespace in (".", "..")):
        raise ValueError("Invalid observation namespace")
    observation_prefix = f"{dataset}/{scene}"
    if observation_namespace:
        observation_prefix += f"/{observation_namespace}"
    height, width = frame.mask_local.shape
    if frame.depth_m.shape != (height, width):
        raise ValueError("Mask and depth dimensions differ")
    mask_flat = frame.mask_local.ravel()
    depth_flat = frame.depth_m.ravel()
    rotation = frame.camera_to_world[:3, :3]
    translation = frame.camera_to_world[:3, 3]
    observations = []
    for local_id in np.unique(mask_flat):
        local_id = int(local_id)
        if local_id == 0:
            continue
        pixels = np.flatnonzero(mask_flat == local_id)
        rows, cols = np.divmod(pixels, width)
        bbox = (
            int(cols.min()), int(rows.min()),
            int(cols.max()) + 1, int(rows.max()) + 1,
        )
        depth = depth_flat[pixels]
        valid = np.isfinite(depth) & (depth > 0) & (depth < depth_max_m)
        valid_depth = depth[valid].astype(np.float64)
        if len(valid_depth):
            valid_rows = rows[valid]
            valid_cols = cols[valid]
            camera_xyz = np.column_stack((
                (valid_cols - frame.camera.cx) * valid_depth / frame.camera.fx,
                (valid_rows - frame.camera.cy) * valid_depth / frame.camera.fy,
                valid_depth,
            ))
            world_xyz = camera_xyz @ rotation.T + translation
            centroid = tuple(float(value) for value in world_xyz.mean(axis=0))
            bounds_min = tuple(float(value) for value in world_xyz.min(axis=0))
            bounds_max = tuple(float(value) for value in world_xyz.max(axis=0))
        else:
            centroid = bounds_min = bounds_max = None
        observations.append(RawInstanceObservation(
            observation_id=f"{observation_prefix}/f{frame.frame_id:06d}/m{local_id:03d}",
            frame_id=frame.frame_id,
            mask_local_id=local_id,
            source_mask_sha256=source_mask_sha256,
            pixel_count=int(len(pixels)),
            projectable_pixel_count=int(len(valid_depth)),
            bbox_xyxy_exclusive=bbox,
            world_centroid_m=centroid,
            world_aabb_min_m=bounds_min,
            world_aabb_max_m=bounds_max,
        ))
    return tuple(observations)

