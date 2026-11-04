"""Geometry-bound visible support, independent of map instance labels.

Uses the existing P0.1 surface-to-pixel projection convention and defaults.
Unlike that mapper helper, keeps visible points labelled background as V_t.
"""
from dataclasses import asdict, dataclass
import numpy as np


REFERENCE_PROJECTOR_SHA256 = "affb77e3458572984bc621d79709ff5a4ffaa9804a9ebb1902e321fd0895de62"


@dataclass(frozen=True)
class ProjectionSettings:
    depth_tolerance_m: float = 0.015
    max_depth_m: float = 10.0
    point_chunk_size: int = 250_000

    def __post_init__(self):
        if (not np.isfinite(self.depth_tolerance_m) or self.depth_tolerance_m <= 0
                or not np.isfinite(self.max_depth_m) or self.max_depth_m <= 0
                or self.point_chunk_size < 1):
            raise ValueError("Invalid projection settings")

    def record(self):
        return dict(asdict(self), convention="surface_to_nearest_pixel_depth_checked",
                    rounding="numpy.rint", reference_projector_sha256=REFERENCE_PROJECTOR_SHA256)


def project_frame(frame, xyz, settings):
    """Return V_t and per-original-mask S_t in this exact surface index space."""
    if len(xyz) > np.iinfo(np.int32).max:
        raise ValueError("Surface exceeds the supported int32 index range")
    height, width = frame.depth_m.shape
    k = frame.intrinsic
    visible_parts, pixel_parts, label_parts = [], [], []
    stats = dict(surface_points_tested=len(xyz), in_front_points=0,
                 inside_image_points=0, valid_depth_points=0,
                 visible_depth_consistent_points=0, background_visible_points=0,
                 positive_mask_evidence_points=0)
    for start in range(0, len(xyz), settings.point_chunk_size):
        world = xyz[start:start + settings.point_chunk_size].astype(np.float64, copy=False)
        camera = (world - frame.pose[:3, 3]) @ frame.pose[:3, :3]
        z = camera[:, 2]
        front = np.isfinite(z) & (z > 0) & (z < settings.max_depth_m)
        stats["in_front_points"] += int(front.sum())
        rel = np.flatnonzero(front)
        if not len(rel):
            continue
        p = camera[front]
        uf = k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2]
        vf = k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]
        finite = (np.isfinite(uf) & np.isfinite(vf)
                  & (np.abs(uf) < np.iinfo(np.int32).max)
                  & (np.abs(vf) < np.iinfo(np.int32).max))
        rel, zf = rel[finite], p[finite, 2]
        u, v = np.rint(uf[finite]).astype(np.int32), np.rint(vf[finite]).astype(np.int32)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        stats["inside_image_points"] += int(inside.sum())
        rel, zf, u, v = rel[inside], zf[inside], u[inside], v[inside]
        observed = frame.depth_m[v, u]
        valid = np.isfinite(observed) & (observed > 0) & (observed < settings.max_depth_m)
        stats["valid_depth_points"] += int(valid.sum())
        consistent = valid & (np.abs(observed - zf) <= settings.depth_tolerance_m)
        rel, u, v = rel[consistent], u[consistent], v[consistent]
        if len(rel):
            labels = frame.mask_local[v, u]
            visible_parts.append((start + rel).astype(np.int32))
            pixel_parts.append((v * width + u).astype(np.int32))
            label_parts.append(labels.astype(np.int32))
    visible = np.concatenate(visible_parts) if visible_parts else np.empty(0, np.int32)
    pixels = np.concatenate(pixel_parts) if pixel_parts else np.empty(0, np.int32)
    labels = np.concatenate(label_parts) if label_parts else np.empty(0, np.int32)
    stats["visible_depth_consistent_points"] = len(visible)
    stats["background_visible_points"] = int((labels == 0).sum())
    stats["positive_mask_evidence_points"] = int((labels > 0).sum())
    ids = np.array([r["mask_local_id"] for r in frame.observations], np.int32)
    positive = labels > 0
    order = np.argsort(labels[positive], kind="stable")
    grouped_labels = labels[positive][order]
    support = visible[positive][order]
    offsets = np.r_[np.searchsorted(grouped_labels, ids), len(support)].astype(np.int64)
    return dict(visible_surface_indices=visible, visible_pixel_indices=pixels,
                visible_mask_local_ids=labels, mask_local_ids=ids,
                mask_offsets=offsets, mask_surface_indices=support), stats
