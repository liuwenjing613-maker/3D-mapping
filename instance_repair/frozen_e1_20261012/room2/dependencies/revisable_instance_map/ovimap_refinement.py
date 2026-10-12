"""Faithful Python port of OVI-MAP's frameToSegmentsCropFormer mask fusion.

Source: official OVI-MAP scripts/utils/common_scannet_nyu.py at
f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424. Only the 2D mask
partition is reproduced here; 3D association remains our own baseline.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FusedRegion:
    local_id: int
    geometry_id: int
    cropformer_id: int
    rule: str
    pixel_count: int
    overlap_ratio: float
    original_geometry_pixels: int
    residual_geometry_pixels: int


def fuse_ovimap_masks(cropformer: np.ndarray, geometry: np.ndarray):
    """Return positive instance labels, all regions, and partition statistics.

    Implements upstream thresholds and iteration order exactly: geometric area
    >=100; split if overlap >0.9 of the CropFormer instance and <0.5 of the
    current geometric region; otherwise label residual if overlap >=0.2 of it.
    A geometric region without a winning instance is background (label zero).
    """
    if cropformer.shape != geometry.shape or cropformer.ndim != 2:
        raise ValueError("CropFormer and depth mask shapes differ")
    if cropformer.dtype.kind not in "ui" or geometry.dtype.kind not in "ui":
        raise TypeError("Input masks must contain integer labels")
    raw_counts = np.bincount(cropformer.ravel().astype(np.int64))
    output = np.zeros(geometry.shape, dtype=np.uint16)
    regions = []
    background_regions = []
    small_geometry_pixels = 0
    assigned_pixels = 0
    geometry_pixels_considered = 0
    kept_geometry_regions = 0
    empty_majority_regions = 0

    for geometry_id in np.unique(geometry):
        geometry_id = int(geometry_id)
        if geometry_id == 0:
            continue
        original_mask = geometry == geometry_id
        original_area = int(np.count_nonzero(original_mask))
        if original_area < 100:
            small_geometry_pixels += original_area
            continue
        geometry_pixels_considered += original_area
        kept_geometry_regions += 1
        remaining = original_mask.copy()
        remaining_area = original_area
        candidates = Counter(cropformer[original_mask].reshape(-1))
        max_overlap_area = 0
        max_candidate_id = 0
        split_regions = []

        for candidate_id, overlap_area in candidates.items():
            candidate_id = int(candidate_id)
            if candidate_id == 0:
                continue
            candidate_area = int(raw_counts[candidate_id])
            if overlap_area > 0.9 * candidate_area and overlap_area < 0.5 * remaining_area:
                split_mask = remaining & (cropformer == candidate_id)
                split_count = int(np.count_nonzero(split_mask))
                if split_count != overlap_area:
                    raise AssertionError("OVI-MAP split candidate support changed")
                split_regions.append((candidate_id, split_mask, float(overlap_area / candidate_area)))
                remaining[split_mask] = False
                remaining_area -= overlap_area
            elif max_overlap_area < overlap_area:
                max_overlap_area = overlap_area
                max_candidate_id = candidate_id

        for candidate_id, mask, ratio in split_regions:
            local_id = len(regions) + 1
            if local_id > 65535:
                raise OverflowError("Too many fused regions for uint16 PNG")
            count = int(np.count_nonzero(mask))
            if np.any(output[mask]):
                raise AssertionError("Fused regions overlap")
            output[mask] = local_id
            assigned_pixels += count
            regions.append(FusedRegion(local_id, geometry_id, candidate_id, "split", count,
                                       ratio, original_area, remaining_area))

        # Upstream evaluates this condition even if the residual becomes empty.
        # Empty regions have no image support, so record them without assigning ID.
        if max_overlap_area >= 0.2 * remaining_area:
            if not remaining_area:
                empty_majority_regions += 1
            if remaining_area:
                local_id = len(regions) + 1
                if local_id > 65535:
                    raise OverflowError("Too many fused regions for uint16 PNG")
                if np.any(output[remaining]):
                    raise AssertionError("Fused regions overlap")
                output[remaining] = local_id
                assigned_pixels += remaining_area
                regions.append(FusedRegion(local_id, geometry_id, max_candidate_id,
                                           "majority", remaining_area,
                                           float(max_overlap_area / remaining_area),
                                           original_area, remaining_area))
        else:
            background_regions.append({"geometry_id": geometry_id,
                                       "pixel_count": remaining_area,
                                       "winning_overlap": max_overlap_area})

    if assigned_pixels != int(np.count_nonzero(output)):
        raise AssertionError("Pixel accounting differs from output label mask")
    if len(np.unique(output)) - 1 != len(regions):
        raise AssertionError("Fused labels are not dense and unique")
    background_pixels = int(sum(x["pixel_count"] for x in background_regions))
    if assigned_pixels + background_pixels != geometry_pixels_considered:
        raise AssertionError("Geometric segment partition does not close")
    stats = {
        "geometry_segments_kept": kept_geometry_regions,
        "small_geometry_pixels": small_geometry_pixels,
        "geometry_pixels_considered": geometry_pixels_considered,
        "refined_instance_pixels": assigned_pixels,
        "geometric_background_pixels": background_pixels,
        "split_regions": sum(x.rule == "split" for x in regions),
        "majority_regions": sum(x.rule == "majority" for x in regions),
        "unassigned_region_count": len(background_regions),
        "empty_majority_regions": empty_majority_regions,
    }
    return output, tuple(regions), tuple(background_regions), stats
