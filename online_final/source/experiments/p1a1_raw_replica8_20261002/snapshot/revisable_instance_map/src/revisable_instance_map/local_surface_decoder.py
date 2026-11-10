"""Conservative, single-pass P0 surface decoder. Raw evidence is never mutated."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

CONFIRMED = 2
TENTATIVE = 1
UNOBSERVED = 0

@dataclass(frozen=True)
class DecoderSettings:
    neighbors: int = 20
    radius_m: float = 0.025
    normal_cos: float = 0.92
    plane_distance_m: float = 0.004
    rgb_distance: float = 0.22
    max_own_votes: int = 8
    min_alt_neighbors: int = 9
    min_alt_fraction: float = 0.80
    max_own_neighbors: int = 2
    fill_radius_m: float = 0.018
    fill_normal_cos: float = 0.95
    fill_plane_distance_m: float = 0.003
    fill_rgb_distance: float = 0.18
    min_unknown_neighbors: int = 12
    min_tentative_neighbors: int = 8
    min_fill_fraction: float = 0.92

def query_surface_neighbors(tree, xyz, normals, rgb, target, k, radius, normal_cos,
                            plane_distance, rgb_distance):
    """Geometric and appearance edges; each target is evaluated against frozen neighbors."""
    if not len(target):
        return np.empty((0, k), np.int32), np.empty((0, k), bool)
    distance, index = tree.query(xyz[target], k=k + 1, workers=8)
    index = index[:, 1:].astype(np.int32)
    distance = distance[:, 1:]
    delta = xyz[index] - xyz[target, None]
    dot = np.einsum("ijk,ik->ij", normals[index], normals[target])
    plane0 = np.abs(np.einsum("ijk,ik->ij", delta, normals[target]))
    plane1 = np.abs(np.einsum("ijk,ijk->ij", delta, normals[index]))
    color = np.linalg.norm(rgb[index] - rgb[target, None], axis=2)
    valid = ((distance <= radius) & (np.abs(dot) >= normal_cos) &
             (plane0 <= plane_distance) & (plane1 <= plane_distance) &
             (color <= rgb_distance) & (index != target[:, None]))
    return index, valid

def pair_vote_lookup(pair_point, pair_instance, pair_votes, point, instance):
    if not len(point):
        return np.empty(0, np.int32)
    base = int(max(pair_instance.max(initial=0), instance.max(initial=0))) + 1
    keys = pair_point.astype(np.int64) * base + pair_instance
    if np.any(keys[1:] < keys[:-1]):
        order = np.argsort(keys)
        keys, pair_votes = keys[order], pair_votes[order]
    query = point.astype(np.int64) * base + instance
    at = np.searchsorted(keys, query)
    good = (at < len(keys)) & (keys[np.minimum(at, len(keys)-1)] == query)
    result = np.zeros(len(query), np.int32)
    result[good] = pair_votes[at[good]]
    return result

def dominant_alternative(target, neighbor_idx, neighbor_valid, labels, votes):
    choice = np.full(len(target), -1, np.int32)
    alt_count = np.zeros(len(target), np.int16)
    own_count = np.zeros(len(target), np.int16)
    fraction = np.zeros(len(target), np.float32)
    median_votes = np.zeros(len(target), np.float32)
    for row, point in enumerate(target):
        neighbors = neighbor_idx[row, neighbor_valid[row]]
        ids = labels[neighbors]
        keep = ids > 0
        ids, neighbors = ids[keep], neighbors[keep]
        if not len(ids):
            continue
        unique, counts = np.unique(ids, return_counts=True)
        own = labels[point]
        own_count[row] = counts[unique == own][0] if np.any(unique == own) else 0
        counts[unique == own] = 0
        best = int(np.argmax(counts))
        if counts[best] == 0:
            continue
        choice[row] = unique[best]
        alt_count[row] = counts[best]
        fraction[row] = counts[best] / len(ids)
        median_votes[row] = float(np.median(votes[neighbors[ids == unique[best]]]))
    return choice, alt_count, own_count, fraction, median_votes

def decode_confirmed(xyz, normals, rgb, evidence, pair, settings, tree=None):
    tree = tree or cKDTree(xyz)
    state = evidence["state"]
    own_votes = evidence["top1_votes"]
    labels = np.where(state == CONFIRMED, evidence["top1_instance_id"], -1).astype(np.int32)
    target = np.flatnonzero((state == CONFIRMED) & (own_votes <= settings.max_own_votes) &
                            (evidence["top2_votes"] > 0)).astype(np.int32)
    neighbor_idx, valid = query_surface_neighbors(
        tree, xyz, normals, rgb, target, settings.neighbors, settings.radius_m,
        settings.normal_cos, settings.plane_distance_m, settings.rgb_distance)
    # Only original confirmed, well-supported neighbors can drive a correction.
    valid &= (state[neighbor_idx] == CONFIRMED) & (own_votes[neighbor_idx] >= 3)
    alt, count, own_count, fraction, med_votes = dominant_alternative(
        target, neighbor_idx, valid, labels, own_votes)
    plausible = ((alt > 0) & (count >= settings.min_alt_neighbors) &
                 (own_count <= settings.max_own_neighbors) &
                 (fraction >= settings.min_alt_fraction) & (med_votes >= 4))
    alt_votes = pair_vote_lookup(pair["surface_point_index"], pair["instance_id"],
                                 pair["frame_votes"], target, np.maximum(alt, 0))
    selected = plausible & (alt_votes >= 1)
    changed = target[selected]
    support_neighbors = np.where(
        valid[selected] & (labels[neighbor_idx[selected]] == alt[selected, None]),
        neighbor_idx[selected], -1).astype(np.int32)
    updated = labels.copy()
    updated[changed] = alt[selected]
    diagnostics = {
        "candidate_count": int(len(target)),
        "neighbor_consensus_count": int(np.count_nonzero(plausible)),
        "direct_alternative_evidence_count": int(np.count_nonzero(selected)),
        "changed_points": int(len(changed)),
        "changed_indices": changed,
        "old_instance_id": labels[changed],
        "new_instance_id": alt[selected],
        "old_votes": own_votes[changed],
        "new_direct_votes": alt_votes[selected],
        "neighbor_fraction": fraction[selected],
        "support_neighbor_indices": support_neighbors,
    }
    return updated, diagnostics

def fill_small_holes(xyz, normals, rgb, evidence, decoded_labels, settings, tree=None):
    tree = tree or cKDTree(xyz)
    state = evidence["state"]
    target = np.flatnonzero((state == UNOBSERVED) | (state == TENTATIVE)).astype(np.int32)
    neighbor_idx, valid = query_surface_neighbors(
        tree, xyz, normals, rgb, target, settings.neighbors, settings.fill_radius_m,
        settings.fill_normal_cos, settings.fill_plane_distance_m,
        settings.fill_rgb_distance)
    valid &= (decoded_labels[neighbor_idx] > 0) & (evidence["top1_votes"][neighbor_idx] >= 3)
    labels = decoded_labels[neighbor_idx]
    out = decoded_labels.copy()
    filled_unknown = []
    promoted_tentative = []
    unknown_support = []
    tentative_support = []
    for row, point in enumerate(target):
        ids = labels[row, valid[row]]
        minimum = settings.min_unknown_neighbors if state[point] == UNOBSERVED else settings.min_tentative_neighbors
        if len(ids) < minimum:
            continue
        unique, counts = np.unique(ids, return_counts=True)
        best = int(np.argmax(counts))
        if counts[best] / len(ids) < settings.min_fill_fraction:
            continue
        candidate = int(unique[best])
        if state[point] == TENTATIVE and candidate != evidence["top1_instance_id"][point]:
            continue
        out[point] = candidate
        support = np.where(valid[row] & (labels[row] == candidate),
                           neighbor_idx[row], -1).astype(np.int32)
        if state[point] == UNOBSERVED:
            filled_unknown.append(point)
            unknown_support.append(support)
        else:
            promoted_tentative.append(point)
            tentative_support.append(support)
    return out, {
        "candidate_count": int(len(target)),
        "filled_unknown": np.asarray(filled_unknown, np.int32),
        "promoted_tentative": np.asarray(promoted_tentative, np.int32),
        "support_neighbor_indices": np.asarray(
            unknown_support + tentative_support, dtype=np.int32).reshape(-1, settings.neighbors),
        "conflict_points_untouched": int(np.sum(state == 3)),
    }
