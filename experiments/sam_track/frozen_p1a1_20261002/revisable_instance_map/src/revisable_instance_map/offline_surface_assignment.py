"""Offline, abstaining surface assignment from immutable P0 observations.

No GT or reference geometry is accepted by this module. All original published
labels remain fixed. U/T/CONFLICT share one surface graph; evidence restricts
candidate acceptance and transit, not the geometric component construction.
"""
from dataclasses import asdict, dataclass, replace
from collections import Counter
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, coo_matrix, eye, vstack, hstack
from scipy.sparse.csgraph import connected_components, dijkstra

UNOBSERVED, TENTATIVE, CONFIRMED, CONFLICT = range(4)
REJECTION_NAMES = {
    0: 'accepted', 1: 'no_reachable_reliable_seed',
    2: 'uncertain_normal_orientation', 3: 'own_candidate_incompatible',
    4: 'coherent_competing_evidence', 5: 'path_budget_exceeded',
    6: 'different_instance_margin_insufficient', 7: 'insufficient_seed_patch',
    8: 'path_record_overflow',
}


@dataclass(frozen=True)
class AssignmentSettings:
    # Fourth nearest point gives a stable sampling scale on TSDF edge vertices,
    # where the nearest point can be an almost coincident vertex.
    scale_neighbor: int = 4
    scale_clip_low: float = 0.75
    scale_clip_high: float = 1.25
    edge_radius_scale: float = 1.65
    normal_cos: float = 0.90
    plane_tolerance_scale: float = 0.18
    color_cost_weight: float = 0.35
    color_cost_scale: float = 0.25
    normal_cost_weight: float = 0.75
    edge_cost_floor_scale: float = 0.15
    seed_min_votes: int = 4
    seed_min_confidence: float = 0.80
    seed_patch_radius_scale: float = 2.5
    seed_min_patch_neighbors: int = 6
    conflict_min_patch_neighbors: int = 6
    path_budget_scale: float = 3.0
    conflict_path_budget_scale: float = 2.0
    margin_scale: float = 0.50
    best_second_ratio: float = 0.75
    conflict_margin_scale: float = 0.75
    conflict_best_second_ratio: float = 0.65
    conflict_min_vote_ratio_to_top: float = 0.50
    coherent_min_points: int = 3
    coherent_min_votes: int = 4
    coherent_min_neighbor_fraction: float = 0.30
    coherent_alt_vote_ratio: float = 0.50
    hole_max_points: int = 64
    hole_max_diameter_scale: float = 4.0
    hole_min_boundary_points: int = 8
    hole_min_reliable_boundary_points: int = 8
    hole_min_sectors: int = 6
    hole_max_angular_gap_deg: float = 100.0
    point_min_sectors: int = 5
    point_max_angular_gap_deg: float = 140.0
    hole_plane_rms_scale: float = 0.18
    hole_normal_cos: float = 0.93
    direction_sectors: int = 8
    max_path_points: int = 64
    query_chunk: int = 25000
    workers: int = 8


def validate_inputs(xyz, normals, rgb, evidence, pair, original_labels):
    n = len(xyz)
    if n < 1 or xyz.shape != (n, 3) or normals.shape != (n, 3) or rgb.shape != (n, 3):
        raise ValueError('Invalid surface array shape')
    if not all(np.isfinite(a).all() for a in (xyz, normals, rgb)):
        raise ValueError('Non-finite surface')
    if (np.linalg.norm(normals, axis=1) < 0.5).any() or rgb.min() < 0 or rgb.max() > 1.00001:
        raise ValueError('Invalid normals or RGB')
    for name in ('state', 'top1_instance_id', 'top1_votes', 'total_frame_votes', 'confidence'):
        if evidence[name].shape != (n,):
            raise ValueError('Invalid evidence shape: ' + name)
    state = evidence['state']
    if not np.isin(state, [0, 1, 2, 3]).all():
        raise ValueError('Unknown state')
    expected = np.where(state == CONFIRMED, evidence['top1_instance_id'], -1)
    if not np.array_equal(original_labels, expected) or (original_labels[state == CONFIRMED] <= 0).any():
        raise ValueError('Source is not the original confirmed-only P0')
    p, ids, votes = [np.asarray(pair[k]) for k in ('surface_point_index', 'instance_id', 'frame_votes')]
    if p.ndim != 1 or ids.shape != p.shape or votes.shape != p.shape:
        raise ValueError('Invalid pair arrays')
    if not all(np.issubdtype(a.dtype, np.integer) for a in (p, ids, votes)):
        raise ValueError('Noninteger pair data')
    if ((p < 0) | (p >= n)).any() or (ids <= 0).any() or (votes <= 0).any():
        raise ValueError('Invalid candidate pair')
    base = int(ids.max(initial=0)) + 1
    keys = p.astype(np.int64) * base + ids
    ordered = np.sort(keys)
    if (np.diff(ordered) == 0).any():
        raise ValueError('Duplicate point/candidate pair')
    totals = np.bincount(p, weights=votes, minlength=n)
    maximum = np.zeros(n, dtype=np.int32)
    np.maximum.at(maximum, p, votes)
    if not np.array_equal(totals, evidence['total_frame_votes']) or not np.array_equal(maximum, evidence['top1_votes']):
        raise ValueError('Full candidate votes differ from raw evidence')
    if ((state == UNOBSERVED) != (totals == 0)).any():
        raise ValueError('U state contradicts candidate evidence')
    confidence = np.divide(maximum, totals, out=np.zeros(n, float), where=totals > 0)
    if not np.allclose(confidence, evidence['confidence'], atol=1e-6, rtol=0):
        raise ValueError('Evidence confidence differs from complete votes')
    # A valid top ID must actually occur with the reported vote count.
    order = np.argsort(keys)
    q = np.flatnonzero(totals > 0)
    query = q.astype(np.int64) * base + evidence['top1_instance_id'][q]
    at = np.searchsorted(keys[order], query)
    if (at >= len(keys)).any() or not np.array_equal(keys[order][at], query):
        raise ValueError('Top candidate missing from complete votes')
    if not np.array_equal(votes[order][at], maximum[q]):
        raise ValueError('Top candidate vote mismatch')


def tangent_basis(normals):
    axis = np.zeros_like(normals)
    axis[:, 0] = 1
    use_y = np.abs(normals[:, 0]) > 0.8
    axis[use_y] = [0, 1, 0]
    u = np.cross(normals, axis)
    u /= np.linalg.norm(u, axis=1)[:, None]
    return u, np.cross(normals, u)


def angular_coverage(vectors, normal, sectors=8):
    if not len(vectors):
        return 0, 360.0
    u, v = tangent_basis(np.asarray(normal)[None])
    x, y = vectors @ u[0], vectors @ v[0]
    keep = x * x + y * y > 1e-14
    if not keep.any():
        return 0, 360.0
    angles = np.sort(np.mod(np.arctan2(y[keep], x[keep]), 2 * np.pi))
    bins = np.floor(angles * sectors / (2 * np.pi)).astype(int)
    gap = np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))
    return int(np.unique(bins).size), float(np.degrees(gap))


def surface_edges(tree, xyz, normals, rgb, scale, query, settings):
    """All radius neighbors, signed normals, and both local tangent planes.

    No nearest-k truncation: competing boundaries and region exits are retained.
    Coordinates and normals are frozen; appearance only changes edge cost.
    """
    rows, cols, costs = [], [], []
    uncertain = np.zeros(len(query), bool)
    for start in range(0, len(query), settings.query_chunk):
        q = query[start:start + settings.query_chunk]
        neighbors = tree.query_ball_point(xyz[q], settings.edge_radius_scale * scale[q],
                                          workers=settings.workers, return_sorted=True)
        lengths = np.fromiter((len(x) for x in neighbors), np.int32, len(q))
        r = np.repeat(np.arange(len(q)), lengths)
        j = np.concatenate(neighbors).astype(np.int32) if lengths.sum() else np.empty(0, np.int32)
        p = q[r]
        delta = xyz[j] - xyz[p]
        distance = np.linalg.norm(delta, axis=1)
        local = np.minimum(scale[p], scale[j])
        dot = np.einsum('ij,ij->i', normals[p], normals[j])
        plane0 = np.abs(np.einsum('ij,ij->i', delta, normals[p]))
        plane1 = np.abs(np.einsum('ij,ij->i', delta, normals[j]))
        geometric = ((j != p) & (distance <= settings.edge_radius_scale * local) &
                     (plane0 <= settings.plane_tolerance_scale * local) &
                     (plane1 <= settings.plane_tolerance_scale * local))
        # A close opposing sheet can indicate a thin object or inconsistent
        # orientation. Neither is a safe place to infer ownership.
        opposite = geometric & (dot <= -settings.normal_cos)
        uncertain[start + np.unique(r[opposite])] = True
        keep = geometric & (dot >= settings.normal_cos)
        color = np.linalg.norm(rgb[j[keep]] - rgb[p[keep]], axis=1)
        cost = np.maximum(distance[keep], settings.edge_cost_floor_scale * local[keep]) * (
            1 + settings.normal_cost_weight * (1 - np.clip(dot[keep], 0, 1)) +
            settings.color_cost_weight * np.minimum(color / settings.color_cost_scale, 3))
        rows.append((r[keep] + start).astype(np.int32))
        cols.append(j[keep])
        costs.append(cost.astype(np.float64))
    if not rows:
        return csr_matrix((len(query), len(xyz)), dtype=float), uncertain
    matrix = coo_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(len(query), len(xyz))).tocsr()
    matrix.sort_indices()
    return matrix, uncertain


def candidate_evidence(target, target_graph, evidence, pair, all_ids, settings):
    n = len(evidence['state'])
    local = np.full(n, -1, np.int32)
    local[target] = np.arange(len(target))
    p = pair['surface_point_index']
    keep = local[p] >= 0
    votes = coo_matrix((pair['frame_votes'][keep].astype(np.float64),
                       (local[p[keep]], np.searchsorted(all_ids, pair['instance_id'][keep]))),
                      shape=(len(target), len(all_ids))).tocsr()
    neighborhood = target_graph.copy()
    neighborhood.data[:] = 1
    neighborhood = neighborhood + eye(len(target), format='csr')
    present = votes.copy()
    present.data[:] = 1
    counts = (neighborhood @ present).tocsr()
    mass = (neighborhood @ votes).tocsr()
    row_index = np.repeat(np.arange(len(target)), np.diff(counts.indptr))
    degree = np.diff(neighborhood.indptr)
    # counts and mass have identical positive support, since all votes > 0.
    if not (np.array_equal(counts.indices, mass.indices) and np.array_equal(counts.indptr, mass.indptr)):
        # Sparse multiplication does not promise identical ordering.
        counts.sort_indices()
        mass.sort_indices()
    assert np.array_equal(counts.indices, mass.indices) and np.array_equal(counts.indptr, mass.indptr)
    coherent = ((counts.data >= settings.coherent_min_points) &
                (counts.data >= settings.coherent_min_neighbor_fraction * degree[row_index]) &
                (mass.data >= settings.coherent_min_votes))
    coherence = mass.copy()
    coherence.data[~coherent] = 0
    coherence.eliminate_zeros()
    best_id = np.full(len(target), -1, np.int32)
    best_mass, second_mass = np.zeros(len(target)), np.zeros(len(target))
    # At most a handful of observed IDs normally occur in one neighborhood.
    for row in np.flatnonzero(np.diff(coherence.indptr)):
        lo, hi = coherence.indptr[row:row + 2]
        values, ids = coherence.data[lo:hi], coherence.indices[lo:hi]
        k = int(np.argmax(values))
        best_id[row], best_mass[row] = all_ids[ids[k]], values[k]
        if len(values) > 1:
            second_mass[row] = np.max(np.delete(values, k))
    return votes.tocsc(), mass.tocsc(), best_id, best_mass, second_mass


def candidate_rules(instance, target, evidence, all_ids, votes, mass,
                    coherent_id, coherent_best, coherent_second, settings):
    col = int(np.searchsorted(all_ids, instance))
    own_votes = votes[:, col].toarray().ravel()
    local_mass = mass[:, col].toarray().ravel()
    state = evidence['state'][target]
    compatible = ((state == UNOBSERVED) |
                  ((state == TENTATIVE) & (evidence['top1_instance_id'][target] == instance)) |
                  ((state == CONFLICT) & (own_votes > 0) &
                   (own_votes >= settings.conflict_min_vote_ratio_to_top * evidence['top1_votes'][target])))
    other_mass = np.where(coherent_id == instance, coherent_second, coherent_best)
    coherent_competitor = ((other_mass >= settings.coherent_min_votes) &
                          (other_mass >= settings.coherent_alt_vote_ratio * np.maximum(local_mass, 1)))
    return compatible, coherent_competitor, own_votes.astype(np.int32)


def trace_paths(predecessors, sources, reached, global_nodes, xyz, maximum):
    paths = np.full((len(reached), maximum), -1, np.int32)
    length = np.zeros(len(reached), float)
    cursor = reached.copy()
    alive = np.ones(len(reached), bool)
    for step in range(maximum):
        rows = np.flatnonzero(alive)
        if not len(rows):
            break
        here = cursor[rows]
        paths[rows, step] = global_nodes[here]
        parent = predecessors[here]
        roots = parent == -9999
        if roots.any():
            if not np.array_equal(here[roots], sources[reached[rows[roots]]]):
                raise AssertionError('Path root is not its recorded original source')
            alive[rows[roots]] = False
        moving = rows[~roots]
        if len(moving):
            length[moving] += np.linalg.norm(xyz[global_nodes[cursor[moving]]] -
                                              xyz[global_nodes[parent[~roots]]], axis=1)
            cursor[moving] = parent[~roots]
    return paths, length, alive


def certify_holes(xyz, normals, target, neighborhood, target_graph, original_labels,
                  reliable_seed, normal_uncertain, winners, eligible, settings, global_scale):
    m = len(target)
    accepted = np.zeros(m, bool)
    component_id = np.full(m, -1, np.int32)
    if not m:
        return accepted, component_id, {}
    count, components = connected_components(target_graph, directed=False)
    order = np.argsort(components, kind='stable')
    sizes = np.bincount(components, minlength=count)
    starts = np.r_[0, np.cumsum(sizes)]
    stats = Counter()
    stats['total_components'] = int(count)
    for label in np.flatnonzero(sizes <= settings.hole_max_points):
        region = order[starts[label]:starts[label + 1]]
        points = target[region]
        component_id[region] = label
        stats['size_candidates'] += 1
        if normal_uncertain[region].any():
            stats['uncertain_normals'] += 1
            continue
        if np.linalg.norm(np.ptp(xyz[points], axis=0)) > settings.hole_max_diameter_scale * global_scale:
            stats['too_wide'] += 1
            continue
        neighbors = np.unique(np.concatenate([neighborhood.indices[neighborhood.indptr[r]:neighborhood.indptr[r + 1]] for r in region]))
        boundary = neighbors[original_labels[neighbors] > 0]
        if len(boundary) < settings.hole_min_boundary_points:
            stats['insufficient_boundary'] += 1
            continue
        ids = np.unique(original_labels[boundary])
        # Include ALL published boundary points, even weak competitors.
        if len(ids) != 1:
            stats['competing_boundary_labels'] += 1
            continue
        identity = int(ids[0])
        if np.sum(reliable_seed[boundary]) < settings.hole_min_reliable_boundary_points:
            stats['insufficient_original_stable_boundary'] += 1
            continue
        if not (eligible[region].all() and (winners[region] == identity).all()):
            stats['state_evidence_or_absolute_support_failed'] += 1
            continue
        support = np.r_[points, boundary]
        normal = normals[support].mean(0)
        norm = np.linalg.norm(normal)
        if norm < 0.9:
            stats['high_curvature'] += 1
            continue
        normal /= norm
        if (normals[support] @ normal < settings.hole_normal_cos).any():
            stats['high_curvature'] += 1
            continue
        centered = xyz[support] - xyz[support].mean(0)
        if np.sqrt(np.mean((centered @ normal) ** 2)) > settings.hole_plane_rms_scale * global_scale:
            stats['nonplanar_or_multilayer'] += 1
            continue
        sectors, gap = angular_coverage(xyz[boundary] - xyz[points].mean(0), normal, settings.direction_sectors)
        if sectors < settings.hole_min_sectors or gap > settings.hole_max_angular_gap_deg:
            stats['open_region_boundary'] += 1
            continue
        closed = True
        # A component cannot hide an exit into a larger U/T/C region: those
        # points already share its connected-component ID. This additional
        # pointwise directional test detects physical edges and sampling gaps.
        for r, p in zip(region, points):
            ns = neighborhood.indices[neighborhood.indptr[r]:neighborhood.indptr[r + 1]]
            sectors, gap = angular_coverage(xyz[ns] - xyz[p], normals[p], settings.direction_sectors)
            if sectors < settings.point_min_sectors or gap > settings.point_max_angular_gap_deg:
                closed = False
                break
        if not closed:
            stats['physical_edge_or_sampling_gap'] += 1
            continue
        accepted[region] = True
        stats['accepted_components'] += 1
        stats['accepted_points'] += len(region)
    return accepted, component_id, dict(stats)


def assign_surface(xyz, normals, rgb, evidence, pair, original_labels,
                   settings=None, progress=None):
    settings = settings or AssignmentSettings()
    started = time.perf_counter()
    progress = progress or (lambda *a: None)
    validate_inputs(xyz, normals, rgb, evidence, pair, original_labels)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    state = evidence['state']
    target = np.flatnonzero((original_labels < 0) & np.isin(state, [UNOBSERVED, TENTATIVE, CONFLICT])).astype(np.int32)
    m, n = len(target), len(xyz)
    if not m:
        return {'holes_only': original_labels.copy(), 'holes_geodesic': original_labels.copy()}, {
            'surface_point_index': target}, {'target_points': 0, 'settings': asdict(settings)}
    tree = cKDTree(xyz)
    sample = np.linspace(0, n - 1, min(n, 20000), dtype=int)
    sample_dist, _ = tree.query(xyz[sample], k=min(n, settings.scale_neighbor + 1), workers=settings.workers)
    global_scale = float(np.median(sample_dist[:, -1]))
    if not np.isfinite(global_scale) or global_scale <= 1e-6:
        raise ValueError('Surface sampling scale undefined')
    scale = np.empty(n, np.float32)
    for start in range(0, n, settings.query_chunk):
        dist, _ = tree.query(xyz[start:start + settings.query_chunk], k=min(n, settings.scale_neighbor + 1), workers=settings.workers)
        scale[start:start + len(dist)] = np.clip(dist[:, -1], settings.scale_clip_low * global_scale,
                                               settings.scale_clip_high * global_scale)
    progress('sampling_scale_m', global_scale)
    neighborhood, uncertain = surface_edges(tree, xyz, normals, rgb, scale, target, settings)
    target_graph = neighborhood[:, target].tocsr()
    if (target_graph != target_graph.T).nnz:
        raise AssertionError('Geometric target graph must be symmetric')
    boundary = np.unique(neighborhood.indices[original_labels[neighborhood.indices] > 0])
    possible_core = boundary[(evidence['top1_votes'][boundary] >= settings.seed_min_votes) &
                             (evidence['confidence'][boundary] >= settings.seed_min_confidence)]
    seed_neighbors, seed_uncertain = surface_edges(
        tree, xyz, normals, rgb, scale, possible_core,
        replace(settings, edge_radius_scale=settings.seed_patch_radius_scale))
    seed_row = np.repeat(np.arange(len(possible_core)), np.diff(seed_neighbors.indptr))
    seed_col = seed_neighbors.indices
    good_neighbor = ((original_labels[seed_col] == original_labels[possible_core[seed_row]]) &
                     (evidence['top1_votes'][seed_col] >= settings.seed_min_votes) &
                     (evidence['confidence'][seed_col] >= settings.seed_min_confidence))
    patch_count = np.bincount(seed_row[good_neighbor], minlength=len(possible_core))
    core = (patch_count >= settings.seed_min_patch_neighbors) & ~seed_uncertain
    seeds = possible_core[core]
    reliable_seed = np.zeros(n, bool)
    reliable_seed[seeds] = True
    patch_size = np.zeros(n, np.int32)
    patch_size[possible_core] = patch_count
    # Every original published boundary identity competes, even when it cannot
    # safely originate an assignment. Dropping weak cores here would let a
    # nearby large instance absorb their missing surface by default.
    source_boundary = boundary
    seed_count = len(source_boundary)
    seed_ids = original_labels[source_boundary]
    reliable_boundary = reliable_seed[source_boundary]
    # Seeds have outbound edges only. Published regions, including every
    # competing instance, are endpoints and cannot become transit corridors.
    seed_to_target = neighborhood[:, source_boundary].T.tocsr()
    graph = vstack([hstack([target_graph, csr_matrix((m, seed_count))], format='csr'),
                    hstack([seed_to_target, csr_matrix((seed_count, seed_count))], format='csr')], format='csr')
    graph.sort_indices()
    global_nodes = np.r_[target, source_boundary]
    all_ids = np.unique(np.r_[pair['instance_id'], original_labels[original_labels > 0]])
    votes, mass, coherent_id, coherent_best, coherent_second = candidate_evidence(
        target, target_graph, evidence, pair, all_ids, settings)
    best, second = np.full(m, np.inf), np.full(m, np.inf)
    best_id, second_id = np.full(m, -1, np.int32), np.full(m, -1, np.int32)
    shadow_best, shadow_second = np.full(m, np.inf), np.full(m, np.inf)
    shadow_best_id, shadow_second_id = np.full(m, -1, np.int32), np.full(m, -1, np.int32)
    best_source = np.full(m, -1, np.int32)
    source_patch = np.zeros(m, np.int16)
    direct_votes = np.zeros(m, np.int32)
    compatible_best, competing_best = np.zeros(m, bool), np.zeros(m, bool)
    physical = np.full(m, np.inf)
    paths = np.full((m, settings.max_path_points), -1, np.int32)
    overflow = np.zeros(m, bool)
    reachable_candidates = np.zeros(m, np.int16)
    reachable_boundary_competitors = np.zeros(m, np.int16)
    tt_row = np.repeat(np.arange(m), np.diff(target_graph.indptr))
    tt_len = target_graph.nnz
    # hstack/vstack keep exactly the target graph followed by source edges.
    assert np.array_equal(graph.indptr[:m + 1], target_graph.indptr)
    budget = settings.path_budget_scale * global_scale
    progress('graph', {'targets': m, 'target_edges': target_graph.nnz,
                       'reliable_boundary_seeds': int(reliable_boundary.sum()),
                       'all_published_boundary_points': seed_count, 'boundary_ids': int(np.unique(seed_ids).size)})
    for iteration, identity in enumerate(np.unique(seed_ids)):
        compatible, competing, own = candidate_rules(identity, target, evidence, all_ids, votes, mass,
                                                     coherent_id, coherent_best, coherent_second, settings)
        transit = compatible & ~competing & ~uncertain
        data = graph.data.copy()
        data[:tt_len][~transit[tt_row]] = 0
        candidate_graph = csr_matrix((data, graph.indices.copy(), graph.indptr.copy()), shape=graph.shape)
        candidate_graph.eliminate_zeros()
        boundary_nodes = np.flatnonzero(seed_ids == identity).astype(np.int32) + m
        boundary_distance = dijkstra(candidate_graph, directed=True, indices=boundary_nodes,
                                     min_only=True, return_predecessors=False, limit=budget)[:m]
        reached = np.isfinite(boundary_distance)
        reachable_boundary_competitors[reached] += 1
        improve = reached & (boundary_distance < shadow_best - 1e-12)
        shadow_second[improve], shadow_second_id[improve] = shadow_best[improve], shadow_best_id[improve]
        other_shadow = reached & ~improve & (boundary_distance < shadow_second)
        shadow_second[other_shadow], shadow_second_id[other_shadow] = boundary_distance[other_shadow], identity
        shadow_best[improve], shadow_best_id[improve] = boundary_distance[improve], identity
        source_nodes = np.flatnonzero((seed_ids == identity) & reliable_boundary).astype(np.int32) + m
        if not len(source_nodes):
            continue
        distances, predecessor, source = dijkstra(candidate_graph, directed=True, indices=source_nodes,
                                                   min_only=True, return_predecessors=True, limit=budget)
        distances = distances[:m]
        finite = np.isfinite(distances)
        reachable_candidates[finite] += 1
        better = finite & (distances < best - 1e-12)
        # Each iteration handles a different identity, so second place never
        # means merely another source point belonging to the same object.
        second[better], second_id[better] = best[better], best_id[better]
        other = finite & ~better & (distances < second)
        second[other], second_id[other] = distances[other], identity
        rows = np.flatnonzero(better)
        if len(rows):
            recorded, length, too_long = trace_paths(predecessor, source, rows, global_nodes, xyz,
                                                     settings.max_path_points)
            best[rows], best_id[rows] = distances[rows], identity
            best_source[rows] = global_nodes[source[rows]]
            source_patch[rows] = patch_size[best_source[rows]]
            direct_votes[rows] = own[rows]
            compatible_best[rows], competing_best[rows] = compatible[rows], competing[rows]
            paths[rows], physical[rows], overflow[rows] = recorded, length, too_long
        if iteration % 20 == 0:
            progress('candidate_search', {'completed': iteration + 1, 'ids': int(np.unique(seed_ids).size)})
    del graph, seed_neighbors
    shadow_other = np.where(shadow_best_id == best_id, shadow_second, shadow_best)
    shadow_other_id = np.where(shadow_best_id == best_id, shadow_second_id, shadow_best_id)
    closer_competitor = shadow_other < second
    second[closer_competitor], second_id[closer_competitor] = shadow_other[closer_competitor], shadow_other_id[closer_competitor]
    conflict = state[target] == CONFLICT
    point_budget = np.where(conflict, settings.conflict_path_budget_scale,
                            settings.path_budget_scale) * global_scale
    gap = np.where(conflict, settings.conflict_margin_scale, settings.margin_scale) * global_scale
    ratio = np.where(conflict, settings.conflict_best_second_ratio, settings.best_second_ratio)
    distinguishable = (second > best + gap) & (best <= ratio * second)
    minimum_patch = np.where(conflict, settings.conflict_min_patch_neighbors, settings.seed_min_patch_neighbors)
    reason = np.zeros(m, np.uint8)
    # Apply in reverse priority, so the primary reason is stable and meaningful.
    reason[overflow] = 8
    reason[source_patch < minimum_patch] = 7
    reason[~distinguishable] = 6
    reason[best > point_budget + 1e-12] = 5
    reason[competing_best] = 4
    reason[~compatible_best] = 3
    reason[uncertain] = 2
    reason[~np.isfinite(best)] = 1
    eligible = reason == 0
    holes, component_ids, hole_stats = certify_holes(
        xyz, normals, target, neighborhood, target_graph, original_labels,
        reliable_seed, uncertain, best_id, eligible, settings, global_scale)
    variants = {}
    for name, accepted in [('holes_only', holes), ('holes_geodesic', eligible)]:
        labels = original_labels.copy()
        labels[target[accepted]] = best_id[accepted]
        if not np.array_equal(labels[original_labels > 0], original_labels[original_labels > 0]):
            raise AssertionError('Original published identity changed')
        if not set(labels[labels > 0]).issubset(set(original_labels[original_labels > 0])):
            raise AssertionError('An identity without an original core was invented')
        variants[name] = labels
    origin = np.zeros(m, np.uint8)
    origin[eligible] = 3
    origin[holes] = 2
    accepted_rows = np.flatnonzero(eligible)
    selected_paths = paths[accepted_rows]
    lengths = np.sum(selected_paths >= 0, axis=1).astype(np.int32)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    path_vertices = selected_paths[selected_paths >= 0]
    if len(accepted_rows):
        assert np.all(physical[eligible] <= best[eligible] + 1e-8)
        assert np.all(physical[eligible] <= point_budget[eligible] + 1e-8)
        assert np.all(original_labels[best_source[eligible]] == best_id[eligible])
        assert np.all(evidence['state'][best_source[eligible]] == CONFIRMED)
        assert np.all(reliable_seed[best_source[eligible]])
    detail = {
        'surface_point_index': target, 'original_state': state[target].copy(),
        'proposed_instance_id': best_id, 'second_instance_id': second_id,
        'source_seed_index': best_source, 'source_seed_patch_neighbors': source_patch,
        'weighted_path_cost_m': best, 'geometric_path_length_m': physical,
        'second_weighted_path_cost_m': second, 'reachable_candidate_count': reachable_candidates,
        'reachable_published_boundary_candidate_count': reachable_boundary_competitors,
        'nearest_published_boundary_instance_id': shadow_best_id,
        'nearest_published_boundary_cost_m': shadow_best,
        'selected_direct_frame_votes': direct_votes, 'normal_orientation_uncertain': uncertain,
        'evidence_compatible': compatible_best, 'coherent_competitor': competing_best,
        'rejection_reason': reason, 'assignment_route': origin,
        'hole_component_id': component_ids, 'accepted_path_target_rows': accepted_rows,
        'accepted_path_offsets': offsets, 'accepted_path_surface_indices': path_vertices,
    }
    stats = {
        'target_points': m, 'sampling_scale_m': global_scale, 'path_budget_m': budget,
        'conflict_path_budget_m': settings.conflict_path_budget_scale * global_scale,
        'target_graph_directed_edges': target_graph.nnz,
        'reliable_original_boundary_seeds': int(reliable_boundary.sum()),
        'original_seed_instance_ids': np.unique(seed_ids[reliable_boundary]).astype(int).tolist(),
        'all_original_published_boundary_points': seed_count,
        'original_published_boundary_competing_ids': np.unique(seed_ids).astype(int).tolist(),
        'weak_published_boundaries_preserved_as_competitors': True,
        'normal_orientation_uncertain_points': int(uncertain.sum()),
        'holes': hole_stats, 'settings': asdict(settings), 'seconds': time.perf_counter() - started,
        'by_state': {}, 'original_published_labels_unchanged': True,
        'inferred_points_are_seeds': False, 'raw_evidence_updated': False,
        'complete_candidate_votes_used': True,
    }
    for value, name in [(UNOBSERVED, 'U'), (TENTATIVE, 'T'), (CONFLICT, 'CONFLICT')]:
        mask = state[target] == value
        stats['by_state'][name] = {
            'target': int(mask.sum()), 'accepted': int((mask & eligible).sum()),
            'holes': int((mask & holes).sum()), 'propagation': int((mask & eligible & ~holes).sum()),
            'retained': int((mask & ~eligible).sum()),
            'rejections': {REJECTION_NAMES[int(k)]: int(v) for k, v in zip(*np.unique(reason[mask & ~eligible], return_counts=True))},
        }
    return variants, detail, stats
