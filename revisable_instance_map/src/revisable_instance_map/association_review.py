"""Read-only local evidence views and raw, bidirectional RGB-D verification."""
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
import copy
from dataclasses import replace
import hashlib

import numpy as np

from .assignment_ledger import fingerprint
from .observations import source_pixel_indices


class _ReadMap(Mapping):
    def __init__(self, base, getter):
        self.base, self.getter = base, getter

    def __getitem__(self, key):
        return self.getter(key)

    def __iter__(self):
        return iter(self.base)

    def __len__(self):
        return len(self.base)

    def get(self, key, default=None):
        return self.getter(key) if key in self.base else default


class EvidenceView:
    """Subtract observation references locally, without mutating the real cache."""
    def __init__(self, evidence, excluded_observation_ids=()):
        self.evidence = evidence
        self.excluded_ids = frozenset(excluded_observation_ids)
        if not self.excluded_ids.issubset(evidence.entries):
            raise ValueError("Unknown excluded contribution")
        refs, voxel_frames, instance_frames = Counter(), defaultdict(Counter), defaultdict(Counter)
        for key in self.excluded_ids:
            entry = evidence.entries[key]
            if entry.voxels:
                instance_frames[entry.instance_id][entry.frame_id] += 1
            for voxel in entry.voxels:
                refs[(entry.frame_id, voxel, entry.instance_id)] += 1
                voxel_frames[voxel][entry.frame_id] += 1
        self._cache = {}
        self._refs, self._voxel_frames = refs, voxel_frames
        self.voxel_counts = _ReadMap(evidence.voxel_counts, self._counts)
        self.voxel_totals = _ReadMap(evidence.voxel_totals, lambda v: sum(self._counts(v).values()))
        self.voxel_frame_references = _ReadMap(evidence.voxel_frame_references,
            lambda v: evidence.voxel_frame_references[v] - voxel_frames[v])
        self.instance_frame_references = _ReadMap(evidence.instance_frame_references,
            lambda k: evidence.instance_frame_references[k] - instance_frames[k])

    def _counts(self, voxel):
        if voxel not in self._cache:
            result = self.evidence.voxel_counts[voxel].copy()
            for frame, count in self._voxel_frames.get(voxel, {}).items():
                for instance in tuple(result):
                    removed = self._refs.get((frame, voxel, instance), 0)
                    if removed and removed == self.evidence.frame_voxel_references[(frame, voxel, instance)]:
                        result[instance] -= 1
                        if result[instance] == 0:
                            del result[instance]
            self._cache[voxel] = result
        return self._cache[voxel]

    def scorer(self, engine):
        result = copy.copy(engine)
        result.identity_evidence = self
        result.voxel_to_instances = _ReadMap(engine.voxel_to_instances, lambda v: set(self._counts(v)))
        result.instances = engine.instances.copy()
        affected = {self.evidence.entries[k].instance_id for k in self.excluded_ids}
        for key in affected:
            original = engine.instances[key]
            result.instances[key] = replace(original, observation_ids=[o for o in original.observation_ids if o not in self.excluded_ids])
        result.candidate_support_traces, result.candidate_weight_diagnostics = {}, {}
        return result


def score_complete(engine, frame, observation, voxels, parameters, excluded_ids=()):
    scorer = EvidenceView(engine.identity_evidence, excluded_ids).scorer(engine) if excluded_ids else copy.copy(engine)
    scorer.candidate_support_traces, scorer.candidate_weight_diagnostics = {}, {}
    candidates = scorer._score_candidates(frame, observation, voxels, candidate_limit=8, output_limit=8)
    counts = getattr(scorer, "_retrieved_candidate_counts", {}) if voxels else {}

    def unresolved_bound(rows):
        scored = {c["instance_id"] for c in rows}
        bounds = [(.5 * (count / len(voxels)) + .5, key) for key, count in counts.items()
                  if key not in scored and count / len(voxels) >= engine.min_geometric_coverage]
        return max(bounds, default=(0., None))

    upper, _ = unresolved_bound(candidates)
    threshold = candidates[0]["score"] - engine.ambiguity_margin - 2e-6 if candidates else -1
    if upper >= threshold and parameters["expand_candidates"] and len(counts) > 8:
        n = min(len(counts), parameters["candidate_budget"])
        candidates = scorer._score_candidates(frame, observation, voxels, candidate_limit=n, output_limit=n)
        upper, _ = unresolved_bound(candidates)
    complete = not counts or not any(
        key not in {c["instance_id"] for c in candidates}
        and count / len(voxels) >= engine.min_geometric_coverage
        and .5 * count / len(voxels) + .5 >= (candidates[0]["score"] - engine.ambiguity_margin - 2e-6 if candidates else -1)
        for key, count in counts.items())
    diagnostics = scorer.candidate_weight_diagnostics.get(observation.observation_id, {}).copy()
    diagnostics["relative_weight_scope"] = "scored_"+str(len(candidates))+"_of_"+str(len(counts))+"_spatial_candidates"
    return {"candidates": candidates, "candidate_count_retrieved": len(counts),
            "candidate_count_scored": len(candidates), "candidate_count_truncated": len(counts) - len(candidates),
            "candidate_complete": complete, "unscored_score_upper_bound": upper,
            "read_map_version": engine.map_version,
            **diagnostics}


def qualified(engine, candidate):
    return (candidate["geometric_coverage"] >= engine.min_geometric_coverage
            and candidate["score"] >= engine.min_total_score
            and (candidate["visible_support_points"] < 10 or candidate["visible_overlap"] is None
                 or candidate["visible_overlap"] >= engine.min_visible_overlap))


def reliable(engine, scores, parameters):
    rows = scores["candidates"]
    if not rows:
        return None, "no_spatial_candidate"
    best = rows[0]
    if not scores["candidate_complete"]:
        return None, "candidate_incomplete"
    if not qualified(engine, best):
        return None, "weak_absolute_match"
    competitor = next((c for c in rows[1:] if c["geometric_coverage"] >= engine.min_geometric_coverage), None)
    if competitor and best["score"] - competitor["score"] < engine.ambiguity_margin:
        return None, "low_margin"
    if (best["visible_overlap"] is None or best["visible_support_points"] < parameters["min_visible_points"]
            or best["visible_overlap"] < engine.min_visible_overlap):
        return None, "visibility_unverified"
    if 1 - best["source_single_frame_fraction"] < parameters["mature_fraction"]:
        return None, "single_frame_local_support"
    return best["instance_id"], "reliable_old_instance"


class RawViewVerifier:
    def __init__(self, frame_loader, engine, parameters):
        self.loader, self.engine, self.parameters = frame_loader, engine, parameters
        self.frames, self.points_cache, self.pair_cache = OrderedDict(), OrderedDict(), OrderedDict()
        self.hash_cache = {}
        self.poses = {}
        self.max_available_frame = -1

    def receive(self, frame):
        self.max_available_frame = max(self.max_available_frame, frame.frame_id)
        self.frames[frame.frame_id] = frame
        self.poses[frame.frame_id] = frame.camera_to_world.copy()
        while len(self.frames) > self.parameters["frame_cache_size"]:
            self.frames.popitem(last=False)

    def frame(self, frame_id):
        if frame_id > self.max_available_frame:
            raise ValueError("Cannot load a future view")
        if frame_id not in self.frames:
            self.receive(self.loader(frame_id))
        self.frames.move_to_end(frame_id)
        return self.frames[frame_id]

    def source_hash(self, frame):
        camera = frame.camera
        signature = (id(frame.depth_m), id(frame.camera_to_world), id(frame.mask_local),
                     camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy)
        cached = self.hash_cache.get(frame.frame_id)
        if (cached and cached[0] == signature and not frame.depth_m.flags.writeable
                and not frame.camera_to_world.flags.writeable and not frame.mask_local.flags.writeable):
            return cached[1]
        digest = hashlib.sha256()
        for array in (frame.depth_m, frame.camera_to_world, frame.camera.intrinsic, frame.mask_local):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(str((frame.camera.width, frame.camera.height)).encode())
        result = digest.hexdigest()
        self.hash_cache[frame.frame_id] = (signature, result)
        return result

    def points(self, record):
        key = record.observation_id
        if key not in self.points_cache:
            frame = self.frame(record.source_frame_id)
            if self.source_hash(frame) != record.depth_pose_intrinsics_hash:
                raise ValueError("Immutable depth/pose/intrinsics changed")
            pixels = source_pixel_indices(frame, record.observation)
            if self.engine.valid_first_sampling:
                depth = frame.depth_m.ravel()[pixels]
                pixels = pixels[np.isfinite(depth) & (depth > 0) & (depth < 10)]
            if len(pixels):
                pixels = pixels[np.linspace(0, len(pixels)-1, min(len(pixels), self.engine.max_points_per_observation), dtype=np.int64)]
            depth = frame.depth_m.ravel()[pixels].astype(np.float64)
            valid = np.isfinite(depth) & (depth > 0) & (depth < 10)
            rows, cols = np.divmod(pixels[valid], frame.camera.width)
            z = depth[valid]
            xyz = np.column_stack(((cols-frame.camera.cx)*z/frame.camera.fx,
                                   (rows-frame.camera.cy)*z/frame.camera.fy, z))
            world = xyz @ frame.camera_to_world[:3, :3].T + frame.camera_to_world[:3, 3]
            self.points_cache[key] = world
            while len(self.points_cache) > self.parameters["point_cache_size"]:
                self.points_cache.popitem(last=False)
        self.points_cache.move_to_end(key)
        return self.points_cache[key]

    def _direction(self, source, target):
        frame = self.frame(target.source_frame_id)
        xyz = (self.points(source) - frame.camera_to_world[:3, 3]) @ frame.camera_to_world[:3, :3]
        front = xyz[:, 2] > 0
        xyz = xyz[front]
        if not len(xyz):
            return {"visible": 0, "agreement": None, "multi_region": False}
        z = xyz[:, 2]
        u = np.rint(frame.camera.fx * xyz[:, 0] / z + frame.camera.cx).astype(np.int32)
        v = np.rint(frame.camera.fy * xyz[:, 1] / z + frame.camera.cy).astype(np.int32)
        inside = (u >= 0) & (u < frame.camera.width) & (v >= 0) & (v < frame.camera.height)
        u, v, z = u[inside], v[inside], z[inside]
        depth = frame.depth_m[v, u]
        visible = np.isfinite(depth) & (depth > 0) & (depth < 10) & (np.abs(depth-z) <= self.engine.depth_tolerance_m)
        labels = frame.mask_local[v[visible], u[visible]]
        count = len(labels)
        hist = Counter(int(x) for x in labels)
        significant = [k for k, n in hist.items() if k > 0 and n >= self.parameters["min_shared_points"]
                       and n / max(count, 1) >= self.parameters["multi_region_fraction"]]
        return {"visible": count, "agreement": float(np.mean(labels == target.observation.mask_local_id)) if count else None,
                "multi_region": len(significant) > 1, "significant_regions": significant}

    def compare(self, left, right, region_identities=None):
        key = (left.observation_id, right.observation_id,
               fingerprint(region_identities) if region_identities else "")
        if key not in self.pair_cache:
            a, b = self._direction(left, right), self._direction(right, left)
            if region_identities:
                for direction, target in ((a, right), (b, left)):
                    identities = [region_identities.get(target.source_frame_id, {}).get(k)
                                  for k in direction.get("significant_regions", ())]
                    if identities and all(k is not None for k in identities) and len(set(identities)) == 1:
                        # Frozen independent identity evidence can establish that
                        # several local masks are fragments of the same existing ID.
                        direction["multi_region"] = False
            minimum = self.parameters["min_shared_points"]
            if a["visible"] < minimum or b["visible"] < minimum:
                status, score = "UNKNOWN", None
            elif a["multi_region"] or b["multi_region"]:
                status, score = "MULTI_REGION_CONTRADICTION", min(a["agreement"], b["agreement"])
            else:
                score = min(a["agreement"], b["agreement"])
                status = "COMPATIBLE" if score >= self.parameters["raw_agreement"] else "CONTRADICTION"
            self.pair_cache[key] = {"status": status, "score": score, "forward": a, "backward": b}
            while len(self.pair_cache) > self.parameters["pair_cache_size"]:
                self.pair_cache.popitem(last=False)
        self.pair_cache.move_to_end(key)
        return self.pair_cache[key]

    def information_groups(self, records):
        """Merge near-repeat views; a group is not a statistical independence claim."""
        representatives, groups = [], {}
        for record in sorted(records, key=lambda r: (r.source_frame_id, r.observation_id)):
            if record.source_frame_id not in self.poses:
                self.frame(record.source_frame_id)
            pose = self.poses[record.source_frame_id]
            group = None
            for representative, group_id in representatives:
                if representative.source_frame_id == record.source_frame_id:
                    group = group_id
                    break
                old_pose = self.poses[representative.source_frame_id]
                translation = np.linalg.norm(pose[:3, 3] - old_pose[:3, 3])
                cosine = np.clip((np.trace(pose[:3, :3].T @ old_pose[:3, :3]) - 1) / 2, -1, 1)
                angle = np.degrees(np.arccos(cosine))
                novelty = len(set(record.voxels) - set(representative.voxels)) / max(len(record.voxels), 1)
                if translation < self.parameters["new_view_translation_m"] and angle < self.parameters["new_view_angle_deg"] and novelty < self.parameters["new_support_fraction"]:
                    group = group_id
                    break
            if group is None:
                group = "V:" + record.observation_id
                representatives.append((record, group))
            groups[record.observation_id] = group
        return groups
