"""Conservative, no-repair online association of raw mask proposals.

A voxel index retrieves spatial candidates. Current-view depth agreement and
mask overlap score visibility. Decisions use only previously committed frames.
The raw observation catalog is the authority; voxel support is a rebuildable cache.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .frame_io import Frame
from .observations import RawInstanceObservation, source_pixel_indices


_NEIGHBORS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if dx * dx + dy * dy + dz * dz <= 2
)


@dataclass
class InstanceState:
    instance_id: int
    observation_ids: list[str] = field(default_factory=list)
    voxel_list: list[tuple[int, int, int]] = field(default_factory=list)
    voxel_set: set[tuple[int, int, int]] = field(default_factory=set)
    first_frame_id: int = -1
    last_frame_id: int = -1


@dataclass(frozen=True)
class ObservationSupport:
    observation_id: str
    frame_id: int
    source_mask_sha256: str
    instance_id: int
    voxels: tuple[tuple[int, int, int], ...]


class OnlineVoxelAssociator:
    def __init__(
        self,
        voxel_size_m: float = 0.03,
        max_points_per_observation: int = 512,
        max_visible_support_points: int = 2048,
        min_geometric_coverage: float = 0.25,
        min_total_score: float = 0.35,
        min_visible_overlap: float = 0.20,
        ambiguity_margin: float = 0.05,
        depth_tolerance_m: float = 0.10,
        allow_multiple_observations_per_instance_per_frame: bool = False,
        improved_association: bool = False,
        valid_first_sampling: bool | None = None,
    ):
        if voxel_size_m <= 0 or max_points_per_observation < 1:
            raise ValueError("Invalid voxel or sampling configuration")
        self.voxel_size_m = voxel_size_m
        self.max_points_per_observation = max_points_per_observation
        self.max_visible_support_points = max_visible_support_points
        self.min_geometric_coverage = min_geometric_coverage
        self.min_total_score = min_total_score
        self.min_visible_overlap = min_visible_overlap
        self.ambiguity_margin = ambiguity_margin
        self.depth_tolerance_m = depth_tolerance_m
        self.allow_multiple_observations_per_instance_per_frame = allow_multiple_observations_per_instance_per_frame
        self.improved_association = improved_association
        self.valid_first_sampling = improved_association if valid_first_sampling is None else valid_first_sampling
        self.instances: dict[int, InstanceState] = {}
        self.observation_support: dict[str, ObservationSupport] = {}
        self.voxel_to_instances: dict[tuple[int, int, int], set[int]] = defaultdict(set)
        self.next_instance_id = 1
        self.last_processed_frame_id = -1

    def _sample_observation_voxels(
        self, frame: Frame, observation: RawInstanceObservation
    ) -> tuple[tuple[int, int, int], ...]:
        pixels = source_pixel_indices(frame, observation)
        if self.valid_first_sampling:
            depth_all = frame.depth_m.ravel()[pixels]
            valid_pixels = np.isfinite(depth_all) & (depth_all > 0) & (depth_all < 10.0)
            pixels = pixels[valid_pixels]
            if not len(pixels):
                return ()
        sample_count = min(len(pixels), self.max_points_per_observation)
        pixels = pixels[np.linspace(0, len(pixels) - 1, num=sample_count, dtype=np.int64)]
        rows, cols = np.divmod(pixels, frame.camera.width)
        depth = frame.depth_m.ravel()[pixels].astype(np.float64)
        valid = np.isfinite(depth) & (depth > 0) & (depth < 10.0)
        if not np.any(valid):
            return ()
        rows, cols, depth = rows[valid], cols[valid], depth[valid]
        camera_xyz = np.column_stack((
            (cols - frame.camera.cx) * depth / frame.camera.fx,
            (rows - frame.camera.cy) * depth / frame.camera.fy,
            depth,
        ))
        world_xyz = camera_xyz @ frame.camera_to_world[:3, :3].T + frame.camera_to_world[:3, 3]
        voxels = np.unique(np.floor(world_xyz / self.voxel_size_m).astype(np.int32), axis=0)
        return tuple(tuple(int(value) for value in row) for row in voxels)

    def _candidate_counts(self, voxels: tuple[tuple[int, int, int], ...]):
        counts = Counter()
        local_support = defaultdict(set)
        for x, y, z in voxels:
            nearby_ids = set()
            for dx, dy, dz in _NEIGHBORS:
                key = (x + dx, y + dy, z + dz)
                for instance_id in self.voxel_to_instances.get(key, ()):
                    nearby_ids.add(instance_id)
                    local_support[instance_id].add(key)
            counts.update(nearby_ids)
        return counts, local_support

    def _visible_overlap(
        self, frame: Frame, observation: RawInstanceObservation,
        local_support: set[tuple[int, int, int]],
    ) -> tuple[float | None, int]:
        if not local_support:
            return None, 0
        support = np.asarray(sorted(local_support), dtype=np.float64)
        step = max(1, (len(support) + self.max_visible_support_points - 1)
                   // self.max_visible_support_points)
        support = support[::step]
        world_xyz = (support + 0.5) * self.voxel_size_m
        camera_xyz = (world_xyz - frame.camera_to_world[:3, 3]) @ frame.camera_to_world[:3, :3]
        z = camera_xyz[:, 2]
        front = z > 0
        if not np.any(front):
            return None, 0
        camera_xyz, z = camera_xyz[front], z[front]
        u = np.rint(frame.camera.fx * camera_xyz[:, 0] / z + frame.camera.cx).astype(np.int32)
        v = np.rint(frame.camera.fy * camera_xyz[:, 1] / z + frame.camera.cy).astype(np.int32)
        inside_image = (u >= 0) & (u < frame.camera.width) & (v >= 0) & (v < frame.camera.height)
        if not np.any(inside_image):
            return None, 0
        u, v, z = u[inside_image], v[inside_image], z[inside_image]
        observed_depth = frame.depth_m[v, u]
        visible = np.isfinite(observed_depth) & (observed_depth > 0)
        visible &= np.abs(observed_depth - z) <= self.depth_tolerance_m
        visible_count = int(np.count_nonzero(visible))
        if visible_count == 0:
            return None, 0
        overlap = float(np.mean(frame.mask_local[v[visible], u[visible]] == observation.mask_local_id))
        return overlap, visible_count

    def _score_candidates(
        self, frame: Frame, observation: RawInstanceObservation,
        voxels: tuple[tuple[int, int, int], ...],
    ) -> list[dict]:
        if not voxels:
            return []
        counts, local_support = self._candidate_counts(voxels)
        ranked = []
        for instance_id, matched in counts.most_common(8):
            geometric = matched / len(voxels)
            visible_overlap, visible_count = self._visible_overlap(
                frame, observation, local_support[instance_id]
            )
            score = geometric if visible_overlap is None else (
                0.5 * geometric + 0.5 * visible_overlap
            )
            ranked.append({
                "instance_id": instance_id,
                "geometric_coverage": round(geometric, 6),
                "visible_overlap": None if visible_overlap is None else round(visible_overlap, 6),
                "visible_support_points": visible_count,
                "prior_observation_count": len(self.instances[instance_id].observation_ids),
                "score": round(score, 6),
            })
        ranked.sort(key=lambda item: (-item["score"], -item["prior_observation_count"], item["instance_id"]))
        return ranked[:8] if self.improved_association else ranked[:3]

    def _commit(
        self, instance_id: int, observation: RawInstanceObservation,
        voxels: tuple[tuple[int, int, int], ...],
    ) -> None:
        if instance_id not in self.instances:
            self.instances[instance_id] = InstanceState(
                instance_id=instance_id, first_frame_id=observation.frame_id
            )
        if observation.observation_id in self.observation_support:
            raise ValueError("Observation committed twice")
        self.observation_support[observation.observation_id] = ObservationSupport(
            observation.observation_id, observation.frame_id,
            observation.source_mask_sha256, instance_id, voxels)
        instance = self.instances[instance_id]
        instance.observation_ids.append(observation.observation_id)
        instance.last_frame_id = observation.frame_id
        for voxel in voxels:
            if voxel not in instance.voxel_set:
                instance.voxel_set.add(voxel)
                instance.voxel_list.append(voxel)
                self.voxel_to_instances[voxel].add(instance_id)

    def process_frame(
        self, frame: Frame, observations: tuple[RawInstanceObservation, ...]
    ) -> list[dict]:
        if self.improved_association:
            return self._process_frame_improved(frame, observations)
        if frame.frame_id <= self.last_processed_frame_id:
            raise ValueError("Frames must be processed once in increasing order")
        if len({item.mask_local_id for item in observations}) != len(observations):
            raise ValueError("Duplicate local mask IDs in one frame")
        proposals = []
        for observation in observations:
            if observation.frame_id != frame.frame_id:
                raise ValueError("Observation belongs to another frame")
            voxels = self._sample_observation_voxels(frame, observation)
            candidates = self._score_candidates(frame, observation, voxels)
            if not voxels:
                reason = "new_no_projectable_sample"
                qualified = False
            elif not candidates:
                reason = "new_no_spatial_candidate"
                qualified = False
            else:
                best = candidates[0]
                qualified = (
                    best["geometric_coverage"] >= self.min_geometric_coverage
                    and best["score"] >= self.min_total_score
                    and (
                        best["visible_support_points"] < 10
                        or best["visible_overlap"] is None
                        or best["visible_overlap"] >= self.min_visible_overlap
                    )
                )
                ambiguous = (
                    len(candidates) > 1
                    and best["score"] - candidates[1]["score"] < self.ambiguity_margin
                    and candidates[1]["geometric_coverage"] >= self.min_geometric_coverage
                )
                reason = (
                    "new_low_score" if not qualified
                    else "matched_ambiguous" if ambiguous
                    else "matched"
                )
            proposals.append({
                "observation": observation,
                "voxels": voxels,
                "candidates": candidates,
                "qualified": qualified,
                "reason": reason,
            })

        # The original fixed-mask baseline allows at most one region per existing
        # instance per frame. The optional multi-region switch is diagnostic for
        # the OVI-MAP front end; neither mode is the future revisable solver.
        claimed_instances = set()
        decisions = [None] * len(proposals)
        order = sorted(
            range(len(proposals)),
            key=lambda i: (
                -(proposals[i]["candidates"][0]["score"] if proposals[i]["candidates"] else -1),
                proposals[i]["observation"].mask_local_id,
            ),
        )
        for index in order:
            proposal = proposals[index]
            best = proposal["candidates"][0] if proposal["candidates"] else None
            if proposal["qualified"] and (self.allow_multiple_observations_per_instance_per_frame or best["instance_id"] not in claimed_instances):
                instance_id = best["instance_id"]
                claimed_instances.add(instance_id)
                reason = proposal["reason"]
            else:
                instance_id = self.next_instance_id
                self.next_instance_id += 1
                reason = "new_same_frame_conflict" if proposal["qualified"] else proposal["reason"]
            decisions[index] = {
                "frame_id": frame.frame_id,
                "observation_id": proposal["observation"].observation_id,
                "mask_local_id": proposal["observation"].mask_local_id,
                "instance_id": instance_id,
                "decision": reason,
                "sampled_voxels": len(proposal["voxels"]),
                "candidates": proposal["candidates"],
            }
        # Only now can this frame influence later frames.
        for proposal, decision in zip(proposals, decisions):
            self._commit(decision["instance_id"], proposal["observation"], proposal["voxels"])
        self.last_processed_frame_id = frame.frame_id
        return decisions


    def _process_frame_improved(
        self, frame: Frame, observations: tuple[RawInstanceObservation, ...]
    ) -> list[dict]:
        """Score against the previous frame, then claim the best eligible old ID.

        Parent proposals are the units of association and birth. Geometric child
        regions are recorded separately and never create IDs in this mode.
        """
        if frame.frame_id <= self.last_processed_frame_id:
            raise ValueError("Frames must be processed once in increasing order")
        if len({obs.mask_local_id for obs in observations}) != len(observations):
            raise ValueError("Duplicate local mask IDs in one frame")
        proposals = []
        for obs in observations:
            if obs.frame_id != frame.frame_id:
                raise ValueError("Observation belongs to another frame")
            voxels = self._sample_observation_voxels(frame, obs)
            candidates = self._score_candidates(frame, obs, voxels)
            eligible = [item for item in candidates if (
                item["geometric_coverage"] >= self.min_geometric_coverage
                and item["score"] >= self.min_total_score
                and (item["visible_support_points"] < 10
                     or item["visible_overlap"] is None
                     or item["visible_overlap"] >= self.min_visible_overlap)
            )]
            proposals.append((obs, voxels, candidates, eligible))

        # No state is changed until every proposal has been scored.
        claimed = set()
        decisions = [None] * len(proposals)
        order = sorted(range(len(proposals)), key=lambda i: (
            -(proposals[i][3][0]["score"] if proposals[i][3] else -1),
            proposals[i][0].mask_local_id,
        ))
        for i in order:
            obs, voxels, candidates, eligible = proposals[i]
            selected = next((candidate for candidate in eligible if (
                self.allow_multiple_observations_per_instance_per_frame
                or candidate["instance_id"] not in claimed
            )), None)
            if selected is None:
                instance_id = self.next_instance_id
                self.next_instance_id += 1
                reason = (
                    "new_no_projectable_sample" if not voxels else
                    "new_no_spatial_candidate" if not candidates else
                    "new_same_frame_conflict" if eligible else
                    "new_low_score"
                )
            else:
                instance_id = selected["instance_id"]
                claimed.add(instance_id)
                other = next((candidate for candidate in eligible if
                    candidate["instance_id"] != instance_id), None)
                ambiguous = (other is not None and
                             abs(selected["score"] - other["score"]) < self.ambiguity_margin)
                reason = "matched_ambiguous" if ambiguous else "matched"
            decisions[i] = {
                "frame_id": frame.frame_id,
                "observation_id": obs.observation_id,
                "mask_local_id": obs.mask_local_id,
                "instance_id": instance_id,
                "decision": reason,
                "sampled_voxels": len(voxels),
                "selected_candidate_id": None if selected is None else instance_id,
                "candidates": candidates,
            }
        for (obs, voxels, _, _), decision in zip(proposals, decisions):
            self._commit(decision["instance_id"], obs, voxels)
        self.last_processed_frame_id = frame.frame_id
        return decisions

    def reassign_observations(self, assignment_updates: dict[str, int]) -> None:
        """Atomically rebuild the candidate index after an identity revision.

        Source observations and sampled support remain unchanged; region edits
        require regenerating support from their immutable source masks.
        """
        if not set(assignment_updates).issubset(self.observation_support):
            raise ValueError("Cannot reassign an unknown observation")
        if any(instance_id <= 0 for instance_id in assignment_updates.values()):
            raise ValueError("Instance IDs must be positive")
        revised = {}
        instances = {}
        index = defaultdict(set)
        for obs_id, old in self.observation_support.items():
            instance_id = assignment_updates.get(obs_id, old.instance_id)
            entry = ObservationSupport(obs_id, old.frame_id, old.source_mask_sha256,
                                       instance_id, old.voxels)
            revised[obs_id] = entry
            if instance_id not in instances:
                instances[instance_id] = InstanceState(instance_id, first_frame_id=old.frame_id)
            state = instances[instance_id]
            state.observation_ids.append(obs_id)
            state.first_frame_id = min(state.first_frame_id, old.frame_id)
            state.last_frame_id = max(state.last_frame_id, old.frame_id)
            for voxel in old.voxels:
                if voxel not in state.voxel_set:
                    state.voxel_set.add(voxel)
                    state.voxel_list.append(voxel)
                    index[voxel].add(instance_id)
        self.observation_support = revised
        self.instances = instances
        self.voxel_to_instances = index
        self.next_instance_id = max(instances, default=0) + 1
