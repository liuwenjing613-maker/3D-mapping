"""Conservative, no-repair online association of raw mask proposals.

A voxel index retrieves spatial candidates. Current-view depth agreement and
mask overlap score visibility. Decisions use only previously committed frames.
The raw observation catalog is the authority; voxel support is a rebuildable cache.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .frame_io import Frame
from .observations import RawInstanceObservation, source_pixel_indices
from .identity_evidence import FrameIdentityEvidence, ObservationSupport, validate_instance_id


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
        association_mode: str = "legacy",
        relative_weight_temperature: float = 0.1,
        null_candidate_score: float | None = None,
    ):
        if voxel_size_m <= 0 or max_points_per_observation < 1:
            raise ValueError("Invalid voxel or sampling configuration")
        if association_mode not in ("legacy", "binary-ledger", "probabilistic"):
            raise ValueError("Unknown identity association mode")
        if not np.isfinite(relative_weight_temperature) or relative_weight_temperature <= 0:
            raise ValueError("Relative-weight temperature must be finite and positive")
        if null_candidate_score is not None and not np.isfinite(null_candidate_score):
            raise ValueError("Null candidate score must be finite")
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
        self.association_mode = association_mode
        self.relative_weight_temperature = relative_weight_temperature
        self.null_candidate_score = min_total_score if null_candidate_score is None else null_candidate_score
        self.identity_evidence = None if association_mode == "legacy" else FrameIdentityEvidence()
        self.instances: dict[int, InstanceState] = {}
        self.observation_support = {} if self.identity_evidence is None else self.identity_evidence.entries
        self.voxel_to_instances: dict[tuple[int, int, int], set[int]] = defaultdict(set)
        self.next_instance_id = 1
        self.last_processed_frame_id = -1
        self.map_version = 0
        self.revision_events = []
        self.candidate_support_traces = {}
        self.candidate_weight_diagnostics = {}
        self._last_scoring_map_version = 0

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
        self._probability_sources = defaultdict(list)
        for query_index, (x, y, z) in enumerate(voxels):
            nearby_ids = set()
            maxima = {}
            for dx, dy, dz in _NEIGHBORS:
                key = (x + dx, y + dy, z + dz)
                for instance_id in self.voxel_to_instances.get(key, ()):
                    nearby_ids.add(instance_id)
                    local_support[instance_id].add(key)
                    if self.identity_evidence is not None:
                        evidence = self.identity_evidence
                        votes = evidence.voxel_counts[key][instance_id]
                        total = evidence.voxel_totals[key]
                        probability = votes / total
                        previous = maxima.get(instance_id)
                        # Exact max; deterministic provenance for equal probabilities.
                        if previous is None or (-probability, -votes, key) < (-previous[0], -previous[1], previous[2]):
                            maxima[instance_id] = (probability, votes, key, total,
                                                  len(evidence.voxel_frame_references[key]))
            counts.update(nearby_ids)
            for instance_id, (probability, votes, key, total, frames) in maxima.items():
                self._probability_sources[instance_id].append((query_index, key, votes, total, frames))
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
        candidate_limit: int = 8, output_limit: int | None = None,
    ) -> list[dict]:
        if not voxels:
            return []
        counts, local_support = self._candidate_counts(voxels)
        self._retrieved_candidate_counts = counts
        ranked = []
        for instance_id, matched in counts.most_common(candidate_limit):
            geometric = matched / len(voxels)
            sources = self._probability_sources.get(instance_id, [])
            probability_geometric = sum(item[2] / item[3] for item in sources) / len(voxels)
            visible_overlap, visible_count = self._visible_overlap(
                frame, observation, local_support[instance_id]
            )
            geometry_score = probability_geometric if self.association_mode == "probabilistic" else geometric
            score = geometry_score if visible_overlap is None else (
                0.5 * geometry_score + 0.5 * visible_overlap
            )
            ranked.append({
                "instance_id": instance_id,
                "geometric_coverage": round(geometric, 6),
                "visible_overlap": None if visible_overlap is None else round(visible_overlap, 6),
                "visible_support_points": visible_count,
                "prior_observation_count": len(self.instances[instance_id].observation_ids),
                "score": round(score, 6),
            })
            if self.identity_evidence is not None:
                source_votes = [item[2] for item in sources]
                ranked[-1].update({
                    "probabilistic_geometric_consistency": round(probability_geometric, 6),
                    "prior_support_frame_count": len(self.identity_evidence.instance_frame_references[instance_id]),
                    "source_frame_votes_min": min(source_votes),
                    "source_frame_votes_max": max(source_votes),
                    "source_single_frame_fraction": round(sum(v == 1 for v in source_votes) / len(source_votes), 6),
                    "visibility_fallback": visible_overlap is None,
                })
        ranked.sort(key=lambda item: (-item["score"], -item["prior_observation_count"], item["instance_id"]))
        if self.identity_evidence is not None:
            scores = np.asarray([item["score"] for item in ranked] + [self.null_candidate_score])
            # Relative diagnostic weights only; neither gating nor fusion uses q.
            weights = np.exp((scores - scores.max()) / self.relative_weight_temperature)
            weights /= weights.sum()
            self.candidate_weight_diagnostics[observation.observation_id] = {
                "candidate_relative_weights": [dict(instance_id=item["instance_id"], weight=float(weight))
                                               for item, weight in zip(ranked, weights[:-1])],
                "null_relative_weight": float(weights[-1]),
                "relative_weight_scope": "spatial_top8_before_output_truncation",
            }
            self.candidate_support_traces[observation.observation_id] = [
                (item["instance_id"], voxels, self._probability_sources[item["instance_id"]]) for item in ranked
            ]
        if output_limit is not None:
            return ranked[:output_limit]
        return ranked[:8] if self.improved_association else ranked[:3]

    def _commit(
        self, instance_id: int, observation: RawInstanceObservation,
        voxels: tuple[tuple[int, int, int], ...],
    ) -> None:
        validate_instance_id(instance_id)
        if observation.observation_id in self.observation_support:
            raise ValueError("Observation committed twice")
        voxels = tuple(dict.fromkeys(voxels))
        entry = ObservationSupport(observation.observation_id, observation.frame_id,
                                   observation.source_mask_sha256, int(instance_id), voxels)
        if self.identity_evidence is None:
            self.observation_support[observation.observation_id] = entry
        else:
            self.identity_evidence.add(entry)
        self.next_instance_id = max(self.next_instance_id, instance_id + 1)
        if instance_id not in self.instances:
            self.instances[instance_id] = InstanceState(
                instance_id=instance_id, first_frame_id=observation.frame_id
            )
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
        self._validate_frame(frame, observations)
        previous_version = self.map_version
        self._last_scoring_map_version = previous_version
        self.candidate_support_traces = {}
        self.candidate_weight_diagnostics = {}
        if self.improved_association:
            decisions = self._process_frame_improved(frame, observations)
            return self._finish_frame(decisions, previous_version)
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
        return self._finish_frame(decisions, previous_version)


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

    def _validate_frame(self, frame, observations):
        if frame.frame_id <= self.last_processed_frame_id:
            raise ValueError("Frames must be processed once in increasing order")
        if len({item.mask_local_id for item in observations}) != len(observations):
            raise ValueError("Duplicate local mask IDs in one frame")
        ids = [item.observation_id for item in observations]
        if len(set(ids)) != len(ids) or any(key in self.observation_support for key in ids):
            raise ValueError("Duplicate or previously committed observation ID")
        if any(item.frame_id != frame.frame_id for item in observations):
            raise ValueError("Observation belongs to another frame")

    def _finish_frame(self, decisions, previous_version):
        self.map_version += 1
        if self.identity_evidence is not None:
            for decision in decisions:
                observation_id = decision["observation_id"]
                decision.update(self.candidate_weight_diagnostics.get(observation_id, {
                    "candidate_relative_weights": [], "null_relative_weight": 1.0,
                    "relative_weight_scope": "spatial_top8_before_output_truncation",
                }))
                decision.update({"association_mode": self.association_mode,
                                 "map_version_before": previous_version,
                                 "map_version_after": self.map_version,
                                 "assignment_version": 0})
        return decisions

    def _rebuild_instance_index(self):
        instances, index = {}, defaultdict(set)
        for obs_id, entry in self.observation_support.items():
            instance_id = entry.instance_id
            if instance_id not in instances:
                instances[instance_id] = InstanceState(instance_id, first_frame_id=entry.frame_id)
            state = instances[instance_id]
            state.observation_ids.append(obs_id)
            state.first_frame_id = min(state.first_frame_id, entry.frame_id)
            state.last_frame_id = max(state.last_frame_id, entry.frame_id)
            for voxel in entry.voxels:
                if voxel not in state.voxel_set:
                    state.voxel_set.add(voxel)
                    state.voxel_list.append(voxel)
                    index[voxel].add(instance_id)
        self.instances, self.voxel_to_instances = instances, index

    def reassign_observations(self, assignment_updates: dict[str, int],
                              reason: str = "explicit_identity_revision") -> None:
        """Withdraw/reapply affected votes, then rebuild the binary spatial index.

        This explicit operation does not replay subsequent association decisions.
        P0 must be rematerialized with the revised assignment table before export.
        Source masks and projected voxels remain unchanged.
        """
        if not set(assignment_updates).issubset(self.observation_support):
            raise ValueError("Cannot reassign an unknown observation")
        for instance_id in assignment_updates.values():
            validate_instance_id(instance_id)
        if self.identity_evidence is None:
            changes = [(old, replace(old, instance_id=int(assignment_updates[obs_id]),
                                     assignment_version=old.assignment_version + 1))
                       for obs_id, old in self.observation_support.items()
                       if obs_id in assignment_updates and assignment_updates[obs_id] != old.instance_id]
            for old, new in changes:
                self.observation_support[old.observation_id] = new
        else:
            changes = self.identity_evidence.reassign(assignment_updates)
        if not changes:
            return
        previous_version = self.map_version
        self.map_version += 1
        for old, new in changes:
            self.next_instance_id = max(self.next_instance_id, new.instance_id + 1)
            self.revision_events.append({
                "event_id": len(self.revision_events) + 1, "type": "REASSIGN",
                "observation_id": old.observation_id, "frame_id": old.frame_id,
                "old_instance_id": old.instance_id, "new_instance_id": new.instance_id,
                "old_assignment_version": old.assignment_version,
                "new_assignment_version": new.assignment_version,
                "map_version_before": previous_version, "map_version_after": self.map_version,
                "reason": reason,
            })
        self._rebuild_instance_index()

    def checkpoint_parameters(self):
        return {name: getattr(self, name) for name in (
            "voxel_size_m", "max_points_per_observation", "max_visible_support_points",
            "min_geometric_coverage", "min_total_score", "min_visible_overlap",
            "ambiguity_margin", "depth_tolerance_m",
            "allow_multiple_observations_per_instance_per_frame", "improved_association",
            "valid_first_sampling", "association_mode", "relative_weight_temperature", "null_candidate_score",
        )}

    def save_checkpoint(self, path):
        path = Path(path)
        entries = list(self.observation_support.values())
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary, format_version=np.asarray([2], np.int32),
            observation_ids=np.asarray([entry.observation_id for entry in entries], dtype=str),
            frame_ids=np.asarray([entry.frame_id for entry in entries], np.int32),
            source_mask_sha256=np.asarray([entry.source_mask_sha256 for entry in entries], dtype=str),
            instance_ids=np.asarray([entry.instance_id for entry in entries], np.int32),
            assignment_versions=np.asarray([entry.assignment_version for entry in entries], np.int32),
            offsets=np.r_[0, np.cumsum([len(entry.voxels) for entry in entries], dtype=np.int64)],
            voxel_coordinates=np.asarray([voxel for entry in entries for voxel in entry.voxels], np.int32).reshape(-1, 3),
            voxel_size_m=np.asarray([self.voxel_size_m], np.float64),
            next_instance_id=np.asarray([self.next_instance_id], np.int64),
            last_processed_frame_id=np.asarray([self.last_processed_frame_id], np.int64),
            map_version=np.asarray([self.map_version], np.int64),
            parameters_json=np.asarray([json.dumps(self.checkpoint_parameters(), sort_keys=True)]),
            revision_events_json=np.asarray([json.dumps(self.revision_events, sort_keys=True)]),
        )
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["format_version"][0]) != 2:
                raise ValueError("Unsupported identity checkpoint format")
            result = cls(**json.loads(str(data["parameters_json"][0])))
            ids, offsets, coordinates = data["observation_ids"], data["offsets"], data["voxel_coordinates"]
            fields = ("frame_ids", "source_mask_sha256", "instance_ids", "assignment_versions")
            # NpzFile re-decompresses a member on every access. Load these arrays
            # once so restoring N observations does not repeatedly read N records.
            columns = {key: data[key] for key in fields}
            if (len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(coordinates)
                or np.any(np.diff(offsets) < 0) or any(len(columns[key]) != len(ids) for key in fields)):
                raise ValueError("Malformed identity checkpoint")
            for i, obs_id in enumerate(ids):
                entry = ObservationSupport(str(obs_id), int(columns["frame_ids"][i]),
                    str(columns["source_mask_sha256"][i]), int(columns["instance_ids"][i]),
                    tuple(tuple(int(v) for v in row) for row in coordinates[offsets[i]:offsets[i + 1]]),
                    int(columns["assignment_versions"][i]))
                if result.identity_evidence is None:
                    validate_instance_id(entry.instance_id)
                    if entry.observation_id in result.observation_support:
                        raise ValueError("Duplicate observation in checkpoint")
                    result.observation_support[entry.observation_id] = entry
                else:
                    result.identity_evidence.add(entry)
            result.next_instance_id = int(data["next_instance_id"][0])
            result.last_processed_frame_id = int(data["last_processed_frame_id"][0])
            result.map_version = int(data["map_version"][0])
            result.revision_events = json.loads(str(data["revision_events_json"][0]))
        used_ids = [entry.instance_id for entry in result.observation_support.values()]
        used_ids.extend(event[key] for event in result.revision_events for key in ("old_instance_id", "new_instance_id"))
        if result.next_instance_id <= max(used_ids, default=0):
            raise ValueError("Checkpoint would reuse a persistent instance ID")
        result._rebuild_instance_index()
        return result

    def save_identity_counts(self, path):
        if self.identity_evidence is None:
            raise ValueError("Identity evidence is disabled")
        self.identity_evidence.validate()
        np.savez_compressed(path, **self.identity_evidence.export_arrays(),
                            voxel_size_m=np.asarray([self.voxel_size_m], np.float64),
                            map_version=np.asarray([self.map_version], np.int64))

    def save_candidate_support_trace(self, path, frame_id):
        observation_ids, candidate_ids, lengths = [], [], []
        queries, sources, votes, totals, frames = [], [], [], [], []
        for obs_id, candidates in self.candidate_support_traces.items():
            for instance_id, voxels, rows in candidates:
                observation_ids.append(obs_id)
                candidate_ids.append(instance_id)
                lengths.append(len(rows))
                for query_index, source, count, total, unique_frames in rows:
                    queries.append(voxels[query_index])
                    sources.append(source)
                    votes.append(count)
                    totals.append(total)
                    frames.append(unique_frames)
        np.savez_compressed(path,
            observation_ids=np.asarray(observation_ids, dtype=str),
            candidate_ids=np.asarray(candidate_ids, np.int32),
            offsets=np.r_[0, np.cumsum(lengths, dtype=np.int64)],
            query_voxels=np.asarray(queries, np.int32).reshape(-1, 3),
            source_voxels=np.asarray(sources, np.int32).reshape(-1, 3),
            source_instance_frame_votes=np.asarray(votes, np.int32),
            source_total_frame_instance_votes=np.asarray(totals, np.int32),
            source_unique_support_frames=np.asarray(frames, np.int32),
            frame_id=np.asarray([frame_id], np.int32),
            map_version_queried=np.asarray([self._last_scoring_map_version], np.int64))
