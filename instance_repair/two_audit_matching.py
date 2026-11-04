"""Fixed-representative, geometry-only association of original mask observations."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import chain
import math

import numpy as np

from .two_audit_sources import canonical_hash


@dataclass(frozen=True)
class TrackSettings:
    min_common_points: int = 25
    min_covisible_iou: float = 0.55
    min_observations: int = 3
    min_view_groups: int = 2
    view_translation_m: float = 0.1
    view_axis_degrees: float = 10.0

    def __post_init__(self):
        for value in (self.min_common_points, self.min_observations, self.min_view_groups):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("Counts must be integers")
        if (self.min_common_points < 1 or self.min_observations < 1
                or self.min_view_groups < 1 or not 0 < self.min_covisible_iou <= 1
                or not math.isfinite(self.view_translation_m) or self.view_translation_m <= 0
                or not 0 < self.view_axis_degrees <= 180):
            raise ValueError("Invalid stable-track settings")

    def record(self):
        return dict(asdict(self), representative="first_eligible_observation_fixed",
                    view_grouping="greedy_fixed_view_representatives",
                    within_frame="simultaneous_unique_assignment",
                    ambiguity="defer_without_assignment",
                    map_instance_ids_used=False)


def intersection_count(a, b):
    """Exact intersection size for sorted unique integer arrays."""
    if not len(a) or not len(b):
        return 0
    if len(a) > len(b):
        a, b = b, a
    locations = np.searchsorted(b, a)
    in_range = locations < len(b)
    return int(np.count_nonzero(b[locations[in_range]] == a[in_range]))


def pair_evidence(a, visible_a, b, visible_b, settings, *,
                  shared=None, common_a=None, common_b=None):
    """S_a ∩ V_b and S_b ∩ V_a; insufficient visibility is never disagreement."""
    shared = intersection_count(a, b) if shared is None else int(shared)
    ca = intersection_count(a, visible_b) if common_a is None else int(common_a)
    cb = intersection_count(b, visible_a) if common_b is None else int(common_b)
    if shared > min(ca, cb) or ca > len(a) or cb > len(b):
        raise ValueError("Invalid intersection counts/visibility relationship")
    union = ca + cb - shared
    enough = ca >= settings.min_common_points and cb >= settings.min_common_points
    iou = float(shared / union) if enough else None
    state = ("INSUFFICIENT_COMMON_SUPPORT" if not enough else
             "MATCH" if iou >= settings.min_covisible_iou else "BELOW_IOU")
    return dict(common_a=ca, common_b=cb, intersection=shared, union=union,
                covisible_iou=iou, common_fraction_a=float(ca / len(a)) if len(a) else 0.,
                common_fraction_b=float(cb / len(b)) if len(b) else 0., state=state)


def camera_view(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Invalid camera pose")
    axis = pose[:3, 2]
    norm = np.linalg.norm(axis)
    if norm <= 0:
        raise ValueError("Camera optical axis has zero length")
    return pose[:3, 3].copy(), axis / norm


def view_distance(position, axis, representative):
    distance = float(np.linalg.norm(position - representative["position_m"]))
    cosine = float(np.clip(np.dot(axis, representative["optical_axis"]), -1., 1.))
    angle = float(np.degrees(np.arccos(cosine)))
    return distance, angle


def different_view(distance, angle, settings):
    # The tiny slack only handles floating-point equality at the approved limits.
    return (distance >= settings.view_translation_m - 1e-12
            or angle >= settings.view_axis_degrees - 1e-9)


@dataclass(frozen=True)
class Observation:
    key: str
    mask_local_id: int
    support: np.ndarray


class Track:
    def __init__(self, index, observation, frame_id, pose, geometry_id, settings):
        self.index = index
        self.track_id = f"T{index + 1:06d}"
        self.anchor = Observation(observation.key, observation.mask_local_id,
                                  observation.support.copy())
        self.anchor.support.setflags(write=False)
        self.anchor_frame = frame_id
        self.geometry_id, self.settings = geometry_id, settings
        self.track_key = canonical_hash(dict(geometry_id=geometry_id,
                                             representative_key=observation.key,
                                             settings=settings.record()))
        self.members, self.view_groups, self.member_frames = [], [], set()
        self.first_stable_frame_id = None
        self.admit(observation, frame_id, pose, evidence=None)

    @property
    def stable(self):
        return (len(self.members) >= self.settings.min_observations
                and len(self.view_groups) >= self.settings.min_view_groups)

    def admit(self, observation, frame_id, pose, evidence):
        if frame_id in self.member_frames:
            raise ValueError("A track may contain at most one observation per frame")
        if evidence is not None and evidence["state"] != "MATCH":
            raise ValueError("Only a match to the fixed representative may be admitted")
        position, axis = camera_view(pose)
        chosen, distance, angle = None, 0., 0.
        for group in self.view_groups:
            distance, angle = view_distance(position, axis, group)
            if not different_view(distance, angle, self.settings):
                chosen = group
                break
        if chosen is None:
            chosen = dict(view_group_id=len(self.view_groups) + 1,
                          representative_frame_id=frame_id,
                          representative_observation_key=observation.key,
                          position_m=position.tolist(), optical_axis=axis.tolist(),
                          member_frame_ids=[])
            self.view_groups.append(chosen)
            distance = angle = 0.
        chosen["member_frame_ids"].append(frame_id)
        self.member_frames.add(frame_id)
        self.members.append(dict(observation_key=observation.key, frame_id=frame_id,
                                 mask_local_id=int(observation.mask_local_id),
                                 surface_support_count=len(observation.support),
                                 view_group_id=chosen["view_group_id"],
                                 distance_to_view_representative_m=distance,
                                 axis_angle_to_view_representative_deg=angle,
                                 fixed_representative_evidence=evidence))
        if self.stable and self.first_stable_frame_id is None:
            self.first_stable_frame_id = frame_id

    def record(self):
        scores = [m["fixed_representative_evidence"]["covisible_iou"]
                  for m in self.members[1:]]
        return dict(track_id=self.track_id, track_key=self.track_key,
                    geometry_id=self.geometry_id,
                    status="STABLE" if self.stable else "PROVISIONAL",
                    representative_observation_key=self.anchor.key,
                    representative_frame_id=self.anchor_frame,
                    representative_mask_local_id=int(self.anchor.mask_local_id),
                    representative_support_count=len(self.anchor.support),
                    observation_count=len(self.members), independent_view_groups=len(self.view_groups),
                    first_stable_frame_id=self.first_stable_frame_id,
                    minimum_admitted_covisible_iou=min(scores) if scores else None,
                    members=self.members, view_groups=self.view_groups)


class TrackBuilder:
    """Exact positive-intersection index; no spatial approximation or map IDs."""
    def __init__(self, geometry_id, surface_count, settings=None):
        if surface_count <= 0 or not geometry_id:
            raise ValueError("An explicit nonempty geometry index space is required")
        self.geometry_id, self.surface_count = geometry_id, int(surface_count)
        self.settings = settings or TrackSettings()
        self.postings = [None] * self.surface_count
        self.tracks, self.anchor_visibility = [], {}
        self.seen_observations = set()
        self.last_frame = -1
        self.stats = Counter()

    def candidate_intersections(self, support):
        # A positive IoU match must share at least one exact surface index.
        # No ID-based or bounding-box pruning is used.
        return Counter(chain.from_iterable(self.postings[int(point)] or ()
                                           for point in support))

    def _check_indices(self, array):
        a = np.asarray(array)
        if (a.ndim != 1 or a.dtype.kind not in "iu"
                or (len(a) and (a[0] < 0 or a[-1] >= self.surface_count
                                or np.any(np.diff(a.astype(np.int64)) <= 0)))):
            raise ValueError("Support/visibility must be sorted unique indices in this geometry")

    def process_frame(self, frame_id, visible, observations, pose, *, geometry_id,
                      evidence_sink=None):
        if geometry_id != self.geometry_id:
            raise ValueError("Cannot mix geometry index spaces")
        if frame_id <= self.last_frame:
            raise ValueError("Frames must arrive once in strictly increasing order")
        camera_view(pose)
        self._check_indices(visible)
        observations = sorted(observations, key=lambda x: (x.mask_local_id, x.key))
        if len({x.mask_local_id for x in observations}) != len(observations):
            raise ValueError("Duplicate original local mask ID in a frame")
        keys = [x.key for x in observations]
        if len(set(keys)) != len(keys) or set(keys) & self.seen_observations:
            raise ValueError("An original observation may be consumed only once")
        for item in observations:
            if not item.key or item.mask_local_id <= 0:
                raise ValueError("Invalid original observation identity")
            self._check_indices(item.support)
            if intersection_count(item.support, visible) != len(item.support):
                raise ValueError("Mask support is not a subset of valid visibility")

        decisions, proposals, match_details, common_a_cache = [], defaultdict(list), {}, {}
        old_track_count = len(self.tracks)
        for oi, item in enumerate(observations):
            row = dict(observation_key=item.key, frame_id=int(frame_id),
                       mask_local_id=int(item.mask_local_id), support_count=len(item.support),
                       geometry_id=self.geometry_id, track_id=None,
                       matched_track_ids=[], shared_point_candidates=0,
                       insufficient_common_comparisons=0, sufficient_common_comparisons=0)
            if len(item.support) < self.settings.min_common_points:
                row.update(status="DEFERRED_INSUFFICIENT_SUPPORT",
                           evidence_state="INSUFFICIENT_TOTAL_SUPPORT")
                decisions.append(row)
                continue
            shared_counts = self.candidate_intersections(item.support)
            row["shared_point_candidates"] = len(shared_counts)
            common_b_cache, matches = {}, []
            for ti in sorted(shared_counts):
                if ti >= old_track_count:
                    raise AssertionError("Same-frame anchors must not enter candidate matching")
                track = self.tracks[ti]
                if ti not in common_a_cache:
                    common_a_cache[ti] = intersection_count(track.anchor.support, visible)
                af = track.anchor_frame
                if af not in common_b_cache:
                    common_b_cache[af] = intersection_count(item.support, self.anchor_visibility[af])
                evidence = pair_evidence(
                    track.anchor.support, self.anchor_visibility[af],
                    item.support, visible, self.settings, shared=shared_counts[ti],
                    common_a=common_a_cache[ti], common_b=common_b_cache[af])
                self.stats["comparisons"] += 1
                self.stats["pair_" + evidence["state"]] += 1
                if evidence["state"] == "INSUFFICIENT_COMMON_SUPPORT":
                    row["insufficient_common_comparisons"] += 1
                else:
                    row["sufficient_common_comparisons"] += 1
                if evidence_sink is not None:
                    evidence_sink(dict(evidence, observation_key=item.key, frame_id=int(frame_id),
                                       track_id=track.track_id,
                                       representative_observation_key=track.anchor.key,
                                       representative_frame_id=track.anchor_frame))
                if evidence["state"] == "MATCH":
                    matches.append(ti)
                    match_details[(oi, ti)] = evidence
            row["matched_track_ids"] = [self.tracks[ti].track_id for ti in matches]
            if len(matches) > 1:
                row.update(status="DEFERRED_AMBIGUOUS", evidence_state="MULTIPLE_MATCHES")
            elif len(matches) == 1:
                row.update(status="PROPOSED_MATCH", evidence_state="UNIQUE_MATCH")
                proposals[matches[0]].append(oi)
            else:
                state = ("INSUFFICIENT_COMMON_SUPPORT"
                         if row["insufficient_common_comparisons"] and
                         not row["sufficient_common_comparisons"] else
                         "NO_ACCEPTED_MATCH" if shared_counts else
                         "NO_SHARED_SUPPORT_CANDIDATES")
                row.update(status="NEW_TRACK", evidence_state=state)
            decisions.append(row)

        # Resolve the entire frame together; mask enumeration cannot win a tie.
        for ti, candidates in proposals.items():
            if len(candidates) != 1:
                for oi in candidates:
                    decisions[oi].update(status="DEFERRED_SAME_FRAME_CONFLICT",
                                         evidence_state="MULTIPLE_OBSERVATIONS_FOR_ONE_TRACK")
                continue
            oi = candidates[0]
            self.tracks[ti].admit(observations[oi], frame_id, pose, match_details[(oi, ti)])
            decisions[oi].update(status="MATCHED_TRACK", track_id=self.tracks[ti].track_id)
        for oi, row in enumerate(decisions):
            if row["status"] != "NEW_TRACK":
                continue
            if frame_id not in self.anchor_visibility:
                self.anchor_visibility[frame_id] = visible.copy()
                self.anchor_visibility[frame_id].setflags(write=False)
            track = Track(len(self.tracks), observations[oi], frame_id, pose,
                          self.geometry_id, self.settings)
            self.tracks.append(track)
            for point in track.anchor.support:
                posting = self.postings[int(point)]
                if posting is None:
                    self.postings[int(point)] = [track.index]
                else:
                    posting.append(track.index)
            row["track_id"] = track.track_id
        for row in decisions:
            self.stats[row["status"]] += 1
        self.stats["frames"] += 1
        self.stats["observations"] += len(observations)
        self.seen_observations.update(keys)
        self.last_frame = frame_id
        return decisions

    def records(self):
        return [track.record() for track in self.tracks]
