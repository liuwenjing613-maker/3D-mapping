"""Stage 3 set arithmetic: stable masks separated in the same observed frame."""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from .two_audit_map_inputs import check_points
from .two_audit_matching import TrackSettings, camera_view, different_view, view_distance
from .two_audit_sources import canonical_hash


@dataclass(frozen=True)
class AnomalySettings:
    min_target_points: int = 25
    min_joint_view_groups: int = 2
    view_translation_m: float = 0.1
    view_axis_degrees: float = 10.

    def __post_init__(self):
        TrackSettings(min_common_points=self.min_target_points,
                      min_view_groups=self.min_joint_view_groups,
                      view_translation_m=self.view_translation_m,
                      view_axis_degrees=self.view_axis_degrees)

    def record(self):
        return dict(asdict(self), require_same_frame_distinct_raw_masks=True,
                    assigned_stable_members_only=True, coverage_ratio_threshold=None,
                    label_source="pre_repair_native", map_modified=False,
                    repair_triggered=False, decision="review_only")


def histogram(ids):
    keys, counts = np.unique(ids, return_counts=True)
    return {int(key): int(count) for key, count in zip(keys, counts)}


def add_joint_view(groups, frame_id, observation_keys, pose, settings):
    position, axis = camera_view(pose)
    compatible = [g for g in groups
                  if not different_view(*view_distance(position, axis, g), settings)]
    if compatible:
        group = compatible[0]
    else:
        group = dict(view_group_id=len(groups)+1, representative_frame_id=frame_id,
                     representative_observation_keys=observation_keys,
                     position_m=position.tolist(), optical_axis=axis.tolist(),
                     member_frame_ids=[])
        groups.append(group)
    group["member_frame_ids"].append(frame_id)
    return group["view_group_id"]


class StableMapAudit:
    def __init__(self, ids, stable_tracks, geometry_id, native_sha256, settings=None):
        self.ids = np.asarray(ids)
        if self.ids.ndim != 1 or self.ids.dtype.kind not in "iu" or not len(self.ids):
            raise ValueError("Invalid native instance labels")
        self.settings = settings or AnomalySettings()
        self.geometry_id, self.native_sha256 = geometry_id, native_sha256
        self.tracks = {t["track_id"]: t for t in stable_tracks}
        if len(self.tracks) != len(stable_tracks) or any(t["status"] != "STABLE" for t in stable_tracks):
            raise ValueError("Only distinct stable tracks are eligible")
        self.members = {}
        for track in stable_tracks:
            if track["geometry_id"] != geometry_id:
                raise ValueError("Stable tracks use another geometry")
            for member in track["members"]:
                key = member["observation_key"]
                if key in self.members:
                    raise ValueError("Original observation assigned to more than one track")
                self.members[key] = (track["track_id"], member)
        self.seen, self.last_frame = set(), -1
        self.coverage, self.pair_checks, self.pairs = [], [], {}
        self.support_parts = defaultdict(list)
        self.stats = Counter()
        self.instance_sizes = histogram(self.ids)

    def consume_frame(self, frame_id, visible, observations, pose, *, geometry_id):
        if geometry_id != self.geometry_id or frame_id <= self.last_frame:
            raise ValueError("Geometry mismatch or non-increasing frame order")
        check_points(visible, len(self.ids))
        camera_view(pose)
        # Each observation is (stable_track_id, raw_key, raw_local_id, sorted support).
        observations = sorted(observations, key=lambda row: row[0])
        if len({r[0] for r in observations}) != len(observations):
            raise ValueError("Multiple same-frame members for one stable track")
        if len({r[2] for r in observations}) != len(observations):
            raise ValueError("Same original mask reused in a joint-view claim")
        all_points = []
        visible_counts = histogram(self.ids[visible])
        by_instance = defaultdict(list)
        for tid, key, mid, support in observations:
            if key not in self.members or key in self.seen:
                raise ValueError("Unassigned, deferred, provisional or duplicate observation")
            owner, member = self.members[key]
            if (tid != owner or member["frame_id"] != frame_id
                    or member["mask_local_id"] != mid or member["surface_support_count"] != len(support)):
                raise ValueError("Observation differs from its confirmed stable membership")
            check_points(support, len(self.ids))
            if len(np.intersect1d(support, visible, assume_unique=True)) != len(support):
                raise ValueError("Joint evidence includes invisible support")
            all_points.append(support)
            counts = histogram(self.ids[support])
            row = dict(track_id=tid, observation_key=key, frame_id=frame_id,
                       mask_local_id=mid, track_view_group_id=member["view_group_id"],
                       support_points=len(support),
                       assigned_points=sum(n for iid, n in counts.items() if iid > 0),
                       unassigned_points=sum(n for iid, n in counts.items() if iid <= 0),
                       instance_coverage=[dict(instance_id=iid, points=count,
                           fraction_of_mask_support=count/len(support),
                           visible_instance_points=visible_counts[iid],
                           fraction_of_visible_instance=count/visible_counts[iid])
                           for iid, count in counts.items()])
            self.coverage.append(row)
            for iid, count in counts.items():
                if iid > 0:
                    by_instance[iid].append((tid, key, mid, len(support), count))
            self.support_parts[tid].append(support.copy())
            self.seen.add(key)
        if all_points:
            flat = np.concatenate(all_points)
            if len(np.unique(flat)) != len(flat):
                raise ValueError("Distinct same-frame raw masks have overlapping surface support")
        for iid in sorted(by_instance):
            for a, b in combinations(by_instance[iid], 2):
                enough = min(a[4], b[4]) >= self.settings.min_target_points
                key = (iid, a[0], b[0])
                pair_key = canonical_hash(dict(geometry_id=self.geometry_id,
                    native_surface_sha256=self.native_sha256, target_instance_id=iid,
                    track_keys=[self.tracks[a[0]]["track_key"], self.tracks[b[0]]["track_key"]]))
                event = dict(pair_key=pair_key, target_instance_id=iid,
                             track_ids=[a[0], b[0]], frame_id=frame_id,
                             observation_keys=[a[1], b[1]], mask_local_ids=[a[2], b[2]],
                             mask_support_points=[a[3], b[3]],
                             target_support_points=[a[4], b[4]],
                             target_support_fractions=[a[4]/a[3], b[4]/b[3]],
                             visible_target_points=visible_counts[iid],
                             target_visible_fractions=[a[4]/visible_counts[iid], b[4]/visible_counts[iid]],
                             same_frame_support_disjoint=True,
                             state="JOINT_SEPARATION_SUPPORT" if enough else "INSUFFICIENT_TARGET_SUPPORT")
                if enough:
                    pair = self.pairs.setdefault(key, dict(pair_key=pair_key, target_instance_id=iid,
                             track_ids=[a[0], b[0]], joint_observations=[], joint_view_groups=[]))
                    event["joint_view_group_id"] = add_joint_view(
                        pair["joint_view_groups"], frame_id, event["observation_keys"], pose, self.settings)
                    pair["joint_observations"].append(event)
                self.pair_checks.append(event)
                self.stats[event["state"]] += 1
        self.stats["processed_frames"] += 1
        self.last_frame = frame_id

    def finish(self):
        if self.seen != set(self.members):
            raise ValueError("Stable members were lost during audit")
        supports = {tid: np.unique(np.concatenate(self.support_parts[tid]))
                    for tid in self.tracks}
        observed = defaultdict(list)
        for row in self.coverage:
            observed[row["track_id"]].append(row)
        track_coverage = []
        for tid, track in self.tracks.items():
            support = supports[tid]
            counts = histogram(self.ids[support])
            links = []
            for iid, count in counts.items():
                members = [(row, part) for row in observed[tid] for part in row["instance_coverage"]
                           if part["instance_id"] == iid]
                links.append(dict(instance_id=iid, unique_points=count,
                    fraction_of_track_union=count/len(support),
                    instance_surface_points=self.instance_sizes[iid],
                    fraction_of_instance_surface=count/self.instance_sizes[iid],
                    observation_count=len(members),
                    track_view_group_count=len({row["track_view_group_id"] for row, _ in members}),
                    observations_with_at_least_25_points=sum(
                        part["points"] >= self.settings.min_target_points for _, part in members)))
            track_coverage.append(dict(track_id=tid, track_key=track["track_key"],
                observation_count=len(track["members"]), unique_support_points=len(support),
                unassigned_unique_points=sum(n for iid, n in counts.items() if iid <= 0),
                instance_coverage=links))
        pairs = []
        for _, pair in sorted(self.pairs.items()):
            view_count = len(pair["joint_view_groups"])
            eligible = view_count >= self.settings.min_joint_view_groups
            pair.update(status="POTENTIAL_OVERMERGE" if eligible else "INSUFFICIENT_INDEPENDENT_VIEWS",
                        joint_frame_count=len(pair["joint_observations"]),
                        independent_joint_view_groups=view_count,
                        review_only=True, repair_triggered=False)
            if eligible:
                group_frame = pair["joint_view_groups"][self.settings.min_joint_view_groups-1]["representative_frame_id"]
                pair["evidence_available_frame_on_this_fixed_geometry"] = max(
                    group_frame, *(self.tracks[tid]["first_stable_frame_id"] for tid in pair["track_ids"]))
            pairs.append(pair)
        self.stats.update(stable_tracks=len(self.tracks), stable_observations=len(self.coverage),
                          observed_pair_instance_comparisons=len(self.pair_checks),
                          pairs_with_joint_support=len(pairs),
                          potential_overmerge_pairs=sum(p["status"] == "POTENTIAL_OVERMERGE" for p in pairs),
                          insufficient_joint_view_pairs=sum(p["status"] != "POTENTIAL_OVERMERGE" for p in pairs))
        return dict(observation_coverage=self.coverage, track_coverage=track_coverage,
                    pair_checks=self.pair_checks, candidate_pairs=pairs,
                    supports=supports, totals=dict(self.stats))


def add_roi_comparison(result, inputs, stable_tracks):
    """No coverage threshold: preserve exact core/context overlap at both scales."""
    tracks = {t["track_id"]: t for t in stable_tracks}
    supports = result["supports"]
    footprint_cache = {}
    for pair in result["candidate_pairs"]:
        tids = tuple(pair["track_ids"])
        if tids not in footprint_cache:
            full = np.union1d(supports[tids[0]], supports[tids[1]])
            footprint_cache[tids] = (full, inputs.roi_overlap(full, per_component=True))
        full, full_overlap = footprint_cache[tids]
        target = full[inputs.ids[full] == pair["target_instance_id"]]
        pair["roi_comparison"] = dict(
            basis="union_of_all_assigned_members_of_both_stable_tracks",
            full_mask_footprint=full_overlap,
            target_instance_footprint=inputs.roi_overlap(target, per_component=True),
            side_full_mask_footprints=[inputs.roi_overlap(supports[tid]) for tid in tids],
            outside_all_selected_roi_context=full_overlap["context_points"] == 0,
            overlap_does_not_prove_error_detection=True,
            actual_error_status="NOT_HUMAN_VERIFIED")
        pair["track_representatives"] = [
            dict(track_id=tid, observation_key=tracks[tid]["representative_observation_key"],
                 frame_id=tracks[tid]["representative_frame_id"]) for tid in tids]
    by_instance = defaultdict(list)
    for pair in result["candidate_pairs"]:
        if pair["status"] == "POTENTIAL_OVERMERGE":
            by_instance[pair["target_instance_id"]].append(pair)
    anomalies = []
    for iid, pairs in sorted(by_instance.items()):
        pair_keys = [p["pair_key"] for p in pairs]
        anomalies.append(dict(target_instance_id=iid, status="POTENTIAL_OVERMERGE",
            review_status="UNREVIEWED", native_instance_surface_points=inputs.instance_sizes[iid],
            pair_count=len(pairs), pair_keys=pair_keys,
            track_ids=sorted({tid for p in pairs for tid in p["track_ids"]}),
            pairs_outside_all_roi_context=sum(p["roi_comparison"]["outside_all_selected_roi_context"] for p in pairs),
            repair_triggered=False, confirmed_error=False))
    result["instance_anomalies"] = anomalies
    eligible = [p for p in result["candidate_pairs"] if p["status"] == "POTENTIAL_OVERMERGE"]
    result["totals"].update(potential_overmerge_instances=len(anomalies),
        potential_pairs_outside_all_roi_context=sum(p["roi_comparison"]["outside_all_selected_roi_context"] for p in eligible),
        potential_instances_with_outside_roi_pair=sum(a["pairs_outside_all_roi_context"] > 0 for a in anomalies),
        roi_relation_counts=dict(Counter(p["roi_comparison"]["full_mask_footprint"]["relation"] for p in eligible)))
    return result
