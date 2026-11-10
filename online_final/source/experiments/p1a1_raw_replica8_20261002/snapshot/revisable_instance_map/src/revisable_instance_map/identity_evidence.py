"""Rebuildable identity counts: one vote per (frame, voxel, instance).

Observation support is the authority. Reference counts allow one overlapping
region to be withdrawn without removing a vote still supplied by another.
Probabilities are empirical support ratios, not calibrated identity posteriors.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ObservationSupport:
    observation_id: str
    frame_id: int
    source_mask_sha256: str
    instance_id: int
    voxels: tuple[tuple[int, int, int], ...]
    assignment_version: int = 0


def validate_instance_id(instance_id):
    if (isinstance(instance_id, bool) or not isinstance(instance_id, Integral)
        or instance_id <= 0 or instance_id > np.iinfo(np.int32).max):
        raise ValueError("Instance IDs must be positive int32 integers")


class FrameIdentityEvidence:
    def __init__(self):
        self.entries: dict[str, ObservationSupport] = {}
        self.frame_voxel_references: dict[tuple, int] = {}
        self.voxel_counts: dict[tuple, Counter] = defaultdict(Counter)
        self.voxel_totals: Counter = Counter()
        self.voxel_frame_references: dict[tuple, Counter] = defaultdict(Counter)
        self.instance_frame_references: dict[int, Counter] = defaultdict(Counter)

    def _add_votes(self, entry):
        if not entry.voxels:
            return
        self.instance_frame_references[entry.instance_id][entry.frame_id] += 1
        for voxel in entry.voxels:
            key = (entry.frame_id, voxel, entry.instance_id)
            previous = self.frame_voxel_references.get(key, 0)
            self.frame_voxel_references[key] = previous + 1
            self.voxel_frame_references[voxel][entry.frame_id] += 1
            if previous == 0:
                self.voxel_counts[voxel][entry.instance_id] += 1
                self.voxel_totals[voxel] += 1

    def _remove_votes(self, entry):
        if not entry.voxels:
            return
        frames = self.instance_frame_references[entry.instance_id]
        frames[entry.frame_id] -= 1
        if frames[entry.frame_id] == 0:
            del frames[entry.frame_id]
        if not frames:
            del self.instance_frame_references[entry.instance_id]
        for voxel in entry.voxels:
            key = (entry.frame_id, voxel, entry.instance_id)
            previous = self.frame_voxel_references[key]
            if previous <= 0:
                raise RuntimeError("Invalid identity reference count")
            if previous > 1:
                self.frame_voxel_references[key] = previous - 1
            else:
                del self.frame_voxel_references[key]
                counts = self.voxel_counts[voxel]
                counts[entry.instance_id] -= 1
                if counts[entry.instance_id] == 0:
                    del counts[entry.instance_id]
                self.voxel_totals[voxel] -= 1
                if not counts:
                    del self.voxel_counts[voxel]
                    del self.voxel_totals[voxel]
            frames = self.voxel_frame_references[voxel]
            frames[entry.frame_id] -= 1
            if frames[entry.frame_id] == 0:
                del frames[entry.frame_id]
            if not frames:
                del self.voxel_frame_references[voxel]

    def add(self, entry):
        validate_instance_id(entry.instance_id)
        if entry.observation_id in self.entries:
            raise ValueError("Observation committed twice")
        if entry.frame_id < 0 or entry.assignment_version < 0:
            raise ValueError("Invalid support frame or assignment version")
        # A sampled voxel contributes once even if callers provide repeated points.
        entry = replace(entry, voxels=tuple(dict.fromkeys(entry.voxels)))
        self.entries[entry.observation_id] = entry
        self._add_votes(entry)
        return entry

    def reassign(self, updates):
        if not set(updates).issubset(self.entries):
            raise ValueError("Cannot reassign an unknown observation")
        for instance_id in updates.values():
            validate_instance_id(instance_id)
        changes = []
        # Validate the entire transaction before withdrawing anything.
        for observation_id, instance_id in updates.items():
            old = self.entries[observation_id]
            if old.instance_id == instance_id:
                continue
            new = replace(old, instance_id=int(instance_id),
                          assignment_version=old.assignment_version + 1)
            changes.append((old, new))
        for old, new in changes:
            self._remove_votes(old)
            self._add_votes(new)
            self.entries[old.observation_id] = new
        return changes

    @classmethod
    def rebuild(cls, entries):
        result = cls()
        for entry in entries:
            result.add(entry)
        return result

    def probability(self, voxel, instance_id):
        total = self.voxel_totals.get(voxel, 0)
        return self.voxel_counts.get(voxel, {}).get(instance_id, 0) / total if total else 0.0

    def export_arrays(self):
        coordinates = sorted(self.voxel_counts)
        lengths, instance_ids, votes = [], [], []
        for voxel in coordinates:
            items = sorted(self.voxel_counts[voxel].items())
            lengths.append(len(items))
            instance_ids.extend(instance_id for instance_id, _ in items)
            votes.extend(count for _, count in items)
        return {
            "voxel_coordinates": np.asarray(coordinates, np.int32).reshape(-1, 3),
            "offsets": np.r_[0, np.cumsum(lengths, dtype=np.int64)],
            "instance_ids": np.asarray(instance_ids, np.int32),
            "frame_votes": np.asarray(votes, np.int32),
            "total_frame_instance_votes": np.asarray([self.voxel_totals[v] for v in coordinates], np.int32),
            "unique_support_frames": np.asarray([len(self.voxel_frame_references[v]) for v in coordinates], np.int32),
        }

    def validate(self):
        """Independently rebuild from sources; intended for checkpoints/audits."""
        rebuilt = self.rebuild(self.entries.values())
        for name in ("frame_voxel_references", "voxel_counts", "voxel_totals",
                     "voxel_frame_references", "instance_frame_references"):
            if getattr(self, name) != getattr(rebuilt, name):
                raise ValueError(f"Identity cache differs from its ledger: {name}")
        return True
