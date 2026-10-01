"""Full immutable sources, versioned assignments and append-only dual-time events."""
from dataclasses import asdict, dataclass, replace
from functools import cached_property
import hashlib
import json

from .identity_evidence import validate_instance_id
from .observations import RawInstanceObservation

ACCEPTED = "ACCEPTED"
PENDING_BIND = "PENDING_BIND"
PENDING_BIRTH = "PENDING_BIRTH"
UNPROJECTABLE = "UNPROJECTABLE"
PENDING = (PENDING_BIND, PENDING_BIRTH)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RawSupportRecord:
    observation: RawInstanceObservation
    voxels: tuple
    depth_pose_intrinsics_hash: str
    sampling_config_hash: str
    support_hash: str

    @property
    def observation_id(self):
        return self.observation.observation_id

    @property
    def source_frame_id(self):
        return self.observation.frame_id

    @cached_property
    def voxel_bounds(self):
        if not self.voxels:
            return None
        columns = tuple(zip(*self.voxels))
        return tuple(min(v) for v in columns), tuple(max(v) for v in columns)


@dataclass(frozen=True)
class AssignmentRecord:
    observation_id: str
    status: str
    persistent_instance_id: int | None
    packet_id: str | None
    assignment_version: int
    first_decision_frame_id: int
    last_decision_frame_id: int
    max_evidence_frame_id: int
    read_map_version: int
    active_since_map_version: int | None
    candidate_records: tuple
    reason: str
    origin_packet_id: str | None = None
    witnesses: tuple = ()


def assignment_from_dict(row):
    return AssignmentRecord(**{**row, "candidate_records": tuple(row["candidate_records"]),
                               "witnesses": tuple(row.get("witnesses", ()))})


class AssignmentLedger:
    def __init__(self):
        self.raw_support_store = {}
        self.assignment_store = {}
        self.events = []

    def precheck(self, new_sources, assignments, decision_frame, read_version):
        sources = {r.observation_id: r for r in new_sources}
        if len(sources) != len(new_sources) or set(sources).intersection(self.raw_support_store):
            raise ValueError("Duplicate raw observation")
        if len({a.observation_id for a in assignments}) != len(assignments):
            raise ValueError("Duplicate assignment in transaction")
        for a in assignments:
            raw = sources.get(a.observation_id, self.raw_support_store.get(a.observation_id))
            if raw is None or a.status not in (*PENDING, ACCEPTED, UNPROJECTABLE):
                raise ValueError("Missing source or invalid status")
            if not raw.source_frame_id <= a.max_evidence_frame_id <= decision_frame:
                raise ValueError("Future evidence or invalid dual time")
            if a.last_decision_frame_id != decision_frame or a.read_map_version != read_version:
                raise ValueError("Stale transaction version")
            if not raw.source_frame_id <= a.first_decision_frame_id <= a.last_decision_frame_id:
                raise ValueError("Invalid decision chronology")
            for witness in a.witnesses:
                record = sources.get(witness["observation_id"], self.raw_support_store.get(witness["observation_id"]))
                if (record is None or witness["frame_id"] != record.source_frame_id
                        or witness["frame_id"] > a.max_evidence_frame_id
                        or witness["frame_id"] == raw.source_frame_id
                        or witness["support_hash"] != record.support_hash
                        or not witness["group_id"]):
                    raise ValueError("Invalid independent witness provenance")
            old = self.assignment_store.get(a.observation_id)
            if old and a.first_decision_frame_id != old.first_decision_frame_id:
                raise ValueError("First decision time changed")
            if old and (old.status == ACCEPTED or a.assignment_version != old.assignment_version + 1):
                raise ValueError("P1-B cannot automatically revise an accepted identity")
            if old is None and a.assignment_version != 0:
                raise ValueError("Invalid first assignment version")
            if a.status == ACCEPTED:
                validate_instance_id(a.persistent_instance_id)
                if a.active_since_map_version != read_version + 1:
                    raise ValueError("Invalid activation time")
            elif a.persistent_instance_id is not None or a.active_since_map_version is not None:
                raise ValueError("Inactive assignment has a formal identity")
            if a.status in PENDING and not a.packet_id:
                raise ValueError("Pending assignment lacks an audit packet")
        if not set(sources).issubset({a.observation_id for a in assignments}):
            raise ValueError("A raw source lacks an assignment")

    def apply(self, sources, assignments, decision_frame, read_version, parameters_hash):
        self.precheck(sources, assignments, decision_frame, read_version)
        self.raw_support_store.update((r.observation_id, r) for r in sources)
        for a in assignments:
            old = self.assignment_store.get(a.observation_id)
            raw = self.raw_support_store[a.observation_id]
            self.assignment_store[a.observation_id] = a
            kind = ("CONFIRM_BIRTH" if a.reason.startswith("confirmed_birth") else "ACCEPT_BIND") if a.status == ACCEPTED else "DEFER"
            self.events.append({
                "event_id": len(self.events) + 1, "transaction_id": read_version + 1,
                "type": kind, "observation_id": a.observation_id,
                "source_frame_id": raw.source_frame_id, "decision_frame_id": decision_frame,
                "max_evidence_frame_id": a.max_evidence_frame_id,
                "read_map_version": read_version, "written_map_version": read_version + 1,
                "old_status": None if old is None else old.status, "new_status": a.status,
                "old_id": None if old is None else old.persistent_instance_id,
                "new_id": a.persistent_instance_id,
                "old_assignment_version": None if old is None else old.assignment_version,
                "new_assignment_version": a.assignment_version,
                "witness_observation_ids": [w["observation_id"] for w in a.witnesses],
                "witness_frame_ids": [w["frame_id"] for w in a.witnesses],
                "evidence_group_ids": [w["group_id"] for w in a.witnesses],
                "excluded_origin_packets": [] if a.origin_packet_id is None else [a.origin_packet_id],
                "rule_version": "P1B-v1", "parameters_hash": parameters_hash,
                "reason": a.reason, "assignment": json.loads(json.dumps(asdict(a))),
            })

    def snapshot_asof(self, frame_id):
        current = {}
        for event in self.events:
            if event["decision_frame_id"] <= frame_id and "assignment" in event:
                current[event["observation_id"]] = assignment_from_dict(event["assignment"])
        return current

    def validate(self):
        if set(self.raw_support_store) != set(self.assignment_store):
            raise ValueError("Full assignment/source coverage differs")
        if self.events and self.snapshot_asof(max(e["decision_frame_id"] for e in self.events)) != self.assignment_store:
            raise ValueError("Assignment store differs from its event history")
        return True
