"""Independent packet index: audit working sets never contribute identity votes."""
from collections import defaultdict
from dataclasses import dataclass

from .association import _NEIGHBORS
from .assignment_ledger import PENDING


@dataclass(frozen=True)
class PendingPacket:
    packet_id: str
    observation_ids: tuple
    last_touched_step: int
    lifecycle: str = "ACTIVE"
    last_review_step: int = -1
    evidence_fingerprint: str = ""
    birth_cohorts: tuple = ()
    review_cursor: int = 0


class PendingIndex:
    def __init__(self):
        self.references = defaultdict(set)
        self.observation_packet = {}
        self.observation_voxels = {}

    def add(self, record, packet_id):
        key = record.observation_id
        if key in self.observation_packet:
            raise ValueError("Duplicate pending index contribution")
        self.observation_packet[key] = packet_id
        self.observation_voxels[key] = record.voxels
        for voxel in record.voxels:
            self.references[voxel].add(key)

    def remove(self, observation_id):
        for voxel in self.observation_voxels.pop(observation_id):
            self.references[voxel].remove(observation_id)
            if not self.references[voxel]:
                del self.references[voxel]
        del self.observation_packet[observation_id]

    def query(self, voxels):
        counts = defaultdict(int)
        for x, y, z in voxels:
            packets = set()
            for dx, dy, dz in _NEIGHBORS:
                for key in self.references.get((x+dx, y+dy, z+dz), ()):
                    packets.add(self.observation_packet[key])
            for packet in packets:
                counts[packet] += 1
        return sorted(counts, key=lambda k: (-counts[k], k))

    @classmethod
    def rebuild(cls, ledger):
        result = cls()
        for key, assignment in ledger.assignment_store.items():
            if assignment.status in PENDING:
                result.add(ledger.raw_support_store[key], assignment.packet_id)
        return result

    def validate(self, ledger):
        rebuilt = self.rebuild(ledger)
        if (self.references != rebuilt.references or self.observation_packet != rebuilt.observation_packet
                or self.observation_voxels != rebuilt.observation_voxels):
            raise ValueError("Pending index differs from its raw ledger")
        return True
