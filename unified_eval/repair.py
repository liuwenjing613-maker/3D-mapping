from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepairEvent:
    timestamp_frame: int
    affected_instance_ids_before: tuple[str, ...]
    affected_instance_ids_after: tuple[str, ...]
    affected_vertex_ids: tuple[int, ...]
    action: str
    evidence_frames: tuple[int, ...]


# RSR/FRR/TTR intentionally remain undefined until repairable events and
# evidence timing are specified from the actual repair algorithm.
