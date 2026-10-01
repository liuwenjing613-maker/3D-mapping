"""P1-B: quarantine, independent raw witnesses, per-observation activation."""
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from .assignment_ledger import (ACCEPTED, PENDING, PENDING_BIND, PENDING_BIRTH, UNPROJECTABLE,
    AssignmentLedger, AssignmentRecord, RawSupportRecord, assignment_from_dict, fingerprint)
from .association import OnlineVoxelAssociator
from .association_review import RawViewVerifier, score_complete, reliable, qualified
from .identity_evidence import ObservationSupport, FrameIdentityEvidence
from .observations import RawInstanceObservation
from .pending_observations import PendingIndex, PendingPacket


DEFAULT_PARAMETERS = {
    "mature_fraction": .5, "min_visible_points": 3, "min_shared_points": 3,
    "raw_agreement": .7, "multi_region_fraction": .2, "anchor_groups": 2,
    "birth_groups": 2, "new_view_translation_m": .02, "new_view_angle_deg": 5.,
    "new_support_fraction": .1, "active_window_steps": 20, "packet_budget": 32,
    "observation_review_budget": 16, "witness_budget": 32, "candidate_budget": 32,
    "expand_candidates": True, "exclude_self_support": True,
    "frame_cache_size": 96, "point_cache_size": 4096, "pair_cache_size": 65536,
}


class DeferredAssociator:
    def __init__(self, frame_loader, mode="B2", parameters=None, **association_parameters):
        if mode not in ("B0", "B1", "B2"):
            raise ValueError("Unknown deferred mode")
        self.mode = mode
        self.parameters = {**DEFAULT_PARAMETERS, **(parameters or {})}
        if set(self.parameters) != set(DEFAULT_PARAMETERS):
            raise ValueError("Unknown P1-B parameter")
        for key in ("min_visible_points", "min_shared_points", "anchor_groups", "birth_groups",
                    "active_window_steps", "packet_budget", "observation_review_budget", "witness_budget",
                    "candidate_budget", "frame_cache_size", "point_cache_size", "pair_cache_size"):
            if not isinstance(self.parameters[key], int) or self.parameters[key] < 1:
                raise ValueError("Invalid positive integer parameter: " + key)
        if self.parameters["candidate_budget"] < 8 or self.parameters["birth_groups"] < 2 or self.parameters["anchor_groups"] < 2:
            raise ValueError("Insufficient candidate or witness budget")
        for key in ("mature_fraction", "raw_agreement", "multi_region_fraction", "new_support_fraction"):
            if not 0 < self.parameters[key] <= 1:
                raise ValueError("Invalid ratio: " + key)
        if self.parameters["new_view_translation_m"] <= 0 or self.parameters["new_view_angle_deg"] <= 0:
            raise ValueError("Invalid view novelty parameters")
        defaults = dict(association_mode="probabilistic", valid_first_sampling=False,
                        allow_multiple_observations_per_instance_per_frame=True)
        self.engine = OnlineVoxelAssociator(**{**defaults, **association_parameters})
        if self.engine.association_mode != "probabilistic" or self.engine.improved_association:
            raise ValueError("P1-B uses the frozen A2 probability path, not improved association")
        self.ledger, self.pending_index = AssignmentLedger(), PendingIndex()
        self.packets, self.review_queue = {}, set()
        self.next_packet_id, self.processing_step = 1, 0
        self.verifier = RawViewVerifier(frame_loader, self.engine, self.parameters)
        self.arrivals, self.review_log, self.repair_tickets = [], [], []
        self.imported_a2_revision_events = []
        self._frame_score_cache = {}
        self.fault_injector = None
        self.parameters_hash = fingerprint({"deferred": self.parameters, "association": self.engine.checkpoint_parameters()})

    @property
    def map_version(self):
        return self.engine.map_version

    def _raw(self, frame, observation):
        voxels = self.engine._sample_observation_voxels(frame, observation)
        input_hash = self.verifier.source_hash(frame)
        sampling_hash = fingerprint({"max_points": self.engine.max_points_per_observation,
            "valid_first": self.engine.valid_first_sampling, "voxel_size_m": self.engine.voxel_size_m})
        support_hash = fingerprint([observation.observation_id, observation.source_mask_sha256,
                                    input_hash, sampling_hash, voxels])
        return RawSupportRecord(observation, voxels, input_hash, sampling_hash, support_hash)

    @staticmethod
    def _near(a, b):
        if not a.voxels or not b.voxels:
            return False
        left_min, left_max = a.voxel_bounds
        right_min, right_max = b.voxel_bounds
        return all(lo <= hi + 1 and other_lo <= other_hi + 1
                   for lo, hi, other_lo, other_hi in zip(left_min, right_max, right_min, left_max))

    def _score_record(self, record, excluded=()):
        key = (self.map_version, record.observation_id, record.support_hash, tuple(sorted(excluded)))
        if key not in self._frame_score_cache:
            self._frame_score_cache[key] = score_complete(self.engine,
                self.verifier.frame(record.source_frame_id), record.observation,
                record.voxels, self.parameters, excluded)
        return self._frame_score_cache[key]

    def _witness_records(self, target, packet, raw, scores, current_keys, assignment):
        candidates = set(current_keys)
        candidates.update(packet.observation_ids[-12:])
        candidates.update(w["observation_id"] for w in assignment.witnesses)
        for candidate in scores["candidates"]:
            state = self.engine.instances.get(candidate["instance_id"])
            if state:
                candidates.update(state.observation_ids[-12:])
                candidates.update(state.observation_ids[:2])
        records = [raw[k] for k in candidates if k in raw and k != target.observation_id
                   and raw[k].source_frame_id != target.source_frame_id and self._near(target, raw[k])]
        known = {w["observation_id"] for w in assignment.witnesses}
        records.sort(key=lambda r: (r.observation_id not in known, r.observation_id not in current_keys,
                                    -r.source_frame_id, r.observation_id))
        return records[:self.parameters["witness_budget"]]

    def _review(self, packet, raw, assignments, current_keys, read_version, decision_frame):
        pending = [k for k in packet.observation_ids if assignments[k].status in PENDING]
        if not pending:
            return {}, packet, [], []
        cursor = packet.review_cursor % len(pending)
        rotated = pending[cursor:] + pending[:cursor]
        selected = rotated[:self.parameters["observation_review_budget"]]
        excluded = [k for k in packet.observation_ids if k in self.engine.observation_support
                    and self.ledger.assignment_store[k].origin_packet_id == packet.packet_id]
        if not self.parameters["exclude_self_support"]:
            excluded = []
        updates, diagnostics, tickets, birth_candidates = {}, [], [], {}
        anchor_cache = {}
        for key in selected:
            record, old = raw[key], assignments[key]
            scores = self._score_record(record, excluded)
            witnesses = self._witness_records(record, packet, raw, scores, current_keys, old)
            checks, compatible = [], []
            for other in witnesses:
                check = self.verifier.compare(record, other)
                checks.append({"observation_id": other.observation_id, "source_frame_id": other.source_frame_id, **check})
                if check["status"] == "MULTI_REGION_CONTRADICTION":
                    tickets.append({"observation_id": key, "witness_observation_id": other.observation_id,
                                    "reason": check["status"], "decision_frame_id": decision_frame})
                if check["status"] == "COMPATIBLE":
                    compatible.append(other)
            groups = self.verifier.information_groups([record, *compatible])
            evidence = []
            by_identity = {}
            for other in compatible:
                if other.observation_id not in anchor_cache:
                    anchor_scores = self._score_record(other, excluded)
                    anchor_cache[other.observation_id] = reliable(self.engine, anchor_scores, self.parameters)[0]
                identity = anchor_cache[other.observation_id]
                witness = {"observation_id": other.observation_id, "frame_id": other.source_frame_id,
                           "group_id": groups[other.observation_id], "anchor_instance_id": identity,
                           "support_hash": other.support_hash}
                evidence.append(witness)
                if identity is not None:
                    by_identity.setdefault(identity, set()).add(witness["group_id"])
            ranked = sorted(by_identity, key=lambda k: (-len(by_identity[k]), k))
            chosen = ranked[0] if ranked else None
            conflict = bool(len(ranked) > 1 and len(by_identity[ranked[1]]) >= len(by_identity[chosen]))
            mixed = any(c["status"] == "MULTI_REGION_CONTRADICTION" for c in checks)
            reason = "awaiting_independent_witnesses"
            identity = None
            if (self.mode == "B2" and chosen is not None and len(by_identity[chosen]) >= self.parameters["anchor_groups"]
                    and not conflict and not mixed):
                identity, reason = chosen, "accepted_bind_independent_raw_witnesses"
                evidence = [w for w in evidence if w["anchor_instance_id"] == identity]
            has_old_explanation = any(qualified(self.engine, c) for c in scores["candidates"])
            status = PENDING_BIND if has_old_explanation or not scores["candidate_complete"] else PENDING_BIRTH
            if identity is None and status == PENDING_BIRTH and not mixed and scores["candidate_complete"]:
                distinct = set(groups.values())
                if len(distinct) >= self.parameters["birth_groups"]:
                    birth_candidates[key] = (evidence, groups)
            if identity is not None:
                status = ACCEPTED
            max_frame = max([record.source_frame_id] + [r.source_frame_id for r in witnesses])
            update = replace(old, status=status, persistent_instance_id=identity,
                assignment_version=old.assignment_version + (0 if key in current_keys else 1),
                last_decision_frame_id=decision_frame, max_evidence_frame_id=max_frame,
                read_map_version=read_version, active_since_map_version=read_version + 1 if identity is not None else None,
                candidate_records=tuple(scores["candidates"]), reason=reason, witnesses=tuple(evidence))
            # Re-evaluating identical evidence is not a new observation or confirmation.
            changed = (status, identity, tuple(evidence)) != (old.status, old.persistent_instance_id, old.witnesses)
            if changed or key in current_keys:
                updates[key] = update
            diagnostics.append({"observation_id": key, "packet_id": packet.packet_id,
                "decision_frame_id": decision_frame, "read_map_version": read_version,
                "excluded_observation_ids": excluded, "candidate_diagnostics": scores,
                "raw_checks": checks, "witnesses": evidence, "mixed": mixed})
        return updates, packet, diagnostics, [(birth_candidates, tickets)]

    def process_frame(self, frame, observations):
        self.engine._validate_frame(frame, observations)
        if any(o.observation_id in self.ledger.raw_support_store for o in observations):
            raise ValueError("Raw observation already received")
        self.verifier.receive(frame)
        self._frame_score_cache.clear()
        read_version = self.map_version
        sources = [self._raw(frame, o) for o in observations]
        if self.mode == "B0":
            return self._process_compatible(frame, observations, sources, read_version)
        raw = {**self.ledger.raw_support_store, **{r.observation_id: r for r in sources}}
        current_keys = {r.observation_id for r in sources}
        planned, packets, next_packet = {}, self.packets.copy(), self.next_packet_id
        queue, arrivals = self.review_queue.copy(), []
        next_instance = self.engine.next_instance_id
        arrival_scores = {r.observation_id: self._score_record(r) for r in sources}
        current_aliases = {r.observation.mask_local_id: reliable(self.engine, arrival_scores[r.observation_id], self.parameters)[0]
                           for r in sources if r.voxels}
        for record in sources:
            scores = arrival_scores[record.observation_id]
            identity, reason = reliable(self.engine, scores, self.parameters) if record.voxels else (None, "unprojectable")
            if identity is not None:
                history = self.engine.instances[identity].observation_ids[-3:]
                aliases = {frame.frame_id: current_aliases}
                for key in history:
                    prior = raw[key]
                    aliases.setdefault(prior.source_frame_id, {})[prior.observation.mask_local_id] = self.ledger.assignment_store[key].persistent_instance_id
                if any(self.verifier.compare(record, raw[k], aliases)["status"] == "MULTI_REGION_CONTRADICTION" for k in history):
                    identity, reason = None, "raw_multi_region_contradiction"
            nearby = self.pending_index.query(record.voxels)
            queue.update(nearby)
            for candidate_packet in nearby:
                packets[candidate_packet] = replace(packets[candidate_packet],
                    last_touched_step=self.processing_step, lifecycle="ACTIVE")
            packet_id = None
            if identity is None and record.voxels:
                for candidate_packet in nearby[:8]:
                    old = packets[candidate_packet]
                    prior = [k for k in old.observation_ids
                             if (planned.get(k) or self.ledger.assignment_store[k]).status in PENDING][-4:]
                    if any(self.verifier.compare(record, raw[k])["status"] == "COMPATIBLE" for k in prior):
                        packet_id = candidate_packet
                        break
                if packet_id is None:
                    packet_id = f"Q{next_packet:08d}"
                    next_packet += 1
                    packets[packet_id] = PendingPacket(packet_id, (), self.processing_step)
                packet = packets[packet_id]
                packets[packet_id] = replace(packet, observation_ids=packet.observation_ids + (record.observation_id,),
                                             last_touched_step=self.processing_step, lifecycle="ACTIVE")
                queue.add(packet_id)
            status = ACCEPTED if identity is not None else (UNPROJECTABLE if not record.voxels else
                PENDING_BIND if not scores["candidate_complete"] or any(qualified(self.engine, c) for c in scores["candidates"])
                else PENDING_BIRTH)
            planned[record.observation_id] = AssignmentRecord(record.observation_id, status, identity, packet_id, 0,
                frame.frame_id, frame.frame_id, frame.frame_id, read_version,
                read_version+1 if identity is not None else None, tuple(scores["candidates"]), reason, packet_id)
            arrivals.append({"frame_id": frame.frame_id, "observation_id": record.observation_id,
                "mask_local_id": record.observation.mask_local_id, "instance_id": identity,
                "decision": reason, "sampled_voxels": len(record.voxels), "status": status,
                "packet_id": packet_id, **scores})
        # A reliable current frame or newly touched local support can wake a packet.
        changed_ids = {a.persistent_instance_id for a in planned.values() if a.status == ACCEPTED}
        for packet_id, packet in packets.items():
            if any(c["instance_id"] in changed_ids for key in packet.observation_ids
                   for c in (planned.get(key) or self.ledger.assignment_store[key]).candidate_records):
                queue.add(packet_id)
        assignments = {**self.ledger.assignment_store, **planned}
        review_logs, repair_tickets = [], []
        order = sorted(queue, key=lambda k: (packets[k].last_review_step, packets[k].last_touched_step, k))
        for packet_id in order[:self.parameters["packet_budget"]]:
            packet = packets[packet_id]
            if self.processing_step - packet.last_touched_step > self.parameters["active_window_steps"] and packet_id not in {
                    q for r in sources for q in self.pending_index.query(r.voxels)}:
                packets[packet_id] = replace(packet, lifecycle="DORMANT")
                queue.discard(packet_id)
                continue
            updates, packet, diagnostics, auxiliary = self._review(packet, raw, assignments, current_keys, read_version, frame.frame_id)
            cohorts = list(packet.birth_cohorts)
            for births, tickets in auxiliary:
                repair_tickets.extend(tickets)
                remaining = sorted(births)
                while remaining:
                    seed = remaining[0]
                    identity = next((i for i, old_seed in cohorts
                                     if self.verifier.compare(raw[seed], raw[old_seed])["status"] == "COMPATIBLE"), None)
                    if identity is None:
                        identity = next_instance
                        next_instance += 1
                        cohorts.append((identity, seed))
                    accepted = [k for k in remaining if k == seed or self.verifier.compare(raw[k], raw[seed])["status"] == "COMPATIBLE"]
                    for key in accepted:
                        old = updates.get(key, assignments[key])
                        witnesses = births[key][0]
                        updates[key] = replace(old, status=ACCEPTED, persistent_instance_id=identity,
                            assignment_version=assignments[key].assignment_version + (0 if key in current_keys else 1),
                            last_decision_frame_id=frame.frame_id, read_map_version=read_version,
                            active_since_map_version=read_version+1, reason="confirmed_birth_raw_multiview",
                            witnesses=tuple(witnesses), max_evidence_frame_id=max(old.max_evidence_frame_id,
                                max([raw[key].source_frame_id]+[w["frame_id"] for w in witnesses])))
                    remaining = [k for k in remaining if k not in accepted]
            planned.update(updates)
            assignments.update(updates)
            pending_count = sum(assignments[k].status in PENDING for k in packet.observation_ids)
            life = "RESOLVED" if pending_count == 0 else "PARTIALLY_RESOLVED" if pending_count < len(packet.observation_ids) else "ACTIVE"
            packets[packet_id] = replace(packet, lifecycle=life, last_review_step=self.processing_step,
                evidence_fingerprint=fingerprint(diagnostics), birth_cohorts=tuple(cohorts),
                review_cursor=packet.review_cursor + len(diagnostics))
            reviewed = {d["observation_id"] for d in diagnostics}
            if not any(assignments[k].status in PENDING and k not in reviewed for k in packet.observation_ids):
                queue.discard(packet_id)
            review_logs.extend(diagnostics)
        for packet_id, packet in list(packets.items()):
            if packet.lifecycle in ("ACTIVE", "PARTIALLY_RESOLVED") and self.processing_step - packet.last_touched_step > self.parameters["active_window_steps"]:
                packets[packet_id] = replace(packet, lifecycle="DORMANT")
        self._commit_transaction(frame, sources, list(planned.values()), packets, queue,
                                 next_packet, next_instance, arrivals, review_logs, repair_tickets)
        return arrivals

    def _commit_transaction(self, frame, sources, assignments, packets, queue, next_packet,
                            next_instance, arrivals, review_logs, tickets):
        version = self.map_version
        self.ledger.precheck(sources, assignments, frame.frame_id, version)
        active = [ObservationSupport(a.observation_id,
            (next((r for r in sources if r.observation_id == a.observation_id), None) or self.ledger.raw_support_store[a.observation_id]).source_frame_id,
            (next((r for r in sources if r.observation_id == a.observation_id), None) or self.ledger.raw_support_store[a.observation_id]).observation.source_mask_sha256,
            a.persistent_instance_id,
            (next((r for r in sources if r.observation_id == a.observation_id), None) or self.ledger.raw_support_store[a.observation_id]).voxels,
            a.assignment_version) for a in assignments if a.status == ACCEPTED]
        old_assignments = {a.observation_id: self.ledger.assignment_store.get(a.observation_id) for a in assignments}
        event_count = len(self.ledger.events)
        old_engine = (self.engine.next_instance_id, self.engine.last_processed_frame_id,
                      self.engine.map_version, self.engine._last_scoring_map_version)
        old_state = (self.packets, self.review_queue, self.next_packet_id, self.processing_step)
        old_lengths = (len(self.arrivals), len(self.review_log), len(self.repair_tickets))
        try:
            self.engine.identity_evidence.activate(active)
            if self.fault_injector:
                self.fault_injector("after_activation")
            self.ledger.apply(sources, assignments, frame.frame_id, version, self.parameters_hash)
            if self.fault_injector:
                self.fault_injector("after_ledger")
            for a in assignments:
                if a.observation_id in self.pending_index.observation_packet and a.status not in PENDING:
                    self.pending_index.remove(a.observation_id)
                elif a.status in PENDING and a.observation_id not in self.pending_index.observation_packet:
                    self.pending_index.add(self.ledger.raw_support_store[a.observation_id], a.packet_id)
            self._finish_commit(frame, active, packets, queue, next_packet, next_instance,
                                arrivals, review_logs, tickets, version, old_engine[0])
            if self.fault_injector:
                self.fault_injector("after_bookkeeping")
        except Exception:
            activated = [e.observation_id for e in active if e.observation_id in self.engine.identity_evidence.entries]
            self.engine.identity_evidence.deactivate(activated)
            for record in sources:
                self.ledger.raw_support_store.pop(record.observation_id, None)
            for key, old in old_assignments.items():
                if old is None:
                    self.ledger.assignment_store.pop(key, None)
                else:
                    self.ledger.assignment_store[key] = old
            del self.ledger.events[event_count:]
            self.pending_index = PendingIndex.rebuild(self.ledger)
            (self.engine.next_instance_id, self.engine.last_processed_frame_id,
             self.engine.map_version, self.engine._last_scoring_map_version) = old_engine
            self.engine._rebuild_instance_index()
            self.packets, self.review_queue, self.next_packet_id, self.processing_step = old_state
            for rows, length in zip((self.arrivals, self.review_log, self.repair_tickets), old_lengths):
                del rows[length:]
            raise

    def _finish_commit(self, frame, active, packets, queue, next_packet, next_instance,
                       arrivals, review_logs, tickets, version, old_next):
        # Identity cache is already committed; append the same contributions to its spatial index.
        for entry in active:
            from .association import InstanceState
            state = self.engine.instances.setdefault(entry.instance_id, InstanceState(entry.instance_id, first_frame_id=entry.frame_id))
            state.observation_ids.append(entry.observation_id)
            state.first_frame_id = min(state.first_frame_id, entry.frame_id)
            state.last_frame_id = max(state.last_frame_id, entry.frame_id)
            for voxel in entry.voxels:
                if voxel not in state.voxel_set:
                    state.voxel_set.add(voxel)
                    state.voxel_list.append(voxel)
                    self.engine.voxel_to_instances[voxel].add(entry.instance_id)
        if self.fault_injector:
            self.fault_injector("after_spatial_index")
        self.engine.next_instance_id = max(next_instance, max((e.instance_id+1 for e in active), default=old_next))
        self.engine.last_processed_frame_id = frame.frame_id
        self.engine.map_version = version + 1
        self.engine._last_scoring_map_version = version
        for key, packet in packets.items():
            old = self.packets.get(key)
            if old is None or old.lifecycle != packet.lifecycle:
                self.ledger.events.append({"event_id": len(self.ledger.events)+1, "transaction_id": version+1,
                    "type": "LINK_PACKET" if old is None else "MARK_DORMANT" if packet.lifecycle == "DORMANT" else "REACTIVATE" if old.lifecycle == "DORMANT" else "PACKET_STATE",
                    "packet_id": key, "old_lifecycle": None if old is None else old.lifecycle,
                    "new_lifecycle": packet.lifecycle, "decision_frame_id": frame.frame_id,
                    "read_map_version": version, "written_map_version": version+1})
        self.packets, self.review_queue = packets, queue
        self.next_packet_id = next_packet
        self.processing_step += 1
        self.arrivals.extend(arrivals)
        self.review_log.extend(review_logs)
        existing = {(t["observation_id"], t["witness_observation_id"]) for t in self.repair_tickets}
        self.repair_tickets.extend(t for t in tickets if (t["observation_id"], t["witness_observation_id"]) not in existing)

    def _process_compatible(self, frame, observations, sources, version):
        old_next, old_frame = self.engine.next_instance_id, self.engine.last_processed_frame_id
        try:
            decisions = self.engine.process_frame(frame, observations)
            assignments = [AssignmentRecord(d["observation_id"], ACCEPTED, d["instance_id"], None, 0,
                frame.frame_id, frame.frame_id, frame.frame_id, version, version+1,
                tuple(d["candidates"]), d["decision"]) for d in decisions]
            self.ledger.apply(sources, assignments, frame.frame_id, version, self.parameters_hash)
        except Exception:
            keys = [r.observation_id for r in sources if r.observation_id in self.engine.observation_support]
            self.engine.identity_evidence.deactivate(keys)
            self.engine.next_instance_id, self.engine.last_processed_frame_id, self.engine.map_version = old_next, old_frame, version
            self.engine._rebuild_instance_index()
            raise
        self.processing_step += 1
        self.arrivals.extend(decisions)
        return decisions

    def validate(self):
        self.ledger.validate()
        self.engine.identity_evidence.validate()
        self.pending_index.validate(self.ledger)
        expected = {key for key, a in self.ledger.assignment_store.items() if a.status == ACCEPTED}
        if set(self.engine.observation_support) != expected:
            raise ValueError("Pending evidence leaked into the active cache")
        for key in expected:
            source, a, active = self.ledger.raw_support_store[key], self.ledger.assignment_store[key], self.engine.observation_support[key]
            if (active.voxels != source.voxels or active.frame_id != source.source_frame_id
                    or active.instance_id != a.persistent_instance_id or active.assignment_version != a.assignment_version):
                raise ValueError("Active contribution differs from its immutable source or assignment")
        return True

    def assignment_rows(self, asof=None):
        assignments = self.ledger.assignment_store if asof is None else self.ledger.snapshot_asof(asof)
        return [{"frame_id": self.ledger.raw_support_store[key].source_frame_id,
                 "mask_local_id": self.ledger.raw_support_store[key].observation.mask_local_id,
                 "instance_id": a.persistent_instance_id, **asdict(a)} for key, a in assignments.items()]

    def save_checkpoint(self, path):
        self.validate()
        path = Path(path)
        sources = list(self.ledger.raw_support_store.values())
        metadata = {"format": "P1B-v1", "mode": self.mode, "parameters": self.parameters,
            "association_parameters": self.engine.checkpoint_parameters(), "map_version": self.map_version,
            "last_processed_frame_id": self.engine.last_processed_frame_id, "next_instance_id": self.engine.next_instance_id,
            "next_packet_id": self.next_packet_id, "processing_step": self.processing_step,
            "assignments": [asdict(a) for a in self.ledger.assignment_store.values()], "events": self.ledger.events,
            "packets": [asdict(p) for p in self.packets.values()], "review_queue": sorted(self.review_queue),
            "arrivals": self.arrivals, "review_log": self.review_log, "repair_tickets": self.repair_tickets,
            "imported_a2_revision_events": self.imported_a2_revision_events,
            "active_observation_order": list(self.engine.observation_support)}
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]),
            observations_json=np.asarray([json.dumps(asdict(r.observation)) for r in sources], dtype=str),
            depth_pose_hash=np.asarray([r.depth_pose_intrinsics_hash for r in sources], dtype=str),
            sampling_hash=np.asarray([r.sampling_config_hash for r in sources], dtype=str),
            support_hash=np.asarray([r.support_hash for r in sources], dtype=str),
            offsets=np.r_[0, np.cumsum([len(r.voxels) for r in sources], dtype=np.int64)],
            voxels=np.asarray([v for r in sources for v in r.voxels], np.int32).reshape(-1, 3))
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path, frame_loader):
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        m = json.loads(str(data["metadata_json"][0]))
        if m["format"] != "P1B-v1":
            raise ValueError("Unsupported checkpoint format")
        result = cls(frame_loader, m["mode"], m["parameters"], **m["association_parameters"])
        for i, row in enumerate(data["observations_json"]):
            observation = RawInstanceObservation(**json.loads(str(row)))
            start, stop = data["offsets"][i:i+2]
            voxels = tuple(tuple(int(x) for x in v) for v in data["voxels"][start:stop])
            record = RawSupportRecord(observation, voxels, str(data["depth_pose_hash"][i]), str(data["sampling_hash"][i]), str(data["support_hash"][i]))
            if fingerprint([observation.observation_id, observation.source_mask_sha256,
                record.depth_pose_intrinsics_hash, record.sampling_config_hash, voxels]) != record.support_hash:
                raise ValueError("Checkpoint raw support hash changed")
            if observation.observation_id in result.ledger.raw_support_store:
                raise ValueError("Duplicate checkpoint source")
            result.ledger.raw_support_store[observation.observation_id] = record
        if len({a["observation_id"] for a in m["assignments"]}) != len(m["assignments"]):
            raise ValueError("Duplicate checkpoint assignment")
        result.ledger.assignment_store = {a["observation_id"]: assignment_from_dict(a) for a in m["assignments"]}
        result.ledger.events = m["events"]
        result.packets = {p["packet_id"]: PendingPacket(**{**p, "observation_ids": tuple(p["observation_ids"]),
                             "birth_cohorts": tuple(tuple(x) for x in p["birth_cohorts"])}) for p in m["packets"]}
        result.pending_index = PendingIndex.rebuild(result.ledger)
        active_order = m["active_observation_order"]
        if len(active_order) != len(set(active_order)) or set(active_order) != {k for k,a in result.ledger.assignment_store.items() if a.status == ACCEPTED}:
            raise ValueError("Invalid active contribution order")
        for key in active_order:
            a = result.ledger.assignment_store[key]
            if a.status == ACCEPTED:
                raw = result.ledger.raw_support_store[key]
                result.engine.identity_evidence.add(ObservationSupport(key, raw.source_frame_id,
                    raw.observation.source_mask_sha256, a.persistent_instance_id, raw.voxels, a.assignment_version))
        result.engine._rebuild_instance_index()
        for field in ("map_version", "last_processed_frame_id", "next_instance_id"):
            setattr(result.engine, field, m[field])
        result.next_packet_id, result.processing_step = m["next_packet_id"], m["processing_step"]
        result.review_queue = set(m["review_queue"])
        result.arrivals, result.review_log, result.repair_tickets = m["arrivals"], m["review_log"], m["repair_tickets"]
        result.imported_a2_revision_events = m.get("imported_a2_revision_events", [])
        result.verifier.max_available_frame = result.engine.last_processed_frame_id
        used_ids = [a.persistent_instance_id or 0 for a in result.ledger.assignment_store.values()]
        used_ids += [i for p in result.packets.values() for i, _ in p.birth_cohorts]
        if result.engine.next_instance_id <= max(used_ids, default=0):
            raise ValueError("Checkpoint would reuse an instance ID")
        if result.next_packet_id <= max((int(k[1:]) for k in result.packets), default=0):
            raise ValueError("Checkpoint would reuse a packet ID")
        result.validate()
        return result

    @classmethod
    def from_a2_checkpoint(cls, path, frame_loader, observations, mode="B2", parameters=None):
        """Import sources as accepted; retain revisions without inventing old decision times."""
        original = OnlineVoxelAssociator.from_checkpoint(path)
        if original.association_mode != "probabilistic":
            raise ValueError("Only A2 checkpoints can seed the compatibility converter")
        catalog = {o.observation_id: o for o in observations}
        if len(catalog) != len(observations) or set(catalog) != set(original.observation_support):
            raise ValueError("A2 import requires precisely its full immutable source catalog")
        result = cls(frame_loader, mode, parameters, **original.checkpoint_parameters())
        result.engine = original
        result.verifier.engine = original
        result.verifier.max_available_frame = original.last_processed_frame_id
        result.processing_step = original.map_version - len({e["map_version_after"] for e in original.revision_events})
        result.imported_a2_revision_events = original.revision_events.copy()
        result.parameters_hash = fingerprint({"deferred": result.parameters, "association": original.checkpoint_parameters()})
        for key, entry in original.observation_support.items():
            observation = catalog[key]
            if observation.frame_id != entry.frame_id or observation.source_mask_sha256 != entry.source_mask_sha256:
                raise ValueError("A2 import provenance differs")
            frame = result.verifier.frame(entry.frame_id)
            raw = result._raw(frame, observation)
            if raw.voxels != entry.voxels:
                raise ValueError("A2 import source projection differs")
            result.ledger.raw_support_store[key] = raw
            # P1-A revisions lacked decision-frame metadata. Import the accepted snapshot
            # at import time; never claim a revised label was known in an earlier prefix.
            decision = original.last_processed_frame_id
            assignment = AssignmentRecord(key, ACCEPTED, entry.instance_id, None, entry.assignment_version,
                decision, decision, decision, max(original.map_version-1,0), original.map_version,
                (), "imported_A2_accepted_snapshot")
            result.ledger.assignment_store[key] = assignment
            result.ledger.events.append({"event_id":len(result.ledger.events)+1, "type":"IMPORT_ACCEPTED",
                "source_frame_id":entry.frame_id, "decision_frame_id":decision,
                "max_evidence_frame_id":decision, "read_map_version":max(original.map_version-1,0),
                "written_map_version":original.map_version, "observation_id":key,
                "new_id":entry.instance_id, "assignment":json.loads(json.dumps(asdict(assignment)))})
        result.validate()
        return result
