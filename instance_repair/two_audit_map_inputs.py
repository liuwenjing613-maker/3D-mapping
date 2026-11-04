"""Immutable pre-repair native IDs and original three-state ROI membership."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .two_audit_matching import intersection_count
from .two_audit_sources import file_record, verify_record


def read_ids(path, cache):
    with np.load(path, allow_pickle=False) as archive:
        cache.require_geometry(archive["xyz_m"])
        ids = archive["instance_id"]
    if ids.shape != (cache.manifest["geometry"]["shape"][0],) or ids.dtype.kind not in "iu":
        raise ValueError("Instance labels must be one integer per certified surface point")
    ids.setflags(write=False)
    return ids


def check_points(points, size):
    a = np.asarray(points)
    if (a.ndim != 1 or a.dtype.kind not in "iu"
            or (len(a) and (a[0] < 0 or a[-1] >= size
                            or np.any(np.diff(a.astype(np.int64)) <= 0)))):
        raise ValueError("ROI membership must be sorted unique valid surface indices")


def check_offsets(offsets, points, count):
    if (offsets.dtype.kind not in "iu" or offsets.shape != (count+1,)
            or offsets[0] != 0 or offsets[-1] != len(points)
            or np.any(np.diff(offsets.astype(np.int64)) < 0)):
        raise ValueError("Invalid ROI membership offsets")


class AuditMapInputs:
    """No arbitrary label-file override: IDs must be locked by the raw cache."""
    def __init__(self, cache):
        self.cache = cache
        mapping = Path(cache.manifest["mapping_root"]).resolve(strict=True)
        self.mapping = mapping
        self.records = {}

        def lock(path, expected=None):
            record = file_record(path)
            if expected is not None and record["sha256"] != expected:
                raise ValueError(f"Input source hash mismatch: {path}")
            self.records[record["path"]] = record
            return record

        self.lock = lock
        native = mapping / "baseline/native/instance_surface.npz"
        self.native_record = lock(native)
        matched = [r for r in cache.manifest["locked_sources"]
                   if Path(r["resolved_path"]) == native.resolve()]
        if len(matched) != 1 or matched[0]["sha256"] != self.native_record["sha256"]:
            raise ValueError("Native instance labels differ from the cache's pre-repair snapshot")
        verify_record(matched[0])
        self.ids = read_ids(native, cache)
        keys, counts = np.unique(self.ids, return_counts=True)
        self.instance_sizes = dict(zip(map(int, keys), map(int, counts)))

        candidate_root = mapping.parent / "candidates"
        self.catalog_root = candidate_root / "catalog" / cache.manifest["scene"]
        manifest_path = self.catalog_root / "candidate_manifest.json"
        lock(manifest_path)
        self.roi_manifest = m = json.loads(manifest_path.read_text())
        verification_path = self.catalog_root / "verification.json"
        lock(verification_path)
        verification = json.loads(verification_path.read_text())
        if (m["scene"] != cache.manifest["scene"]
                or m["checkpoint"] != cache.manifest["checkpoint"]
                or m["history_frames"] != cache.manifest["checkpoint"]
                or m.get("no_GT_used") is not True
                or m.get("current_checkpoint_repaired") is not False
                or verification.get("status") != "PASS"
                or verification.get("scene") != m["scene"]
                or verification.get("no_GT_used") is not True
                or verification.get("no_map_change") is not True):
            raise ValueError("ROI catalog is not the matching unrepaired checkpoint")
        if cache.manifest["mode"] == "offline" and (
                m.get("repair_applied") is not False or m.get("previous_committed_checkpoint") != 0):
            raise ValueError("Offline ROI catalog must precede all repair")
        frames = cache.manifest["frame_ids"]
        if m.get("scoring_history_frame_ids") != frames:
            raise ValueError("ROI scoring history differs from the observed prefix")
        search = m.get("candidate_search_frame_ids", [])
        if not search or any(fid not in cache.frames for fid in search):
            raise ValueError("ROI search references unobserved/future frames")
        if len(set(search)) != len(search):
            raise ValueError("Duplicate ROI search frames")

        freeze_path = candidate_root / "mapping_input_freeze.json"
        lock(freeze_path)
        freeze = json.loads(freeze_path.read_text())
        if (freeze.get("status") != "FROZEN" or freeze.get("checkpoint") != m["checkpoint"]
                or freeze.get("map_is_this_branch") is not True):
            raise ValueError("ROI mapping input is not frozen to this branch/checkpoint")
        frozen = freeze["source_hashes"]
        if frozen.get(str(native)) != self.native_record["sha256"]:
            raise ValueError("ROI and audit do not refer to the same native label snapshot")
        for field, relative in [("surface", "baseline/final/instance_surface.npz"),
                                ("evidence", "baseline/surface_evidence.npz"),
                                ("graph", "geometry/review_surface_graph_n02_p08.npz"),
                                ("camera_config", "config.json")]:
            original = mapping / relative
            expected = frozen.get(str(original))
            if not expected or expected != m[field+"_sha256"]:
                raise ValueError(f"ROI {field} is not bound to the frozen mapping")
            lock(original, expected)
            lock(m[field+"_file"], expected)
        if verification.get("baseline_sha256") != m["surface_sha256"]:
            raise ValueError("ROI verification refers to another map")
        self.final_ids = read_ids(m["surface_file"], cache)
        with np.load(m["evidence_file"], allow_pickle=False) as archive:
            cache.require_geometry(archive["xyz_m"])
            state = archive["state"]
        if state.shape != self.ids.shape or state.dtype.kind not in "iu":
            raise ValueError("ROI state array does not align with the geometry")

        membership_path = self.catalog_root / "membership.npz"
        lock(membership_path, m["membership_sha256"])
        with np.load(membership_path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in
                      ("core_offsets", "core_point_index", "context_offsets", "context_point_index")}
        components = m["unique_components"]
        for kind in ("core", "context"):
            check_offsets(arrays[kind+"_offsets"], arrays[kind+"_point_index"], len(components))
        entries = m["entries"]
        if len({e["id"] for e in entries}) != len(entries):
            raise ValueError("Duplicate ROI aliases")
        by_uid = {}
        for entry in entries:
            if entry["group"] not in ("CONFLICT", "U", "T"):
                raise ValueError("Unexpected ROI source group")
            for view in entry.get("views", []):
                if view["frame"] not in cache.frames:
                    raise ValueError("ROI view references a future/unobserved frame")
            by_uid.setdefault(entry["uid"], []).append(entry)

        self.components = []
        self.core_union = np.zeros(len(self.ids), bool)
        self.context_union = np.zeros(len(self.ids), bool)
        used_uids = set()
        for i, component in enumerate(components):
            if component["component_index"] != i or component["uid"] in used_uids:
                raise ValueError("ROI component indexing/identity is inconsistent")
            used_uids.add(component["uid"])
            pair = {}
            for kind in ("core", "context"):
                lo, hi = map(int, arrays[kind+"_offsets"][i:i+2])
                points = arrays[kind+"_point_index"][lo:hi].copy()
                check_points(points, len(self.ids))
                if len(points) != component[kind+"_points"]:
                    raise ValueError("ROI point count differs from manifest")
                pair[kind] = points
            core, context = pair["core"], pair["context"]
            if not len(core) or intersection_count(core, context) != len(core):
                raise ValueError("ROI core is empty or outside its context")
            digest = hashlib.sha256(core.astype("<i4").tobytes()).hexdigest()
            if digest != component["core_sha256"]:
                raise ValueError("ROI core content hash differs")
            if np.any(self.final_ids[core] > 0) or not np.isin(state[core], [0, 1, 3]).all():
                raise ValueError("ROI core does not match the recorded three-state alarm definition")
            measured = {name: int(np.count_nonzero(state[core] == value))
                        for name, value in (("CONFLICT", 3), ("U", 0), ("T", 1))}
            if measured != component["counts"]:
                raise ValueError("ROI state counts differ from original evidence")
            aliases = by_uid.get(component["uid"], [])
            if not aliases or {e["id"] for e in aliases} != {e["id"] for e in component["aliases"]}:
                raise ValueError("ROI aliases are not bound to their component")
            if any(e["component_index"] != i or e["core_sha256"] != digest for e in aliases):
                raise ValueError("ROI alias membership mismatch")
            pair.update(roi_uid=component["uid"],
                        aliases=[dict(id=e["id"], group=e["group"]) for e in aliases],
                        selected_view_frame_ids=sorted({v["frame"] for e in aliases for v in e.get("views", [])}))
            self.components.append(pair)
            self.core_union[core] = True
            self.context_union[context] = True
        if set(by_uid) != used_uids or verification.get("unique_ROIs") != len(components):
            raise ValueError("ROI registry coverage is incomplete")
        self.verify_unchanged()

    def verify_unchanged(self):
        for record in self.records.values():
            verify_record(record)

    def roi_overlap(self, points, *, per_component=False):
        check_points(points, len(self.ids))
        core = int(np.count_nonzero(self.core_union[points]))
        context = int(np.count_nonzero(self.context_union[points]))
        relation = "CORE_OVERLAP" if core else "CONTEXT_ONLY" if context else "NO_GEOMETRIC_OVERLAP"
        result = dict(point_count=len(points), core_points=core, context_points=context,
                      core_fraction=core/len(points) if len(points) else 0.,
                      context_fraction=context/len(points) if len(points) else 0.,
                      relation=relation)
        if per_component:
            matches = []
            for component in self.components:
                a, b = (intersection_count(points, component[k]) for k in ("core", "context"))
                if a or b:
                    matches.append(dict(roi_uid=component["roi_uid"], aliases=component["aliases"],
                                        selected_view_frame_ids=component["selected_view_frame_ids"],
                                        core_points=a, context_points=b))
            result["components"] = matches
        return result

    def record(self):
        return dict(label_source="pre_repair_baseline_native_instance_surface",
                    native_surface=self.native_record,
                    roi_catalog=str(self.catalog_root),
                    roi_components=len(self.components),
                    roi_aliases=len(self.roi_manifest["entries"]),
                    roi_core_points=int(self.core_union.sum()),
                    roi_context_points=int(self.context_union.sum()),
                    roi_overlap_is_not_detection_or_seed_acceptance=True,
                    sources=list(self.records.values()))
