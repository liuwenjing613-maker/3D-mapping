"""Build/read the first two-audit stage: raw observations, S_t and V_t.

No stability decisions, anomaly detection, selection overrides or repair writes.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np

from .two_audit_projection import ProjectionSettings, project_frame
from .two_audit_sources import (MappingSource, canonical_hash, file_record,
                                geometry_identity, sha256_file, verify_prefix,
                                verify_record)

SCHEMA_VERSION = 1


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                            allow_nan=False) + "\n")


def implementation_records():
    root = Path(__file__).parent
    return [file_record(root / name) for name in
            ("two_audit_sources.py", "two_audit_projection.py", "two_audit_cache.py")]


def check_pack(pack, frame_record, observations, geometry):
    if str(pack["geometry_id"].item()) != geometry["geometry_id"]:
        raise ValueError("Frame pack belongs to another geometry index space")
    if int(pack["frame_id"].item()) != frame_record["frame_id"]:
        raise ValueError("Frame pack ID mismatch")
    if str(pack["source_mask_sha256"].item()) != frame_record["sources"]["mask"]["sha256"]:
        raise ValueError("Frame pack original-mask provenance mismatch")
    v = pack["visible_surface_indices"]
    pixels, labels = pack["visible_pixel_indices"], pack["visible_mask_local_ids"]
    ids, offsets, support = (pack[k] for k in
                              ("mask_local_ids", "mask_offsets", "mask_surface_indices"))
    if any(x.ndim != 1 or x.dtype.kind not in "iu"
           for x in (v, pixels, labels, ids, offsets, support)):
        raise ValueError("Malformed frame pack arrays")
    count = geometry["shape"][0]
    if len(v) != len(pixels) or len(v) != len(labels):
        raise ValueError("Visible point/pixel/label arrays are not aligned")
    if len(v) and (v[0] < 0 or v[-1] >= count or np.any(np.diff(v) <= 0)):
        raise ValueError("Visible surface indices must be sorted, unique and in range")
    h, w = frame_record["image_shape"]
    if len(pixels) and (pixels.min() < 0 or pixels.max() >= h * w):
        raise ValueError("Visible pixel indices out of bounds")
    expected_ids = np.array([x["mask_local_id"] for x in observations], np.int32)
    if not np.array_equal(ids, expected_ids) or np.any(ids <= 0) or np.any(np.diff(ids) <= 0):
        raise ValueError("Per-mask support must include every original mask exactly once")
    if len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(support):
        raise ValueError("Invalid per-mask support offsets")
    if np.any(np.diff(offsets) < 0):
        raise ValueError("Per-mask offsets must be monotonic")
    if not set(map(int, np.unique(labels))) <= set(map(int, ids)) | {0}:
        raise ValueError("Visible labels are not original frame-local masks")
    positive = labels > 0
    order = np.argsort(labels[positive], kind="stable")
    grouped = labels[positive][order]
    expected_offsets = np.r_[np.searchsorted(grouped, ids), int(positive.sum())]
    if (not np.array_equal(support, v[positive][order])
            or not np.array_equal(offsets, expected_offsets)):
        raise ValueError("Mask support is not the labelled subset of valid visibility")
    if any(int(offsets[i + 1] - offsets[i]) != row["surface_support_count"]
           for i, row in enumerate(observations)):
        raise ValueError("Observation and packed support counts differ")
    stats = frame_record["projection"]
    if (stats["visible_depth_consistent_points"] != len(v)
            or stats["positive_mask_evidence_points"] != int(positive.sum())
            or stats["background_visible_points"] != int((labels == 0).sum())):
        raise ValueError("Projection statistics disagree with the support arrays")


def build_cache(mapping_root, output, *, mode, checkpoint, settings=None):
    settings = settings or ProjectionSettings()
    source = MappingSource(mapping_root, mode, checkpoint)
    output = Path(output).absolute()
    # Never let an invocation place artifacts inside an input mapping snapshot.
    if output.resolve().is_relative_to(source.root):
        raise ValueError("Output must be separate from the read-only mapping snapshot")
    if output.exists():
        raise FileExistsError(f"Cache output already exists; nothing overwritten: {output}")
    implementation = implementation_records()
    request = dict(schema_version=SCHEMA_VERSION, stage="raw_observation_cache",
                   mode=mode, dataset=source.dataset, scene=source.scene,
                   checkpoint=source.checkpoint, raw_cutoff=source.cutoff,
                   frame_ids=list(source.frame_ids), geometry=source.geometry,
                   mapping_root=str(source.root), projection=settings.record(),
                   locked_sources=list(source.locked_files.values()),
                   prefix_sources=source.prefix_sources,
                   implementation=implementation)
    cache_id = canonical_hash(request)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    write_json(output / "request.json", dict(request, cache_id=cache_id))
    started = time.perf_counter()
    totals = dict(frames=0, observations=0, zero_surface_support_observations=0,
                  no_valid_depth_observations=0, visible_point_observations=0,
                  positive_point_observations=0, background_point_observations=0)
    manifest_files = {}

    def progress(status, **extra):
        value = dict(status=status, stage="raw_observation_cache", cache_id=cache_id,
                     completed_frames=totals["frames"], total_frames=len(source.frame_ids),
                     observations=totals["observations"],
                     seconds=time.perf_counter() - started, **extra)
        write_json(output / "status.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    try:
        progress("RUNNING")
        observation_path, frame_path = output / "mask_observations.jsonl", output / "frames.jsonl"
        with observation_path.open("w", encoding="utf-8") as obs_stream, \
                frame_path.open("w", encoding="utf-8") as frame_stream:
            for fid in source.frame_ids:
                frame = source.load_frame(fid, settings.max_depth_m)
                pack, stats = project_frame(frame, source.xyz, settings)
                pack.update(frame_id=np.array(fid, np.int64),
                            geometry_id=np.array(source.geometry["geometry_id"]),
                            source_mask_sha256=np.array(frame.sources["mask"]["sha256"]))
                relative = f"frames/f{fid:06d}.npz"
                rows = []
                for i, original in enumerate(frame.observations):
                    count = int(pack["mask_offsets"][i + 1] - pack["mask_offsets"][i])
                    flags = []
                    if original["valid_depth_pixels"] == 0:
                        flags.append("no_valid_depth")
                    if count == 0:
                        flags.append("no_surface_support")
                    if original["touches_image_border"]:
                        flags.append("touches_image_border")
                    rows.append(dict(original, cache_id=cache_id,
                                     geometry_id=source.geometry["geometry_id"],
                                     surface_support_count=count, support_file=relative,
                                     frame_metadata_id=fid, quality_flags=flags))
                frame_record = dict(frame_id=fid, geometry_id=source.geometry["geometry_id"],
                                    image_shape=[source.height, source.width],
                                    camera_to_world=frame.pose.tolist(),
                                    intrinsic=frame.intrinsic.tolist(),
                                    sources=frame.sources, mask_count=len(rows),
                                    support_file=relative, projection=stats)
                check_pack(pack, frame_record, rows, source.geometry)
                np.savez_compressed(output / relative, **pack)
                frame_record["support_sha256"] = sha256_file(output / relative)
                for row in rows:
                    json_line(obs_stream, row)
                json_line(frame_stream, frame_record)
                totals["frames"] += 1
                totals["observations"] += len(rows)
                totals["zero_surface_support_observations"] += sum(
                    row["surface_support_count"] == 0 for row in rows)
                totals["no_valid_depth_observations"] += sum(
                    row["valid_depth_pixels"] == 0 for row in rows)
                totals["visible_point_observations"] += stats["visible_depth_consistent_points"]
                totals["positive_point_observations"] += stats["positive_mask_evidence_points"]
                totals["background_point_observations"] += stats["background_visible_points"]
                if totals["frames"] % 25 == 0 or totals["frames"] == len(source.frame_ids):
                    progress("RUNNING", last_frame_id=fid)
        progress("VERIFYING_SOURCES")
        source.verify_unchanged()
        for record in implementation:
            verify_record(record)
        for path in (observation_path, frame_path, output / "request.json"):
            manifest_files[path.name] = sha256_file(path)
        manifest = dict(request, cache_id=cache_id, status="COMPLETE", totals=totals,
                        artifact_sha256=manifest_files, seconds=time.perf_counter() - started,
                        original_masks_only=True, current_map_labels_used_as_evidence=False,
                        GT_used=False, future_image_inputs_used=False,
                        sources_unchanged=True, map_modified=False,
                        stability_or_repair_decisions_made=False)
        # This is the only completion marker and is published after all checks.
        write_json(output / "manifest.json", manifest)
        progress("COMPLETE")
        return manifest
    except Exception as exc:
        progress("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


class ObservationCache:
    """Reader that requires an explicit geometry ID for every point-index query."""
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        m = self.manifest
        if m.get("status") != "COMPLETE" or m.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Cache is not a supported complete observation cache")
        for name, expected in m["artifact_sha256"].items():
            if Path(name).name != name or sha256_file(self.root / name) != expected:
                raise ValueError(f"Cache metadata hash mismatch: {name}")
        request = json.loads((self.root / "request.json").read_text(encoding="utf-8"))
        expected_id = request.pop("cache_id")
        if canonical_hash(request) != expected_id or expected_id != m["cache_id"]:
            raise ValueError("Cache identity does not match the locked request")
        if any(m[k] != value for k, value in request.items()):
            raise ValueError("Manifest differs from its locked request")
        with (self.root / "frames.jsonl").open(encoding="utf-8") as stream:
            frame_rows = [json.loads(line) for line in stream if line.strip()]
        self.frames = {row["frame_id"]: row for row in frame_rows}
        if len(self.frames) != len(frame_rows) or list(self.frames) != m["frame_ids"]:
            raise ValueError("Cache frames are not the declared exact prefix")
        self.observations, self.by_frame = {}, defaultdict(list)
        with (self.root / "mask_observations.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key, fid = row["observation_key"], row["frame_id"]
                if key in self.observations or fid not in self.frames:
                    raise ValueError("Duplicate/out-of-prefix original observation")
                identity = {k: row[k] for k in
                            ("dataset", "scene", "frame_id", "mask_local_id",
                             "source_mask_sha256", "source_model")}
                if canonical_hash(identity) != key:
                    raise ValueError("Original observation identity hash mismatch")
                if (row["cache_id"] != m["cache_id"]
                        or row["geometry_id"] != m["geometry"]["geometry_id"]):
                    raise ValueError("Observation refers to another cache/geometry")
                self.observations[key] = row
                self.by_frame[fid].append(row)
        if len(self.observations) != m["totals"]["observations"]:
            raise ValueError("Observation count mismatch")
        for fid, frame in self.frames.items():
            if len(self.by_frame[fid]) != frame["mask_count"]:
                raise ValueError("Frame mask count mismatch")

    @property
    def geometry_id(self):
        return self.manifest["geometry"]["geometry_id"]

    def require_geometry(self, xyz):
        actual = geometry_identity(xyz)["geometry_id"]
        if actual != self.geometry_id:
            raise ValueError("Geometry changed; recompute projections for this checkpoint")
        return actual

    def read_frame(self, fid, *, geometry_id):
        if geometry_id != self.geometry_id:
            raise ValueError("Point indices cannot be reused with another geometry")
        row = self.frames[fid]
        expected_relative = f"frames/f{fid:06d}.npz"
        if row["support_file"] != expected_relative:
            raise ValueError("Unexpected frame support path")
        path = self.root / expected_relative
        if sha256_file(path) != row["support_sha256"]:
            raise ValueError(f"Frame support hash mismatch: {fid}")
        with np.load(path, allow_pickle=False) as archive:
            pack = {k: archive[k] for k in archive.files}
        check_pack(pack, row, self.by_frame[fid], self.manifest["geometry"])
        return pack

    def support(self, observation_key, *, geometry_id):
        row = self.observations[observation_key]
        pack = self.read_frame(row["frame_id"], geometry_id=geometry_id)
        i = int(np.searchsorted(pack["mask_local_ids"], row["mask_local_id"]))
        lo, hi = map(int, pack["mask_offsets"][i:i + 2])
        return pack["mask_surface_indices"][lo:hi]

    def visible(self, frame_id, *, geometry_id):
        return self.read_frame(frame_id, geometry_id=geometry_id)["visible_surface_indices"]

    def validate(self, *, verify_sources=False):
        visible = positive = 0
        for fid in self.frames:
            pack = self.read_frame(fid, geometry_id=self.geometry_id)
            visible += len(pack["visible_surface_indices"])
            positive += len(pack["mask_surface_indices"])
        totals = self.manifest["totals"]
        if (visible != totals["visible_point_observations"]
                or positive != totals["positive_point_observations"]):
            raise ValueError("Cache aggregate support counts differ")
        if verify_sources:
            for record in self.manifest["locked_sources"]:
                verify_record(record)
            for record in self.manifest["prefix_sources"]:
                verify_prefix(record)
            for row in self.frames.values():
                for record in row["sources"].values():
                    verify_record(record)
        return dict(status="PASS", stage="raw_observation_cache_only",
                    cache_id=self.manifest["cache_id"], geometry_id=self.geometry_id,
                    totals=totals, source_files_verified=verify_sources,
                    map_modified=False, stability_or_repair_decisions_made=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--mapping-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--mode", choices=("offline", "online-v2"), required=True)
    build.add_argument("--checkpoint", type=int, required=True,
                       help="Number of processed mapping frames, not raw frame ID")
    verify = commands.add_parser("verify")
    verify.add_argument("--cache", type=Path, required=True)
    verify.add_argument("--verify-sources", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build_cache(args.mapping_root, args.output, mode=args.mode,
                             checkpoint=args.checkpoint)
        result = dict(status=result["status"], cache_id=result["cache_id"],
                      totals=result["totals"], seconds=result["seconds"])
    else:
        result = ObservationCache(args.cache).validate(verify_sources=args.verify_sources)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
