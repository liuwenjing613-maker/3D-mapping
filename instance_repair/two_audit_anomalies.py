"""Read-only two-audit overmerge review on a certified pre-repair TSDF snapshot."""
import argparse
from collections import defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
import time

import numpy as np

from .two_audit_anomaly_core import AnomalySettings, StableMapAudit, add_roi_comparison
from .two_audit_cache import ObservationCache, json_line, write_json
from .two_audit_map_inputs import AuditMapInputs
from .two_audit_matching import TrackSettings
from .two_audit_sources import canonical_hash, file_record, sha256_file, verify_record
from .two_audit_tracks import validate_tracks

VERSION = "two-audit-instance-anomalies-v1"


def implementation_records():
    here = Path(__file__).parent
    return [file_record(here/name) for name in (
        "two_audit_anomalies.py", "two_audit_anomaly_core.py", "two_audit_map_inputs.py",
        "two_audit_tracks.py", "two_audit_matching.py", "two_audit_cache.py",
        "two_audit_sources.py", "two_audit_projection.py")]


def load_inputs(tracks_root, verify_sources):
    tracks_root = Path(tracks_root).resolve(strict=True)
    validation = validate_tracks(tracks_root, verify_inputs=verify_sources)
    manifest = json.loads((tracks_root/"manifest.json").read_text())
    if manifest["settings"] != TrackSettings().record():
        raise ValueError("Anomaly v1 requires the confirmed stage-2 stability settings")
    cache = ObservationCache(manifest["cache_root"])
    stable = json.loads((tracks_root/"stable_tracks.json").read_text())
    inputs = AuditMapInputs(cache)
    records = [file_record(tracks_root/name) for name in
               ["manifest.json", *manifest["artifact_sha256"]]]
    records += [file_record(cache.root/name) for name in
                ("manifest.json", "request.json", "frames.jsonl", "mask_observations.jsonl")]
    return cache, stable, inputs, manifest, records, validation


def derive(cache, stable, inputs, settings, progress=None):
    collector = StableMapAudit(inputs.ids, stable, cache.geometry_id,
                               inputs.native_record["sha256"], settings)
    by_frame = defaultdict(list)
    for track in stable:
        for member in track["members"]:
            by_frame[member["frame_id"]].append((track["track_id"], member))
    for i, fid in enumerate(cache.frames):
        rows = []
        visible = np.empty(0, np.int32)
        if by_frame[fid]:
            pack = cache.read_frame(fid, geometry_id=cache.geometry_id)
            visible = pack["visible_surface_indices"]
            for tid, member in by_frame[fid]:
                index = int(np.searchsorted(pack["mask_local_ids"], member["mask_local_id"]))
                lo, hi = map(int, pack["mask_offsets"][index:index+2])
                rows.append((tid, member["observation_key"], member["mask_local_id"],
                             pack["mask_surface_indices"][lo:hi]))
        collector.consume_frame(fid, visible, rows, cache.frames[fid]["camera_to_world"],
                                geometry_id=cache.geometry_id)
        if progress and (i % 100 == 0 or i+1 == len(cache.frames)):
            progress("MEASURING", processed_frames=i+1, total_frames=len(cache.frames),
                     stable_observations=len(collector.coverage), pair_checks=len(collector.pair_checks))
    result = collector.finish()
    if progress:
        progress("COMPARING_ORIGINAL_ROIS", pairs=len(result["candidate_pairs"]))
    add_roi_comparison(result, inputs, stable)
    for row in result["observation_coverage"]:
        row["source_mask_sha256"] = cache.observations[row["observation_key"]]["source_mask_sha256"]
    for pair in result["candidate_pairs"]:
        values = [min(event["target_support_fractions"]) for event in pair["joint_observations"]]
        pair["minimum_joint_target_fraction"] = min(values)
        pair["median_joint_target_fraction"] = float(np.median(values))
    return result


def support_arrays(result):
    tids = list(result["supports"])
    arrays = [result["supports"][tid] for tid in tids]
    return dict(track_ids=np.asarray(tids, dtype="<U16"),
                offsets=np.r_[0, np.cumsum([len(a) for a in arrays], dtype=np.int64)],
                surface_indices=np.concatenate(arrays).astype(np.int32) if arrays else np.empty(0, np.int32))


def make_review_samples(result, cache):
    eligible = [p for p in result["candidate_pairs"] if p["status"] == "POTENTIAL_OVERMERGE"]
    ranked = sorted(eligible, key=lambda p: (-p["independent_joint_view_groups"],
                    -p["joint_frame_count"], p["target_instance_id"], p["track_ids"]))
    outside = [p for p in ranked if p["roi_comparison"]["outside_all_selected_roi_context"]]
    weak = sorted(eligible, key=lambda p: (p["minimum_joint_target_fraction"], p["pair_key"]))
    selected = {}
    for category, pairs in [("outside_roi_context", outside[:10]),
                            ("most_joint_views", ranked[:10]), ("low_target_fraction", weak[:10])]:
        for pair in pairs:
            sample = selected.setdefault(pair["pair_key"], dict(
                pair_key=pair["pair_key"], target_instance_id=pair["target_instance_id"],
                track_ids=pair["track_ids"], categories=[], status=pair["status"],
                independent_joint_view_groups=pair["independent_joint_view_groups"],
                minimum_joint_target_fraction=pair["minimum_joint_target_fraction"],
                roi_comparison=pair["roi_comparison"], joint_view_samples=[]))
            sample["categories"].append(category)
    pair_map = {p["pair_key"]: p for p in eligible}
    for key, sample in selected.items():
        pair = pair_map[key]
        events = {event["frame_id"]: event for event in pair["joint_observations"]}
        for group in pair["joint_view_groups"][:4]:
            fid = group["representative_frame_id"]
            sample["joint_view_samples"].append(dict(
                events[fid], source_mask=cache.frames[fid]["sources"]["mask"]))
    return dict(status="AWAITING_HUMAN_REVIEW", samples=list(selected.values()),
                interpretation="Potential overmerge is a review candidate, not a confirmed error. ROI overlap is geometric coverage, not proof of prior detection.")


def build_anomalies(tracks_root, output):
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(f"Output exists; nothing overwritten: {output}")
    tracks_root = Path(tracks_root).resolve(strict=True)
    input_manifest = json.loads((tracks_root/"manifest.json").read_text())
    cache_manifest = json.loads((Path(input_manifest["cache_root"])/"manifest.json").read_text())
    protected = [tracks_root, Path(input_manifest["cache_root"]).resolve(),
                 Path(cache_manifest["mapping_root"]).resolve().parent]
    if any(output.resolve().is_relative_to(path) for path in protected):
        raise ValueError("Audit output must be outside all input cache/map/candidate directories")
    implementation = implementation_records()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()

    def progress(status, **extra):
        row = dict(status=status, stage="read_only_instance_anomalies",
                   seconds=time.perf_counter()-start, **extra)
        write_json(output/"status.json", row)
        print(json.dumps(row), flush=True)

    try:
        progress("VALIDATING_SOURCES")
        cache, stable, inputs, track_manifest, records, validation = load_inputs(tracks_root, True)
        settings = AnomalySettings()
        request = dict(version=VERSION, tracks_root=str(tracks_root), tracks_run_id=track_manifest["run_id"],
            cache_id=cache.manifest["cache_id"], cache_root=str(cache.root.resolve()),
            geometry_id=cache.geometry_id, scene=cache.manifest["scene"], mode=cache.manifest["mode"],
            checkpoint=cache.manifest["checkpoint"], raw_cutoff=cache.manifest["raw_cutoff"],
            frame_ids=cache.manifest["frame_ids"], settings=settings.record(),
            map_inputs=inputs.record(), input_records=records,
            implementation=implementation, numpy_version=np.__version__)
        run_id = canonical_hash(request)
        write_json(output/"request.json", dict(request, run_id=run_id))
        write_json(output/"input_validation.json", validation)
        result = derive(cache, stable, inputs, settings, progress)
        json_files = dict(track_instance_coverage=result["track_coverage"],
                          candidate_pairs=result["candidate_pairs"],
                          instance_anomalies=result["instance_anomalies"],
                          review_samples=make_review_samples(result, cache))
        for name, value in json_files.items():
            write_json(output/(name+".json"), value)
        with (output/"observation_instance_coverage.jsonl").open("w", encoding="utf-8") as stream:
            for row in result["observation_coverage"]:
                json_line(stream, row)
        with (output/"same_frame_pair_checks.jsonl.gz").open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                    for row in result["pair_checks"]:
                        json_line(stream, row)
        np.savez_compressed(output/"stable_track_supports.npz", **support_arrays(result))
        fields = ["target_instance_id", "track_a", "track_b", "status", "joint_frame_count",
                  "independent_joint_view_groups", "minimum_joint_target_fraction",
                  "median_joint_target_fraction", "full_footprint_roi_relation",
                  "full_footprint_roi_context_fraction", "target_footprint_roi_context_fraction",
                  "outside_all_selected_roi_context", "pair_key"]
        with (output/"pair_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for pair in result["candidate_pairs"]:
                r = pair["roi_comparison"]
                writer.writerow(dict(target_instance_id=pair["target_instance_id"],
                    track_a=pair["track_ids"][0], track_b=pair["track_ids"][1], status=pair["status"],
                    joint_frame_count=pair["joint_frame_count"],
                    independent_joint_view_groups=pair["independent_joint_view_groups"],
                    minimum_joint_target_fraction=pair["minimum_joint_target_fraction"],
                    median_joint_target_fraction=pair["median_joint_target_fraction"],
                    full_footprint_roi_relation=r["full_mask_footprint"]["relation"],
                    full_footprint_roi_context_fraction=r["full_mask_footprint"]["context_fraction"],
                    target_footprint_roi_context_fraction=r["target_instance_footprint"]["context_fraction"],
                    outside_all_selected_roi_context=r["outside_all_selected_roi_context"], pair_key=pair["pair_key"]))
        progress("VERIFYING_OUTPUT")
        inputs.verify_unchanged()
        for record in records+implementation:
            verify_record(record)
        files = ["request.json", "input_validation.json", "observation_instance_coverage.jsonl",
                 "same_frame_pair_checks.jsonl.gz", "stable_track_supports.npz", "pair_summary.csv"]
        files += [name+".json" for name in json_files]
        manifest = dict(request, run_id=run_id, status="COMPLETE", totals=result["totals"],
            artifact_sha256={name:sha256_file(output/name) for name in files},
            seconds=time.perf_counter()-start, GT_used=False, map_modified=False,
            candidate_generation_uses_roi=False, coverage_ratio_gate_used=False,
            repair_triggered=False, new_roi_written=False,
            human_instance_quality_status="NOT_ANNOTATED", repair_effectiveness="NOT_EVALUATED")
        write_json(output/"manifest.json", manifest)
        progress("COMPLETE", totals=result["totals"])
        return manifest
    except Exception as exc:
        progress("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def validate_anomalies(root, *, verify_inputs=False):
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest.get("version") != VERSION or manifest.get("status") != "COMPLETE":
        raise ValueError("Unsupported/incomplete anomaly output")
    for name, expected in manifest["artifact_sha256"].items():
        if Path(name).name != name or sha256_file(root/name) != expected:
            raise ValueError(f"Anomaly artifact hash mismatch: {name}")
    request = json.loads((root/"request.json").read_text())
    run_id = request.pop("run_id")
    if run_id != canonical_hash(request) or run_id != manifest["run_id"]:
        raise ValueError("Anomaly run identity mismatch")
    if any(manifest[k] != value for k, value in request.items()):
        raise ValueError("Anomaly manifest differs from its locked request")
    for record in request["input_records"]+request["implementation"]+request["map_inputs"]["sources"]:
        verify_record(record)
    cache, stable, inputs, tracks, _, _ = load_inputs(request["tracks_root"], verify_inputs)
    if (cache.geometry_id != request["geometry_id"] or cache.manifest["cache_id"] != request["cache_id"]
            or tracks["run_id"] != request["tracks_run_id"] or inputs.record() != request["map_inputs"]):
        raise ValueError("Anomaly sources changed")
    settings = AnomalySettings()
    if request["settings"] != settings.record():
        raise ValueError("Anomaly v1 settings differ from the confirmed design")
    result = derive(cache, stable, inputs, settings)
    for name, expected in [("track_instance_coverage.json", result["track_coverage"]),
                           ("candidate_pairs.json", result["candidate_pairs"]),
                           ("instance_anomalies.json", result["instance_anomalies"]),
                           ("review_samples.json", make_review_samples(result, cache))]:
        if json.loads((root/name).read_text()) != expected:
            raise ValueError(f"Recomputed evidence differs: {name}")
    with (root/"observation_instance_coverage.jsonl").open() as stream:
        if [json.loads(line) for line in stream] != result["observation_coverage"]:
            raise ValueError("Original-observation coverage replay differs")
    with gzip.open(root/"same_frame_pair_checks.jsonl.gz", "rt") as stream:
        if [json.loads(line) for line in stream] != result["pair_checks"]:
            raise ValueError("Same-frame pair evidence replay differs")
    with np.load(root/"stable_track_supports.npz", allow_pickle=False) as archive:
        for name, expected in support_arrays(result).items():
            if not np.array_equal(archive[name], expected):
                raise ValueError("Stable-track support union differs")
    if manifest["totals"] != result["totals"]:
        raise ValueError("Anomaly totals differ from replay")
    inputs.verify_unchanged()
    return dict(status="PASS", version=VERSION, run_id=run_id, totals=result["totals"],
                full_geometry_and_membership_replay=True, raw_sources_reverified=verify_inputs,
                GT_used=False, map_modified=False, repair_triggered=False,
                confirmed_errors_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--tracks", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--audit", required=True, type=Path)
    verify.add_argument("--verify-inputs", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build_anomalies(args.tracks, args.output)
        result = {k:result[k] for k in ("status", "run_id", "totals", "seconds")}
    else:
        result = validate_anomalies(args.audit, verify_inputs=args.verify_inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
