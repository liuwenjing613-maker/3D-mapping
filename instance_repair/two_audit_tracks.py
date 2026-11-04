"""Stage 2: build auditable stable observation tracks from the shared raw cache."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import gzip
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from .two_audit_cache import ObservationCache, json_line, write_json
from .two_audit_matching import (Observation, TrackBuilder, TrackSettings,
                                 camera_view, different_view, view_distance)
from .two_audit_sources import canonical_hash, file_record, sha256_file, verify_record

VERSION = "two-audit-stable-tracks-v1"


def implementation_records():
    root = Path(__file__).parent
    return [file_record(root / name) for name in (
        "two_audit_matching.py", "two_audit_tracks.py", "two_audit_cache.py",
        "two_audit_sources.py", "two_audit_projection.py")]


def review_sample(track, cache):
    selected = [track["members"][0]]
    keys = {selected[0]["observation_key"]}
    by_key = {m["observation_key"]: m for m in track["members"]}
    for group in track["view_groups"][1:4]:
        row = by_key[group["representative_observation_key"]]
        if row["observation_key"] not in keys:
            selected.append(row)
            keys.add(row["observation_key"])
    last = track["members"][-1]
    if last["observation_key"] not in keys:
        selected.append(last)
    return dict(track_id=track["track_id"], status=track["status"],
                observation_count=track["observation_count"],
                independent_view_groups=track["independent_view_groups"],
                minimum_admitted_covisible_iou=track["minimum_admitted_covisible_iou"],
                representative_support_count=track["representative_support_count"],
                non_border_member_count=track["non_border_member_count"],
                sample_observations=[dict(m, source_mask=cache.frames[m["frame_id"]]["sources"]["mask"])
                                     for m in selected])


def build_tracks(cache_root, output, *, settings=None):
    settings = settings or TrackSettings()
    cache = ObservationCache(cache_root)
    output = Path(output).absolute()
    protected = (cache.root.resolve(), Path(cache.manifest["mapping_root"]).resolve())
    if any(output.resolve().is_relative_to(path) for path in protected):
        raise ValueError("Track output must be separate from the input cache and map")
    if output.exists():
        raise FileExistsError(f"Output already exists; nothing overwritten: {output}")
    source_records = [file_record(cache.root / name) for name in
                      ("manifest.json", "request.json", "mask_observations.jsonl", "frames.jsonl")]
    implementation = implementation_records()
    request = dict(version=VERSION, stage="stable_observation_tracks", cache_root=str(cache.root.resolve()),
                   cache_id=cache.manifest["cache_id"], geometry_id=cache.geometry_id,
                   surface_count=cache.manifest["geometry"]["shape"][0],
                   dataset=cache.manifest["dataset"], scene=cache.manifest["scene"],
                   mode=cache.manifest["mode"], checkpoint=cache.manifest["checkpoint"],
                   raw_cutoff=cache.manifest["raw_cutoff"], frame_ids=cache.manifest["frame_ids"],
                   settings=settings.record(), source_records=source_records,
                   implementation=implementation, numpy_version=np.__version__)
    run_id = canonical_hash(request)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "request.json", dict(request, run_id=run_id))
    builder = TrackBuilder(cache.geometry_id, request["surface_count"], settings)
    start = time.perf_counter()

    def progress(status, **kw):
        row = dict(status=status, stage=request["stage"], run_id=run_id,
                   processed_frames=builder.stats["frames"], total_frames=len(cache.frames),
                   observations=builder.stats["observations"], tracks=len(builder.tracks),
                   stable_tracks=sum(t.stable for t in builder.tracks),
                   comparisons=builder.stats["comparisons"], seconds=time.perf_counter()-start, **kw)
        write_json(output / "status.json", row)
        print(json.dumps(row), flush=True)

    try:
        progress("VALIDATING_INPUT")
        source_validation = cache.validate(verify_sources=True)
        write_json(output / "input_validation.json", source_validation)
        progress("RUNNING")
        pending_samples = []
        with ExitStack() as stack:
            decision_stream = stack.enter_context((output / "observation_decisions.jsonl").open("w"))
            pending_stream = stack.enter_context((output / "pending_observations.jsonl").open("w"))
            raw = stack.enter_context((output / "pair_evidence.jsonl.gz").open("wb"))
            compressed = stack.enter_context(gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                                           compresslevel=6, mtime=0))
            evidence_stream = stack.enter_context(io.TextIOWrapper(compressed, encoding="utf-8"))
            for fid in cache.frames:
                pack = cache.read_frame(fid, geometry_id=cache.geometry_id)
                original = cache.by_frame[fid]
                observations = []
                for i, row in enumerate(original):
                    lo, hi = map(int, pack["mask_offsets"][i:i+2])
                    observations.append(Observation(row["observation_key"], row["mask_local_id"],
                                                    pack["mask_surface_indices"][lo:hi]))
                decisions = builder.process_frame(
                    fid, pack["visible_surface_indices"], observations,
                    np.asarray(cache.frames[fid]["camera_to_world"], np.float64),
                    geometry_id=cache.geometry_id,
                    evidence_sink=lambda row: json_line(evidence_stream, row))
                for decision in decisions:
                    row = cache.observations[decision["observation_key"]]
                    decision.update(cache_id=cache.manifest["cache_id"],
                                    input_quality_flags=row["quality_flags"],
                                    source_mask_sha256=row["source_mask_sha256"])
                    json_line(decision_stream, decision)
                    if decision["status"].startswith("DEFERRED_"):
                        json_line(pending_stream, decision)
                        if decision["status"] != "DEFERRED_INSUFFICIENT_SUPPORT" and len(pending_samples) < 10:
                            pending_samples.append(decision)
                if builder.stats["frames"] % 25 == 0 or fid == cache.manifest["raw_cutoff"]:
                    progress("RUNNING", last_frame_id=fid)

        tracks = builder.records()
        for track in tracks:
            track["non_border_member_count"] = sum(
                not cache.observations[m["observation_key"]]["touches_image_border"]
                for m in track["members"])
        stable = [t for t in tracks if t["status"] == "STABLE"]
        totals = dict(builder.stats)
        assigned = sum(len(t["members"]) for t in tracks)
        totals.update(tracks=len(tracks), stable_tracks=len(stable),
                      provisional_tracks=len(tracks)-len(stable),
                      assigned_observations=assigned,
                      deferred_observations=totals["observations"]-assigned)
        if totals["observations"] != len(cache.observations):
            raise AssertionError("Not all raw observations were processed")
        if assigned != totals.get("NEW_TRACK", 0) + totals.get("MATCHED_TRACK", 0):
            raise AssertionError("Assignment counts do not balance")
        if totals["deferred_observations"] != sum(
                value for key, value in totals.items() if key.startswith("DEFERRED_")):
            raise AssertionError("Deferred observations were lost")
        write_json(output / "tracks.json", tracks)
        write_json(output / "stable_tracks.json", stable)
        fields = ["track_id", "status", "observation_count", "independent_view_groups",
                  "representative_frame_id", "representative_mask_local_id",
                  "representative_support_count", "minimum_admitted_covisible_iou",
                  "non_border_member_count", "first_stable_frame_id"]
        with (output / "track_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tracks)
        longest = sorted(stable, key=lambda t: (-t["observation_count"], t["track_id"]))[:6]
        smallest = sorted(stable, key=lambda t: (t["representative_support_count"], t["track_id"]))[:6]
        one_view = sorted((t for t in tracks if t["observation_count"] >= settings.min_observations
                           and t["independent_view_groups"] == 1),
                          key=lambda t: (-t["observation_count"], t["track_id"]))[:6]
        write_json(output / "review_samples.json",
                   dict(stable_long=[review_sample(t, cache) for t in longest],
                        stable_small_support=[review_sample(t, cache) for t in smallest],
                        repeated_single_view=[review_sample(t, cache) for t in one_view],
                        deferred_ambiguities=pending_samples,
                        interpretation="Stability is observational support, not instance truth or a repair decision."))

        progress("VERIFYING_OUTPUT")
        for record in source_records + implementation:
            verify_record(record)
        for frame in cache.frames.values():
            if sha256_file(cache.root / frame["support_file"]) != frame["support_sha256"]:
                raise ValueError("Input support pack changed during track construction")
        names = ("request.json", "input_validation.json", "observation_decisions.jsonl",
                 "pending_observations.jsonl", "pair_evidence.jsonl.gz", "tracks.json",
                 "stable_tracks.json", "track_summary.csv", "review_samples.json")
        hashes = {name: sha256_file(output / name) for name in names}
        manifest = dict(request, run_id=run_id, status="COMPLETE", totals=totals,
                        artifact_sha256=hashes, seconds=time.perf_counter()-start,
                        original_masks_only=True, input_cache_unchanged=True,
                        map_instance_ids_used=False, GT_used=False, map_modified=False,
                        anomaly_detection_implemented=False, repair_decisions_made=False,
                        pending_observations_assigned=False)
        write_json(output / "manifest.json", manifest)
        progress("COMPLETE")
        return manifest
    except Exception as exc:
        progress("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise


def validate_tracks(root, *, verify_inputs=False):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "COMPLETE" or manifest.get("version") != VERSION:
        raise ValueError("Unsupported or incomplete stable-track output")
    for name, digest in manifest["artifact_sha256"].items():
        if Path(name).name != name or sha256_file(root / name) != digest:
            raise ValueError(f"Track artifact hash mismatch: {name}")
    request = json.loads((root / "request.json").read_text())
    run_id = request.pop("run_id")
    if canonical_hash(request) != run_id or manifest["run_id"] != run_id:
        raise ValueError("Track run identity mismatch")
    if any(manifest[k] != value for k, value in request.items()):
        raise ValueError("Manifest differs from its locked request")
    cache = ObservationCache(request["cache_root"])
    if cache.geometry_id != request["geometry_id"] or cache.manifest["cache_id"] != request["cache_id"]:
        raise ValueError("Track input cache/geometry mismatch")
    for record in request["source_records"]:
        verify_record(record)
    settings = TrackSettings(**{k: request["settings"][k]
                               for k in TrackSettings.__dataclass_fields__})
    tracks = json.loads((root / "tracks.json").read_text())
    stable = json.loads((root / "stable_tracks.json").read_text())
    if stable != [t for t in tracks if t["status"] == "STABLE"]:
        raise ValueError("Stable subset differs from the full track registry")
    track_map, owner = {}, {}
    for i, track in enumerate(tracks):
        tid = track["track_id"]
        if tid != f"T{i+1:06d}" or tid in track_map or track["geometry_id"] != cache.geometry_id:
            raise ValueError("Invalid track identity")
        track_map[tid] = track
        members, groups = track["members"], track["view_groups"]
        if not members or track["representative_observation_key"] != members[0]["observation_key"]:
            raise ValueError("Fixed representative was changed")
        frames = [m["frame_id"] for m in members]
        if frames != sorted(set(frames)):
            raise ValueError("Track has duplicate or out-of-order frames")
        if track["observation_count"] != len(members) or track["independent_view_groups"] != len(groups):
            raise ValueError("Track counts are inconsistent")
        anchor_key = members[0]["observation_key"]
        expected_key = canonical_hash(dict(geometry_id=cache.geometry_id,
                                          representative_key=anchor_key, settings=settings.record()))
        if expected_key != track["track_key"]:
            raise ValueError("Track key does not bind the fixed representative")
        replay_groups, first_stable, member_groups = [], None, []
        for j, member in enumerate(members):
            key = member["observation_key"]
            if key in owner or key not in cache.observations:
                raise ValueError("An original observation was duplicated or invented")
            raw = cache.observations[key]
            if any(member[k] != raw[k] for k in ("frame_id", "mask_local_id", "surface_support_count")):
                raise ValueError("Track member metadata differs from the original cache")
            owner[key] = tid
            evidence = member["fixed_representative_evidence"]
            if j == 0:
                if evidence is not None or raw["surface_support_count"] < settings.min_common_points:
                    raise ValueError("Invalid initial representative")
            else:
                _check_pair(evidence, settings)
                if evidence["state"] != "MATCH":
                    raise ValueError("Non-matching observation was admitted")
            position, axis = camera_view(cache.frames[member["frame_id"]]["camera_to_world"])
            compatible = [g for g in replay_groups
                          if not different_view(*view_distance(position, axis, g), settings)]
            if compatible:
                group = compatible[0]
            else:
                group = dict(view_group_id=len(replay_groups)+1, position_m=position.tolist(),
                             optical_axis=axis.tolist(), representative_frame_id=member["frame_id"],
                             representative_observation_key=key, member_frame_ids=[])
                replay_groups.append(group)
            group["member_frame_ids"].append(member["frame_id"])
            member_groups.append(group["view_group_id"])
            if member["view_group_id"] != group["view_group_id"]:
                raise ValueError("View grouping differs from the approved fixed-representative rule")
            if j+1 >= settings.min_observations and len(replay_groups) >= settings.min_view_groups and first_stable is None:
                first_stable = member["frame_id"]
        if groups != replay_groups or track["first_stable_frame_id"] != first_stable:
            raise ValueError("View representatives/stability onset differ")
        is_stable = len(members) >= settings.min_observations and len(groups) >= settings.min_view_groups
        if (track["status"] == "STABLE") != is_stable:
            raise ValueError("Declared stability does not meet the approved thresholds")

    decisions, counts = {}, Counter()
    with (root / "observation_decisions.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row["observation_key"]
            if key in decisions or key not in cache.observations:
                raise ValueError("Decision coverage contains duplicates/unknown observations")
            decisions[key] = row
            counts[row["status"]] += 1
            if row["track_id"] != owner.get(key):
                raise ValueError("Decision and track assignment disagree")
            if row["status"].startswith("DEFERRED_") and key in owner:
                raise ValueError("Deferred evidence was used to support stability")
            if row["status"] == "DEFERRED_INSUFFICIENT_SUPPORT" and row["support_count"] >= settings.min_common_points:
                raise ValueError("Sufficient global support marked insufficient")
    if set(decisions) != set(cache.observations):
        raise ValueError("Original observations missing from the decision ledger")
    comparisons, pair_counts, matches = 0, Counter(), {}
    with gzip.open(root / "pair_evidence.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            _check_pair(row, settings)
            key, tid = row["observation_key"], row["track_id"]
            if key not in decisions or tid not in track_map:
                raise ValueError("Unknown observation/track in pair evidence")
            if track_map[tid]["representative_frame_id"] >= row["frame_id"]:
                raise ValueError("Pair evidence used a same-frame or future representative")
            comparisons += 1
            pair_counts["pair_"+row["state"]] += 1
            if row["state"] == "MATCH":
                matches.setdefault(key, []).append(tid)
    for key, row in decisions.items():
        if row["matched_track_ids"] != sorted(matches.get(key, [])):
            raise ValueError("Decision match list differs from measured pairs")
        if row["status"] == "DEFERRED_AMBIGUOUS" and len(row["matched_track_ids"]) < 2:
            raise ValueError("Ambiguity requires at least two eligible matches")
    totals = manifest["totals"]
    expected = dict(observations=len(decisions), frames=len(cache.frames), tracks=len(tracks),
                    stable_tracks=len(stable), provisional_tracks=len(tracks)-len(stable),
                    assigned_observations=len(owner), deferred_observations=len(decisions)-len(owner),
                    comparisons=comparisons, **counts, **pair_counts)
    if any(totals.get(k, 0) != value for k, value in expected.items()):
        raise ValueError("Track summary totals do not balance")
    with (root / "pending_observations.jsonl").open() as stream:
        pending = [json.loads(line) for line in stream]
    if pending != [r for r in decisions.values() if r["status"].startswith("DEFERRED_")]:
        raise ValueError("Deferred observations were lost or changed")
    if verify_inputs:
        cache.validate(verify_sources=True)
    return dict(status="PASS", version=VERSION, run_id=run_id, totals=totals,
                every_original_observation_accounted=True, pending_counted_as_stable=False,
                input_sources_reverified=verify_inputs, GT_used=False, map_modified=False,
                repair_decisions_made=False)


def _check_pair(row, settings):
    a, b, overlap, union = (row[k] for k in ("common_a", "common_b", "intersection", "union"))
    if min(a, b, overlap, union) < 0 or overlap > min(a, b) or union != a+b-overlap:
        raise ValueError("Invalid common-visibility set counts")
    enough = min(a, b) >= settings.min_common_points
    if not enough:
        if row["state"] != "INSUFFICIENT_COMMON_SUPPORT" or row["covisible_iou"] is not None:
            raise ValueError("Insufficient common visibility was treated as a decision")
    else:
        score = overlap / union
        state = "MATCH" if score >= settings.min_covisible_iou else "BELOW_IOU"
        if row["state"] != state or abs(row["covisible_iou"] - score) > 1e-12:
            raise ValueError("Pair classification does not match the approved IoU threshold")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--cache", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--tracks", type=Path, required=True)
    verify.add_argument("--verify-inputs", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        result = build_tracks(args.cache, args.output)
        result = dict(status=result["status"], run_id=result["run_id"],
                      totals=result["totals"], seconds=result["seconds"])
    else:
        result = validate_tracks(args.tracks, verify_inputs=args.verify_inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
