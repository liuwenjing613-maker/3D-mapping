"""A shared final publication stage, never an online state transition.

The producer writes a scene-complete manifest AFTER its last export/repair.
The run command waits for the entire producer before consuming that manifest.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .core import Settings, complete

POLICY = "scene-final-B-20261101"
FRAME_IDS = list(range(0, 2000, 5))
WORKFLOWS = {"no_repair", "main", "auto-v3-swap", "online-v1", "online-v2"}
SCENES = {"room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"}
INPUT_KEYS = {"surface", "native", "evidence", "normal_geometry"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def write_scene_complete(path, *, scene, workflow, frame_ids, inputs,
                         final_processing_complete, final_repair_complete=False):
    """Producer hook: call ONLY after the scene's final export/repair has returned.

    This is not a checkpoint hook. Pass actually processed frame IDs, not planned
    IDs. Repaired workflows should prefer manifest_from_commit's receipt audit.
    """
    m = dict(schema_version=1, status="PASS", stage="scene_complete", scope="replica400",
             scene=scene, workflow=workflow, frame_ids=list(frame_ids), pending_stages=[],
             completion_history=[], final_processing_complete=final_processing_complete,
             final_repair_complete=final_repair_complete,
             inputs={k: str(Path(v).resolve()) for k, v in inputs.items()},
             input_sha256={k: sha(v) for k, v in inputs.items()})
    validate_manifest(m)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
        f.write("\n")
    return m


def manifest_from_commit(root, *, scene, workflow):
    """Adapt an existing terminal 400-frame repair commit, without changing it."""
    require(workflow in WORKFLOWS - {"no_repair"}, "repair workflow required")
    root = Path(root).resolve()
    mapping = read(root / "mapping/mapping_complete.json")
    commit_path = root / "repair/commit_complete.json"
    commit = read(commit_path)
    require(mapping.get("status") == "PASS" and mapping.get("frame_ids") == FRAME_IDS,
            "mapping is not a completed 400-frame scene")
    require(commit.get("status") == "PASS" and commit.get("state") == "COMMITTED_BRANCH_ONLY"
            and commit.get("processed_frames") == 400 and commit.get("last_raw_frame") == 1995,
            "last scene repair has not committed")
    require(sha(commit["final_map"]) == commit["final_map_sha256"], "commit final map changed")
    require(sha(commit["audit"]) == commit["audit_sha256"], "commit audit changed")
    audit = read(commit["audit"])
    for key in ("surface_journal_reproduces_all_candidates_and_votes", "surface_evidence_fields_exact",
                "native_labels_exact", "strict_max_one_vote_per_frame_surface_point"):
        require(audit.get(key) is True, "commit audit failed: " + key)
    surface = Path(commit["final_map"]).resolve()
    native = surface.parent.parent / "native"
    inputs = dict(surface=surface, native=native / "instance_surface.npz",
                  evidence=native / "surface_evidence.npz",
                  normal_geometry=root / "mapping/geometry/surface.ply")
    hashes = {k: sha(v) for k, v in inputs.items()}
    for key in ("surface", "evidence"):
        require(audit["input_sha256"].get(str(inputs[key])) == hashes[key], "unbound committed " + key)
    m = dict(schema_version=1, status="PASS", stage="scene_complete", scope="replica400",
             scene=scene, workflow=workflow, frame_ids=mapping["frame_ids"], pending_stages=[],
             completion_history=[], final_processing_complete=True, final_repair_complete=True,
             inputs={k: str(v) for k, v in inputs.items()}, input_sha256=hashes,
             source_commit=str(commit_path), source_commit_sha256=sha(commit_path))
    return validate_manifest(m)


def validate_manifest(m):
    require(m.get("schema_version") == 1, "unsupported scene manifest")
    require(m.get("workflow") in WORKFLOWS, "unknown workflow")
    require(m.get("scene") in SCENES, "unknown Replica scene")
    require(m.get("status") == "PASS" and m.get("stage") == "scene_complete",
            "B requires scene_complete, not a mapping pause/checkpoint")
    require(m.get("scope") == "replica400" and m.get("frame_ids") == FRAME_IDS,
            "B requires all 400 scene frames (0,5,...,1995), not a prefix")
    require(m.get("pending_stages") == [], "final processing is unfinished")
    require(m.get("completion_history") == [], "B has already been applied")
    require(m.get("final_processing_complete") is True, "final export is unfinished")
    if m["workflow"] != "no_repair":
        require(m.get("final_repair_complete") is True, "last repair is unfinished")
    require(set(m.get("inputs", {})) == INPUT_KEYS, "four source artifacts required")
    require(set(m.get("input_sha256", {})) == INPUT_KEYS, "source hashes required")
    for value in m["inputs"].values():
        require(Path(value).is_absolute(), "use absolute source paths")
    return m


def finalize(manifest_path, output):
    """Publish B once; preserve every input and never write an online checkpoint."""
    require(__debug__, "run without -O: the frozen B core uses invariant assertions")
    manifest_path = Path(manifest_path).resolve()
    m = validate_manifest(read(manifest_path))
    inputs = {k: Path(v).resolve() for k, v in m["inputs"].items()}
    hashes = {k: sha(v) for k, v in inputs.items()}
    require(hashes == m["input_sha256"], "scene inputs changed after final processing")
    output = Path(output).resolve()
    require(output != manifest_path.parent, "publish into a separate B directory")
    require(all(output != p.parent for p in inputs.values()), "do not overwrite source directories")
    signature = dict(policy=POLICY, manifest_sha256=sha(manifest_path),
                     input_sha256=hashes, core_sha256=sha(Path(__file__).with_name("core.py")),
                     pipeline_sha256=sha(Path(__file__)),
                     settings=asdict(Settings()))
    receipt_path = output / "scene_finalization.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt.get("signature") == signature, "output belongs to another input/version")
        for name, digest in receipt["output_sha256"].items():
            require(sha(output / name) == digest, "published artifact changed: " + name)
        return receipt  # Idempotent: no second call to complete().
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".finalizing"
    # Exclusive creation also prevents two callers from running B concurrently.
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        require(not any(p != lock for p in output.iterdir()),
                "incomplete/nonempty output: inspect it and use a new directory")
        started = time.perf_counter()
        surface, native, evidence = (load(inputs[k]) for k in ("surface", "native", "evidence"))
        require("scene_completion_policy" not in surface, "B output cannot be a B input")
        for key in ("xyz_m", "rgb"):
            require(np.array_equal(surface[key], native[key]) and
                    np.array_equal(surface[key], evidence[key]), "point order mismatch: " + key)
        require(np.array_equal(native["instance_id"],
                               np.where(evidence["state"] == 2, evidence["top1_instance_id"], -1)),
                "native labels do not match original confirmed evidence")
        original = surface["instance_id"]
        confirmed = native["instance_id"] > 0
        require(np.array_equal(original[confirmed], native["instance_id"][confirmed]),
                "final labels changed confirmed evidence")
        if inputs["normal_geometry"].suffix == ".npz":
            geometry = load(inputs["normal_geometry"])
            xyz, normals = geometry["xyz_m"], geometry["normals"]
        else:
            import open3d as o3d
            cloud = o3d.io.read_point_cloud(str(inputs["normal_geometry"]))
            xyz, normals = np.asarray(cloud.points), np.asarray(cloud.normals)
        require(xyz.shape == surface["xyz_m"].shape and
                np.allclose(xyz, surface["xyz_m"], rtol=0, atol=1e-5), "normal geometry mismatch")
        labels, diagnostics, stats = complete(surface["xyz_m"], normals, original, evidence)
        require(np.array_equal(labels[original != -1], original[original != -1]),
                "B modified an assigned label")
        np.savez_compressed(output / "instance_surface.npz", **{
            **surface, "instance_id": labels, "scene_completion_policy": np.asarray(POLICY)})
        np.savez_compressed(output / "local_diagnostics.npz", **diagnostics)
        stats.update(scene=m["scene"], workflow=m["workflow"],
                     total_surface_points=len(labels), baseline_unassigned=int((original == -1).sum()),
                     remaining_unassigned=int((labels == -1).sum()), GT_read=False,
                     seconds_including_io=time.perf_counter() - started)
        write(output / "completion_stats.json", stats)
        require({k: sha(v) for k, v in inputs.items()} == hashes and
                sha(manifest_path) == signature["manifest_sha256"], "inputs changed while running B")
        receipt = dict(status="PASS", stage="scene_published", policy=POLICY,
                       scene=m["scene"], workflow=m["workflow"], signature=signature,
                       frame_count=400, final_surface=str(output / "instance_surface.npz"),
                       pre_B_surface=str(inputs["surface"]), online_state_updated=False,
                       completion_history=[POLICY], C_enabled=False,
                       newly_assigned=stats["newly_assigned"],
                       output_sha256={n: sha(output / n) for n in
                                      ("instance_surface.npz", "local_diagnostics.npz", "completion_stats.json")})
        temporary = output / "scene_finalization.json.tmp"
        write(temporary, receipt)
        temporary.replace(receipt_path)  # Only this receipt marks publication complete.
        return receipt
    finally:
        lock.unlink()


def run_scene(manifest_path, output, command, cwd=None):
    """Run the whole producer, including final repair/export, then publish B."""
    require(bool(command), "whole-scene producer command required")
    require(not Path(manifest_path).exists(), "new run requires a fresh completion manifest")
    subprocess.run(command, cwd=cwd, check=True)
    return finalize(manifest_path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--cwd")
            p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("committed", help="publish a completed 400-frame repaired scene")
    p.add_argument("--root", required=True, help="terminal 400/A or 400/B directory")
    p.add_argument("--scene", required=True, choices=sorted(SCENES))
    p.add_argument("--workflow", required=True, choices=sorted(WORKFLOWS - {"no_repair"}))
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cwd")
    p.add_argument("command", nargs=argparse.REMAINDER,
                   help="optional whole-scene producer, run before inspecting its terminal commit")
    p = sub.add_parser("no-repair", help="run complete mapping/final export, then publish B")
    p.add_argument("--scene", required=True, choices=sorted(SCENES))
    p.add_argument("--mapping-receipt", required=True, help="producer receipt with status PASS and actual frame_ids")
    for key in sorted(INPUT_KEYS):
        p.add_argument("--" + key.replace("_", "-"), required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cwd")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        result = run_scene(args.manifest, args.out, command, args.cwd)
    elif args.action == "committed":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if command:
            require(not Path(args.manifest).exists(), "new producer requires a fresh manifest")
            subprocess.run(command, cwd=args.cwd, check=True)
        m = manifest_from_commit(args.root, scene=args.scene, workflow=args.workflow)
        path = Path(args.manifest)
        if path.exists():
            require(read(path) == m, "completion manifest changed")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as f:
                json.dump(m, f, indent=2)
                f.write("\n")
        result = finalize(args.manifest, args.out)
    elif args.action == "no-repair":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        require(bool(command), "whole-scene mapping and final export command required")
        require(not Path(args.manifest).exists(), "new producer requires a fresh manifest")
        subprocess.run(command, cwd=args.cwd, check=True)
        mapping = read(args.mapping_receipt)
        require(mapping.get("status") == "PASS", "mapping did not pass")
        write_scene_complete(args.manifest, scene=args.scene, workflow="no_repair",
                             frame_ids=mapping.get("frame_ids", []),
                             inputs={k: getattr(args, k) for k in INPUT_KEYS},
                             final_processing_complete=True)
        result = finalize(args.manifest, args.out)
    else:
        result = finalize(args.manifest, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
