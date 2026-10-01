#!/usr/bin/env python3
"""Apply explicit observation identity updates and republish the same P0 surface.

Never edits the input run. This is an evidence revision interface, not an
automatic repair policy or a replay of subsequent online decisions.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revisable_instance_map.association import OnlineVoxelAssociator
from rebuild_p0_from_regions import rebuild_surface
from run_baseline_association import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("association-dir", "updates-json", "source-surface-dir", "observations", "config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    updates = json.loads(args.updates_json.read_text())
    if not isinstance(updates, dict):
        raise ValueError("Updates must be an observation_id -> instance_id JSON object")
    checkpoint = args.association_dir / "observation_support_3cm.npz"
    state = OnlineVoxelAssociator.from_checkpoint(checkpoint)
    if state.identity_evidence is None:
        raise ValueError("Revision export requires a binary-ledger or probabilistic run")
    rows = [json.loads(line) for line in (args.association_dir / "associations.jsonl").read_text().splitlines()]
    original_assigned = {row["observation_id"]: row["instance_id"] for row in rows}
    if len(original_assigned) != len(rows) or original_assigned != {key: entry.instance_id for key, entry in state.observation_support.items()}:
        raise ValueError("Input association table and checkpoint differ")
    state.reassign_observations(updates, args.reason)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    association_dir = args.output_dir / "association"
    association_dir.mkdir()
    for row in rows:
        entry = state.observation_support[row["observation_id"]]
        row.setdefault("initial_instance_id", row["instance_id"])
        row["instance_id"] = entry.instance_id
        row["assignment_version"] = entry.assignment_version
        row["published_map_version"] = state.map_version
    association_file = association_dir / "associations.jsonl"
    association_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    revised_checkpoint = association_dir / "observation_support_3cm.npz"
    state.save_checkpoint(revised_checkpoint)
    state.save_identity_counts(association_dir / "identity_voxel_counts_3cm.npz")
    events = association_dir / "identity_revision_events.jsonl"
    events.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in state.revision_events))
    selection = json.loads(args.config.read_text())["frame_selection"]
    frame_count = len([frame_id for frame_id in range(selection["start"], selection["stop_exclusive"], selection["stride"])
                       if frame_id <= state.last_processed_frame_id])
    surface = rebuild_surface(args.source_surface_dir, args.observations, association_file,
        revised_checkpoint, args.config, args.output_dir / "surface_p0",
        frame_count)
    write_json(args.output_dir / "revision_manifest.json", {
        "status": "PASS", "map_version": state.map_version, "next_instance_id": state.next_instance_id,
        "input_checkpoint_sha256": sha256_file(checkpoint), "updates_sha256": sha256_file(args.updates_json),
        "association_decisions_sha256": sha256_file(association_file),
        "identity_checkpoint_sha256": sha256_file(revised_checkpoint),
        "surface_instance_sha256": surface["instance_surface_sha256"],
        "source_mask_and_projected_support_unchanged": True,
        "subsequent_online_decisions_replayed": False,
        "revision_event_count": len(state.revision_events), "reason": args.reason,
    })
    print(f"PASS: explicit identity revision, map version {state.map_version}")


if __name__ == "__main__":
    main()
