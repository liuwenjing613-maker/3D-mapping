#!/usr/bin/env python3
"""Hold raw-parent instance IDs fixed while changing observation pixel support."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-associations', type=Path, required=True)
    ap.add_argument('--target-observations', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = {}
    for line in args.raw_associations.read_text().splitlines():
        row = json.loads(line)
        key = (row['frame_id'], row['mask_local_id'])
        if key in source:
            raise ValueError('Duplicate source parent')
        source[key] = row
    rows = []
    for line in args.target_observations.read_text().splitlines():
        obs = json.loads(line)
        key = (obs['frame_id'], obs['mask_local_id'])
        if key not in source:
            raise ValueError(f'Target lacks raw parent: {key}')
        old = source[key]
        rows.append({'frame_id': obs['frame_id'], 'mask_local_id': obs['mask_local_id'],
                     'observation_id': obs['observation_id'], 'instance_id': old['instance_id'],
                     'decision': 'fixed_raw_parent_assignment',
                     'source_raw_observation_id': old['observation_id']})
    if len({r['observation_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate target observation')
    path = args.output_dir / 'associations.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = {'status': 'PASS', 'purpose': 'fixed_raw_parent_association_for_materialization_ablation',
              'frame_count': len({row['frame_id'] for row in rows}),
              'decision_count': len(rows), 'instance_count': len({row['instance_id'] for row in rows}),
              'missing_source_parents': len(source) - len(rows),
              'raw_association_sha256': sha(args.raw_associations),
              'target_observation_catalog_sha256': sha(args.target_observations),
              'associations_sha256': sha(path), 'ground_truth_used': False}
    (args.output_dir / 'association_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
