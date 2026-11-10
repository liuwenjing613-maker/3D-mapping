#!/usr/bin/env python3
"""Record exact source-mask / depth-region intersections without hard births.

A geometric region can overlap several CropFormer parents. Its majority label is
never treated as authoritative provenance. Raw and refined PNGs remain immutable.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-config', type=Path, required=True)
    ap.add_argument('--refined-config', type=Path, required=True)
    ap.add_argument('--raw-observations', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    raw_cfg = json.loads(args.raw_config.read_text())
    refined_cfg = json.loads(args.refined_config.read_text())
    if raw_cfg['frame_selection'] != refined_cfg['frame_selection']:
        raise ValueError('Frame selections differ')
    if raw_cfg['scene'] != refined_cfg['scene']:
        raise ValueError('Scene mismatch')
    fs = raw_cfg['frame_selection']
    frames = range(fs['start'], fs['stop_exclusive'], fs['stride'])
    catalog = defaultdict(dict)
    for line in args.raw_observations.read_text().splitlines():
        row = json.loads(line)
        catalog[row['frame_id']][row['mask_local_id']] = row
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'parent_child_lineage.jsonl'
    temp = path.with_suffix('.jsonl.tmp')
    orphan_path = args.output_dir / 'background_refined_regions.jsonl'
    orphan_temp = orphan_path.with_suffix('.jsonl.tmp')
    totals = defaultdict(int)
    with temp.open('w', encoding='utf-8') as out, orphan_temp.open('w', encoding='utf-8') as orphan_out:
        for frame_id in frames:
            raw_path = Path(raw_cfg['source']['mask_root']) / raw_cfg['source']['mask_pattern'].format(frame=frame_id)
            ref_path = Path(refined_cfg['source']['mask_root']) / refined_cfg['source']['mask_pattern'].format(frame=frame_id)
            raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
            refined = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
            if raw is None or refined is None or raw.shape != refined.shape:
                raise ValueError(f'Mask load/shape error in frame {frame_id}')
            raw_sha, ref_sha = digest(raw_path), digest(ref_path)
            stride = int(refined.max()) + 1
            pair, counts = np.unique(raw.astype(np.int64).ravel() * stride + refined.astype(np.int64).ravel(), return_counts=True)
            support = defaultdict(dict)
            for code, count in zip(pair.tolist(), counts.tolist()):
                raw_id, ref_id = divmod(code, stride)
                support[raw_id][ref_id] = count
            if set(support) - {0} != set(catalog[frame_id]):
                raise ValueError(f'Raw IDs disagree in frame {frame_id}')
            for raw_id, obs in sorted(catalog[frame_id].items()):
                parts = support[raw_id]
                if sum(parts.values()) != obs['pixel_count'] or obs['source_mask_sha256'] != raw_sha:
                    raise ValueError(f'Raw provenance mismatch in frame {frame_id}, mask {raw_id}')
                record = {
                    'parent_observation_id': obs['observation_id'],
                    'frame_id': frame_id,
                    'parent_mask_local_id': raw_id,
                    'raw_mask_sha256': raw_sha,
                    'refined_mask_sha256': ref_sha,
                    'raw_pixel_count': obs['pixel_count'],
                    'residual_pixel_count': parts.get(0, 0),
                    'children': [{'refined_local_id': child_id, 'pixel_count': count}
                                 for child_id, count in sorted(parts.items()) if child_id > 0],
                }
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                totals['parents'] += 1
                totals['residual_pixels'] += parts.get(0, 0)
                totals['parent_pixels'] += obs['pixel_count']
                totals['child_intersections'] += len(record['children'])
            for child_id, count in sorted(support.get(0, {}).items()):
                if child_id > 0:
                    orphan_out.write(json.dumps({'frame_id': frame_id, 'refined_local_id': child_id,
                                                 'pixel_count': count, 'raw_mask_sha256': raw_sha,
                                                 'refined_mask_sha256': ref_sha}) + '\n')
                    totals['orphan_refined_pixels'] += count
                    totals['orphan_intersections'] += 1
            totals['frames'] += 1
    temp.replace(path)
    orphan_temp.replace(orphan_path)
    report = dict(totals)
    report.update({'lineage_sha256': digest(path), 'background_regions_sha256': digest(orphan_path),
                   'raw_config_sha256': digest(args.raw_config),
                   'refined_config_sha256': digest(args.refined_config),
                   'raw_catalog_sha256': digest(args.raw_observations),
                   'ground_truth_used': False, 'raw_proposals_are_birth_units': True})
    (args.output_dir / 'lineage_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    sys.exit(main())
