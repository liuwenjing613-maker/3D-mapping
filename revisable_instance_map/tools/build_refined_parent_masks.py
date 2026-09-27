#!/usr/bin/env python3
"""Ablation: retain only refined pixels, grouped by their actual raw source ID."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/home/chenkejun/CVPR/revisable_instance_map')
DATA = Path('/data/chenkejun/CVPR/revisable_instance_map')
raw_cfg = json.loads((ROOT / 'configs/replica_room0_stride5.json').read_text())
ref_cfg = json.loads((ROOT / 'configs/replica_room0_stride5_ovimap_refined.json').read_text())
out = DATA / 'ovimap_refined_regrouped_by_source_room0_stride5'
out.mkdir(parents=True, exist_ok=True)
records = []
raw_retained = 0
raw_removed = 0
for frame_id in range(0, 2000, 5):
    raw_path = Path(raw_cfg['source']['mask_root']) / raw_cfg['source']['mask_pattern'].format(frame=frame_id)
    ref_path = Path(ref_cfg['source']['mask_root']) / ref_cfg['source']['mask_pattern'].format(frame=frame_id)
    raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
    ref = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
    if raw is None or ref is None or raw.shape != ref.shape:
        raise ValueError(f'Mask error in frame {frame_id}')
    kept = np.where(ref > 0, raw, 0).astype(raw.dtype)
    if np.count_nonzero((kept > 0) & (raw == 0)):
        raise AssertionError('Added pixels have no raw source')
    output = out / f'frame{frame_id:06d}.png'
    if not cv2.imwrite(str(output), kept):
        raise OSError(output)
    raw_retained += int(np.count_nonzero(kept))
    raw_removed += int(np.count_nonzero((raw > 0) & (kept == 0)))
    records.append({'frame': frame_id, 'mask_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                    'instances': int(np.count_nonzero(np.unique(kept))),
                    'raw_mask_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    'refined_mask_sha256': hashlib.sha256(ref_path.read_bytes()).hexdigest()})
(out / 'frames.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in records))
cfg = json.loads(json.dumps(raw_cfg))
cfg['purpose'] = 'ablation_depth_refined_regrouped_by_actual_raw_source'
cfg['source']['mask_root'] = str(out)
cfg['mask']['observation_namespace'] = 'ovimap-parent-regrouped-v1'
cfg['mask']['provenance'] = 'raw mask intersected with positive OVI refined support; raw parent is birth unit'
(ROOT / 'configs/replica_room0_stride5_ovimap_parent_regrouped.json').write_text(json.dumps(cfg, indent=2) + '\n')
print(json.dumps({'frames': len(records), 'retained_pixels': raw_retained,
                  'removed_pixels': raw_removed, 'ground_truth_used': False}), flush=True)
