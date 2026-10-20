#!/usr/bin/env python3
"""Freeze the room0 no-repair ablation inputs, outputs, code and evaluation."""
import hashlib
import json
from pathlib import Path

HOME = Path('/home/chenkejun/CVPR/revisable_instance_map')
DATA = Path('/data/chenkejun/CVPR/revisable_instance_map')
NAMES = [
    'room0_no_repair', 'room0_ovimap_refined', 'room0_ovimap_refined_multi',
    'raw_multi', 'parent_raw_multi_final_vf_dedup',
    'ovimap_parent_fixed_raw', 'ovimap_parent_regrouped_single',
    'ovimap_parent_regrouped_multi', 'ovimap_parent_regrouped_multi_vf_dedup',
    'ovimap_parent_regrouped_fallback_multi',
]
CODE = [
    'src/revisable_instance_map/association.py',
    'tools/run_baseline_association.py',
    'tools/materialize_instance_map.py',
    'tools/build_parent_child_lineage.py',
    'tools/build_refined_parent_masks.py',
    'tools/remap_parent_associations.py',
    'tools/adapt_instance_map_replica_ca.py',
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    comparison = {}
    protocols, gts = set(), set()
    for name in NAMES:
        root = DATA / f'evaluation_{name}'
        metric = json.loads((root / 'metrics/metrics.json').read_text())
        adapter = json.loads((root / 'adapter/adapter_manifest.json').read_text())
        protocols.add(adapter['protocol_sha256'])
        gts.add(adapter['gt_sha256'])
        pq = metric['CA_PQ']
        comparison[name] = {
            'AP50': metric['CA_AP50_uniform'], 'PQ': pq['PQ'],
            'TP': pq['TP'], 'FP': pq['FP'], 'FN': pq['FN'],
            'metrics_sha256': sha(root / 'metrics/metrics.json'),
            'prediction_sha256': sha(root / 'adapter/canonical_prediction.npz'),
        }
    if len(protocols) != 1 or len(gts) != 1:
        raise ValueError('Evaluation protocol or GT differs across ablations')
    source_paths = {
        'raw_config': HOME / 'configs/replica_room0_stride5.json',
        'refined_config': HOME / 'configs/replica_room0_stride5_ovimap_refined.json',
        'regrouped_config': HOME / 'configs/replica_room0_stride5_ovimap_parent_regrouped.json',
        'raw_observations': DATA / 'raw_observations_room0_stride5/observations.jsonl',
        'regrouped_observations': DATA / 'observations_ovimap_parent_regrouped/observations.jsonl',
        'parent_child_lineage': DATA / 'parent_child_lineage_room0_stride5/parent_child_lineage.jsonl',
        'background_regions': DATA / 'parent_child_lineage_room0_stride5/background_refined_regions.jsonl',
        'regrouped_mask_manifest': DATA / 'ovimap_refined_regrouped_by_source_room0_stride5/frames.jsonl',
        'association_decisions': DATA / 'association_ovimap_parent_regrouped_multi/associations.jsonl',
        'observation_support_3cm': DATA / 'association_ovimap_parent_regrouped_multi/observation_support_3cm.npz',
        'instance_surface': DATA / 'map_ovimap_parent_regrouped_multi_vf_dedup/instance_surface.npz',
        'surface_support_votes_1cm': DATA / 'map_ovimap_parent_regrouped_multi_vf_dedup/surface_support_votes_1cm.npz',
        'tsdf_surface': DATA / 'geometry_full400_1cm_capacity100k/surface.ply',
    }
    manifest = {
        'status': 'PASS', 'scene': 'Replica/room0', 'frame_count': 400,
        'ground_truth_used_in_mapping': False,
        'evaluation_protocol_sha256': next(iter(protocols)),
        'evaluation_gt_sha256': next(iter(gts)),
        'source_files': {name: {'path': str(path), 'sha256': sha(path)} for name, path in source_paths.items()},
        'code_sha256': {name: sha(HOME / name) for name in CODE},
        'comparison': comparison,
        'recommended': 'ovimap_parent_regrouped_multi_vf_dedup',
        'scope': 'room0 only; no cross-scene generalization claim',
    }
    path = DATA / 'parent_regrouped_v1_manifest.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)
    print(json.dumps({'status': 'PASS', 'variants': len(comparison),
                      'protocol_sha256': next(iter(protocols)),
                      'gt_sha256': next(iter(gts)),
                      'recommended_AP50': comparison['ovimap_parent_regrouped_multi_vf_dedup']['AP50']}), flush=True)


if __name__ == '__main__':
    main()
