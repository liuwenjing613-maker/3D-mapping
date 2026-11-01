"""Server-only migration verification; never reads GT or runs a producer again."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scene_completion.pipeline import (FRAME_IDS, finalize, load, manifest_from_commit,
                                       read, sha, write, write_scene_complete)
from unified_eval.repair_profile import load_surface_export

OUT = Path('/data/chenkejun/CVPR/results/scene_final_b_publish_20261101')
SCENES = ['room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']
PRIOR = dict(no_repair=Path('/data/chenkejun/CVPR/results/uct_completion_b_replica8_20261101'),
             main=Path('/data/chenkejun/CVPR/results/uct_completion_main_replica8_20261101'))
SOURCE = Path('/data/chenkejun/CVPR/results/ovimap_aligned_main_onlinev2_20261101/main')


def main():
    rows = []
    code = ROOT / 'scene_completion/core.py'
    original = Path('/home/chenkejun/CVPR/experiments/uct_completion_bc_20261101/completion.py')
    assert sha(code) == sha(original)
    for workflow in ['no_repair', 'main']:
        for scene in SCENES:
            started = time.perf_counter()
            folder = OUT / workflow / scene
            folder.mkdir(parents=True, exist_ok=True)
            manifest = folder / 'scene_complete.json'
            freeze = read(PRIOR[workflow] / scene / 'B/prediction_freeze.json')
            assert freeze['frame_count'] == 400 and freeze['input_scope_verified'] is True
            if workflow == 'main':
                m = manifest_from_commit(SOURCE / scene / '400/A', scene=scene, workflow=workflow)
                if not manifest.exists():
                    write(manifest, m)
                else:
                    assert read(manifest) == m
            else:
                # Explicit migration of previously verified full-scene experiments.
                # This is not a production shortcut for creating completion receipts.
                assert freeze['frame_list_sha256'] == '937af0bde6876c9b8fc9fdccbef609641a0ec95b68b867f33720c2be678a1215'
                inputs = {k: freeze['source_files'][old] for k, old in
                          [('surface','A'), ('native','native'), ('evidence','evidence'), ('normal_geometry','normal_geometry')]}
                assert all(sha(v) == freeze['source_sha256']['A' if k == 'surface' else k] for k,v in inputs.items())
                if not manifest.exists():
                    write_scene_complete(manifest, scene=scene, workflow=workflow,
                                         frame_ids=FRAME_IDS, inputs=inputs, final_processing_complete=True)
            result = finalize(manifest, folder / 'B')
            expected = load(PRIOR[workflow] / scene / 'B/instance_surface.npz')
            actual = load(folder / 'B/instance_surface.npz')
            for key in expected:
                np.testing.assert_array_equal(expected[key], actual[key], err_msg=(workflow, scene, key))
            for name in ['local_diagnostics.npz']:
                a, b = load(PRIOR[workflow] / scene / 'B' / name), load(folder / 'B' / name)
                assert a.keys() == b.keys()
                for key in a:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=(workflow, scene, key))
            xyz, labels, inventory, *_ = load_surface_export(folder / 'B/instance_surface.npz')
            np.testing.assert_array_equal(xyz, expected['xyz_m'])
            np.testing.assert_array_equal(labels, expected['instance_id'])
            np.testing.assert_array_equal(inventory, expected['native_instance_ids'])
            assert result == finalize(manifest, folder / 'B')
            row = dict(workflow=workflow, scene=scene, points=len(labels),
                       newly_assigned=result['newly_assigned'], original_fields_exact=True,
                       diagnostics_exact=True, evaluator_loader_exact=True, idempotent=True,
                       seconds=time.perf_counter()-started)
            rows.append(row)
            write(OUT / 'progress.json', dict(status='RUNNING', finished=len(rows), last=row))
            print(json.dumps(row), flush=True)
    result = dict(status='PASS', policy='scene-final-B-20261101', regression_cases=16,
                  core_sha256=sha(code), original_core_byte_identical=True,
                  GT_read=False, new_metrics_claimed=False, rows=rows)
    write(OUT / 'validation.json', result)
    print('ALL PASS', flush=True)


if __name__ == '__main__':
    main()
