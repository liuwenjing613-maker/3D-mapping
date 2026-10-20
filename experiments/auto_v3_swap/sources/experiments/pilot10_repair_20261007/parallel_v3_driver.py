"""Schedule independent maps concurrently; reuse the unchanged frozen evaluator."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, time, traceback
import evaluate_pilot_v3 as evaluator


def main():
    e = evaluator
    out = e.ROOT / 'v3'
    freeze = json.loads((out / 'evaluation_freeze.json').read_text())
    assert freeze['code_sha256'] == e.sha(e.__file__)
    assert json.loads((e.ROOT / 'repair/status.json').read_text())['status'] == 'PASS'
    for row in freeze['predictions'].values():
        assert e.sha(row['path']) == row['sha256']
    human_dir = e.ROOT / 'human_seeds_20ad6139511b'
    manifest = json.loads((human_dir / 'seed_manifest.json').read_text())
    assert e.sha(human_dir / 'source_human_export.json') == manifest['source_export_sha256']
    parser = e.ArgumentParser()
    e.add_protocol_args(parser)
    protocol, raw, debug = e.load_protocol(e.PROTOCOL, parser.parse_args(['--config', str(e.PROTOCOL), *e.FLAGS]))
    jobs = {}
    for seed in manifest['seeds']:
        scene, uid = seed['scene'], seed['case_uid']
        jobs['baseline:' + scene] = (scene, e.BASE / f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz', out / 'baseline' / scene)
        if seed['status'] == 'READY':
            for condition in ['seed_only', 'seed_plus_short_track']:
                jobs[uid + ':' + condition] = (scene, e.ROOT / 'repair' / uid / condition / 'final/instance_surface.npz', out / uid / condition)
    status = {'status': 'RUNNING', 'total_maps': len(jobs), 'completed_maps': [],
              'flags': e.FLAGS, 'predictions_frozen_before_GT': True, 'workers': 3}
    e.dump(out / 'status.json', status)
    e.dump(out / 'parallel_provenance.json', {'driver_sha256': e.sha(__file__),
        'unchanged_evaluator_sha256': e.sha(e.__file__), 'workers': 3,
        'existing_frozen_predictions_verified_before_GT': True})
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {pool.submit(e.evaluate, scene, surface, destination, protocol): key
                 for key, (scene, surface, destination) in jobs.items()}
        for task in as_completed(tasks):
            task.result()
            status['completed_maps'].append(tasks[task])
            e.dump(out / 'status.json', status)
    # Every map is now cached. The exact original aggregation/target/damage
    # routine checks the cached hashes and produces its final report.
    e.main()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        evaluator.dump(evaluator.ROOT / 'v3/failure.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        raise
