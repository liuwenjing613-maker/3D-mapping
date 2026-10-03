"""Read-only verification of the committed experiment archive (stdlib only)."""
from pathlib import Path
import ast
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'artifact_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        file = root / entry['path']
        assert file.is_file(), file
        payload = file.read_bytes()
        assert len(payload) == entry['bytes'], file
        assert hashlib.sha256(payload).hexdigest() == entry['sha256'], file
        assert file.suffix.lower() not in {'.ply', '.pth', '.pt', '.npz', '.zip'}, file
        if file.suffix == '.py':
            ast.parse(payload.decode('utf-8'), filename=str(file))
    ranked = json.loads((root / 'roi_review/evidence/roi_ranked.json').read_text(encoding='utf-8'))
    assert len(ranked) == 1352
    assert sum(r['core_points'] for r in ranked) == 145379
    assert sum(r['viewable'] for r in ranked) == 1150
    selection = json.loads((root / 'roi_review/evidence/review_selection.json').read_text(encoding='utf-8'))
    assert len(selection['rendered']) == 32
    base = root / 'cropformer_zoom'
    inputs = json.loads((base / 'evidence/input_manifest.json').read_text(encoding='utf-8'))
    records = {r['key']: r for r in map(json.loads, (base / 'evidence/inference_records.jsonl').read_text(encoding='utf-8').splitlines())}
    crops = [j for j in inputs['jobs'] if j['kind'] == 'ROI_crop']
    controls = [r for r in records.values() if r['kind'] == 'full_frame_control']
    assert len(crops) == 159 and len(controls) == 79 and len(records) == 238
    for job in crops:
        key = job['key']
        files = [
            (base / 'native/inputs' / (key + '_rgb.png'), job['RGB_file_sha256']),
            (base / 'native/inputs' / (key + '_baseline.png'), job['baseline_file_sha256']),
            (base / 'native/inference' / (key + '.png'), records[key]['mask_file_sha256']),
        ]
        for file, expected in files:
            assert hashlib.sha256(file.read_bytes()).hexdigest() == expected, file
    assert all(r['frozen_cache_comparison']['identical_label_image'] for r in controls)
    audit = json.loads((base / 'evidence/visual_review_audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'PASS' and audit['no_depth_refinement'] and audit['no_GT_used']
    frozen = json.loads((base / 'evidence/frozen_map_hash_audit.json').read_text(encoding='utf-8'))
    assert frozen['unchanged']
    print(json.dumps({'status': 'PASS', 'archived_files': len(manifest['files']), 'ROI_candidates': len(ranked), 'selected_ROIs': 32, 'crop_inputs_outputs_SHA256_verified': len(crops), 'pixel_identical_full_frame_control_records': len(controls), 'note': 'validates archive and recorded controls; does not rerun inference or access source maps'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
