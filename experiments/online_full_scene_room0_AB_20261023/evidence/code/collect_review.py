"""Package completed-run evidence without copying datasets or model weights."""
from pathlib import Path
import hashlib, json, shutil, tarfile

HERE = Path(__file__).resolve().parent
OUT = Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    audit = json.loads((OUT / 'verification/causal_audit.json').read_text())
    assert audit['status'] == 'PASS'
    review = OUT / 'review'
    review.mkdir(exist_ok=True)
    files = ['run_plan.json', 'adapter_audit.json', 'status_A.json', 'status_B.json',
             'stage_timings_A.jsonl', 'stage_timings_B.jsonl', 'verification/causal_audit.json',
             'verification/current_protocol_preflight.json']
    checkpoint_files = ['mapping/mapping_complete.json', 'mapping/mapping_timing.json', 'mapping/restore_audit.json',
                        'candidates/selection_complete.json', 'candidates/online_qa.json',
                        'tracking/complete.json', 'tracking/cache_fingerprint_audit.json',
                        'repair/commit_complete.json']
    for group in ['A', 'B']:
        files.append(f'deliverables/room0_{group}_400_export.json')
        for n in ([300, 400] if group == 'A' else [200, 300, 400]):
            files += [f'{n}/{group}/{p}' for p in checkpoint_files]
            files += [f'{n}/{group}/mapping/frame{n-99}_state_audit.json']
        e = OUT / 'evaluation_r2' / group
        assert json.loads((e / 'evaluation_complete.json').read_text())['status'] == 'PASS'
        files += [str(p.relative_to(OUT)) for p in e.rglob('*.json')]
    manifest = []
    for rel in files:
        src, dst = OUT / rel, review / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst)
        manifest.append({'path': rel, 'source': str(src), 'sha256': sha(src), 'bytes': src.stat().st_size})
    for src in HERE.glob('*.py'):
        dst = review / 'code' / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    (review / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = OUT / 'room0_AB_400_review.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(review, arcname='review')
    print(json.dumps({'status': 'PASS', 'files': len(files), 'review': str(review),
                      'archive': str(archive), 'archive_bytes': archive.stat().st_size, 'sha256': sha(archive)}))

if __name__ == '__main__':
    main()
