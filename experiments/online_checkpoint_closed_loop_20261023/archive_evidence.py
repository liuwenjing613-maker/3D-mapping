"""Archive small completed evidence plus hashes; keep numerical data on the data disk."""
from pathlib import Path
import hashlib,json,shutil,tarfile

HERE=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()

def main():
    assert read(OUT/'validation_tail_status.json')['status']=='COMPLETE'
    assert (OUT/'final_audit.json').exists() and (OUT/'current_source_audit.json').exists()
    root=HERE/'evidence';root.mkdir(exist_ok=True)
    paths=list(OUT.glob('*.json'))
    for mode in ['reference','checkpoint','fixed_reference','fixed_checkpoint']:
        base=OUT/mode
        rels=['input_fence_audit.json','tracking/cache_fingerprint_audit.json','tracking/tracking_queue_complete.json',
            'tracking/gate_complete.json','repair/repair_complete.json','repair/commit_complete.json',
            'repair/commit_audit.json']
        if mode.startswith('fixed_'):rels+=['fixed_surface_control.json']
        else:rels+=['mapping/prior_commit.json','candidates/online_qa.json',
            'candidates/exact_model_cache_audit.json','candidates/selection_complete.json','tracking/complete.json',
            'repair_execution_timing.json','commit_execution_timing.json','frame201/mapping/frame201_state_audit.json',
            'frame201/mapping/restore_audit.json','frame201/mapping/mapping_complete.json']
        paths.extend(base/rel for rel in rels if (base/rel).exists())
        commit=read(base/'repair/commit_complete.json')
        paths.append(Path(commit['audit']))
    manifest=[]
    for p in sorted(set(paths)):
        rel=p.relative_to(OUT);dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        # Some audit files include very large per-input inventories: preserve their hash and source path.
        size=p.stat().st_size
        copied=size<=2_000_000
        if copied:
            shutil.copyfile(p,dest)
            assert sha(p)==sha(dest)
        manifest.append(dict(relative_path=str(rel),source=str(p),sha256=sha(p),bytes=size,copied=copied))
    (root/'manifest.json').write_text(json.dumps(dict(status='PASS',entries=manifest),indent=2)+'\n')
    bundle=OUT/'review_bundle.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        for p in sorted(root.rglob('*')):
            if p.is_file():archive.add(p,arcname=str(p.relative_to(HERE)),recursive=False)
        for p in sorted(HERE.glob('*')):
            if p.is_file() and (p.suffix in ['.py','.md'] or p.name=='.gitignore'):
                archive.add(p,arcname=p.name,recursive=False)
    print(json.dumps(dict(status='PASS',entries=len(manifest),copied=sum(r['copied'] for r in manifest),
        copied_bytes=sum(r['bytes'] for r in manifest if r['copied']),bundle=str(bundle),bundle_sha256=sha(bundle))))

if __name__=='__main__':main()
