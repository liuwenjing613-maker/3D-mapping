"""Path-only adapters for the authorized original/checkpoint closed-loop comparison."""
from pathlib import Path
import ast
import hashlib
import json

REPO=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023')
SRC=REPO/'experiments/online_room0_repair50_to100_20261022/continuation300'
OLD_CODE=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/continuation300')
OLD_ROOT=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300/200/A')
HERE=REPO/'experiments/online_checkpoint_closed_loop_20261023'
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
MAPS=Path('/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2)+'\n')

rows=[]
for mode in ['reference','checkpoint']:
    code=HERE/mode
    root=OUT/mode
    assert not code.exists() and not root.exists(), 'Fresh isolated paths required'
    code.mkdir(parents=True)
    root.mkdir(parents=True)
    mapping=MAPS/(mode+'_200_A')/'mapping'
    assert json.loads((mapping/'mapping_complete.json').read_text())['status']=='PASS'
    (root/'mapping').symlink_to(mapping,target_is_directory=True)
    files=[p for p in SRC.glob('*.py') if p.name not in ['map_continue.py','map_support.py']]
    files += list((SRC/'selection_guard').glob('*.py'))
    files += [p for p in (SRC/'selection_code/200/A').rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.relative_to(SRC/'selection_code/200/A')!=Path('config.json')]
    files += [p for p in (SRC/'tracking_code/200/A').iterdir() if p.is_file()]
    for p in files:
        rel=p.relative_to(SRC)
        s=p.read_text()
        # Relocate the currently executing checkpoint only; historical cache roots stay read-only.
        t=s.replace(str(OLD_CODE),str(code)).replace(str(OLD_ROOT),str(root))
        if p.name=='common_continue.py':
            assert "CONT=BASEC/'continuation300'" in t and "ROOT=TOP/'continuation300'/str(CHECKPOINT)/GROUP" in t
            t=t.replace("CONT=BASEC/'continuation300'",f'CONT=Path({str(code)!r})')
            t=t.replace("ROOT=TOP/'continuation300'/str(CHECKPOINT)/GROUP",f'ROOT=Path({str(root)!r})')
        if p.name=='selection_setup.py':
            assert 'caches=[str(ROOT),' in t
            t=t.replace('caches=[str(ROOT),','caches=[str(ROOT),str(B.resolve()),')
        if p.name=='selection_cache.py':
            # Provenance points to the original archived inference source, never a new adapter.
            t=t.replace("(CONT/'selection_code'/str(n)/g,root)",f"(Path({str(OLD_CODE)!r})/'selection_code'/str(n)/g,root)")
        if p.suffix=='.py':ast.parse(t)
        dest=code/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(t)
        rows.append(dict(mode=mode,file=str(rel),source=str(p),source_sha256=sha(p),adapter_sha256=sha(dest),
                         numerical_changes=False,changed=t!=s))
    # track_case has no path literals needing adaptation: keep its fingerprint byte-identical.
    assert sha(code/'track_case.py')==sha(SRC/'track_case.py')
    assert sha(code/'tracking_code/200/A/track_case.py')==sha(SRC/'track_case.py')
    for name in ['repair_continue.py','commit_continue.py','scene_continue.py','track_prepare.py','track_gate.py','track_pure_gate.py']:
        assert ast.dump(ast.parse((code/name).read_text()))==ast.dump(ast.parse((SRC/name).read_text())),name
write(OUT/'adapter_audit.json',dict(status='PASS',scope='Path relocation and exact cache-source lookup only; numerical stage functions unchanged',files=rows))
print(json.dumps(dict(status='PASS',files=len(rows),code=str(HERE),results=str(OUT)),indent=2))
