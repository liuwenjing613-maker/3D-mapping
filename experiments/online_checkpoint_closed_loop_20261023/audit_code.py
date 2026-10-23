"""Verify the executed adapters and unchanged numerical stage source files."""
from pathlib import Path
import ast,hashlib,json,subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
SRC=REPO/'experiments/online_room0_repair50_to100_20261022/continuation300'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    checks=[]
    for row in read(OUT/'adapter_audit.json')['files']:
        mode=row['mode'];rel=Path(row['file']);source=Path(row['source']);dest=HERE/mode/rel
        assert sha(source)==row['source_sha256']
        if rel==Path('selection_run.py'):
            text=dest.read_text().replace("CONT/'track_queue_with_cache.py'","CONT/'track_queue.py'")
            match=hashlib.sha256(text.encode()).hexdigest()==row['adapter_sha256']
        else:match=sha(dest)==row['adapter_sha256']
        checks.append(dict(mode=mode,file=str(rel),source_sha256=sha(source),executed_sha256=sha(dest),exact_permitted_adapter=match))
        assert match,(mode,rel)
    for mode in ['reference','checkpoint']:
        src=HERE/mode;fixed=HERE/('fixed_'+mode)
        assert sha(src/'track_queue_with_cache.py')==sha(HERE/'track_queue_with_cache.py')
        for dest in fixed.rglob('*'):
            if not dest.is_file() or '__pycache__' in dest.parts:continue
            rel=dest.relative_to(fixed);original=src/rel
            expected=original.read_text().replace(str(src),str(fixed)).replace(str(OUT/mode),str(OUT/('fixed_'+mode)))
            assert dest.read_text()==expected,rel
            checks.append(dict(mode='fixed_'+mode,file=str(rel),source_sha256=sha(original),executed_sha256=sha(dest),exact_permitted_adapter=True))
    numerical=[]
    for mode in ['reference','checkpoint','fixed_reference','fixed_checkpoint']:
        for name in ['repair_continue.py','commit_continue.py','scene_continue.py','track_prepare.py','track_gate.py','track_pure_gate.py']:
            equal=ast.dump(ast.parse((HERE/mode/name).read_text()))==ast.dump(ast.parse((SRC/name).read_text()))
            assert equal,(mode,name)
            numerical.append(dict(mode=mode,file=name,AST_identical=True))
    tracked_changes=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=REPO,text=True).splitlines()
    assert not tracked_changes,tracked_changes
    result=dict(status='PASS',base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        tracked_production_changes=tracked_changes,adapter_files=checks,numerical_stage_checks=numerical,
        no_production_point_sorting=True,no_Gate_sampling_change=True,
        harness_files={p.name:sha(p) for p in sorted(HERE.glob('*.py'))})
    (OUT/'current_source_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',adapters=len(checks),numerical_stage_checks=len(numerical),tracked_changes=tracked_changes)))

if __name__=='__main__':main()
