"""Isolate the authorized room0/A+B continuation to 300 and 400; numerical kernels unchanged."""
from pathlib import Path
import ast,hashlib,json,runpy

REPO=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023')
SRC=REPO/'experiments/online_room0_repair50_to100_20261022/continuation300'
OLD=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/continuation300')
HERE=REPO/'experiments/online_full_scene_room0_AB_20261023'
OUT=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
PRIOR=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023/checkpoint')
PREVIOUS_CODE=REPO/'experiments/online_checkpoint_closed_loop_20261023'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def required(text,a,b):assert a in text,a;return text.replace(a,b)

def main():
    assert not OUT.exists(),'Fresh result directory required'
    OUT.mkdir(parents=True);HERE.mkdir(parents=True,exist_ok=True)
    prior=json.loads((PRIOR/'repair/commit_complete.json').read_text())
    assert prior['status']=='PASS' and prior['processed_frames']==200 and prior['last_raw_frame']==995
    for key in ['checkpoint','identity_counts','source_pixel_support','journal_manifest','final_map','geometry']:
        assert sha(Path(prior[key]))==prior[key+'_sha256']
    names=['common_continue.py','bootstrap_continue.py','scene_continue.py','map_support.py','map_continue.py',
        'selection_runtime.py','selection_setup.py','selection_run.py','selection_cache.py','selection_fence.py','selection_qa.py',
        'track_prepare.py','track_queue.py','track_gate.py','track_batch.py','track_pure_gate.py','track_case.py',
        'repair_continue.py','commit_continue.py']
    files=[SRC/n for n in names]+list((SRC/'map_guard').glob('*.py'))+list((SRC/'selection_guard').glob('*.py'))
    audit=[]
    for p in files:
        rel=p.relative_to(SRC);text=p.read_text()
        if p.name=='common_continue.py':
            text=required(text,'CHECKPOINT in (200,300)','CHECKPOINT in (200,300,400)')
            text=required(text,"CONT=BASEC/'continuation300'",f'CONT=Path({str(HERE)!r})')
            text=required(text,"ROOT=TOP/'continuation300'/str(CHECKPOINT)/GROUP",f'ROOT=Path({str(OUT)!r})/str(CHECKPOINT)/GROUP')
        elif p.name=='map_support.py':
            text=required(text,'TARGET in (200,300)','TARGET in (200,300,400)')
            text=required(text,"C=BASEC/'continuation300'",f'C=Path({str(HERE)!r})')
            start=text.index('def prior_paths():');end=text.index('\ndef setup():',start)
            text=text[:start]+f'''def prior_paths():
 if TARGET==200:
  return TOP/'repair100'/GROUP/'commit_complete.json',TOP/'prefix100'
 if TARGET==300 and GROUP=='A':
  return Path({str(PRIOR/'repair/commit_complete.json')!r}),Path({str(PRIOR/'mapping')!r}).resolve()
 return Path({str(OUT)!r})/str(TARGET-100)/GROUP/'repair/commit_complete.json',Path({str(OUT)!r})/str(TARGET-100)/GROUP/'mapping'
'''+text[end:]
            text=required(text,'cache=[str(previous),','cache=[str(previous.resolve()),')
        elif p.name=='selection_setup.py':
            text=required(text,'caches=[str(ROOT),',"caches=list(old['allowed_prefix_cache_roots'])+[str(ROOT),str(B.resolve()),str(ROOT.parent/('B' if GROUP=='A' else 'A')/'tracking'),")
        elif p.name=='selection_cache.py':
            text=required(text,"(CONT/'selection_code'/str(n)/g,root)",f"(Path({str(OLD)!r})/'selection_code'/str(n)/g,root)")
        elif p.name=='selection_run.py':
            text=required(text,"CONT/'track_queue.py'","CONT/'track_queue_with_cache.py'")
        ast.parse(text);dest=HERE/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
        audit.append(dict(file=str(rel),source=str(p),source_sha256=sha(p),executed_sha256=sha(dest),changed=text!=p.read_text()))
    for name in ['map_continue.py','repair_continue.py','commit_continue.py','scene_continue.py','track_prepare.py','track_gate.py','track_batch.py','track_pure_gate.py','track_case.py']:
        assert sha(HERE/name)==sha(SRC/name),name
    # Use the original candidate adapter generator with only output roots and checkpoint bounds extended.
    source=(SRC/'selection_build.py').read_text()
    source=required(source,"CONT=BASE/'continuation300';TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')",
        f"CONT=Path({str(HERE)!r});TOP=Path({str(OUT)!r})")
    source=required(source,"SRC=BASE/'prefix100';SR=TOP/'prefix100'",
        "SRC=BASE/'prefix100';SR=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/prefix100')")
    source=required(source,'for n in [200,300]:','for n in [200,300,400]:')
    source=required(source,"out=TOP/'continuation300'/str(n)/g/'candidates'","out=TOP/str(n)/g/'candidates'")
    (HERE/'selection_build.py').write_text(source)
    runpy.run_path(str(HERE/'selection_build.py'),run_name='__main__')
    # The generated guard is the same contract as the production guard.
    assert sha(HERE/'selection_guard/sitecustomize.py')==sha(SRC/'selection_guard/sitecustomize.py')
    for n in [200,300,400]:
        for group in ['A','B']:
            dest=HERE/'tracking_code'/str(n)/group;dest.mkdir(parents=True)
            for p in (SRC/'tracking_code/300'/group).iterdir():
                if p.is_file():(dest/p.name).write_bytes(p.read_bytes())
            assert sha(dest/'track_case.py')==sha(SRC/'track_case.py')
    wrapper=(PREVIOUS_CODE/'track_queue_with_cache.py').read_text()
    wrapper=required(wrapper,"SOURCES=[TOP/'continuation300/200/A/tracking',TOP/'continuation300/200/B/tracking']",
        "SOURCES=[ROOT.parent/('B' if GROUP=='A' else 'A')/'tracking']+[TOP/'continuation300'/str(n)/g/'tracking' for n in [200,300] if n<=CHECKPOINT for g in ['A','B']]")
    wrapper=required(wrapper,"assert CHECKPOINT==200 and GROUP=='A'","assert CHECKPOINT in (200,300,400) and GROUP in ('A','B')")
    (HERE/'track_queue_with_cache.py').write_text(wrapper)
    bmap=Path('/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023/checkpoint_200_B/mapping')
    bdone=json.loads((bmap/'mapping_complete.json').read_text());assert bdone['status']=='PASS' and bdone['surface_policy']=='checkpoint'
    bdest=OUT/'200/B/mapping';bdest.parent.mkdir(parents=True);bdest.symlink_to(bmap,target_is_directory=True)
    plan=dict(status='READY',scene='room0',branches=['A','B'],branch_start={'A':'verified repair200 commit','B':'verified optimized mapping200; perform independent B repair200 first'},B_mapping200_source=str(bmap),B_mapping200_sha256=sha(bmap/'mapping_complete.json'),full_mapping_frames=400,source_raw_range=[0,1999],
        mapping_frame_ids=list(range(0,2000,5)),last_mapping_raw_frame=1995,
        inherited_checkpoint=200,inherited_commit=str(PRIOR/'repair/commit_complete.json'),inherited_commit_sha256=sha(PRIOR/'repair/commit_complete.json'),
        executed_checkpoints={'A':[300,400],'B':[200,300,400]},prior_repair_checkpoints=[50,100,200],surface_policy='checkpoint',
        final400_repair_and_commit=True,numerical_algorithms_unchanged=True,Gate_sampling_unchanged=True,
        no_candidate_unification=True,strict_cache_reuse_only=True,code=str(HERE),results=str(OUT))
    write(OUT/'run_plan.json',plan)
    write(OUT/'adapter_audit.json',dict(status='PASS',scope='Output locations, checkpoint bounds and strict cache lookup only',
        unchanged_numerical_files=[r for r in audit if not r['changed']],files=audit,
        all_code_sha256={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file()}))
    print(json.dumps(plan,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
