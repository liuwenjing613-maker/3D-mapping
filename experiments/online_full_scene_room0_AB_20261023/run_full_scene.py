"""Run checkpoint mapping, real selection/tracking, repair and commit in causal order."""
from pathlib import Path
import json,os,subprocess,time,traceback,sys

HERE=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
GROUP=sys.argv[1];assert GROUP in ('A','B')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'

def read(p):return json.loads(p.read_text())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def run(n,stage,script,env):
    root=OUT/str(n)/GROUP;root.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    with (root/(stage+'.log')).open('ab',buffering=0) as log:
        p=subprocess.Popen([PY,'-B','-u',str(HERE/script)],cwd=HERE,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(OUT/('status_'+GROUP+'.json'),dict(status='RUNNING',checkpoint=n,stage=stage,pid=p.pid,started_unix=time.time()))
        rc=p.wait()
    result=dict(checkpoint=n,stage=stage,returncode=rc,seconds=time.perf_counter()-start)
    with (OUT/('stage_timings_'+GROUP+'.jsonl')).open('a') as f:f.write(json.dumps(result)+'\n')
    assert rc==0,(n,stage,rc)

def main():
    for n in ([300,400] if GROUP=='A' else [200,300,400]):
        root=OUT/str(n)/GROUP
        env={**os.environ,'CVPR_CHECKPOINT':str(n),'CVPR_TRACK_GROUP':GROUP,'CVPR_MAPPING_RUN_ROOT':str(root),
            'CVPR_SURFACE_POLICY':'checkpoint','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
        env.pop('CVPR_ONLINE_GUARD',None);env.pop('PYTHONPATH',None)
        if not (root/'mapping/mapping_complete.json').exists():
            if not (root/'mapping/mapping_plan.json').exists():run(n,'mapping_setup','map_support.py',env)
            guarded={**env,'CVPR_ONLINE_GUARD':str(root/'mapping/input_guard.json'),'PYTHONPATH':str(HERE/'map_guard')+':'+str(HERE)}
            run(n,'mapping','map_continue.py',guarded)
        m=read(root/'mapping/mapping_complete.json')
        assert m['status']=='PASS' and m['frame_count']==n and m['surface_rebuild_frames']==[n]
        first=read(root/f'mapping/frame{n-99}_state_audit.json')
        assert first['status']=='PASS' and first['repaired_state_used']
        if not (root/'selection_tracking_status.json').exists():run(n,'selection_tracking','selection_run.py',env)
        assert read(root/'selection_tracking_status.json')['status']=='PASS'
        guarded={**env,'CVPR_ONLINE_GUARD':str(root/'input_guard.json'),'PYTHONPATH':str(HERE/'selection_guard')+':'+str(HERE)}
        for stage,script,marker in [('repair','repair_continue.py','repair/repair_complete.json'),('commit','commit_continue.py','repair/commit_complete.json')]:
            if not (root/marker).exists():run(n,stage,script,guarded)
            assert read(root/marker)['status']=='PASS'
        commit=read(root/'repair/commit_complete.json')
        assert commit['processed_frames']==n and commit['last_raw_frame']==(n-1)*5
        write(OUT/f'checkpoint{n}_{GROUP}_complete.json',dict(status='PASS',checkpoint=n,commit=str(root/'repair/commit_complete.json'),
            mapping_surface_rebuild_frames=m['surface_rebuild_frames'],repaired_state_used_in_first_frame=True))
    write(OUT/('status_'+GROUP+'.json'),dict(status='COMPLETE',scene='room0',branch=GROUP,processed_frames=400,last_raw_frame=1995,
        final_commit=str(OUT/'400'/GROUP/'repair/commit_complete.json')))

if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/('status_'+GROUP+'.json'),dict(status='FAILED',traceback=traceback.format_exc()));raise
