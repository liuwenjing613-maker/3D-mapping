"""Run the validated camera-aware candidate frontend; stop before ABC/tracking repair."""
from pathlib import Path
import os,sys,json,subprocess,time,traceback,fcntl
ROOTC=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
ROOTO=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
method,scene,n,group=sys.argv[1:5];n=int(n)
assert method in ['main','online'] and group in ['A','B']
branch=ROOTC/method/scene;root=ROOTO/method/scene/str(n)/group
root.mkdir(parents=True,exist_ok=True)
lock=(root/'candidate_frontend_driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ.update(CVPR_CHECKPOINT=str(n),CVPR_TRACK_GROUP=group,CVPR_SCENE=scene,CVPR_GPU_POOL='0,1,2,3,4,5,6,7',PYTHONDONTWRITEBYTECODE='1')
os.environ.pop('CVPR_ONLINE_GUARD',None);os.environ.pop('PYTHONPATH',None)
sys.path.insert(0,str(ROOTC));sys.path.insert(0,str(branch))
from common_continue import *
from selection_runtime import run
from source_guard import check
def status(state,**kw):dump(root/'candidate_frontend_driver_status.json',dict(status=state,scene=scene,method=method,checkpoint=n,group=group,pid=os.getpid(),unix=time.time(),**kw))
try:
    check()
    while not (B/'mapping_complete.json').exists():
        failure=root/'mapping_driver_status.json'
        if failure.exists() and read(failure).get('status')=='FAILED':raise RuntimeError('Mapping failed: '+str(failure))
        status('WAITING_MAPPING');time.sleep(5)
    assert read(B/'mapping_complete.json')['status']=='PASS'
    if not (CANDIDATE_ROOT/'setup_complete.json').exists():
        status('RUNNING',stage='selection_setup')
        with (root/'selection_setup.log').open('a') as f:
            subprocess.run([PY,'-B','-u',str(branch/'selection_setup.py')],cwd=branch,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
    sc=branch/'selection_code'/str(n)/group
    steps=[('input_fence',branch/'selection_fence.py',[],PY,False,root/'input_fence_audit.json'),
        ('catalog',sc/'catalog/build_candidates.py',[scene],PY,False,CANDIDATE_ROOT/'catalog'/scene/'candidate_manifest.json'),
        ('prepare',sc/'frontend/prepare_inputs.py',[],PY,False,CANDIDATE_ROOT/'frontend/input_manifest.json'),
        ('exact_model_cache',branch/'selection_cache.py',[],PY,False,CANDIDATE_ROOT/'exact_model_cache_audit.json'),
        ('cropformer',sc/'frontend/infer_cropformer.py',[],CROP,True,CANDIDATE_ROOT/'frontend/cropformer/audit.json'),
        ('sam',sc/'frontend/infer_sam.py',[],SAM,True,CANDIDATE_ROOT/'frontend/sam2_1/audit.json'),
        ('summary',sc/'frontend/summarize_frontend.py',[],PY,False,CANDIDATE_ROOT/'frontend/pipeline_complete.json'),
        ('overmerge',sc/'overmerge/score.py',[],PY,False,CANDIDATE_ROOT/'overmerge/audit.json'),
        ('intersection',sc/'intersection/compute.py',[],PY,False,CANDIDATE_ROOT/'intersection/web/intersection.json'),
        ('versions',sc/'versions/compute.py',[],PY,False,CANDIDATE_ROOT/'versions/selection.json'),
        ('semantic_classify',sc/'semantic_route/classify.py',[],PY,True,CANDIDATE_ROOT/'semantic_route/semantic.json'),
        ('semantic_cabinet',sc/'semantic_route/refine_cabinet.py',[],PY,True,CANDIDATE_ROOT/'semantic_route/cabinet_context_rule.json'),
        ('semantic_route',sc/'semantic_route/route.py',[],PY,False,CANDIDATE_ROOT/'semantic_route/selection.json'),
        ('grouping_r2',sc/'grouping_r2/compute.py',[],PY,False,CANDIDATE_ROOT/'grouping_r2/audit.json')]
    for name,script,args,python,gpu,marker in steps:
        events=CANDIDATE_ROOT/'process_timing_events.jsonl'
        previous=[json.loads(s) for s in events.read_text().splitlines()] if events.exists() else []
        if marker.exists() and any(x['stage']==name and x['status']=='PASS' for x in previous):continue
        check();status('RUNNING',stage=name)
        run(name,script,args,python=python,gpu=gpu,out=CANDIDATE_ROOT)
    status('PASS',stage='CANDIDATE_FRONTEND_ONLY_COMPLETE',repair_complete=False,ABC='AWAITING_CAMERA_ADAPTER_VALIDATION',tracking_complete=False)
except Exception:
    status('FAILED',traceback=traceback.format_exc());raise
