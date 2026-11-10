"""Reconstruct actual original sampled voxels; verify every saved observation ledger entry."""
from pathlib import Path
import sys,os,json,time,ast,fcntl,traceback
import numpy as np
from collections import defaultdict
from types import SimpleNamespace
from dbscan_common import C,O,SRC,COUNTS,read,load,save,dump,sha
sys.path.insert(0,str(SRC))
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.observations import RawInstanceObservation,source_pixel_indices
scene=sys.argv[1];n=COUNTS[scene];root=O/'main'/scene/str(n)/'A/mapping';out=O/'dbscan_inputs/no_repair'/scene;out.mkdir(parents=True,exist_ok=True)
lock=(out/'build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state(status,**kw):dump(out/'status.json',dict(status=status,scene=scene,pid=os.getpid(),unix=time.time(),**kw))
try:
    from source_guard import check
    check();start=time.time()
    if (out/'complete.json').exists():
        d=read(out/'complete.json');assert d['status']=='PASS' and sha(out/'source_pixel_support.npz')==d['output_sha256'];sys.exit(0)
    receipt=read(root/'mapping_complete.json');assert receipt['status']=='PASS' and receipt['frame_ids']==list(range(0,n*5,5)) and not receipt['repair_applied']
    paths=[root/'config.json',root/'association/observations.jsonl',root/'association/observation_support_3cm.npz',root/'association/associations.jsonl',root/'mapping_complete.json']
    hashes={str(p):sha(p) for p in paths}
    ledger=load(root/'association/observation_support_3cm.npz');params=json.loads(str(ledger['parameters_json'][0]));assoc=SimpleNamespace(**params)
    # Extract this single already-audited sampler; do not import the commit driver's side effects.
    source=C/'main'/scene/'commit_continue.py';text=source.read_text();tree=ast.parse(text);node=next(t for t in tree.body if isinstance(t,ast.FunctionDef) and t.name=='sample_pixels')
    function=ast.get_source_segment(text,node);namespace=dict(np=np,source_pixel_indices=source_pixel_indices);exec(compile(function,str(source),'exec'),namespace);sample=namespace['sample_pixels']
    obs={};byframe=defaultdict(list)
    for line in (root/'association/observations.jsonl').read_text().splitlines():
        row=json.loads(line);item=RawInstanceObservation(**row);obs[item.observation_id]=item
    sourceframes=ReplicaFrameSource(root/'config.json');assert list(sourceframes.frame_ids)==receipt['frame_ids']
    assert set(obs)==set(map(str,ledger['observation_ids']))
    for k,fid in enumerate(ledger['frame_ids']):byframe[int(fid)].append(k)
    samples=[None]*len(obs);pixels=[None]*len(obs);frame_hashes=[]
    for index,fid in enumerate(sourceframes.frame_ids,1):
        fr=sourceframes.load_frame(fid)
        for k in byframe[fid]:
            item=obs[str(ledger['observation_ids'][k])];assert item.frame_id==fid and item.source_mask_sha256==str(ledger['source_mask_sha256'][k])
            px,vox=sample(fr,item,assoc);lo,hi=map(int,ledger['offsets'][k:k+2]);np.testing.assert_array_equal(np.unique(vox,axis=0),ledger['voxel_coordinates'][lo:hi])
            samples[k]=vox;pixels[k]=px
        if index%100==0:state('RUNNING',frames=index,total=n);print('EVIDENCE',scene,index,n,flush=True)
    off=np.r_[0,np.cumsum([len(v) for v in samples])].astype(np.int64)
    save(out/'source_pixel_support.npz',observation_ids=ledger['observation_ids'],instance_ids=ledger['instance_ids'],frame_ids=ledger['frame_ids'],layer_index=np.zeros(len(obs),np.int32),offsets=off,sampled_voxel_coordinates=np.concatenate(samples),sampled_pixel_indices=np.concatenate(pixels),voxel_size_m=ledger['voxel_size_m'])
    for p,h in hashes.items():assert sha(p)==h
    report=dict(status='PASS',scene=scene,frames=n,observations=len(obs),sampled_pixels=int(off[-1]),all_saved_observation_support_exact=True,only_original_observations=True,repair_history_created=False,GT_used=False,source_hashes=hashes,sampler_source=str(source),sampler_source_sha256=sha(source),sampler_function_sha256=__import__('hashlib').sha256(function.encode()).hexdigest(),output_sha256=sha(out/'source_pixel_support.npz'),seconds=time.time()-start)
    dump(out/'complete.json',report);state('PASS',frames=n,total=n);print('ORIGINAL_EVIDENCE_PASS',scene,flush=True)
except Exception:state('FAILED',traceback=traceback.format_exc());raise
