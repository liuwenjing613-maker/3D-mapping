from pathlib import Path
import sys,json,subprocess,hashlib
C=Path(__file__).resolve().parent;O=Path('/data/chenkejun/CVPR/results')/C.name
scene=sys.argv[1]
method=sys.argv[2]
if method=='no_repair':surface=O/'published/no_repair'/scene/'final_B/instance_surface.npz'
elif method=='OVIMAP':surface=O/'ovimap'/scene/'200/ovi/f000995.npz'
else:raise ValueError(method)
out=O/'evaluation'/method/scene;out.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
predhash=sha(surface)
provenance=dict(input_scope_verified=False,frame_count=200,frame_list_sha256=hashlib.sha256(json.dumps(list(range(0,1000,5)),separators=(',',':')).encode()).hexdigest(),prediction_frozen_before_GT=True,GT_used_for_mapping=False,selection_manifest=str(O/'inputs'/scene/'selection.json'),selection_sha256=sha(O/'inputs'/scene/'selection.json'),prediction_sha256=predhash,instance_only=True,evaluation_adapter='Explicit ScanNet scope; unchanged main default R2 config and scorer; separate geometry-dependent correspondences')
source=out/'source_provenance.json';source.write_text(json.dumps(provenance,indent=2))
cache=O/'correspondence'/(scene+'_'+method+'.npz')
with (out/'adapt.log').open('a') as f:subprocess.run([sys.executable,'-B',str(C/'evaluate_surface.py'),'adapt-surface','--gt',str(O/'ground_truth'/scene/'ground_truth/gt.npz'),'--surface',str(surface),'--fixed-correspondence',str(cache),'--source-provenance',str(source),'--method-name',method,'--method-commit','source_sha256:'+sha(C/'source_lock.json'),'--out',str(out/'r2')],cwd='/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009',stdout=f,stderr=subprocess.STDOUT,check=True)
assert sha(surface)==predhash
(out/'complete.json').write_text(json.dumps(dict(status='PASS',prediction=str(surface),prediction_sha256=predhash,method=method,scene=scene)))
