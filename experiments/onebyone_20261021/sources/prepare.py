from pathlib import Path
import ast,hashlib,json,subprocess,sys
H=Path('/home/chenkejun/CVPR');D=Path('/data/chenkejun/CVPR/results')
C=H/'experiments/frame_point_replacement_20261021';R=D/'frame_point_replacement_20261021'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
assert C.is_dir(); R.mkdir(exist_ok=False)
scenes=['room0','room1','room2','office0','office1','office2','office3','office4'];config={};sources={}
for group,root,counts in [('automatic',D/'replica8_swap34_20261020',[35,22,35,19,22,23,22,18]),('manual',D/'manual76_partial_20261020',[17,6,28,1,10,2,7,5])]:
    config[group]={}
    for scene,count in zip(scenes,counts):
        src=(root/scene/'e1_repair').resolve();part=(root/scene/'partial').resolve()
        rf=read(src/'e1/region_freeze.json');sc=next(Path(p).parent for p in rf['code_sha256'] if p.endswith('/observation_ops.py'))
        assert len(read(src/'confirmed_selection.json')['included'])==count
        assert read(part/'mapping_complete.json')['status']=='PASS'
        tx=part/'transactions'
        if group=='automatic' and scene=='room2':tx=D/'e1_partial_autov2_20261019/room2/transactions'
        assert len(list(tx.glob('f*.npz')))==400
        config[group][scene]=dict(source=str(src),source_code=str(sc),partial=str(part),partial_transactions=str(tx),selected=count)
        for f in ['confirmed_selection.json','global_association.json','pipeline_code_freeze.json','e1/region_freeze.json','original_repair/predictions_freeze.json']:
            sources[str(src/f)]=sha(src/f)
        (R/group/scene).mkdir(parents=True)
save(C/'config.json',config); save(R/'input_lock.json',dict(status='PASS',source_sha256=sources,config_sha256=sha(C/'config.json')))
for p in C.glob('*.py'):ast.parse(p.read_text())
p=subprocess.run([sys.executable,'-B',str(C/'test_point_replace.py')],cwd=C,capture_output=True,text=True)
print(p.stdout);print(p.stderr);assert p.returncode==0
save(R/'tests_complete.json',json.loads(p.stdout))
save(R/'deployment_code_freeze.json',{str(p):sha(p) for p in C.glob('*.py')})
print('PREPARED',C,R)
