"""Inspect exact field differences after basic root normalization."""
from pathlib import Path
from collections import Counter
import json

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
def read(p):return json.loads(p.read_text())
def differences(a,b,path=''):
    if type(a)!=type(b):yield path,a,b
    elif isinstance(a,dict):
        for k in sorted(a.keys()|b.keys()):
            if k not in a or k not in b:yield path+'/'+k,a.get(k),b.get(k)
            else:yield from differences(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        if len(a)!=len(b):yield path+'/length',len(a),len(b)
        for i,(x,y) in enumerate(zip(a,b)):yield from differences(x,y,path+'/'+str(i))
    elif a!=b:yield path,a,b

rows=[]
for mode in ['fixed_reference','fixed_checkpoint']:
    root=OUT/mode;commit=read(root/'repair/commit_complete.json')
    line=read(Path(commit['observation_lineage']))['entries']
    rows.append(json.loads(json.dumps(line).replace(str(root/'mapping'),'<MAPPING>').replace(str((root/'mapping').resolve()),'<MAPPING>').replace(str(root),'<RUN>')))
diff=list(differences(*rows))
result=dict(field_counts=dict(Counter(p.rsplit('/',1)[-1] for p,_,_ in diff)),difference_count=len(diff),examples=diff[:12])
(OUT/'fixed_lineage_path_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
