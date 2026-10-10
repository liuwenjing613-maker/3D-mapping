"""Extract final unassigned U/T/C points from verified strict-baseline PLYs."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'exports_20261010'
TYPES={'float':'<f4','double':'<f8','uchar':'u1','int':'<i4'}
COLORS={0:(111,124,137),1:(231,180,76),3:(204,91,101)}
NAMES={0:'UNOBSERVED',1:'TENTATIVE',3:'CONFLICT'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_ply(path):
    with Path(path).open('rb') as f:
        lines=[]
        while True:
            line=f.readline().decode('ascii').strip()
            if not line:raise ValueError('Incomplete header')
            lines.append(line)
            if line=='end_header':break
        assert lines[:2]==['ply','format binary_little_endian 1.0']
        elements=[x for x in lines if x.startswith('element ')]
        assert len(elements)==1 and elements[0].startswith('element vertex ')
        count=int(elements[0].split()[2]);properties=[x for x in lines if x.startswith('property ')]
        dtype=np.dtype([(x.split()[2],TYPES[x.split()[1]]) for x in properties])
        payload=f.read();assert len(payload)==count*dtype.itemsize
        return np.frombuffer(payload,dtype=dtype),properties
def main():
    sources=json.loads((OUT/'strict_baseline_ply_complete.json').read_text(encoding='utf-8'))
    evaluation=json.loads((ROOT/'evaluation_r2/evaluation_complete.json').read_text(encoding='utf-8'))
    comparison_path=ROOT/'evaluation_r2/comparison.json'
    assert sha(comparison_path)==evaluation['comparison_sha256']
    comparison=json.loads(comparison_path.read_text(encoding='utf-8'));reports=[]
    for record in sources['files']:
        scene=record['scene'];source=OUT/Path(record['path']).name
        assert sha(source)==record['sha256']
        vertices,properties=read_ply(source)
        mask=vertices['instance_id']<=0
        assert np.all(np.isin(vertices['native_state'][mask],[0,1,3]))
        selected=vertices[mask].copy()
        expected=comparison['states'][scene]['strict_baseline_final']
        assert len(selected)==expected['three_state_total']==record['gray_points']
        counts={}
        for state,color in COLORS.items():
            choose=selected['native_state']==state;count=int(choose.sum())
            assert count==expected['counts_U_T_assigned_C'][state]
            counts[NAMES[state]]=count
            for j,name in enumerate(('red','green','blue')):selected[name][choose]=color[j]
        target=OUT/('P1-A1-strict-vote_'+scene+'_three_states.ply')
        assert not target.exists()
        header=['ply','format binary_little_endian 1.0',
            'comment P1-A1 strict-vote baseline, final unassigned points only (instance_id <= 0)',
            'comment native_state: 0 UNOBSERVED blue-gray; 1 TENTATIVE yellow; 3 CONFLICT red',
            'comment Coordinates, IDs, source indices and original RGB preserved exactly',
            'element vertex '+str(len(selected)),*properties,'end_header']
        with target.open('wb') as f:f.write(('\n'.join(header)+'\n').encode('ascii'));f.write(selected.tobytes())
        check,_=read_ply(target)
        for field in selected.dtype.names:
            np.testing.assert_array_equal(check[field],selected[field])
            if field not in ('red','green','blue'):np.testing.assert_array_equal(check[field],vertices[field][mask])
        assert sha(source)==record['sha256']
        reports.append({'scene':scene,'file':target.name,'source_sha256':record['sha256'],'sha256':sha(target),
            'points':len(selected),'counts':counts,'bytes':target.stat().st_size,'roundtrip_exact':True})
    result={'status':'PASS','condition':'strict_baseline_final','filter':'final instance_id <= 0',
        'state_colors_RGB':{NAMES[k]:v for k,v in COLORS.items()},'files':reports}
    (OUT/'three_states_export_complete.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
