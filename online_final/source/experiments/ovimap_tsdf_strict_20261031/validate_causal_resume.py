from pathlib import Path
import json,hashlib,numpy as np
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
C=Path('/home/chenkejun/CVPR/experiments/ovimap_tsdf_strict_20261031')
def arr_equal(a,b):
 with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
  assert set(x.files)==set(y.files),(str(a),x.files,y.files)
  for k in x.files:np.testing.assert_array_equal(x[k],y[k])
def main():
 normal=O/'ours/room0_5_smoke_v2';resumed=O/'ours/room0_5_resume_v2'
 for p in [normal,resumed]:assert (p/'complete.json').exists()
 for i in range(5):
  name='f%06d'% (i*5)
  arr_equal(normal/'maps'/(name+'.npz'),resumed/'maps'/(name+'.npz'))
  assert json.loads((normal/'decisions'/(name+'.json')).read_text())==json.loads((resumed/'decisions'/(name+'.json')).read_text())
 for k in ['association','votes']:arr_equal(normal/'checkpoints'/('f000020_'+k+'.npz'),resumed/'checkpoints'/('f000020_'+k+'.npz'))
 a=O/'native/room0_2_trace';b=O/'native/room0_5_trace'
 for i in range(2):
  for kind in ['geometry','ovi']:arr_equal(a/kind/('f%06d.npz'%(i*5)),b/kind/('f%06d.npz'%(i*5)))
 ar=[json.loads(x) for x in (a/'prefix_audit.jsonl').read_text().splitlines()]
 br=[json.loads(x) for x in (b/'prefix_audit.jsonl').read_text().splitlines()]
 for x,y in zip(ar,br):assert x['tsdf_sha256']==y['tsdf_sha256']
 hashes=json.loads((C/'original_hashes.json').read_text())
 for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 report={'status':'PASS','resume_maps_decisions_votes_bit_exact':True,'prefix_2_vs_5_bit_exact':True,'original_code_and_binary_unchanged':True}
 (O/'causal_resume_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
