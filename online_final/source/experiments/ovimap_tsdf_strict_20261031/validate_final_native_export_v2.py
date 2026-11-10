"""Final native PLY cross-check. Coordinate precision ambiguities are reported explicitly."""
from pathlib import Path
import hashlib,json,re,time
import numpy as np
import plyfile
from scipy.spatial import cKDTree
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
def read_vertices(mesh):
 with Path(mesh).open('rb') as f:
  properties=[];inside=False;count=None;ascii_format=False
  while True:
   line=f.readline().decode('ascii').strip()
   if line=='end_header':break
   if line=='format ascii 1.0':ascii_format=True
   if line.startswith('element '):
    fields=line.split();inside=fields[1]=='vertex'
    if inside:count=int(fields[2])
   if inside and line.startswith('property '):properties.append(line.split()[-1])
  assert ascii_format and count is not None
  names=['x','y','z','red','green','blue'];columns=[properties.index(k) for k in names]
  values=np.loadtxt(f,dtype=np.float32,usecols=columns,max_rows=count,ndmin=2)
 assert values.shape==(count,6)
 return values[:,:3].copy(),values[:,3:].astype(np.int64)

def validate(scene):
 root=O/'native'/f'{scene}_400_trace';mesh=root/'cropformer_inst/instance_mesh_400.ply'
 xyz,colors=read_vertices(mesh)
 logs=''.join(p.read_text(errors='replace') for p in (root/'log').glob('*INFO*'))
 palette={}
 for ident,r,g,b in re.findall(r'Instance:\s*(\d+)\s+Color:\s*\((\d+),(\d+),(\d+)\)',logs):
  key=(int(r)<<16)+(int(g)<<8)+int(b)
  assert key not in palette or palette[key]==int(ident)
  palette[key]=int(ident)
 assert palette
 # Upstream background gray is known even though LogMeshColors omits ID 0.
 background=(200<<16)+(200<<8)+200
 assert palette.get(background,0)==0
 palette[background]=0
 pk=np.array(sorted(palette),np.int64);pv=np.array([palette[int(k)] for k in pk],np.int32)
 packed=(colors[:,0]<<16)+(colors[:,1]<<8)+colors[:,2]
 ix=np.searchsorted(pk,packed);known=ix<len(pk);known[known]&=pk[ix[known]]==packed[known]
 decoded=np.full(len(xyz),-1,np.int32);decoded[known]=pv[ix[known]];decoded[decoded<=0]=-1
 with np.load(root/'ovi/f001995.npz') as z:
  target=z['xyz_m'];expected=z['instance_id'];inventory=z['native_instance_ids']
 distance,index=cKDTree(target).query(xyz,k=2,workers=2)
 assert distance[:,0].max()<1e-5
 # ASCII PLY rounding can make neighboring floats indistinguishable.
 # A coordinate group is certified only when its nearest assignment is separated by >2e-5 m.
 ambiguous=distance[:,1]<=distance[:,0]+2e-5
 point=index[:,0];lo=np.full(len(target),np.iinfo(np.int32).max,np.int32);hi=np.full(len(target),-1,np.int32)
 np.minimum.at(lo,point,decoded);np.maximum.at(hi,point,decoded)
 counts=np.bincount(point,minlength=len(target));tainted=np.zeros(len(target),bool);tainted[point[ambiguous]]=True
 unlogged=np.zeros(len(target),bool);unlogged[point[~known]]=True
 certified=(counts>0)&~tainted&~unlogged
 consensus=np.where((lo==hi)&(lo>0),lo,-1)
 mismatches=np.flatnonzero(certified&(consensus!=expected))
 report={'status':'PASS' if not len(mismatches) else 'FAIL','scene':scene,
  'certified_coordinate_groups':int(certified.sum()),'total_coordinate_groups':len(target),
  'rounding_ambiguous_groups':int(tainted.sum()),'uncovered_groups':int(np.sum(counts==0)),
  'label_mismatches_in_certified_groups':len(mismatches),
  'max_ply_rounding_error_m':float(distance[:,0].max()),
  'integer_inventory':len(inventory),'positive_palette_inventory':len(np.unique(decoded[decoded>0])),
  'unknown_palette_vertices':int(np.sum(~known)),
  'unlogged_palette_coordinate_groups':int(unlogged.sum()),
  'unlogged_native_ids':sorted(set(inventory.tolist())-set(palette.values())),
  'verification_scope':'only decodable, unambiguous coordinate groups; missing palette entries are not background',
  'mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),
  'interpretation':'independent official palette export check on coordinate groups uniquely resolved despite ASCII PLY rounding'}
 report['mismatch_examples']=[{'point':int(i),'native':int(expected[i]),'palette':int(consensus[i])} for i in mismatches[:10]]
 dst=O/'native_final_export_validation_v2';dst.mkdir(exist_ok=True)
 (dst/(scene+'.json')).write_text(json.dumps(report,indent=2))
 print('FINAL_EXPORT',scene,report['status'],len(mismatches),flush=True)
 return report
def main():
 pending=set(['room0','room1','room2','office0','office1','office2','office3','office4']);results={}
 while pending:
  for scene in sorted(pending):
   log=O/'driver'/(scene+'_0.log')
   if log.exists() and 'NATIVE_DONE' in log.read_text():
    results[scene]=validate(scene);pending.remove(scene)
  if pending:time.sleep(10)
 report={'status':'PASS' if all(r['status']=='PASS' for r in results.values()) else 'FAIL','scenes':results}
 (O/'native_final_export_validation_v2.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
