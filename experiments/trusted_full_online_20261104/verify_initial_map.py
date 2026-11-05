"""Independent strict-vote replay on the exact full-depth prefix."""
from alignment_common import *
import sys
sys.path.insert(0,str(OLD_C))
from strict_votes import nearest,frame_votes,count_keys,BASE
from scipy.spatial import cKDTree
sys.path.insert(0,str(SRC))
from revisable_instance_map.surface_evidence import reduce_surface_votes,CONFIRMED
FB=Path('/data/chenkejun/CVPR/results/ovimap_full_depth_baseline_20261101')
def verify(method,scene,n,group):
 dst=O/method/scene/str(n)/group/'mapping'
 current=load(dst/'baseline/native/instance_surface.npz')
 geom=load(O/'attributes'/scene/str(n)/'normals.npz')
 np.testing.assert_array_equal(current['xyz_m'],geom['xyz_m'])
 history=OLD/'ours'/f'{scene}_400_main'
 if n==400:
  ref=load(FB/'scenes'/scene/'strict_instance_surface.npz')
  np.testing.assert_array_equal(current['xyz_m'],ref['xyz_m'])
  np.testing.assert_array_equal(current['instance_id'],ref['instance_id'])
  original=load(FB/'scenes'/scene/'strict_votes.npz');keys=original['keys'];votes=original['votes']
 else:
  cp=read(history/'checkpoint.json');tree=cKDTree(current['xyz_m']);arrays=[]
  for fid in range(0,n*5,5):
   rel=f'history/f{fid:06d}.npz';p=history/rel;assert sha(p)==cp['history_sha256'][rel]
   h=load(p);point,valid=nearest(tree,h['world']);arrays.append(frame_votes(point[valid],h['ids'][valid]))
  keys,votes=count_keys(arrays)
  ev=reduce_surface_votes(keys,votes,len(current['xyz_m']),BASE,2,.67)
  np.testing.assert_array_equal(current['instance_id'],np.where(ev['state']==CONFIRMED,ev['top1_instance_id'],-1))
 pair=load(dst/'baseline/surface_instance_frame_votes.npz')
 k=pair['surface_point_index'].astype(np.int64)*BASE+pair['instance_id'];order=np.argsort(k)
 np.testing.assert_array_equal(k[order],keys);np.testing.assert_array_equal(pair['frame_votes'][order],votes)
 rows=[json.loads(x) for x in (dst/'association/associations.jsonl').read_text().splitlines()]
 for fid in range(0,n*5,5):
  old=read(history/'decisions'/f'f{fid:06d}.json')
  assert [(x['mask_local_id'],x['instance_id']) for x in rows if x['frame_id']==fid]==[(x['mask_local_id'],x['instance_id']) for x in old]
 filled=load(dst/'baseline/final/instance_surface.npz');native=current['instance_id']
 np.testing.assert_array_equal(filled['instance_id'][native>0],native[native>0])
 if n==400:
  reference=load(FB/'scenes'/scene/'instance_surface.npz')
  for key in ['xyz_m','rgb','instance_id']:np.testing.assert_array_equal(filled[key],reference[key])
 dump(dst/'independent_initial_audit.json',dict(status='PASS',scene=scene,frames=n,method=method,group=group,full_depth_xyz_exact=True,all_causal_association_IDs_exact=True,all_strict_votes_exact=True,original_unfilled_labels_exact=True,hole_fill_preserves_all_confirmed_IDs=True,GT_used=False))
if __name__=='__main__':verify(sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4])
