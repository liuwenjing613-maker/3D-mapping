from pathlib import Path
import json,sys,numpy as np
from scipy.spatial import cKDTree
from baseline import get_frame,O,C,SNAP,load
from strict_votes import BASE
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/p1a1_one_vote_20261009')
from one_vote import effective_keys
from revisable_instance_map.surface_evidence import project_frame_regions
def main():
 native=O/'native/room0_20_trace';ours=O/'ours/room0_20_validation_v2'
 assert (ours/'complete.json').exists()
 config=json.loads((ours/'config.json').read_text())
 rows=[json.loads(x) for x in (native/'prefix_audit.jsonl').read_text().splitlines()]
 frames=[]
 for row in rows:
  fid=row['frame_id'];frame,_,_=get_frame('room0',fid,row,config)
  dec=json.loads((ours/'decisions'/('f%06d.json'%fid)).read_text())
  table=np.zeros(int(frame.mask_local.max())+1,np.int64)
  for d in dec:table[d['mask_local_id']]=d['instance_id']
  frames.append((frame,table))
 for count in [1,2,5,10,20]:
  fid=(count-1)*5;xyz=load(native/'geometry'/('f%06d.npz'%fid))['xyz_m'];tree=cKDTree(xyz);allkeys=[]
  for frame,table in frames[:count]:
   points,local,stats=project_frame_regions(frame,tree,pixel_stride=2,max_distance_m=.015,workers=2)
   allkeys.append(effective_keys(points,table[local],BASE,'abstain'))
  keys,votes=np.unique(np.concatenate(allkeys),return_counts=True)
  cache=load(ours/'checkpoints'/('f%06d_votes.npz'%fid))
  np.testing.assert_array_equal(keys,cache['keys']);np.testing.assert_array_equal(votes,cache['votes'])
  print('FROZEN_REFERENCE_MATCH',count,flush=True)
 report={'status':'PASS','checkpoints':[1,2,5,10,20],'reference':'frozen project_frame_regions + original one_vote.effective_keys, freshly loaded raw RGB-D/masks and own association decisions','all_keys_and_counts_exact':True}
 (O/'frozen_reference_validation.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
