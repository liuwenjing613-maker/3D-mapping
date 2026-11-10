import tempfile
from pathlib import Path
import numpy as np
from strict_votes import *
def check(x,h,c):
 k,v=c.refresh(x,h);a,b=oracle(x,h)
 np.testing.assert_array_equal(k,a);np.testing.assert_array_equal(v,b)
 a,b=full_replay(x,h);np.testing.assert_array_equal(k,a);np.testing.assert_array_equal(v,b)
def main():
 cases=0;rng=np.random.default_rng(731)
 with tempfile.TemporaryDirectory() as t:
  for trial in range(35):
   c=IncrementalVotes();h=[]
   xyz=np.unique(rng.integers(-3,4,(20,3)).astype(np.float32)*.01,axis=0)
   for f in range(8):
    if f:
     xyz=np.unique(np.concatenate((xyz[rng.random(len(xyz))>.15],rng.integers(-3,4,(3,3)).astype(np.float32)*.01)),axis=0)
    world=rng.uniform(-.04,.04,(40,3));world=np.concatenate([world,world[:5],world[:5]])
    ids=rng.integers(1,5,len(world),dtype=np.int64)
    h.append(dict(frame_id=f*5,world=world,ids=ids));check(xyz,h,c);cases+=1
    if f==3:c.save(Path(t)/'cache.npz');c=IncrementalVotes.load(Path(t)/'cache.npz')
  xyz=np.array([[0,0,0],[.02,0,0]],np.float32)
  h=[dict(frame_id=0,world=np.array([[0,0,0],[0,0,0],[.02,0,0],[.02,0,0]]),ids=np.array([1,1,1,2]))]
  c=IncrementalVotes();check(xyz,h,c);assert c.votes.tolist()==[1]
  check(np.empty((0,3),np.float32),h,c)
  h.append(dict(frame_id=5,world=np.array([[0,0,0]]),ids=np.array([1])))
  check(xyz,h,c)
 print('VOTE_TEST_PASS',cases+3,flush=True)
if __name__=='__main__':main()
