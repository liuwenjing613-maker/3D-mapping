from pathlib import Path
import json,time,numpy as np
import test_votes
from strict_votes_parallel import IncrementalVotes
from baseline import load,O
test_votes.IncrementalVotes=IncrementalVotes
test_votes.main()
native=O/'native/room0_20_trace';ours=O/'ours/room0_20_validation_v2'
history=[];cache=IncrementalVotes();timings=[]
for count in range(1,21):
 fid=(count-1)*5
 row=load(ours/'history'/('f%06d.npz'%fid));row['frame_id']=int(row['frame_id'][0]);history.append(row)
 xyz=load(native/'geometry'/('f%06d.npz'%fid))['xyz_m']
 start=time.perf_counter();k,v=cache.refresh(xyz,history);timings.append(time.perf_counter()-start)
 ref=load(ours/'checkpoints'/('f%06d_votes.npz'%fid))
 np.testing.assert_array_equal(k,ref['keys']);np.testing.assert_array_equal(v,ref['votes'])
 print('PARALLEL_EXACT',count,round(timings[-1],3),flush=True)
(O/'parallel_validation.json').write_text(json.dumps({'status':'PASS','random_cases':283,'real_prefixes':20,'keys_and_counts_bit_exact':True,'query_seconds':timings},indent=2))
