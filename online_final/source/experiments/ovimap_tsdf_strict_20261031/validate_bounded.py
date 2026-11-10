from pathlib import Path
import json,time,numpy as np
import test_votes
from strict_votes_bounded import IncrementalVotes
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
 print('BOUNDED_EXACT',count,round(timings[-1],3),flush=True)
(O/'bounded_vote_validation.json').write_text(json.dumps({'status':'PASS','random_cases':283,'real_prefixes':20,'keys_and_counts_bit_exact':True,'query_seconds':timings},indent=2))

# Check sparse invalidation on distant stable geometry with a separate brute-force oracle.
from strict_votes import oracle,full_replay
rng=np.random.default_rng(4007)
x=np.column_stack((np.arange(120)*.07,np.zeros(120),np.zeros(120))).astype(np.float32);h=[];c=IncrementalVotes()
for f in range(12):
 x=x.copy();x[f,0]+=.003
 p=x.astype(np.float64)+rng.uniform(-.014,.014,x.shape)
 h.append({'frame_id':f,'world':p,'ids':rng.integers(1,6,len(p),dtype=np.int64)})
 k,v=c.refresh(x,h);a,b=oracle(x,h);np.testing.assert_array_equal(k,a);np.testing.assert_array_equal(v,b)
 if f>0:assert c.stats['stable_points']>100
# Long-enough real geometry where unrestricted distance search had become slow.
root=O/'ours/room0_400_main';cache=IncrementalVotes.load(root/'checkpoints/f000095_votes.npz');h=[]
for i in range(40):
 row=load(root/'history'/('f%06d.npz'%(i*5)));row['frame_id']=int(row['frame_id'][0]);h.append(row)
for count in [30,35,40]:
 x=load(O/'native/room0_400_trace/geometry'/('f%06d.npz'%((count-1)*5)))['xyz_m']
 k,v=cache.refresh(x,h[:count]);a,b=full_replay(x,h[:count])
 np.testing.assert_array_equal(k,a);np.testing.assert_array_equal(v,b);print('BOUNDED_FULL_REPLAY',count,flush=True)
report=json.loads((O/'bounded_vote_validation.json').read_text());report['sparse_oracle_steps']=12;report['longer_prefixes']=[30,35,40]
(O/'bounded_vote_validation.json').write_text(json.dumps(report,indent=2))
