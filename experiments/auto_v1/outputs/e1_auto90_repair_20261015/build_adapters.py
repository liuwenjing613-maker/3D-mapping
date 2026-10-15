"""Create isolated configurable adapters; leave frozen E1 sources untouched."""
from pathlib import Path
import hashlib,json,difflib
C=Path('/home/chenkejun/CVPR/experiments/e1_auto90_repair_20261015')
R='/data/chenkejun/CVPR/results/e1_auto90_repair_20261015'
V=Path('/home/chenkejun/CVPR/experiments/mask_gate_no_foreign_v2_20261010/room2')
F=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009/instance_repair/frozen_e1_20261012/room2')
L=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007/objectwise_repair_20261007')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={**{n:V/n for n in ['run_batch.py','run_strict_repair.py','gate_policy.json','gate_v1.py','strict_vote_ops.py','one_vote.py','test_strict_repair.py','test_history_core.py','joint_batch_ops.py','test_gate_v1.py','test_joint_batch.py']},
 **{n:F/n for n in ['expand_room2.py','run_observation_repair.py','observation_ops.py','adaptive_core.py','test_observation_ops.py','test_adaptive_core.py']},
 'run_history_core.py':F/'dependencies/run_history_core.py','run_case.py':L/'run_case.py','objectwise_ops.py':L/'objectwise_ops.py'}
changes=[]
for name,src in sources.items():
 if not src.exists():
  if name.startswith('test_'):
   print('OPTIONAL_TEST_NOT_FOUND',src);continue
  raise FileNotFoundError(src)
 original=src.read_text();code=original
 if name in ('run_batch.py','run_strict_repair.py','run_history_core.py','expand_room2.py','run_observation_repair.py'):
  code=code.replace('/data/chenkejun/CVPR/results/mask_gate_no_foreign_v2_20261010/room2',R)
  code=code.replace('/data/chenkejun/CVPR/results/room2_observation_repair_20261012',R+'/e1')
 if name=='run_batch.py':
  code=code.replace('human_selection.json','input_selection.json').replace("'total_ranked_cases':30","'source_ROI_rank_slots':60")
  code=code.replace("'human masks'","'approved r4 exact primary masks'")
 if name in ('run_strict_repair.py','run_history_core.py'):
  code=code.replace("'room2':13","'room2':59").replace("'room2':38","'room2':90")
 if name=='expand_room2.py':
  assert '==38 and len(groups)==32' in code
  code=code.replace('==38 and len(groups)==32','==90 and len(groups)==90')
  code=code.replace('all 38 selected masks, 32 frozen persistent identities','all 90 approved unique primary masks, 90 frozen persistent identities')
  code=code.replace("'frozen V2 reused'","'same V2 gate applied to new SAM2.1 tracks'")
 if name=='run_observation_repair.py':
  old="  btx=load(B/'transactions'/f'f{fid:06d}.npz');path=A/'transactions'/f'f{fid:06d}.npz';atx=load(path) if path.exists() else btx;candidate0=atx['new_candidate_keys'];before=atx['new_frame_keys'];np.testing.assert_array_equal(strict_frame(candidate0,base),before)"
  new="""  # New seeds may have frames with no accepted tracks. Reconstruct their exact
  # unchanged original candidate set instead of assuming a transaction exists.
  bpath=B/'transactions'/f'f{fid:06d}.npz'
  if bpath.exists():btx=load(bpath)
  else:
   record=scene.records[fid];z=h.load(record['support_file'],record['support_sha256']);pp=z['surface_point_index'];ll=z['mask_local_id'];tt=np.zeros(int(ll.max())+1,np.int64)
   for mid0 in np.unique(ll):tt[mid0]=scene.lookup[(fid,int(mid0))]
   kk=keyset(pp,tt[ll],base);btx={'new_candidate_keys':kk,'new_frame_keys':strict_frame(kk,base)}
   assert not dec[fid]['accepted_track_ids']
  path=A/'transactions'/f'f{fid:06d}.npz';atx=load(path) if path.exists() else btx;candidate0=atx['new_candidate_keys'];before=atx['new_frame_keys'];np.testing.assert_array_equal(strict_frame(candidate0,base),before)"""
  assert old in code;code=code.replace(old,new)
 dst=C/name
 if dst.exists():assert dst.read_text()==code,name
 else:dst.write_text(code)
 changes.append({'file':name,'source':str(src),'source_sha256':sha(src),'adapter_sha256':sha(dst),'diff':''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile=str(src),tofile=str(dst)))})
(C/'adapter_provenance.json').write_text(json.dumps({'status':'FROZEN','scope':'path and input-count adaptation; unchanged-frame fallback; numerical rules unchanged','files':changes},indent=2))
print('ADAPTERS_READY',len(changes))
