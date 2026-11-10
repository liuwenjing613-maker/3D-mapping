"""Adapt the already verified prefix-100 seed/track/Gate path; algorithms unchanged."""
from pathlib import Path
import hashlib,json,ast
BASE=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022');CONT=BASE/'continuation300';SRC=BASE/'tracking100'
def replace(s,a,b):
 assert a in s,a
 return s.replace(a,b)
def write(rel,s):
 ast.parse(s);(CONT/rel).write_text(s)
def main():
 pure=(SRC/'pure_gate.py').read_text();write('track_pure_gate.py',pure)
 batch=(SRC/'run_batch.py').read_text().replace('from common100 import R as ROOT','from common_continue import R as ROOT, RAW_COUNT, CHECKPOINT')
 batch=batch.replace('==496','==RAW_COUNT').replace('range(496)','range(RAW_COUNT)').replace("'verified_raw_frames':496","'verified_raw_frames':RAW_COUNT").replace("'raw_RGB_files_checked':496","'raw_RGB_files_checked':RAW_COUNT").replace("'label_files_checked':496*len(cases)","'label_files_checked':RAW_COUNT*len(cases)")
 batch=batch.replace("HERE=Path(__file__).resolve().parent","from common_continue import C as HERE")
 write('track_batch.py',batch)
 case=(SRC/'track_case.py').read_text()
 case=replace(case,"raw = json.loads((Path(__file__).parent/'track_config100.json').read_text())","from common_continue import R, RAW_COUNT\n    raw = json.loads((R/'track_config.json').read_text())")
 case=replace(case,'assert total == 496','assert total == RAW_COUNT')
 case=case.replace('time.monotonic()','time.perf_counter()')
 write('track_case.py',case)
 prepare=(SRC/'prepare_group.py').read_text().replace('from bootstrap100 import *','from bootstrap_continue import *')
 prepare=replace(prepare,"ready=read(TOP/'review'/('statusA.json' if GROUP=='A' else 'statusB.json'))","ready=read(CANDIDATE_ROOT/'online_qa.json')")
 prepare=replace(prepare,"inputs['raw_cutoff']==495 and inputs['raw_frames']==496","inputs['raw_cutoff']==CURRENT_RAW and inputs['raw_frames']==RAW_COUNT")
 prepare=replace(prepare,"(GROUP=='A' or fid>=250)","(GROUP=='A' or fid>=OLD_COUNT*5)")
 prepare=replace(prepare,"'committed_history_plus_adopted' if fid<=245 else 'post_repair_online_association'","'committed_history_layers' if fid<OLD_COUNT*5 else 'post_repair_online_association'")
 prepare=replace(prepare,"cp=C/GROUP/'cases'/(uid+'.json')","cp=C/'cases'/(uid+'.json')")
 prepare=prepare.replace('raw_frames=496','raw_frames=RAW_COUNT').replace('raw_prefix=[0,495]','raw_prefix=[0,CURRENT_RAW]').replace('mapping_frames=100','mapping_frames=CHECKPOINT')
 prepare=replace(prepare,"candidate_source_frames=[0,495] if GROUP=='A' else [250,495]","candidate_source_frames=[0,CURRENT_RAW] if GROUP=='A' else [OLD_COUNT*5,CURRENT_RAW]")
 prepare=prepare.replace('repair100_applied=False','repair100_applied=True,current_checkpoint_repaired=False,previous_checkpoint=OLD_COUNT,checkpoint=CHECKPOINT')
 write('track_prepare.py',prepare)
 gate=(SRC/'gate100.py').read_text().replace('from bootstrap100 import *','from bootstrap_continue import *').replace('from pure_gate import evaluate','from track_pure_gate import evaluate')
 gate=replace(gate,"assert read(TRACKROOT/'gate50_equivalence.json')['status']=='PASS'\n assert digest(C/'pure_gate.py')==read(TRACKROOT/'gate50_equivalence.json')['pure_gate_sha256']\n assert read(TRACKROOT/'preflight100.json')['status']=='PASS'","assert read(TOP/'tracking100/gate50_equivalence.json')['status']=='PASS'\n assert digest(CONT/'track_pure_gate.py')==read(TOP/'tracking100/gate50_equivalence.json')['pure_gate_sha256']\n assert read(ROOT/'input_fence_audit.json')['status']=='PASS'")
 gate=replace(gate,"C/'run_batch.py'","CONT/'track_batch.py'")
 gate=gate.replace('raw_cutoff=495','raw_cutoff=CURRENT_RAW').replace('whole_history_mapping_frames=100','whole_history_mapping_frames=CHECKPOINT').replace('range(496)','range(RAW_COUNT)').replace('raw_frames=496','raw_frames=RAW_COUNT').replace('mapping_frames=100','mapping_frames=CHECKPOINT')
 gate=gate.replace("'committed history/adopted layers for frames0..245; current association for250..495'","f'committed layers through processed{OLD_COUNT}; current association for remaining frames through{CHECKPOINT}'")
 gate=gate.replace('repair100_applied=False','repair100_applied=True,current_checkpoint_repaired=False,checkpoint=CHECKPOINT,previous_checkpoint=OLD_COUNT')
 write('track_gate.py',gate)
 # Per-branch code contracts used by the shared LockedModules resolver.
 for n in [200,300]:
  for group in ['A','B']:
   code=CONT/'tracking_code'/str(n)/group;code.mkdir(parents=True,exist_ok=True)
   for name in ['run_case.py','objectwise_ops.py','gate_v1.py','strict_vote_ops.py','joint_batch_ops.py','tracking_policy.json','gate_policy.json']:
    p=code/name
    if not p.exists():p.symlink_to(SRC/name)
    assert p.resolve()==(SRC/name).resolve()
   p=code/'track_case.py'
   if not p.exists():p.symlink_to(CONT/'track_case.py')
 # Confirm original decision functions stayed exact.
 def fns(s):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(s).body if isinstance(x,ast.FunctionDef)}
 a=fns((SRC/'run_batch.py').read_text());b=fns(batch)
 for name in ['inputs','freeze_global_association','metrics_for']:assert a[name]==b[name],name
 audit=dict(status='PASS',original_pure_gate_sha256=hashlib.sha256(pure.encode()).hexdigest(),pure_gate_byte_identical=True,
  exact_unchanged_batch_functions=['inputs','freeze_global_association','metrics_for'],
  track_changes=['current raw prefix config path and bound only','perf_counter timing instead of monotonic'],
  seeds_and_directions='original seed; independent forward/reverse states; dense current history',
  sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CONT.glob('track_*.py')})
 (CONT/'track_adapter_audit.json').write_text(json.dumps(audit,indent=2))
 print('TRACK_ADAPTERS_PASS')
if __name__=='__main__':main()
