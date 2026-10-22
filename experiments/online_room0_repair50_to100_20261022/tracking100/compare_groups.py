"""Read-only A/B pairing by exact source pixels; never run repair or update map."""
from common100 import *
import numpy as np
from PIL import Image
import collections

def group_data(group):
 root=TRACKROOT/group
 assert read(root/'complete.json')['status']=='PASS'
 diag=read(root/'seed_diagnostics.json')
 sig={}
 for case in diag['cases']:
  hashes=case['fingerprint_data']['ordered_seed_binary_sha256']
  assert len(hashes)==len(case['objects'])
  for obj,h in zip(case['objects'],hashes):
   tid=str(obj['track_id'])
   sig[tid]=dict(source_frame=case['fingerprint_data']['seed_frame'],binary_sha256=h,
    case_uid=case['case_uid'],joint_fingerprint=case['fingerprint'],case_old_family_ids=case['old_family_ids'])
 return dict(root=root,summary=read(root/'gate_complete.json'),sig=sig,
  holes=read(root/'ID91_holes.json'),assoc=read(root/'global_association.json'),
  queue=read(root/'tracking_queue_complete.json'),render=read(root/'render_complete.json'),
  gate=read(root/'gate/frame_decisions.json'),freeze=read(root/'input_freeze.json'))

def label(root,folder,fid):
 return np.asarray(Image.open(root/folder/f'f{fid:06d}.png'),np.uint16)

def compare_pair(a,b,at,bt):
 sa,sb=a['sig'][at],b['sig'][bt]
 ar,br=a['summary']['tracks'][at],b['summary']['tracks'][bt]
 raw=[];gate=[]
 for fid in range(496):
  am=label(a['root'],f"cases/{sa['case_uid']}/tracking",fid)==int(at)
  bm=label(b['root'],f"cases/{sb['case_uid']}/tracking",fid)==int(bt)
  union=int(np.count_nonzero(am|bm));intersection=int(np.count_nonzero(am&bm));changed=union-intersection
  raw.append(dict(frame=fid,area_A=int(am.sum()),area_B=int(bm.sum()),changed_pixels=changed,
   intersection=intersection,union=union,IoU=intersection/union if union else None))
  if fid%5==0:
   am=label(a['root'],'gate/accepted_masks',fid)==int(at)
   bm=label(b['root'],'gate/accepted_masks',fid)==int(bt)
   gate.append(dict(frame=fid,area_A=int(am.sum()),area_B=int(bm.sum()),changed_pixels=int(np.count_nonzero(am^bm))))
 identical_joint=sa['joint_fingerprint']==sb['joint_fingerprint']
 total_changed=sum(x['changed_pixels'] for x in raw)
 reuse_for_b=next(x for x in b['queue']['cases'] if x['case_uid']==sb['case_uid'])
 if identical_joint and reuse_for_b.get('inference_reused'):
  assert total_changed==0,('exact reuse differs',at,bt,total_changed)
 accepted_a=set(ar['accepted_frame_ids']);accepted_b=set(br['accepted_frame_ids'])
 details_path=TRACKROOT/f'pair_A{int(at):03d}_B{int(bt):03d}.json'
 dump(details_path,dict(raw_frames=raw,gate_mapping_frames=gate))
 return dict(A=ar['group_id'],B=br['group_id'],A_track_id=int(at),B_track_id=int(bt),
  source_frame=sa['source_frame'],binary_sha256=sa['binary_sha256'],exact_source_seed=True,
  same_joint_context=identical_joint,reused_B_inference=reuse_for_b.get('inference_reused',False),case_old_family_ids_A=sa['case_old_family_ids'],case_old_family_ids_B=sb['case_old_family_ids'],
  raw_compared_frames=496,raw_identical_frames=sum(x['changed_pixels']==0 for x in raw),
  raw_changed_pixels_sum=total_changed,
  raw_mean_IoU_on_nonempty_union=float(np.mean([x['IoU'] for x in raw if x['IoU'] is not None])),
  gate_compared_frames=100,gate_identical_binary_frames=sum(x['changed_pixels']==0 for x in gate),
  gate_changed_pixels_sum=sum(x['changed_pixels'] for x in gate),
  gate_pass_A=len(accepted_a),gate_pass_B=len(accepted_b),
  gate_pass_common=sorted(accepted_a&accepted_b),gate_pass_only_A=sorted(accepted_a-accepted_b),
  gate_pass_only_B=sorted(accepted_b-accepted_a),proposed_ID_A=ar['persistent_id'],proposed_ID_B=br['persistent_id'],
  old_family_A=ar['old_family_id'],old_family_B=br['old_family_id'],details_url=details_path.name)

def main():
 started=time.perf_counter();a,b=group_data('A'),group_data('B')
 assert a['freeze']['original_baseline_source_hashes']==b['freeze']['original_baseline_source_hashes']
 for path,h in a['freeze']['original_baseline_source_hashes'].items():assert digest(path)==h,path
 for group in [a,b]:
  for path,h in group['freeze']['seed_and_config_hashes'].items():assert digest(path)==h,path
 assert a['freeze']['raw_rgb_hashes']==b['freeze']['raw_rgb_hashes']
 bindex=collections.defaultdict(list)
 for tid,info in b['sig'].items():bindex[(info['source_frame'],info['binary_sha256'])].append(tid)
 pairs=[]
 for at,info in a['sig'].items():
  for bt in bindex[(info['source_frame'],info['binary_sha256'])]:
   pairs.append(compare_pair(a,b,at,bt))
 window_comparison=[]
 for pair in pairs:
  at,bt=str(pair['A_track_id']),str(pair['B_track_id'])
  wa=a['summary'].get('window91_observation_gate',{}).get(at)
  wb=b['summary'].get('window91_observation_gate',{}).get(bt)
  if wa is None or wb is None:continue
  raw={x['frame']:x for x in read(TRACKROOT/pair['details_url'])['raw_frames']}
  five=[]
  for ar in wa['five_frontend_error_frames']:
   fid=ar['frame'];br=next(x for x in wb['five_frontend_error_frames'] if x['frame']==fid)
   five.append(dict(frame=fid,raw_prediction_pair=raw[fid],A=ar,B=br))
  window_comparison.append(dict(A=pair['A'],B=pair['B'],case_old_family_ids_A=wa['case_old_family_ids'],
   case_old_family_ids_B=wb['case_old_family_ids'],same_original_seed=True,
   same_joint_context=pair['same_joint_context'],raw_identical_frames_of496=pair['raw_identical_frames'],
   raw_changed_pixels_sum=pair['raw_changed_pixels_sum'],five_frames=five,
   interpretation='Original gate checks any old family in the joint case. ID7 can satisfy the B-case prerequisite when ID91 is absent. A/B Gate-pass differences therefore cannot be interpreted as improved mask quality. Full repair100 partial+E1 has not executed.'))
 def concise(d):
  return dict(objects=d['summary']['objects'],cases=d['summary']['cases'],
   tracking_wall_seconds=d['queue']['seconds'],inferred_cases=d['queue']['inferred_cases'],
   reused_cases=d['queue']['reused_cases'],gate_entry_wall_seconds=d['summary']['seconds'],
   gate_body_seconds_included=d['summary']['gate_seconds'],render_seconds_separate=d['render']['seconds'],
   low_gate_count_tracks=[dict(group_id=x['group_id'],gate_pass_frames=x['accepted_mapping_frames'],initial_gate_pass_frames=sum(f['initial_objects'][str(x['track_id'])]['accepted'] for f in d['gate']['frames']),initially_passed_but_conflict_rejected_frames=sum(f['initial_objects'][str(x['track_id'])]['accepted'] and not f['objects'][str(x['track_id'])]['accepted'] for f in d['gate']['frames']),reasons=x['rejection_reasons'])
    for x in d['summary']['tracks'].values() if x['accepted_mapping_frames']<=10],
   proposed_new_ids=[x for x in d['assoc']['objects'] if x['association']=='distinct_new_identity'],
   window91_observation_gate=d['summary'].get('window91_observation_gate',{}),
   ID91=[dict(track=d['summary']['tracks'][tid],holes={k:v for k,v in h.items() if k!='frames'})
    for tid,h in d['holes']['tracks'].items()])
 result=dict(status='PASS',groups=dict(A=concise(a),B=concise(b)),exact_seed_pairs=pairs,ID91_fiveframe_comparison=window_comparison,gate_rule_evidence_url='gate_case_scope_evidence.json',
  A_only_original_seeds=[a['summary']['tracks'][t]['group_id'] for t in a['sig'] if not any(x['A_track_id']==int(t) for x in pairs)],
  B_only_original_seeds=[b['summary']['tracks'][t]['group_id'] for t in b['sig'] if not any(x['B_track_id']==int(t) for x in pairs)],
  baseline_source_hashes_unchanged=True,source_masks_and_configs_unchanged=True,shared_dense_RGB_identical=True,
  repair100_applied=False,read_only_comparison_seconds=time.perf_counter()-started,
  interpretation='IoU here compares A/B predictions, not ground-truth accuracy. Gate counts depend on visibility and other retained candidates; do not rank groups by overall pass proportion. All proposed IDs remain uncommitted.')
 dump(TRACKROOT/'comparison.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='groups'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
