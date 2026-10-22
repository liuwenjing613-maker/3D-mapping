"""Read-only explanation of unchanged cross-case abstention."""
from common100 import *
import numpy as np,collections
from PIL import Image

def main():
 tick=time.perf_counter()
 summary=read(R/'gate_complete.json');decisions=read(R/'gate/frame_decisions.json')
 association=read(R/'global_association.json')
 canonical={int(t):x['canonical_track_id'] for x in association['objects'] for t in x['member_track_ids']}
 tracks=summary['tracks'];uids=sorted(set(x['case_uid'] for x in tracks.values()))
 cases={x['case_uid']:x for x in read(R/'seed_diagnostics.json')['cases']}
 pairs=collections.defaultdict(lambda:dict(frames_with_overlap=0,overlap_pixels_sum=0,
  dropped_frames_with_overlap=0,overlap_pixels_sum_on_target_dropped_frames=0,source_frame_overlap_pixels=0))
 source={}
 for row in decisions['frames']:
  fid=row['frame'];initial={int(t) for t,m in row['initial_objects'].items() if m['accepted']}
  labels={uid:np.asarray(Image.open(R/'cases'/uid/'tracking'/f'f{fid:06d}.png'),np.uint16).ravel() for uid in uids}
  canonical_masks={}
  for tid in initial:
   m=labels[tracks[str(tid)]['case_uid']]==tid
   c=canonical[tid]
   if c in canonical_masks:canonical_masks[c]|=m
   else:canonical_masks[c]=m.copy()
  coverage=np.zeros(1200*680,np.uint8)
  for m in canonical_masks.values():coverage+=m
  conflicting=coverage>1
  for at,ar in tracks.items():
   aid=int(at);am=labels[ar['case_uid']]==aid
   drop=aid in initial and not row['objects'][at]['accepted']
   if fid==ar['source_frame']:
    source[at]=dict(frame=fid,initial=row['initial_objects'][at],final=row['objects'][at],
     raw_prediction_pixels=int(am.sum()),overlap_abstained_pixels=int(np.count_nonzero(am&conflicting)) if aid in initial else 0,
     final_gate_pixels=int(np.count_nonzero(np.asarray(Image.open(R/'gate/accepted_masks'/f'f{fid:06d}.png'))==aid)))
   if aid not in initial:continue
   for uid,lab in labels.items():
    if uid==ar['case_uid']:continue
    hist=np.bincount(lab[am],minlength=1)
    for bid in initial:
     br=tracks[str(bid)]
     if br['case_uid']!=uid or canonical[bid]==canonical[aid]:continue
     n=int(hist[bid]) if bid<len(hist) else 0
     if not n:continue
     p=pairs[(at,str(bid))];p['frames_with_overlap']+=1;p['overlap_pixels_sum']+=n
     if drop:p['dropped_frames_with_overlap']+=1;p['overlap_pixels_sum_on_target_dropped_frames']+=n
     if fid==ar['source_frame']:p['source_frame_overlap_pixels']=n
 out={}
 for tid,tr in tracks.items():
  initial=sum(row['initial_objects'][tid]['accepted'] for row in decisions['frames'])
  final=sum(row['objects'][tid]['accepted'] for row in decisions['frames'])
  dropped=[row['frame'] for row in decisions['frames'] if row['initial_objects'][tid]['accepted'] and not row['objects'][tid]['accepted']]
  other=[dict(other_track_id=int(bt),other_group_id=tracks[bt]['group_id'],other_proposed_id=tracks[bt]['persistent_id'],**p) for (at,bt),p in pairs.items() if at==tid]
  other.sort(key=lambda p:(-p['overlap_pixels_sum_on_target_dropped_frames'],-p['overlap_pixels_sum']))
  values=dict(initial_accepted_mapping_frames=initial,accepted_mapping_frames=final,overlap_abstention_dropped_frames=len(dropped),
   overlap_abstention_dropped_frame_ids=dropped,source_frame_gate_evidence=source[tid],overlap_partners=other)
  assert final==tr['accepted_mapping_frames']
  out[tid]=dict(group_id=tr['group_id'],track_id=int(tid),**values)
  tr.update(values)
 continuity_path=TOP/'review/square_window_continuity100_summary.json';window={}
 if continuity_path.exists():
  continuity=read(continuity_path)
  errors=sorted(x['frame_id'] for x in continuity['misassociation_frames'])
  for tid,tr in tracks.items():
   if not (tr['persistent_id']==91 or tr['old_family_id']==91):continue
   missing=sorted(row['frame'] for row in decisions['frames'] if 'no_identified_old_family_observation' in row['objects'][tid]['reasons'])
   all_rejected=sorted(row['frame'] for row in decisions['frames'] if not row['objects'][tid]['accepted'])
   families=cases[tr['case_uid']]['old_family_ids'];five_frames=[]
   for fid in errors:
    row=next(x for x in decisions['frames'] if x['frame']==fid)
    with np.load(B/'baseline/frame_support'/f'f{fid:06d}.npz') as support:
     labels=support['candidate_persistent_id']
     counts={str(k):int(np.count_nonzero(labels==k)) for k in sorted(set([91,*families]))}
     present=bool(np.isin(labels,families).any())
    five_frames.append(dict(frame=fid,support_rows_by_old_family=counts,case_any_old_family_present=present,
     target_ID91_present=counts['91']>0,initial_gate=row['initial_objects'][tid],final_gate=row['objects'][tid]))
   window[tid]=dict(group_id=tr['group_id'],case_uid=tr['case_uid'],case_old_family_ids=families,
    old_family_gate_scope='any identified old-family observation in the entire joint case, not per target',
    five_frontend_error_frames=five_frames,missing_old_family_rejection_frames=missing,
    existing_frontend_misassociation_frames=errors,exact_frame_match=missing==errors,
    all_final_rejection_frames=all_rejected,existing_continuity_summary_sha256=digest(continuity_path),
    interpretation='These frames do not adopt the prediction through direct Gate. Repair100 partial+E1 has not run, so this does not establish whether complete historical repair would recover them.')
 summary['window91_observation_gate']=window
 result=dict(status='PASS',group=GROUP,seconds=time.perf_counter()-tick,tracks=out,window91_observation_gate=window,
  counting='Pair overlaps count original predictions of targets accepted by initial Gate in distinct cases and distinct canonical identities. Pixel sums are across 2D frames; a pixel involving 3 targets can occur in multiple pairs. These are not unique 3D points.',
  evaluated_only=True,tracking_modified=False,gate_modified=False,repair100_applied=False)
 dump(R/'gate_conflict_summary.json',result)
 summary['conflict_analysis_seconds_separate']=result['seconds']
 summary['conflict_analysis_path']=str(R/'gate_conflict_summary.json')
 dump(R/'gate_complete.json',summary)
 manifest=R/'web_manifest.json'
 if manifest.exists():
  m=read(manifest);m['summary']=summary;dump(manifest,m)
 print(json.dumps(dict(status='PASS',group=GROUP,seconds=result['seconds'],tracks={t:{k:v for k,v in x.items() if k not in ['source_frame_gate_evidence','overlap_abstention_dropped_frame_ids','overlap_partners']} for t,x in out.items()})),flush=True)
if __name__=='__main__':main()
