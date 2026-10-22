from bootstrap_continue import *
from track_pure_gate import evaluate
import collections,traceback
def main():
 start=time.perf_counter()
 assert read(TOP/'tracking100/gate50_equivalence.json')['status']=='PASS'
 assert digest(CONT/'track_pure_gate.py')==read(TOP/'tracking100/gate50_equivalence.json')['pure_gate_sha256']
 assert read(ROOT/'input_fence_audit.json')['status']=='PASS'
 assert read(R/'tracking_queue_complete.json')['status']=='PASS'
 b=load_script('_read_only100_batch',CONT/'track_batch.py')
 scene,cases,objects,canonical,policy=b.inputs()
 t=time.perf_counter();association=b.freeze_global_association(scene,cases,objects,canonical);assoc_seconds=time.perf_counter()-t
 t=time.perf_counter();tracking=b.validate_tracking(cases);validation_seconds=time.perf_counter()-t
 result=evaluate(b,scene,cases,objects,canonical,policy,tracking,out=R/'gate',
  progress=lambda n,total,seconds:emit('gate',processed_frames=n,total_frames=total,gate_seconds=seconds))
 result.update(group=GROUP,raw_cutoff=CURRENT_RAW,geometry_sha256=digest(B/'geometry/surface.ply'),repair50_applied=True,
  current_identity_source=f'committed layers through processed{OLD_COUNT}; current association for remaining frames through{CHECKPOINT}',
  original_gate_and_thresholds_unchanged=True,whole_history_mapping_frames=CHECKPOINT)
 dump(R/'gate/frame_decisions.json',result)
 bytrack={}
 for tid,obj in objects.items():
  frames=[tracking[obj['case_uid']][fid] for fid in range(RAW_COUNT)]
  decisions=[x for x in result['frames']]
  reasons=collections.Counter(reason for row in decisions for reason in row['objects'][str(tid)]['reasons'])
  pid=next(x['persistent_id'] for x in association['objects'] if x['canonical_track_id']==canonical[tid])
  bytrack[str(tid)]=dict(track_id=tid,group_id=obj['group_id'],mask_id=obj['mask_id'],source_frame=cases[obj['case_uid']]['config']['seed_frame'],
   case_uid=obj['case_uid'],persistent_id=pid,old_family_id=obj['old_family_id'],seed_support_ID91_points=obj.get('seed_support_ID91_points',0),
   seed_projected_points=len(obj['points']),raw_nonempty_frames=sum(x['areas'][str(tid)]>0 for x in frames),raw_frames=RAW_COUNT,
   accepted_mapping_frames=result['counts']['accepted_frames_by_track'][str(tid)],mapping_frames=CHECKPOINT,rejection_reasons=dict(reasons),
   accepted_frame_ids=[x['frame'] for x in decisions if tid in x['accepted_track_ids']])
 scene.verify_unchanged()
 summary=dict(status='PASS',group=GROUP,cases=len(cases),objects=len(objects),seconds=time.perf_counter()-start,
  association_seconds=assoc_seconds,validation_seconds=validation_seconds,gate_seconds=result['seconds'],counts=result['counts'],tracks=bytrack,
  ID91_related_tracks=[tid for tid,row in bytrack.items() if row['persistent_id']==91 or row['old_family_id']==91 or row['seed_support_ID91_points']>0],
  evaluated_only=True,repair50_applied=True,repair100_applied=True,current_checkpoint_repaired=False,checkpoint=CHECKPOINT,previous_checkpoint=OLD_COUNT,partial_executed=False,E1_executed=False,state_committed=False)
 dump(R/'gate_complete.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='tracks'}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'gate_failure.json',dict(traceback=traceback.format_exc()));raise
