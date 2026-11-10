from online_common import *
from collections import Counter
import numpy as np,time,traceback
from PIL import Image
from dataclasses import replace
from scipy.spatial import cKDTree
import sys
sys.path.insert(0,str(SNAPSHOT/'revisable_instance_map/src'))
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.surface_evidence import project_frame_regions,reduce_surface_votes
from one_vote import effective_keys
def main():
 mapping=read(R/'mapping_complete.json')
 assert mapping['status']=='PASS' and mapping['frame_ids']==list(FIDS) and mapping['repair_applied']
 assert not mapping['repair100_applied'] and mapping['stopped_before_second_repair']
 commit=read(R.parent/'committed50.json');audit=read(commit['audit'])
 assert read(R/'frame51_state_audit.json')['repaired_state_used']
 # Recount every current-frame candidate with an independent conflict-count oracle.
 with np.load(R/'baseline/final/instance_surface.npz') as z:xyz=z['xyz_m'];labels=z['instance_id']
 with np.load(R/'baseline/surface_instance_frame_votes.npz') as z:pair={k:z[k] for k in z.files}
 with np.load(R/'association/observation_support_3cm.npz') as z:base=int(z['next_instance_id'][0])
 chunks=[];maxfid=-1
 for fid in FIDS:
  with np.load(R/'baseline/frame_support'/f'f{fid:06d}.npz') as z:p=z['surface_point_index'];ids=z['candidate_persistent_id']
  candidates=np.unique(np.column_stack((p,ids)),axis=0)
  points,n=np.unique(candidates[:,0],return_counts=True);valid=points[n==1]
  chosen=candidates[np.isin(candidates[:,0],valid)]
  keys=np.sort(chosen[:,0].astype(np.int64)*base+chosen[:,1]);chunks.append(keys)
  np.testing.assert_array_equal(keys,effective_keys(p,ids,base,'abstain'))
  assert len(keys)==len(np.unique(keys//base));maxfid=max(maxfid,fid)
 k,v=np.unique(np.concatenate(chunks),return_counts=True)
 expected=pair['surface_point_index'].astype(np.int64)*base+pair['instance_id'];order=np.argsort(expected)
 np.testing.assert_array_equal(k,expected[order]);np.testing.assert_array_equal(v,pair['frame_votes'][order])
 evidence=reduce_surface_votes(k,v,len(xyz),base,2,.67)
 with np.load(R/'baseline/surface_evidence.npz') as z:
  for name,value in evidence.items():np.testing.assert_array_equal(value,z[name])
 # Independently reproject committed history on the changed 100-frame geometry.
 source=ReplicaFrameSource(C/'config.json');tree=cKDTree(xyz);journal=Path(commit['pixel_journal'])
 for fid in (0,70,245):
  fr=source.load_frame(fid);parts=[]
  for suffix in ('history','adopted'):
   lab=np.asarray(Image.open(journal/f'f{fid:06d}_{suffix}.png'))
   p,i,_=project_frame_regions(replace(fr,mask_local=lab),tree,pixel_stride=2,max_distance_m=.015,workers=4)
   parts.append(np.unique(p.astype(np.int64)*base+i))
  raw=np.union1d(*parts)
  with np.load(R/'baseline/frame_support'/f'f{fid:06d}.npz') as z:
   saved=np.unique(z['surface_point_index'].astype(np.int64)*base+z['candidate_persistent_id'])
  np.testing.assert_array_equal(raw,saved)
 O=R/'all_edge_20261021';W=O/'review'
 data=read(W/'data.json');boundary=read(O/'boundary_audit.json');groups=read(O/'v1/grouping.json')
 assert data['prefix']==100 and data['summary']['status']=='PASS'
 assert data['summary']['candidate_source_policy']=='all_observed_mapping_frames_1_to100'
 assert data['summary']['repair50_applied'] and not data['summary']['repair100_applied']
 assert data['rules']['repair50_applied'] and not data['rules']['repair100_applied']
 catalog=read(R/'catalog/room0/candidate_manifest.json')
 assert catalog['history_frames']==100 and catalog['repair50_applied'] and not catalog['repair100_applied']
 assert catalog['surface_file']==str(R/'baseline/final/instance_surface.npz')
 assert catalog['surface_sha256']==sha(catalog['surface_file'])
 metadata=read(R/'metadata_correction_20261022/audit.json')
 assert metadata['status']=='PASS' and metadata['jobs_sha256_before']==metadata['jobs_sha256_after']
 assert metadata['cases_sha256_before']==metadata['cases_sha256_after']
 rows=data['rows'];keep=[x for x in rows if x['status']=='kept']
 assert len(keep)==data['summary']['retained']
 for row in rows:
  assert row['frame'] in FIDS
  assert (row['status']=='edge')==(any(row['edge_pixels'].values()) or bool(row['crop_edges']))
  for key in ('full','zoom','rgb','mask'):assert (W/row[key]).is_file()
 for seed in groups['seeds']:assert not seed['touches_any_edge'] and not seed['artificial_crop_edges']
 for row in keep:assert not any(row['edge_pixels'].values()) and not row['crop_edges']
 for path,digest in groups['source_sha256'].items():assert sha(path)==digest,path
 resolution=read(R/'guard_false_positive_audit.json')
 assert resolution['status']=='PASS' and resolution['future_input_allowlist_unchanged']
 false_events=[x['event'] for x in resolution['false_positive_output_writes']]
 blocked=[];events=0;probes=0;resolved_false_writes=0
 for auditdir in (R/'io_audit',R.parent/'io_audit'):
  for p in auditdir.glob('*.jsonl'):
   for line in p.read_text().splitlines():
    e=json.loads(line)
    if e.get('probe'):probes+=int(e['blocked']);continue
    events+=1
    if e['blocked']:
     if e in false_events:
      assert e['mode']=='w+' and e['reason']=='future_frame_after_current_mapping_step'
      resolved_false_writes+=1
     else:blocked.append(e)
 assert not blocked,blocked[:3]
 assert resolved_false_writes==len(false_events)==1
 result=dict(status='PASS',processed_frames=100,cutoff=495,candidate_source_policy='all_observed_mapping_frames_1_to100',candidate_processed_frame_range=[1,100],candidate_raw_frame_range=[0,495],first_repair=50,second_repair_applied=False,retained=len(keep),
  boundary=boundary['stats'],row_counts=dict(Counter(x['status'] for x in rows)),strict_vote_recount_exact=True,
  committed_history_reprojection_on_100_geometry_exact=True,frame51_used_repaired_state=True,active_and_aux_masks_edge_free=True,
  source_masks_unchanged=True,repair50_state_and_prefix100_provenance_verified=True,metadata_correction_numeric_inputs_unchanged=True,no_GT=True,forbidden_computational_reads=0,guard_false_positive_output_writes=resolved_false_writes,recorded_IO_events=events,intentional_fence_probes=probes,
  data_sha256=sha(W/'data.json'),at=time.time())
 write(R/'online_qa.json',result);write(W/'qa.json',result)
 write(R.parent/'review/statusA.json',dict(status='PASS',arm='A',masks_ready=True,retained=len(keep),qa_path=str(R/'online_qa.json'),retained_manifest=str(O/'v2/seed_manifest.json'),retained_manifest_relative='prefix100/all_edge_20261021/v2/seed_manifest.json',mask_source_directory=str(O/'v1'),page_url='http://127.0.0.1:8798/online-repair/masks/?prefix=100',candidate_source_policy='all_observed_mapping_frames_1_to100',candidate_processed_frame_range=[1,100],candidate_raw_frame_range=[0,495],repair50_applied=True,second_repair_applied=False))
 print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:write(R/'qa_failure.json',dict(traceback=traceback.format_exc()));raise
