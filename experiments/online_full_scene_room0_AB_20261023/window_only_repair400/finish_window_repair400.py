"""Audit both completed repairs, score with main R2, export lossless PLY and evidence."""
from pathlib import Path
from collections import Counter
import hashlib,json,os,subprocess,sys,time,traceback,shutil,tarfile,colorsys
import numpy as np

BASE=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
O=BASE/'window_only_repair400';MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
REVIEW=O/'review';FILES=O/'deliverables';CODE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
HASH={}
def sha(p):
 p=Path(p).resolve();s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns)
 if key not in HASH:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  HASH[key]=h.hexdigest()
 return HASH[key]
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def audit(branch):
 root=O/'400'/branch;old=BASE/'400'/branch
 assert read(root/'repair_pipeline_complete.json')['status']=='PASS'
 commit=read(root/'repair/commit_complete.json');ca=read(commit['audit'])
 checks=['prior_committed_entries_preserved_before_current_repair','every_old_entry_original_sample_voxels_exact','unchanged_supports_preserved',
  'roundtrip_checkpoint_exact','identity_counts_roundtrip_exact','durable_sample_pixel_roundtrip_exact','surface_journal_reproduces_all_candidates_and_votes',
  'baseline_and_final_accumulated_votes_exact','delta_and_rollback_exact','surface_evidence_fields_exact','native_labels_exact',
  'strict_max_one_vote_per_frame_surface_point','original_baseline_and_prior_commit_unchanged']
 assert commit['status']==ca['status']=='PASS' and commit['processed_frames']==400 and commit['last_raw_frame']==1995
 assert all(ca[k] is True for k in checks)
 for k in ['checkpoint','identity_counts','journal_manifest','source_pixel_support','observation_lineage','audit','final_map','geometry']:
  assert sha(commit[k])==commit[k+'_sha256'],k
 assert sha(root/'repair/repair_complete.json')==commit['repair_complete_sha256']
 hashes={**read(root/'run_input_freeze.json')['source_sha256'],**ca['input_sha256'],**ca['source_mask_sha256'],**ca['journal_layer_sha256']}
 for p,h in hashes.items():assert sha(p)==h,p
 for row in read(O/'adapter_audit.json')['unchanged_numerical_files']:
  assert sha(CODE/row['file'])==row['source_sha256']==sha(row['source'])
 blocked=[]
 for p in (root/'io_audit').glob('*.jsonl'):
  for line in p.read_text().splitlines():
   row=json.loads(line)
   if row.get('blocked') and not row.get('probe'):blocked.append(row)
 assert not blocked,blocked[:3]
 assert (root/'mapping').resolve()==(old/'mapping').resolve()
 result=dict(status='PASS',branch=branch,commit_checks={k:ca[k] for k in checks},verified_source_and_journal_hashes=len(hashes),
  forbidden_nonprobe_accesses=0,original_pre_repair400_mapping_shared=True,geometry_reextracted=False,original_solution_unchanged=True)
 save(REVIEW/branch/'audit.json',result);return result

def evaluate(branch,protocol):
 root=O/'400'/branch;old=BASE/'400'/branch;dest=O/'evaluation_r2'/branch;dest.mkdir(parents=True,exist_ok=True)
 nc=read(root/'repair/commit_complete.json');oc=read(old/'repair/commit_complete.json')
 surfaces=dict(before=root/'mapping/baseline/final/instance_surface.npz',original=Path(oc['final_map']),new=Path(nc['final_map']))
 xyz=None
 for p in surfaces.values():
  with np.load(p) as z:
   if xyz is None:xyz=z['xyz_m'].copy()
   else:np.testing.assert_array_equal(xyz,z['xyz_m'])
 gt=Path(protocol['canonical_GT']['room0']['file']);assert sha(gt)==protocol['canonical_GT']['room0']['file_sha256']
 fixed=dest/'fixed_correspondence.npz';oldfixed=BASE/'evaluation_r2'/branch/'fixed_correspondence.npz'
 if not fixed.exists():shutil.copyfile(oldfixed,fixed)
 assert sha(fixed)==sha(oldfixed)
 provenance=dict(input_scope_verified=True,frame_count=400,frame_ids=list(range(0,2000,5)),scene='room0',branch=branch,
  frame_list_sha256=hashlib.sha256(json.dumps(list(range(0,2000,5)),separators=(',',':')).encode()).hexdigest(),
  method='Only cabinet SAM source route disabled; full original repair kernels',
  source_sha256={str(p):sha(p) for p in surfaces.values()},mapping_source=str(root/'mapping'),
  current_commit=str(root/'repair/commit_complete.json'),current_commit_sha256=sha(root/'repair/commit_complete.json'),
  old_commit=str(old/'repair/commit_complete.json'),old_commit_sha256=sha(old/'repair/commit_complete.json'))
 save(dest/'source_provenance.json',provenance)
 timings=[]
 def run(name,args):
  start=time.perf_counter();env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
  env.pop('CVPR_ONLINE_GUARD',None);env.pop('PYTHONPATH',None)
  with (dest/(name+'.log')).open('w') as f:
   subprocess.run([sys.executable,'-B','-m','unified_eval.cli',*map(str,args)],cwd=MAIN,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  timings.append(dict(stage=name,seconds=time.perf_counter()-start))
 for stage,p in surfaces.items():
  if not (dest/stage/'metrics.json').exists():
   run(stage,['adapt-surface','--gt',gt,'--surface',p,'--fixed-correspondence',fixed,'--source-provenance',dest/'source_provenance.json',
    '--method-name',f'room0_{branch}_400_window_source_{stage}','--method-commit','d0c21a9a','--out',dest/stage])
  assert sha(fixed)==sha(oldfixed)
 for name,a,b in [('before_to_new','before','new'),('original_to_new','original','new')]:
  if not (dest/name/'paired_repair_metrics.json').exists():
   run(name,['eval-repair-pair','--gt',gt,'--before',dest/a/'canonical_prediction.npz','--after',dest/b/'canonical_prediction.npz','--out',dest/name])
 # Fresh scoring of the two old surfaces must reproduce archived numerical metrics.
 metrics={stage:read(dest/stage/'metrics.json')['per_scene'][0] for stage in surfaces}
 for stage,oldstage in [('before','before'),('original','after')]:
  prior=read(BASE/'evaluation_r2'/branch/oldstage/'metrics.json')['per_scene'][0]
  for k in ['CA_AP25_uniform','CA_AP50_uniform','CA_AP_uniform','CA_PRF1_0_5','CA_PQ','structure']:
   assert metrics[stage][k]==prior[k],(branch,stage,k)
 for stage in surfaces:
  q=REVIEW/branch/'evaluation_r2'/stage;q.mkdir(parents=True,exist_ok=True)
  for name in ['metrics.json','adapter_manifest.json']:shutil.copyfile(dest/stage/name,q/name)
 for name in ['before_to_new','original_to_new']:
  shutil.copyfile(dest/name/'paired_repair_metrics.json',REVIEW/branch/(name+'.json'))
 with np.load(gt) as z:assert len(xyz)>0
 scope=read(gt.parent/'gt_scope.json');names={x['raw_gt_id']:x['semantic_class'] for x in scope['objects']}
 per_gt={}
 for stage in surfaces:
  with np.load(dest/stage/'overlap_matrix.npz') as z:
   for j,gid in enumerate(z['gt_ids']):
    i=int(np.argmax(z['iou'][:,j]));row=per_gt.setdefault(int(gid),dict(gt_id=int(gid),semantic_class=names[int(gid)]))
    row[stage]=dict(best_iou=float(z['iou'][i,j]),pid=str(z['pred_uids'][i]),intersection=int(z['intersection'][i,j]),gt_size=int(z['gt_size'][j]),pred_size=int(z['pred_size'][i]))
 for row in per_gt.values():
  delta=row['new']['best_iou']-row['original']['best_iou'];row['delta_iou']=delta;row['change']='improved' if delta>1e-12 else 'degraded' if delta< -1e-12 else 'unchanged'
 result=dict(status='PASS',branch=branch,evaluation_profile='object_observed_repair',profile_revision=2,frozen=protocol['frozen'],
  GT_count=72,global_GT_count=350,protocol_config_sha256=protocol['config_sha256'],identical_native_geometry=True,
  fixed_correspondence_sha256=sha(fixed),fixed_correspondence_exactly_original=True,original_metrics_reproduced=True,
  metrics=metrics,per_gt=sorted(per_gt.values(),key=lambda x:x['delta_iou']),GT_changes=dict(Counter(x['change'] for x in per_gt.values())),timings=timings,
  surfaces={k:str(v) for k,v in surfaces.items()})
 save(dest/'comparison.json',result);return result

def tracking_compare(branch):
 old=BASE/'400'/branch/'tracking';new=O/'400'/branch/'tracking'
 def identity(root):
  seeds=read(root/'seed_id_map.json')['objects'];a={x['canonical_track_id']:x for x in read(root/'global_association.json')['objects']}
  return {s['source']['digest']:dict(seed=s,association=a[s['track_id']]) for s in seeds}
 a,b=identity(old),identity(new);assert set(b)<=set(a)
 oldgate=read(old/'gate/frame_decisions.json');newgate=read(new/'gate/frame_decisions.json')
 before={x['frame']:x for x in oldgate['frames']};after={x['frame']:x for x in newgate['frames']};assert set(before)==set(after)
 rows=[];alignment={}
 for digest,n in b.items():
  o=a[digest];ot=o['seed']['track_id'];nt=n['seed']['track_id'];oa=o['association'];na=n['association']
  changed=[f for f in before if before[f]['objects'][str(ot)]['accepted']!=after[f]['objects'][str(nt)]['accepted']]
  op,npid=oa['persistent_id'],na['persistent_id']
  if oa['association']==na['association']=='distinct_new_identity':alignment[str(npid)]=op
  rows.append(dict(mask_digest=digest,old_track=ot,new_track=nt,old_pid=op,new_pid=npid,old_association=oa['association'],new_association=na['association'],
   old_accepted=oldgate['counts']['accepted_frames_by_track'][str(ot)],new_accepted=newgate['counts']['accepted_frames_by_track'][str(nt)],
   Gate_decision_changed_frames=changed,seed_mask_pixels_same=True))
 assert len(set(alignment.values()))==len(alignment)
 queue=read(new/'tracking_queue_complete.json');reuse=read(new/'cache_fingerprint_audit.json')
 result=dict(status='PASS',retained_objects=len(b),removed_objects=len(a)-len(b),cases=queue['total_cases'],inferred_cases=queue['inferred_cases'],reused_cases=queue['reused_cases'],
  queue_seconds=queue['seconds'],rows=rows,display_only_new_ID_to_original_ID=alignment,
  alignment_rule='Only new IDs allocated from the exact same source binary mask; native PLY labels are never changed',cache_fingerprint_audit=reuse)
 save(REVIEW/branch/'tracking_comparison.json',result);return result

def export(branch,ev,tracking):
 arrays={}
 for key,p in ev['surfaces'].items():
  with np.load(p) as z:arrays[key]=(z['xyz_m'].copy(),z['instance_id'].copy())
 xyz,lab=arrays['new'];FILES.mkdir(exist_ok=True)
 dtype=np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4')])
 rec=np.empty(len(lab),dtype=dtype)
 for i,k in enumerate(['x','y','z']):rec[k]=xyz[:,i]
 colors=np.full((len(lab),3),150,np.uint8)
 for pid in np.unique(lab[lab>0]):colors[lab==pid]=np.rint(np.asarray(colorsys.hsv_to_rgb((int(pid)*.6180339887498949)%1,.67,.92))*255).astype(np.uint8)
 for i,k in enumerate(['red','green','blue']):rec[k]=colors[:,i]
 rec['instance_id']=lab
 p=FILES/f'room0_{branch}_400_window_only.ply'
 header=('ply\nformat binary_little_endian 1.0\ncomment Full original coordinates and persistent IDs; display colors only\n'+f'element vertex {len(lab)}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nend_header\n').encode()
 with p.open('wb') as f:f.write(header);rec.tofile(f)
 with p.open('rb') as f:
  assert f.read(len(header))==header;back=np.fromfile(f,dtype)
 assert np.array_equal(back,rec)
 before=arrays['before'][1];original=arrays['original'][1];aligned=lab.copy()
 for nid,oid in tracking['display_only_new_ID_to_original_ID'].items():aligned[lab==int(nid)]=oid
 oldmanifest=read(BASE/'comparison_page/manifest.json');oldpair=next(x for x in oldmanifest['pairs'] if x['key']==f'400_{branch}')
 oldfiles={stage:{**f,'path':'../'+f['path']} for stage,f in zip(['before','original'],oldpair['files'])}
 nf=dict(stage='new',path=p.name,sha256=sha(p),source=ev['surfaces']['new'],source_sha256=sha(ev['surfaces']['new']),instances=len(np.unique(lab[lab>0])))
 cab=xyz[before==7];pairs=[]
 for key,left in [('original_to_new','original'),('before_to_new','before')]:
  rawchange=int(np.count_nonzero(arrays[left][1]!=lab));alignedchange=int(np.count_nonzero(arrays[left][1]!=(aligned if left=='original' else lab)))
  pairs.append(dict(key=f'{branch}_{key}',checkpoint=400,branch=branch,mode=key,points=len(lab),changed=rawchange,aligned_changed=alignedchange,same_geometry=True,
   bounds=np.stack([xyz.min(0),xyz.max(0)]).tolist(),cabinet_bounds=np.stack([cab.min(0),cab.max(0)]).tolist(),
   files=[oldfiles[left],nf],titles=['原方案修复后' if left=='original' else '同一起点：修复前','新方案修复后'],
   display_id_alignment=tracking['display_only_new_ID_to_original_ID'] if left=='original' else {}))
 result=dict(status='PASS',path=str(p),bytes=p.stat().st_size,sha256=sha(p),all_native_points_and_IDs_roundtrip_exact=True,points=len(lab),positive_instances=nf['instances'],pairs=pairs)
 save(REVIEW/branch/'PLY_export.json',result);return result

def main():
 # Quiet dependency wait; the scheduler need not poll the computational stages.
 while True:
  statuses=[read(O/'400'/b/'status.json') if (O/'400'/b/'status.json').exists() else {} for b in ['A','B']]
  if any(s.get('status')=='FAILED' for s in statuses):raise RuntimeError(statuses)
  if all(s.get('status')=='REPAIR_COMPLETE' for s in statuses):break
  time.sleep(60)
 started=time.perf_counter();REVIEW.mkdir(exist_ok=True)
 protocol=read(MAIN/'unified_eval/configs/current_protocol.json')
 assert protocol['evaluation_profile']=='object_observed_repair' and protocol['profile_revision']==2 and protocol['evaluable_GT_count']==350 and protocol['frozen'] is False
 assert protocol['scoring_commit']=='cd260569e89de1e9525f82c45ee10842b80616b0' and protocol['audit_archive_commit']=='d035f68770cefab27c9751e97062c72faa07367c'
 assert sha(MAIN/'unified_eval/configs'/protocol['config_file'])==protocol['config_sha256']
 for k,h in protocol['evaluator_code_sha256'].items():assert sha(MAIN/'unified_eval'/k)==h
 summary=dict(status='PASS',scope='Only checkpoint400 repair rerun; each branch uses its own identical original pre-repair400 surface and prior300 history',
  protocol=protocol,branches={},pairs=[],comparison_is_not_optimization_equivalence_test=True)
 for branch in ['A','B']:
  save(O/'post_status.json',dict(status='RUNNING',stage='audit_evaluate_export',branch=branch))
  a=audit(branch);ev=evaluate(branch,protocol);tr=tracking_compare(branch);ply=export(branch,ev,tr)
  summary['branches'][branch]=dict(audit=a,evaluation=ev,tracking=tr,export=ply,
   repair_pipeline=read(O/'400'/branch/'repair_pipeline_complete.json'),commit=read(O/'400'/branch/'repair/commit_complete.json'))
  summary['pairs'].extend(ply['pairs'])
  for name in ['run_input_freeze.json','tracking_cache_plan.json','repair_pipeline_complete.json']:
   shutil.copyfile(O/'400'/branch/name,REVIEW/branch/name)
 save(REVIEW/'summary.json',summary);save(REVIEW/'manifest.json',dict(pairs=summary['pairs']))
 for p in ['adapter_audit.json','run_plan.json']:shutil.copyfile(O/p,REVIEW/p)
 with tarfile.open(O/'review.tar.gz','w:gz') as tar:
  for p in REVIEW.rglob('*'):
   if p.is_file():tar.add(p,arcname=p.relative_to(REVIEW))
 complete=dict(status='PASS',seconds=time.perf_counter()-started,verified_unique_files=len(HASH),review=str(O/'review.tar.gz'),review_sha256=sha(O/'review.tar.gz'),
  PLY_files=[summary['branches'][b]['export'] for b in ['A','B']])
 save(O/'post_complete.json',complete);save(O/'post_status.json',dict(status='COMPLETE',seconds=complete['seconds']))
 print(json.dumps({k:v for k,v in complete.items() if k!='PLY_files'}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  save(O/'post_status.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
