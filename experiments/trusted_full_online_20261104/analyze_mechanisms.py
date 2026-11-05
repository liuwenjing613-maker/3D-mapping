"""Read-only mechanism audit for the trusted online experiment; no GT projection."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,re,time
import numpy as np
C=Path(__file__).resolve().parent
SCENES=('room0','room1','room2','office0','office1','office2','office3','office4')
PREFIXES=(50,100,200,300,400)
PRIOR={50:0,100:50,200:100,300:200,400:300}
MODES=('normal','multi','single_mismatch','partial_conflict')
OLD=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')
NEW=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104')
SOURCES={};ISSUES=[]
def read(p):
 p=Path(p)
 if not p.exists():return None
 try:
  raw=p.read_bytes();d=json.loads(raw)
 except (OSError,json.JSONDecodeError) as e:
  ISSUES.append({'path':str(p),'issue':str(e)});return None
 SOURCES[str(p)]={'resolved':str(p.resolve()),'sha256':hashlib.sha256(raw).hexdigest()}
 return d
def check(ok,path,message):
 if not ok:ISSUES.append({'path':str(path),'issue':message})
def count_field(value):return len(value) if isinstance(value,(list,dict)) else value if isinstance(value,int) else None
def primary(g):
 p=g.get('primary',g.get('source',g))
 return {'group_id':g.get('id',g.get('group_id')),'mask_id':p.get('id',g.get('mask_id')),'frame':p.get('frame',g.get('frame')),'mask_sha256':p.get('mask_sha256'),'binary_digest':p.get('digest'),'seed_pixels':p.get('area',g.get('seed_pixels'))}
def fingerprint(p):
 digest=p.get('mask_sha256') or p.get('binary_digest')
 return str(p.get('frame'))+':'+str(digest) if digest else None
def no_family_failures(root):
 evidence=[]
 pattern=re.compile(r"AssertionError:\s*\(['\"]NO_CURRENT_OBSERVATION_FAMILY['\"],\s*(\d+),\s*['\"]([^'\"]+)['\"]\)")
 for rel in ('tracking/prepare.log','selection_tracking.log','selection_tracking_failure.json','tracking/prepare_failure.json'):
  p=root/rel
  if not p.exists():continue
  try:
   with p.open('rb') as f:
    f.seek(max(0,p.stat().st_size-262144));tail=f.read().decode('utf8',errors='replace')
   for tid,mid in pattern.findall(tail):evidence.append({'track_id':int(tid),'mask_id':mid,'source':str(p)})
  except OSError as e:ISSUES.append({'path':str(p),'issue':str(e)})
 distinct=sorted(set((e['track_id'],e['mask_id']) for e in evidence))
 return {'observed_distinct_failed_candidates':len(distinct),'candidate_keys':[list(x) for x in distinct],'evidence':evidence,'not_a_total_skipped_count':True}
def partition(root,n,is_new,mapping_complete):
 p=root/'mapping/partition_audit.json'
 if not is_new or n==50:
  return {'status':'NOT_APPLICABLE','reason':'baseline_has_no_partition_policy' if not is_new else 'prefix50_reused_before_first_policy_application','modes':None,'abstentions':None}
 rows=read(p)
 if rows is None:return {'status':'PENDING' if not mapping_complete else 'MISSING_COMPLETED_AUDIT','modes':None,'abstentions':None}
 check(isinstance(rows,list),p,'partition audit must be a list')
 if not isinstance(rows,list):return {'status':'INVALID'}
 counts=Counter();abst=Counter();projected=Counter();by_fitted=defaultdict(lambda:dict(masks=0,projected_unique_points=0,abstained_candidate_pairs=0));risks=[]
 frames=set()
 for a in rows:
  mode=a['mode'];check(mode in MODES,p,'unknown partition mode '+str(mode));counts[mode]+=1
  frame=int(a['frame_id']);frames.add(frame)
  check(PRIOR[n]*5<=frame<=n*5-5 and frame%5==0,p,'partition frame outside current interval')
  npoints=int(a['projected_points']);lost=int(a['abstentions']);abst[mode]+=lost;projected[mode]+=npoints
  # projected_points is unique per mask; abstentions counts raw projected candidate pairs.
  # Their units differ when multiple pixels project to one point. Never divide them.
  key=str(a['fitted_id']);by_fitted[key]['masks']+=1;by_fitted[key]['projected_unique_points']+=npoints;by_fitted[key]['abstained_candidate_pairs']+=lost
  if mode=='single_mismatch':
   positive=sum(map(int,a.get('owner_support',[])))
   risks.append({'frame':frame,'mask_local_id':int(a['mask_local_id']),'fitted_id':int(a['fitted_id']),'significant':a.get('significant',[]),'projected_unique_points':npoints,'positive_unique_owner_support':positive,'outside_clean_positive_owner_points':npoints-positive,'abstained_candidate_pairs':lost})
 return {'status':'PASS' if mapping_complete else 'AVAILABLE_NOT_COMMITTED','source':str(p),'audited_mask_events':len(rows),'audited_frames_with_projected_masks':len(frames),'interval_frames':n-PRIOR[n],'modes':{m:counts[m] for m in MODES},'abstentions':{m:abst[m] for m in MODES},'projected_unique_points_per_mask_summed':{m:projected[m] for m in MODES},'abstention_unit':'projected candidate pairs; duplicates can occur before strict per-point per-frame reduction; repeated across frames and masks','coverage_unit':'sum of distinct projected points within each mask, not distinct scene vertices','by_fitted_id_local_to_this_run':dict(by_fitted),'largest_single_mismatch_events':sorted(risks,key=lambda x:-x['abstained_candidate_pairs'])[:20],'GT_causal_attribution':False}
def summarize(base,scene,n,is_new):
 root=base/'online'/scene/str(n)/'B';m=read(root/'mapping/mapping_complete.json');commit=read(root/'repair/commit_complete.json');prep=read(root/'tracking/prepare_complete.json');selected=read(root/'selection_tracking_status.json')
 mapped=bool(m and m.get('status')=='PASS');committed=bool(commit and commit.get('status')=='PASS');prepared=bool(prep and prep.get('status')=='PASS')
 expected_scope=[PRIOR[n]*5,n*5-5]
 stages={};candidates={}
 for label,rel in [('v2','candidates/v2/seed_manifest.json'),('preflight','candidates/final_preflight/seed_manifest.json'),('final_abc','candidates/final_abc/seed_manifest.json')]:
  d=read(root/rel)
  groups=d.get('groups') if isinstance(d,dict) else None
  if isinstance(groups,list):stages[label]=len(groups);candidates[label]=[primary(g) for g in groups]
  else:stages[label]=None;candidates[label]=None
 seedmap=read(root/'tracking/seed_id_map.json');seed_manifest=read(root/'tracking/seed_manifest.json');gate=read(root/'tracking/gate_complete.json');queue=read(root/'tracking/tracking_queue_complete.json')
 tracked=seedmap.get('objects') if isinstance(seedmap,dict) else None
 tracked_rows=[primary(g) for g in tracked] if isinstance(tracked,list) else None
 if prepared:
  check(prep.get('checkpoint')==n,root/'tracking/prepare_complete.json','checkpoint mismatch')
  check(prep.get('candidate_source_frames')==expected_scope,root/'tracking/prepare_complete.json','candidate source interval mismatch')
  if tracked_rows is not None:check(len(tracked_rows)==prep.get('objects'),root/'tracking/seed_id_map.json','prepared object count disagrees with seed_id_map')
 skip=read(root/'tracking/skipped_no_current_family.json');failures=no_family_failures(root)
 if isinstance(skip,dict):
  items=skip.get('skipped',[]);skip_count=skip.get('skipped_count')
  check(isinstance(items,list) and skip_count==len(items),root/'tracking/skipped_no_current_family.json','skip list/count mismatch')
  reasons=dict(Counter(x.get('reason','UNSPECIFIED') for x in items))
  missing={'status':'EXPLICIT','count':skip_count,'reasons':reasons,'candidates':items,'retained_count':skip.get('retained_count')}
 elif prepared and stages['final_abc'] is not None and prep.get('objects')==stages['final_abc']:
  missing={'status':'VERIFIED_ZERO_BY_COMPLETE_PREPARE','count':0,'reasons':{},'candidates':[],'note':'No explicit skip file; final selected count equals successfully prepared object count.'}
 else:missing={'status':'PENDING' if not prepared else 'UNKNOWN_NOT_ZERO','count':None,'reasons':None,'candidates':None}
 preflight=read(root/'candidates/final_preflight/audit.json');abc=read(root/'candidates/final_abc/audit.json')
 rejected=[a for a in preflight.get('candidates',[]) if a.get('accepted') is False] if isinstance(preflight,dict) else None
 exclusions={'preflight_rejected':rejected,'final_abc_removed_A':abc.get('removed_A') if isinstance(abc,dict) else None,'family':missing,'historical_no_family_assertions':failures}
 prior=read(root/'mapping/prior_commit.json');map_next=None
 cp=root/'mapping/association/observation_support_3cm.npz'
 if mapped and cp.exists():
  with np.load(cp,allow_pickle=False) as z:map_next=int(np.asarray(z['next_instance_id']).ravel()[0])
  SOURCES[str(cp)]={'resolved':str(cp.resolve()),'read_arrays':['next_instance_id'],'file_size':cp.stat().st_size,'hash_from_mapping_source_not_recomputed':True}
 old_next=prior.get('next_instance_id') if isinstance(prior,dict) else None
 commit_next=commit.get('next_instance_id') if committed else None
 ids={'prior_commit_next_id':old_next,'mapping_next_id':map_next,'commit_next_id':commit_next,'allocated_during_mapping':map_next-old_next if map_next is not None and old_next is not None else None,'allocated_or_reserved_during_repair':commit_next-map_next if commit_next is not None and map_next is not None else None,'mapping_persistent_instances':m.get('final',{}).get('persistent_instances') if mapped else None,'note':'ID allocation includes IDs later removed or only reserved; not a GT count or an oversegmentation verdict.'}
 pending=[name for name,done in [('mapping',mapped),('tracking_prepare',prepared),('selection_tracking',bool(selected and selected.get('status')=='PASS')),('repair_commit',committed)] if not done]
 return {'root':str(root),'resolved_root':str(root.resolve()),'prefix50_reused':is_new and n==50,'status':'PASS' if committed else 'PENDING','pending_stages':pending,'candidate_raw_frame_interval':expected_scope,'all_history_mapping_frames':n,'candidate_counts':stages,'final_candidates':candidates['final_abc'],'tracking':{'prepared_objects':prep.get('objects') if prepared else None,'prepared_cases':prep.get('cases') if prepared else None,'seed_manifest_selected_masks':seed_manifest.get('selected_masks') if isinstance(seed_manifest,dict) else None,'queue_total_cases':queue.get('total_cases') if isinstance(queue,dict) else None,'gate_objects':gate.get('objects') if isinstance(gate,dict) else None,'gate_cases':gate.get('cases') if isinstance(gate,dict) else None,'tracked_candidates':tracked_rows},'candidate_exclusions':exclusions,'IDs':ids,'partition':partition(root,n,is_new,mapped)}
def delta(a,b):return a-b if isinstance(a,(int,float)) and isinstance(b,(int,float)) else None
def compare(new,old):
 out={'counts_new_minus_baseline':{k:delta(new['candidate_counts'][k],old['candidate_counts'][k]) for k in new['candidate_counts']}}
 out['counts_new_minus_baseline'].update({k:delta(new['tracking'][k],old['tracking'][k]) for k in ['prepared_objects','prepared_cases','gate_objects','gate_cases']})
 out['counts_new_minus_baseline']['no_family_skipped']=delta(new['candidate_exclusions']['family']['count'],old['candidate_exclusions']['family']['count'])
 out['ID_allocation_new_minus_baseline']={k:delta(new['IDs'][k],old['IDs'][k]) for k in ['allocated_during_mapping','allocated_or_reserved_during_repair','mapping_persistent_instances']}
 a=new['final_candidates'];b=old['final_candidates']
 if a is not None and b is not None and all(fingerprint(x) for x in a+b):
  aa={fingerprint(x):x for x in a};bb={fingerprint(x):x for x in b}
  out['exact_final_candidate_identity']={'key':'raw_frame plus exact mask_sha256 (fallback binary_digest); run-local candidate IDs are not compared','common':len(aa.keys()&bb.keys()),'baseline_only':[bb[x] for x in sorted(bb.keys()-aa.keys())],'new_only':[aa[x] for x in sorted(aa.keys()-bb.keys())]}
 else:out['exact_final_candidate_identity']={'status':'PENDING_OR_MISSING_FINGERPRINT'}
 out['causal_claim']=False
 return out
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--new-root',type=Path,default=NEW);parser.add_argument('--baseline-root',type=Path,default=OLD);parser.add_argument('--out',type=Path)
 args=parser.parse_args();rows=[]
 for scene in SCENES:
  for n in PREFIXES:
   new=summarize(args.new_root,scene,n,True);old=summarize(args.baseline_root,scene,n,False)
   rows.append({'scene':scene,'prefix':n,'group':'B','new':new,'baseline':old,'comparison':compare(new,old)})
 complete=sum(r['new']['status']=='PASS' for r in rows);modes=Counter();abst=Counter();intervals=[]
 for r in rows:
  p=r['new']['partition']
  if p.get('status')=='PASS':
   modes.update(p['modes']);abst.update(p['abstentions']);intervals.append([r['scene'],r['prefix']])
 out={'status':'ERROR' if ISSUES else 'PASS' if complete==len(rows) else 'PENDING','created_unix':time.time(),'scope':'all 8 scenes, online B, same 50/100/200/300/400 prefixes; no GT projection or prefix scoring','completed_new_prefixes':complete,'total_prefixes':len(rows),'pending_prefixes':[[r['scene'],r['prefix'],r['new']['pending_stages']] for r in rows if r['new']['status']!='PASS'],'completed_partition_intervals':intervals,'partition_mode_totals_available_intervals':dict(modes),'abstained_candidate_pair_totals_available_intervals':dict(abst),'limitations':['All-instance mechanism audit; no GT-level causal attribution.','Mask event counts, per-mask unique projected points and candidate-pair abstentions have different units. Never treat their sum as unique 3D vertices.','Missing incomplete prefixes are pending, not zero.','ID values are local to each run; compare counts and exact seed fingerprints, not numeric IDs across runs.','Observed assertions are historical failures, not a total of skipped candidates.','Early prefix diagnostics use no final GT scope.'],'rows':rows,'issues':ISSUES,'source_files':SOURCES,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 target=args.out or args.new_root/'mechanism_audit.json';target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');tmp.replace(target)
 print(json.dumps({'status':out['status'],'completed_new_prefixes':complete,'partition_intervals':len(intervals),'issues':len(ISSUES),'out':str(target)}))
 if ISSUES:raise SystemExit(2)
if __name__=='__main__':main()
