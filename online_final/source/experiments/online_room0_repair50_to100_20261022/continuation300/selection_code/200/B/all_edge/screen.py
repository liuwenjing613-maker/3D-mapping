"""User-authorized full-frame OR artificial-crop boundary gate; no map writes or mask pixel edits."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import sys,json,hashlib,copy,time,os
import numpy as np
from PIL import Image
n=int(sys.argv[1]);mode=sys.argv[2];assert n==200
C=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/continuation300/selection_code/200/B')
R=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300/200/B/candidates')
O=R/'all_edge_20261021';V=O/'v1';S=O/'v2';P=R/'grouping_r2'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def edge_counts(a):
 return dict(left=int(a[:,0].sum()),top=int(a[0,:].sum()),right=int(a[:,-1].sum()),bottom=int(a[-1,:].sum()))
def link(dest,source):
 dest=Path(dest)
 if not dest.exists():dest.symlink_to(Path(source).resolve(),target_is_directory=Path(source).is_dir())
sys.path.insert(0,str(C/'v1'))
from grouping_rules import group_seeds
from noise_filter import shape_features,classify,RULE
from crop_filter import artificial_crop_edges
from source_policy import assert_sam_only_groups
sys.path.insert(0,str(C/'v2'))
from filter_rules import run_filters,semantic_excluded,fixed_primary_groups,iou_duplicate
from semantic_exemption import apply_semantic_exemption
from coverage import measure
# Reuse the already approved direct-containment rule for both prefixes; static code only.
import importlib.util
filter_path=Path('/home/chenkejun/CVPR/experiments/online_room0_prefix50_20261021/v2/filter_rules.py')
spec=importlib.util.spec_from_file_location('accepted_direct_containment',filter_path)
accepted_filter=importlib.util.module_from_spec(spec);spec.loader.exec_module(accepted_filter)
run_filters=accepted_filter.run_filters
if mode=='prepare':
 O.mkdir(parents=True,exist_ok=True);V.mkdir(exist_ok=True);S.mkdir(exist_ok=True)
 assert not (V/'audit.json').exists(),'Variant already exists; do not overwrite'
 started=time.perf_counter();prior=read(P/'grouping.json');pa=read(P/'audit.json')
 assert pa['status']=='PASS'
 locked={}
 for name,h in pa['output_sha256'].items():
  assert sha(P/name)==h
  locked[str(P/name)]=h
 for rel in ['baseline/final/instance_surface.npz','baseline/surface_evidence.npz','v1/seed_manifest.json','v2/selection.json']:
  locked[str(R/rel)]=sha(R/rel)
 with np.load(P/'surface_supports.npz') as z:src_idx=z['indices'];src_off=z['offsets']
 with np.load(P/'pair_metrics.npz') as z:src_inter=z['intersection']
 before=read(R/'v1/grouping.json')
 feature_cache={s['digest']:s['noise_features'] for s in before['seeds']+before['excluded_noise_masks']}
 legacy_by_digest={g['primary']['digest']:g['id'] for g in read(R/'v1/seed_manifest.json')['groups']}
 old_groups={s['digest']:before['groups'][s['group']]['id'] for s in before['seeds']}
 old_keep=set(read(R/'v2/selection.json')['retained_group_ids'])
 active=[];supports=[];old_i=[];edge_audit=[];edge_ex=[];noise_ex=[];raw_keep=[]
 for s0 in prior['seeds']:
  s=copy.deepcopy(s0);path=P/s['mask'];h=sha(path)
  assert h==s['mask_sha256'];locked[str(path)]=h
  a=np.asarray(Image.open(path))>0
  assert a.shape==(680,1200) and int(a.sum())==s['area']
  assert hashlib.sha256(s['frame'].to_bytes(4,'little')+np.packbits(a.ravel()).tobytes()).hexdigest()==s['digest']
  assert 500<=s['frame']<=n*5-5 and s['frame']%5==0
  ec=edge_counts(a);touch=any(ec.values())
  crop_contacts=[];seen_boxes=set()
  for alias in s['aliases']:
   box=tuple(alias['source_box']);key=(alias['model'],box)
   if key in seen_boxes:continue
   seen_boxes.add(key)
   x0,y0,x1,y1=box
   assert 0<=x0<x1<=1200 and 0<=y0<y1<=680
   local={}
   if alias['model']!=0:
    if x0>0:local['left']=int(a[y0:y1,x0].sum())
    if y0>0:local['top']=int(a[y0,x0:x1].sum())
    if x1<1200:local['right']=int(a[y0:y1,x1-1].sum())
    if y1<680:local['bottom']=int(a[y1-1,x0:x1].sum())
   local={edge:count for edge,count in local.items() if count>0}
   assert sorted(local)==sorted(artificial_crop_edges(alias['model'],box,s['bbox'],(1200,680)))
   if local:crop_contacts.append(dict(model=alias['model'],source_box=list(box),edge_pixels=local,crop_key=alias['crop_key']))
  crop_edges=sorted({edge for contact in crop_contacts for edge in contact['edge_pixels']})
  s.update(frame_edge_pixels=ec,touches_full_frame=touch,artificial_crop_edges=crop_edges,crop_contacts=crop_contacts,touches_any_edge=touch or bool(crop_edges),
    legacy_primary_group=legacy_by_digest.get(s['digest']),legacy_member_group=old_groups.get(s['digest']),
    legacy_primary_retained=legacy_by_digest.get(s['digest']) in old_keep,original_index=s0['index'])
  edge_audit.append({k:s[k] for k in ['id','frame','digest','mask','mask_sha256','frame_edge_pixels','touches_full_frame','artificial_crop_edges','crop_contacts','touches_any_edge','legacy_primary_group','legacy_primary_retained']})
  if touch or crop_edges:
   s['boundary_decision']='exclude_full_and_crop_contact' if touch and crop_edges else 'exclude_full_frame_contact' if touch else 'exclude_crop_contact';edge_ex.append(s);continue
  raw_keep.append(s['id'])
  f=feature_cache.get(s['digest'])
  if f is None:f=shape_features(a)
  decision=classify(f);s.update(noise_features=f,noise_decision=decision)
  if decision['exclude']:noise_ex.append(s);continue
  support=src_idx[src_off[s0['index']]:src_off[s0['index']+1]]
  assert len(support)==s['support_count'] and hashlib.sha256(support.tobytes()).hexdigest()==s['support_sha256']
  s['index']=len(active);active.append(s);supports.append(support);old_i.append(s0['index'])
 counts=np.array([len(x) for x in supports],dtype=np.int64)
 assert np.all(counts[:-1]>=counts[1:])
 inter=src_inter[np.ix_(old_i,old_i)].copy();union=counts[:,None]+counts[None,:]-inter
 assert np.array_equal(np.diag(inter),counts)
 passes=(union>0)&(4*inter>union);iou=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
 groups_raw=group_seeds(range(len(active)),lambda a,b:bool(passes[a,b]))
 assert_sam_only_groups(groups_raw,active)
 groups=[];manifest=[]
 for gi,members in enumerate(groups_raw):
  pid=members[0];gid=f'A{gi+1:03d}';vf={}
  for j in members:
   active[j]['group']=gi
   assert not active[j]['touches_full_frame'] and not active[j]['artificial_crop_edges']
   if j!=pid and active[j]['frame']!=active[pid]['frame']:vf.setdefault(active[j]['frame'],[]).append(j)
   if j!=pid:assert len(np.intersect1d(supports[pid],supports[j],assume_unique=True))==inter[pid,j]
  validations=[dict(frame=f,members=js,representative=js[0]) for f,js in sorted(vf.items())]
  g=dict(id=gid,index=gi,primary=pid,members=members,frames=sorted({active[j]['frame'] for j in members}),validation_frames=validations,
   semantic_window_present=any(a.get('semantic_hit') for j in members for a in active[j]['aliases']))
  groups.append(g)
  manifest.append(dict(id=gid,primary=active[pid],auxiliary_ids=[active[j]['id'] for j in members[1:]],validation_frames=validations))
 link(V/'masks',P/'masks')
 rule=dict(candidate_source_policy='new_mapping_frames_101_to200',candidate_raw_frame_range=[500,995],candidate_processed_frame_range=[101,200],full_frame_contact='exclude whole mask if any foreground pixel is at x=0,x=1199,y=0,y=679',crop_contact='exclude whole mask for any artificial source-crop edge contact; all aliases checked',source_eligibility='reuse prefix SAM-only source eligibility',noise_filter=RULE,iou=.25,tracking_requested=False,map_write=False,GT_used=False)
 stats=dict(input_masks=len(prior['seeds']),full_frame_removed=sum(s['touches_full_frame'] for s in edge_ex),crop_removed=sum(bool(s['artificial_crop_edges']) for s in edge_ex),both_edge_removed=sum(s['touches_full_frame'] and bool(s['artificial_crop_edges']) for s in edge_ex),total_edge_removed=len(edge_ex),after_boundary_filter=len(raw_keep),crop_only_after_boundary=sum(bool(s['artificial_crop_edges']) for s in active+noise_ex),
  noise_removed=len(noise_ex),after_noise=len(active),groups=len(groups),primary_crop_only=sum(bool(active[g['primary']]['artificial_crop_edges']) for g in groups))
 np.savez_compressed(V/'surface_supports.npz',indices=np.concatenate(supports).astype(np.int32),offsets=np.r_[0,np.cumsum(counts)])
 np.savez_compressed(V/'pair_metrics.npz',intersection=inter,union=union,iou=iou,passes=passes)
 save(V/'grouping.json',dict(rule=rule,stats=stats,seeds=active,groups=groups,source_sha256=locked,visibility_rule=prior['visibility_rule']))
 save(V/'seed_manifest.json',dict(status='all_edge_screened',map_write=False,tracking_requested=False,rule=rule,groups=manifest))
 save(O/'boundary_audit.json',dict(rule=rule,stats=stats,masks=edge_audit,edge_excluded=edge_ex,noise_excluded=noise_ex,
  boundary_survivor_mask_ids=raw_keep,GT_used=False,cutoff=n*5-5))
 names=['grouping.json','seed_manifest.json','surface_supports.npz','pair_metrics.npz']
 save(V/'audit.json',dict(status='PASS',output_sha256={name:sha(V/name) for name in names},source_sha256=locked,
   full_resolution_binary_border_checked=True,all_active_and_auxiliary_masks_border_free=True,mask_pixels_unchanged=True,stats=stats))
 semcache=S/'semantic/semantic_cache';semcache.mkdir(parents=True,exist_ok=True)
 cached=set()
 for previous_sem in [R/'v2/semantic',R/'full_frame_edge_20261021/v2/semantic']:
  if not (previous_sem/'classification.json').exists():continue
  for p in (previous_sem/'semantic_cache').iterdir():
   if p.suffix in ('.npz','.json'):link(semcache/p.name,p)
  cached|={f['frame'] for f in read(previous_sem/'classification.json')['frames']}
 missing=sorted({g['primary']['frame'] for g in manifest}-cached)
 save(O/'prepare_complete.json',dict(status='PASS',stats=stats,missing_semantic_frames=missing,seconds=time.perf_counter()-started,pid=os.getpid()))
 print('PREPARE_COMPLETE',n,json.dumps(stats),'new_semantic_frames',missing,flush=True)
elif mode=='select':
 started=time.perf_counter();prior=read(V/'grouping.json');manifest=read(V/'seed_manifest.json');sem=read(S/'semantic/classification.json')
 assert sem['source_manifest_sha256']==sha(V/'seed_manifest.json')
 with np.load(V/'surface_supports.npz') as z:indices=z['indices'];offsets=z['offsets']
 with np.load(R/'baseline/final/instance_surface.npz') as z:xyz=z['xyz_m'];labels=z['instance_id']
 with np.load(R/'baseline/surface_evidence.npz') as z:state=z['state'];assert np.array_equal(xyz,z['xyz_m'])
 sr={r['group_id']:r for r in sem['records']};seeds=[];supports={};visible={};vis_indices={};uvs={}
 for tid,g in enumerate(manifest['groups'],1):
  p=g['primary'];gid=g['id'];q=sr[gid];mask=np.asarray(Image.open(V/p['mask']))>0
  assert not any(edge_counts(mask).values()) and not p['artificial_crop_edges'] and sha(V/p['mask'])==p['mask_sha256']
  ix=indices[offsets[p['index']]:offsets[p['index']+1]];supports[gid]=ix
  frame=p['frame']
  if frame not in visible:
   with np.load(R/'overmerge'/f'frame{frame:06d}_visible.npz') as z:vid=z['vertex_index'];uv=z['uv']
   v=np.zeros(len(xyz),bool);v[vid]=True;visible[frame]=v;vis_indices[frame]=vid;uvs[frame]=uv
  assert np.array_equal(vis_indices[frame][mask[uvs[frame][:,1],uvs[frame][:,0]]],ix)
  seed={k:q[k] for k in ['group_id','original_track_id','mask_id','frame','mask_sha256','area','high_excluded_pixels']}
  seed.update(support_count=len(ix),support_sha256=p['support_sha256'],source_model=p['representative']['model'],mask_digest=p['digest'])
  seeds.append(seed)
 remaining=[s for s in seeds if not semantic_excluded(s['high_excluded_pixels'],s['area'])]
 pairs=[];lookup={};byid={s['group_id']:s for s in seeds}
 for sa,sb in combinations(sorted(remaining,key=lambda x:x['group_id']),2):
  a,b=sa['group_id'],sb['group_id'];aa,bb=supports[a],supports[b]
  av=aa[visible[sb['frame']][aa]];bv=bb[visible[sa['frame']][bb]]
  common=int(len(np.intersect1d(av,bv,assume_unique=True)));full=int(len(np.intersect1d(aa,bb,assume_unique=True)))
  assert common==full
  p=dict(a=a,b=b,count_a=len(aa),count_b=len(bb),visible_a=len(av),visible_b=len(bv),intersection=common,union=len(av)+len(bv)-common,original_intersection=full)
  pairs.append(p);lookup[a,b]=p
 ordered=sorted([s['group_id'] for s in remaining],key=lambda g:(-byid[g]['support_count'],byid[g]['frame'],byid[g]['source_model'],byid[g]['mask_digest']))
 dedup=fixed_primary_groups(ordered,lambda a,b:iou_duplicate(lookup[tuple(sorted((a,b)))]['intersection'],lookup[tuple(sorted((a,b)))]['union']))
 coverage=[]
 for members in dedup:
  gid=members[0];s=byid[gid];m=measure(supports[gid],vis_indices[s['frame']],labels,state)
  if m['instance_id'] is None:m['instance_id']=0
  coverage.append(dict(**{k:s[k] for k in ['group_id','original_track_id','mask_id','frame','mask_sha256']},**m))
 data=dict(schema='auto-v2-count-measurements-v1',case_id=f'room0_prefix{n}_all_edge_20261021',seeds=seeds,pairs=pairs,coverage=coverage,GT_used=False)
 save(S/'measurements.json',data)
 selection=apply_semantic_exemption(run_filters(data,allow_chained_containment=True),sem)
 selection['candidate_source_policy']='new_mapping_frames_101_to200'
 selection['candidate_raw_frame_range']=[500,995]
 selection['candidate_processed_frame_range']=[101,200]
 selection['repair50_applied']=True;selection['repair100_applied']=True;selection['current_checkpoint_repaired']=False;selection['previous_committed_checkpoint']=100;selection['scoring_history']=[1,200]
 selection['boundary_filter']='exclude any full-frame OR artificial-crop edge contact before noise, grouping and subsequent selection'
 selection['containment_policy']='direct pairwise >70% remove smaller; inherited user-approved policy'
 save(S/'selection.json',selection);kept=set(selection['retained_group_ids'])
 selected=copy.deepcopy(manifest);selected['groups']=[g for g in manifest['groups'] if g['id'] in kept]
 selected.update(status='complete',mask_source_directory=str(V),GT_used=False,repair_applied=True,repair50_applied=True,repair100_applied=True,current_checkpoint_repaired=False,previous_committed_checkpoint=100)
 save(S/'seed_manifest.json',selected)
 for path,h in prior['source_sha256'].items():assert sha(path)==h,path
 summary=dict(status='PASS',prefix=n,cutoff=n*5-5,candidate_source_policy='new_mapping_frames_101_to200',candidate_processed_frame_range=[101,200],candidate_raw_frame_range=[500,995],initial=prior['stats'],stages=selection['stages'],retained_group_ids=sorted(kept),
  retained=len(kept),retained_crop_only=[g['id'] for g in selected['groups'] if g['primary']['artificial_crop_edges']],
  retained_frame_touch=0,retained_crop_touch=0,all_original_inputs_unchanged=True,mask_pixels_unchanged=True,GT_used=False,future_frames_used=False,repair_applied=True,repair50_applied=True,repair100_applied=True,current_checkpoint_repaired=False,previous_committed_checkpoint=100,
  source_lock_sha256=sha(V/'audit.json'),filter_code_sha256=sha(filter_path),prepare_pid=read(O/'prepare_complete.json')['pid'],select_pid=os.getpid(),seconds=time.perf_counter()-started)
 save(O/'complete.json',summary);print('SELECT_COMPLETE',json.dumps(summary),flush=True)
else:raise ValueError(mode)
