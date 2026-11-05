"""Confirmed one-mask-per-stable-track experiment, reusing current main filters."""
import argparse, copy, hashlib, importlib.util, json, os, subprocess, sys, traceback
from pathlib import Path
from types import SimpleNamespace
from itertools import combinations
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from instance_repair.two_audit_cache import ObservationCache, write_json
from instance_repair.two_audit_sources import file_record, verify_record
from instance_repair.seed_boundary import screen, validate_selected
from instance_repair.seed_final_abc import finalize, validate_selected as validate_abc

ROOT=Path('/data/chenkejun/CVPR/results/two-audit/replica_room0_office2_v1_20261104')
OUT=ROOT/'stable_seed_filter_v1'
ORIGINAL_CODE=Path('/home/chenkejun/CVPR/experiments/unified_full_depth_20261101/main')
ORIGINAL_DATA=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/main')
PROJECTION=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009/instance_repair/frozen_e1_20261012/room0/dependencies/revisable_instance_map/surface_evidence.py')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return file_record(p)['sha256']
def mod(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def write(p,x):write_json(p,x)
def status(scene,stage,**extra):
 OUT.mkdir(exist_ok=True)
 write(OUT/'status.json',dict(status='RUNNING',scene=scene,stage=stage,**extra))
 print(json.dumps(dict(scene=scene,stage=stage,**extra)),flush=True)
def prepare(scene):
 o=OUT/scene
 if (o/'prepared.json').exists():return
 o.mkdir(parents=True,exist_ok=True)
 c=ORIGINAL_CODE/scene/'selection_code/400/A';old=ORIGINAL_DATA/scene/'400/A'
 cache=ObservationCache(ROOT/'observations'/scene)
 trackroot=ROOT/'stable_tracks'/scene;tm=read(trackroot/'manifest.json')
 assert sha(trackroot/'stable_tracks.json')==tm['artifact_sha256']['stable_tracks.json']
 tracks=read(trackroot/'stable_tracks.json');assert len(tracks)==tm['totals']['stable_tracks']
 cfg=read(old/'mapping/config.json');cam=cfg['camera'];raw=cfg['source'];sr=Path(raw['scene_root'])
 poses=np.loadtxt(sr/raw['trajectory']).reshape(-1,4,4)
 baseline=old/'candidates/baseline'
 with np.load(baseline/'final/instance_surface.npz') as z:xyz=z['xyz_m']
 cache.require_geometry(xyz)
 common=mod('original_projection_'+scene,c/'catalog/common.py')
 candidate=o/'candidates';candidate.mkdir(exist_ok=True);(candidate/'masks').mkdir(exist_ok=True)
 visible=o/'overmerge';visible.mkdir(exist_ok=True)
 locks={str(p):sha(p) for p in [trackroot/'stable_tracks.json',old/'mapping/config.json',baseline/'final/instance_surface.npz',baseline/'surface_evidence.npz',c/'catalog/common.py',sr/raw['trajectory']]}
 selected=[]
 for t in tracks:
  assert t['status']=='STABLE'
  j,m=min(enumerate(t['members']),key=lambda jm:(-jm[1]['surface_support_count'],jm[1]['frame_id'],jm[1]['mask_local_id']))
  selected.append(dict(track=t['track_id'],member=j,frame=m['frame_id'],mask_local_id=m['mask_local_id'],
                       selected_support=m['surface_support_count'],observation_key=m['observation_key'],
                       observation_count=len(t['members']),id='M'+t['track_id'][1:]))
 frames=sorted({s['frame'] for s in selected});supports={};seeds=[]
 for no,fid in enumerate(frames):
  fr=cache.frames[fid];verify_record(fr['sources']['mask'])
  labels=np.asarray(Image.open(fr['sources']['mask']['path']))
  depthp=sr/raw['depth_pattern'].format(frame=fid)
  locks[str(depthp)]=sha(depthp);locks[fr['sources']['mask']['path']]=fr['sources']['mask']['sha256']
  dep=np.asarray(Image.open(depthp),np.float64)/cam['depth_png_units_per_meter']
  valid,uv=common.project_visible(xyz,poses[fid],dep,cam)
  idx=np.flatnonzero(valid).astype(np.int32);xy=uv[idx]
  vp=visible/('frame%06d_visible.npz'%fid);np.savez_compressed(vp,vertex_index=idx,uv=xy)
  locks[str(vp)]=sha(vp)
  pack=cache.read_frame(fid,geometry_id=cache.geometry_id)
  for s in [s for s in selected if s['frame']==fid]:
   row=cache.observations[s['observation_key']];assert row['surface_support_count']==s['selected_support']
   mask=labels==s['mask_local_id'];assert int(mask.sum())==row['pixel_count']
   pts=idx[mask[xy[:,1],xy[:,0]]];assert len(pts)>0
   mid=s['id'];mp=candidate/'masks'/(mid+'.png');Image.fromarray(mask.astype(np.uint8)*255).save(mp)
   supports[mid]=pts
   y,x=np.nonzero(mask)
   alias=dict(model=0,source_box=[0,0,1200,680],frame=fid,label=s['mask_local_id'],
              candidate_id=s['observation_key'],source_path=fr['sources']['mask']['path'],
              semantic_hit=False,semantic_family=None,touches_crop=False,touches_frame=row['touches_image_border'],
              crop_key=scene+'_'+s['track'],ROI=s['track'],uid=s['track'],selection_reason='STABLE_TRACK_MAX_SUPPORT')
   seeds.append(dict(id=mid,frame=fid,area=int(mask.sum()),bbox=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1],
       digest=hashlib.sha256(fid.to_bytes(4,'little')+np.packbits(mask.ravel()).tobytes()).hexdigest(),
       aliases=[alias],representative=alias,support_count=len(pts),mask='masks/'+mid+'.png',
       mask_sha256=sha(mp),support_sha256=hashlib.sha256(pts.tobytes()).hexdigest(),rois=[s['track']],candidate_ids=[s['observation_key']],
       track_id=s['track'],selected_member=s['member'],selection_support_count=s['selected_support']))
  if (no+1)%10==0:status(scene,'prepare_supports',frames_done=no+1,frames=len(frames))
 seeds.sort(key=lambda s:(-s['support_count'],s['frame'],s['representative']['model'],s['digest']))
 for i,s in enumerate(seeds):s['index']=i
 counts=np.array([s['support_count'] for s in seeds],np.int64)
 inter=np.zeros((len(seeds),len(seeds)),np.int64)
 for i,a in enumerate(seeds):
  inter[i,i]=a['support_count']
  for j in range(i):inter[i,j]=inter[j,i]=len(np.intersect1d(supports[a['id']],supports[seeds[j]['id']],assume_unique=True))
 np.savez_compressed(candidate/'surface_supports.npz',indices=np.concatenate([supports[s['id']] for s in seeds]),offsets=np.r_[0,np.cumsum(counts)])
 np.savez_compressed(candidate/'pair_metrics.npz',intersection=inter)
 write(candidate/'grouping.json',dict(rule=dict(scene=scene,input='one chosen original full-frame mask per stable track',GT_used=False),
   seeds=seeds,groups=[],source_sha256=locks,visibility_rule=dict(depth_tolerance=.02,vertex_count=len(xyz))))
 write(o/'chosen_masks.json',dict(scene=scene,records=selected,rule='max audit support at 0.015m; earliest frame tie; select before filtering; no fallback',original_filter_support_tolerance=.02))
 stats=screen(candidate/'grouping.json',candidate/'surface_supports.npz',candidate/'pair_metrics.npz',c/'v1',o/'v1')
 # Normalize only path references for the unmodified semantic and ABC modules.
 v=o/'v1';group=read(v/'grouping.json');manifest=read(v/'seed_manifest.json')
 group['rule']['scene']=scene;group['source_sha256'].update(locks)
 for s in group['seeds']:s['mask']='masks/'+Path(s['mask']).name
 byid={s['id']:s for s in group['seeds']}
 for g in manifest['groups']:g['primary']=byid[g['primary']['id']]
 (v/'masks').symlink_to(candidate/'masks',target_is_directory=True)
 write(v/'grouping.json',group);write(v/'seed_manifest.json',manifest)
 audit=read(v/'audit.json');audit['output_sha256']={n:sha(v/n) for n in audit['output_sha256']};write(v/'audit.json',audit)
 full=copy.deepcopy(manifest);full['groups']=[dict(id='P%04d'%(i+1),primary=s,auxiliary_ids=[],validation_frames=[]) for i,s in enumerate(group['seeds'])]
 write(v/'all_pool_manifest.json',full)
 context=dict(grouping=str(v/'grouping.json'),supports=str(v/'surface_supports.npz'),pairs=str(v/'pair_metrics.npz'),
     mask_root=str(v),visible_root=str(visible),geometry=str(old/'mapping/geometry/geometry_prefix400.json'),
     mapping_config=str(old/'mapping/config.json'),semantics=str(o/'abc_semantics/semantics.json'))
 write(o/'abc_context.json',context)
 for p,h in locks.items():assert sha(p)==h
 write(o/'prepared.json',dict(status='PASS',chosen=len(selected),boundary=stats,geometry_id=cache.geometry_id,
      code_hashes={str(p):sha(p) for p in [Path(__file__),c/'v1/noise_filter.py',c/'v1/grouping_rules.py',c/'v2/filter_rules.py',c/'v2/coverage.py',c/'v2/semantic_exemption.py',PROJECTION]}))
 status(scene,'prepared',chosen=len(selected),**stats)
def semantic(scene):
 o=OUT/scene;c=ORIGINAL_CODE/scene/'selection_code/400/A'
 if (o/'semantic/classification.json').exists():return
 status(scene,'semantic')
 # The original model and its frozen configuration are used; no external calls.
 subprocess.run([sys.executable,'-B','-u',str(c/'v2/semantic/run.py'),'--manifest',str(o/'v1/all_pool_manifest.json'),
  '--raw-config',str(c/'config.json'),'--out',str(o/'semantic'),'--config',str(c/'v2/semantic/config.json')],check=True)
def filter_scene(scene):
 o=OUT/scene;c=ORIGINAL_CODE/scene/'selection_code/400/A';old=ORIGINAL_DATA/scene/'400/A'
 if (o/'complete.json').exists():return
 v=o/'v1';pool=read(v/'grouping.json');manifest=read(v/'seed_manifest.json');semantic_all=read(o/'semantic/classification.json')
 rules=mod('filters_'+scene,c/'v2/filter_rules.py');coverage=mod('coverage_'+scene,c/'v2/coverage.py');exempt=mod('exemption_'+scene,c/'v2/semantic_exemption.py')
 sem_byid={s['mask_id']:s for s in semantic_all['records']}
 with np.load(v/'surface_supports.npz') as z:
  ii,off=z['indices'],z['offsets'];support={s['id']:ii[off[s['index']]:off[s['index']+1]] for s in pool['seeds']}
 seeds=[];groups={};semrows=[];bygid={}
 for tid,g in enumerate(manifest['groups'],1):
  p=g['primary'];q=copy.deepcopy(sem_byid[p['id']]);q.update(group_id=g['id'],original_track_id=tid);semrows.append(q);groups[g['id']]=g
  rec={k:q[k] for k in ['group_id','original_track_id','mask_id','frame','mask_sha256','area','high_excluded_pixels']}
  rec.update(support_count=p['support_count'],support_sha256=p['support_sha256'],source_model=0,mask_digest=p['digest'])
  seeds.append(rec);bygid[g['id']]=rec
 visible={}
 for fid in {s['frame'] for s in pool['seeds']}:
  with np.load(o/'overmerge'/('frame%06d_visible.npz'%fid)) as z:visible[fid]=z['vertex_index']
 with np.load(old/'candidates/baseline/final/instance_surface.npz') as z:xyz=z['xyz_m'];labels=z['instance_id']
 with np.load(old/'candidates/baseline/surface_evidence.npz') as z:assert np.array_equal(xyz,z['xyz_m']);state=z['state']
 alive=[s for s in seeds if not rules.semantic_excluded(s['high_excluded_pixels'],s['area'])]
 pairs=[];lookup={}
 for a,b in combinations(sorted(alive,key=lambda s:s['group_id']),2):
  aa,bb=support[a['mask_id']],support[b['mask_id']]
  av=np.intersect1d(aa,visible[b['frame']],assume_unique=True);bv=np.intersect1d(bb,visible[a['frame']],assume_unique=True)
  inter=len(np.intersect1d(aa,bb,assume_unique=True));assert inter==len(np.intersect1d(av,bv,assume_unique=True))
  p=dict(a=a['group_id'],b=b['group_id'],count_a=len(aa),count_b=len(bb),visible_a=len(av),visible_b=len(bv),intersection=inter,union=len(av)+len(bv)-inter,original_intersection=inter)
  pairs.append(p);lookup[(p['a'],p['b'])]=p
 ordered=sorted([s['group_id'] for s in alive],key=lambda g:(-bygid[g]['support_count'],bygid[g]['frame'],bygid[g]['source_model'],bygid[g]['mask_digest']))
 dedup=rules.fixed_primary_groups(ordered,lambda a,b:rules.iou_duplicate(lookup[tuple(sorted((a,b)))]['intersection'],lookup[tuple(sorted((a,b)))]['union']))
 measured=[]
 for gg in dedup:
  s=bygid[gg[0]];m=coverage.measure(support[s['mask_id']],visible[s['frame']],labels,state)
  if m['instance_id'] is None:m['instance_id']=0
  measured.append(dict(**{k:s[k] for k in ['group_id','original_track_id','mask_id','frame','mask_sha256']},**m))
 data=dict(schema='auto-v2-count-measurements-v1',case_id='two-audit-'+scene,seeds=seeds,pairs=pairs,coverage=measured,GT_used=False)
 (o/'v2').mkdir(exist_ok=True);write(o/'v2/measurements.json',data)
 selection=exempt.apply_semantic_exemption(rules.run_filters(data),dict(records=semrows))
 write(o/'v2/selection.json',selection)
 selected=copy.deepcopy(manifest);selected['groups']=[g for g in manifest['groups'] if g['id'] in selection['retained_group_ids']]
 selected.update(status='automatic_v2_complete',scene=scene,GT_used=False,manual_overrides=[])
 write(o/'v2/seed_manifest.json',selected)
 status(scene,'v2_complete',stages=selection['stages'])
 # Existing historical overrides are never transferred by recycled group IDs.
 # Apply none unless exact chosen mask/frame identity exists in the historical ledger.
 history=read(old/'candidates/main_exceptions.json')
 write(o/'historical_exception_check.json',dict(status='PASS',source=str(old/'candidates/main_exceptions.json'),
     source_sha256=sha(old/'candidates/main_exceptions.json'),ledger=history,policy='no old group ID reuse; exact source mask identity only'))
 cfg=read(old/'mapping/config.json');cam=cfg['camera'];raw=cfg['source'];sr=Path(raw['scene_root']);poses=np.loadtxt(sr/raw['trajectory']).reshape(-1,4,4)
 projector=mod('preflight_original',PROJECTION);tree=cKDTree(xyz);kept=[];audit=[]
 for g in selected['groups']:
  p=g['primary'];mask=np.asarray(Image.open(v/p['mask']))>0
  depth=np.asarray(Image.open(sr/raw['depth_pattern'].format(frame=p['frame'])),np.float64)/cam['depth_png_units_per_meter']
  frame=SimpleNamespace(mask_local=mask.astype(np.uint16),depth_m=depth,camera=SimpleNamespace(**cam),camera_to_world=poses[p['frame']])
  pts,ids,stats=projector.project_frame_regions(frame,tree,pixel_stride=2,max_distance_m=.015,workers=4)
  n=len(pts);audit.append(dict(group_id=g['id'],mask_id=p['id'],frame=p['frame'],projected_seed_points=n,accepted=n>0))
  if n:kept.append(g)
 selected['groups']=kept;(o/'final_preflight').mkdir(exist_ok=True);write(o/'final_preflight/seed_manifest.json',selected)
 write(o/'final_preflight/audit.json',dict(status='PASS',candidates=audit,input=len(audit),output=len(kept)))
 validate_selected(o/'final_preflight/seed_manifest.json',v/'grouping.json',v)
 # All legal candidate masks have already been inferred using the original semantic code.
 (o/'abc_semantics').mkdir(exist_ok=True)
 source_hashes={str((v/'grouping.json').resolve()):sha(v/'grouping.json'),str((o/'semantic/classification.json').resolve()):sha(o/'semantic/classification.json')}
 for p,h in semantic_all['input_hashes'].items():assert sha(p)==h;source_hashes[str(Path(p).resolve())]=h
 records={mid:{k:q[k] for k in ['area','exclude','top_class','high_excluded_pixels','high_excluded_fraction','top_class_fraction','decision']} for mid,q in sem_byid.items()}
 assert set(records)=={s['id'] for s in pool['seeds']}
 write(o/'abc_semantics/semantics.json',dict(status='PASS',records=records,source_hashes=source_hashes))
 status(scene,'final_abc',preflight_kept=len(kept))
 result=finalize(o/'final_preflight/seed_manifest.json',o/'abc_context.json',o/'final_abc')
 guard=validate_abc(o/'final_abc/seed_manifest.json',o/'final_abc/receipt.json')
 write(o/'complete.json',dict(status='PASS',handoff=result,guard=guard,algorithm_map_modified=False))
 status(scene,'complete',final=len(read(o/'final_abc/seed_manifest.json')['groups']))
def main():
 a=argparse.ArgumentParser();a.add_argument('--scene',choices=['room0','room1','room2','office0','office1','office2','office3','office4','all'],default='all');args=a.parse_args()
 for s in (['room0','office2'] if args.scene=='all' else [args.scene]):
  prepare(s);semantic(s);filter_scene(s)
 write(OUT/'status.json',dict(status='PASS',stage='complete'))
if __name__=='__main__':
 try:main()
 except Exception:
  OUT.mkdir(exist_ok=True);write(OUT/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
