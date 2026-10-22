"""Measure room0 evidence, then call the byte-identical accepted v2 rules."""
from pathlib import Path
from itertools import combinations
import copy,hashlib,json,sys,time,traceback
import numpy as np
from PIL import Image
from filter_rules import run_filters,semantic_excluded,fixed_primary_groups,iou_duplicate
from coverage import measure
from semantic_exemption import apply_semantic_exemption
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021')
V=R/'v1';O=R/'v2';CASE='room0_online_prefix20_20261021'
BASE=Path('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021/baseline')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def save(name,value):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 assert not (O/'complete.json').exists()
 started=time.time();locked={}
 def lock(p,expected=None):
  d=sha(p)
  if expected:assert d==expected,str(p)
  locked[str(p)]=d;return d
 audit=read(V/'audit.json');assert audit['status']=='PASS'
 for name,h in audit['output_sha256'].items():lock(V/name,h)
 manifest=read(V/'seed_manifest.json');old=read(V/'grouping.json');sem=read(O/'semantic/classification.json')
 lock(O/'semantic/classification.json');assert sem['source_manifest_sha256']==sha(V/'seed_manifest.json')
 assert sem['config']['pixel_confidence']==.7 and sem['config']['mask_fraction']==.9
 assert sem['config']['exclude_classes']==['wall','floor','ceiling','rug']
 for p,h in sem['input_hashes'].items():lock(p,h)
 catalog=read('/data/chenkejun/CVPR/results/online_room0_prefix20_20261021/catalog/room0/candidate_manifest.json')
 lock(BASE/'final/instance_surface.npz',catalog['surface_sha256']);lock(BASE/'surface_evidence.npz',catalog['evidence_sha256'])
 camera_path=Path('/home/chenkejun/CVPR/experiments/online_room0_prefix20_20261021/config.json')
 lock(camera_path,catalog['camera_config_sha256']);cfg=read(camera_path);scene=Path(cfg['source']['scene_root']);camera=cfg['camera']
 trajectory=scene/cfg['source']['trajectory'];lock(trajectory,old['source_sha256'][str(trajectory)]);poses=np.loadtxt(trajectory).reshape(-1,4,4)
 module=Path('/home/chenkejun/CVPR/experiments/online_room0_prefix20_20261021/catalog/common.py');lock(module,old['source_sha256'][str(module)])
 sys.path.insert(0,str(module.parent));import common
 assert common.DEPTH_TOLERANCE==.02
 with np.load(BASE/'final/instance_surface.npz') as z:xyz=z['xyz_m'];labels=z['instance_id'];rgb=z['rgb']
 with np.load(BASE/'surface_evidence.npz') as z:
  assert np.array_equal(xyz,z['xyz_m']) and np.array_equal(rgb,z['rgb']);state=z['state']
 with np.load(V/'surface_supports.npz') as z:indices=z['indices'];offsets=z['offsets']
 with np.load(V/'pair_metrics.npz') as z:original_intersection=z['intersection']
 sr={r['group_id']:r for r in sem['records']};seeds=[];supports={};masks={};groups={}
 for tid,g in enumerate(manifest['groups'],1):
  p=g['primary'];q=sr[g['id']];a=indices[offsets[p['index']]:offsets[p['index']+1]]
  lock(V/p['mask'],p['mask_sha256']);mask=np.asarray(Image.open(V/p['mask']))>0
  assert int(mask.sum())==p['area'] and mask.shape==(680,1200)
  assert hashlib.sha256(p['frame'].to_bytes(4,'little')+np.packbits(mask.ravel()).tobytes()).hexdigest()==p['digest']
  assert len(a)==p['support_count'] and hashlib.sha256(a.tobytes()).hexdigest()==p['support_sha256']
  assert q['original_track_id']==tid and q['mask_id']==p['id'] and q['mask_sha256']==p['mask_sha256']
  seed={k:q[k] for k in ['group_id','original_track_id','mask_id','frame','mask_sha256','area','high_excluded_pixels']}
  seed.update(support_count=len(a),support_sha256=p['support_sha256'],source_model=p['representative']['model'],mask_digest=p['digest'])
  seeds.append(seed);supports[g['id']]=a;masks[g['id']]=mask;groups[g['id']]=g
 visible={};visibility_uv={}
 for no,frame in enumerate(sorted({p['frame'] for p in seeds}),1):
  cache=R/'overmerge'/f'frame{frame:06d}_visible.npz';lock(cache,old['source_sha256'][str(cache)])
  depthp=scene/cfg['source']['depth_pattern'].format(frame=frame);lock(depthp,old['source_sha256'][str(depthp)])
  depth=np.asarray(Image.open(depthp),np.float64)/camera['depth_png_units_per_meter']
  valid,uv=common.project_visible(xyz,poses[frame],depth,camera);idx=np.flatnonzero(valid).astype(np.int32)
  with np.load(cache) as z:assert np.array_equal(idx,z['vertex_index']) and np.array_equal(uv[idx],z['uv'])
  visible[frame]=valid;visibility_uv[frame]=(idx,uv[idx])
  for p in seeds:
   if p['frame']==frame:
    gid=p['group_id'];xy=uv[idx];assert np.array_equal(idx[masks[gid][xy[:,1],xy[:,0]]],supports[gid])
  if no%10==0:print('VERIFIED_VIEWS',no,flush=True)
 remaining=[s for s in seeds if not semantic_excluded(s['high_excluded_pixels'],s['area'])]
 pairs=[];lookup={};byid={s['group_id']:s for s in seeds}
 for sa,sb in combinations(sorted(remaining,key=lambda x:x['group_id']),2):
  a,b=sa['group_id'],sb['group_id'];aa,bb=supports[a],supports[b]
  av=aa[visible[sb['frame']][aa]];bv=bb[visible[sa['frame']][bb]]
  common_n=int(len(np.intersect1d(av,bv,assume_unique=True)));full=int(len(np.intersect1d(aa,bb,assume_unique=True)))
  assert common_n==full==int(original_intersection[groups[a]['primary']['index'],groups[b]['primary']['index']])
  p=dict(a=a,b=b,count_a=len(aa),count_b=len(bb),visible_a=len(av),visible_b=len(bv),intersection=common_n,union=len(av)+len(bv)-common_n,original_intersection=full)
  pairs.append(p);lookup[a,b]=p
 ordered=sorted([s['group_id'] for s in remaining],key=lambda g:(-byid[g]['support_count'],byid[g]['frame'],byid[g]['source_model'],byid[g]['mask_digest']))
 dedup=fixed_primary_groups(ordered,lambda a,b:iou_duplicate(lookup[tuple(sorted((a,b)))]['intersection'],lookup[tuple(sorted((a,b)))]['union']))
 coverage=[]
 for members in dedup:
  gid=members[0];s=byid[gid];idx,_=visibility_uv[s['frame']];m=measure(supports[gid],idx,labels,state)
  # The accepted rules use zero as the absent positive ID; counts remain zero.
  if m['instance_id'] is None:m['instance_id']=0
  coverage.append(dict(**{k:s[k] for k in ['group_id','original_track_id','mask_id','frame','mask_sha256']},**m))
 data=dict(schema='auto-v2-count-measurements-v1',case_id=CASE,seeds=seeds,pairs=pairs,coverage=coverage,baseline_sha256=sha(BASE/'final/instance_surface.npz'),semantic_model=sem['model'],GT_used=False)
 save('measurements.json',data)
 try:selection=apply_semantic_exemption(run_filters(data),sem)
 except ValueError as error:
  save('review_required.json',dict(reason=str(error),measurements_sha256=sha(O/'measurements.json'),scope='No alternate rule silently selected'));raise
 save('selection.json',selection)
 kept=set(selection['retained_group_ids']);selected=copy.deepcopy(manifest);selected['groups']=[]
 for tid,g in enumerate(manifest['groups'],1):
  if g['id'] in kept:selected['groups'].append(dict(copy.deepcopy(g),original_track_id=tid))
 selected.update(status='automatic_v2_complete',scene='room0',mask_source_directory=str(V),original_manifest_sha256=sha(V/'seed_manifest.json'),selection_sha256=sha(O/'selection.json'),manual_overrides=[],GT_used=False)
 save('seed_manifest.json',selected)
 link=O/'masks'
 if not link.exists():link.symlink_to(V/'masks',target_is_directory=True)
 for p,h in locked.items():assert sha(p)==h,p
 save('source_lock.json',dict(source_sha256=locked,code_sha256={p.name:sha(p) for p in C.glob('*.py')},all_binary_masks_and_full_supports_verified=True,source_visible_supports_independently_reprojected=True,GT_used=False))
 save('complete.json',dict(status='PASS',input=len(seeds),retained=len(kept),stages=selection['stages'],manual_overrides=[],seconds=time.time()-started,selection_sha256=sha(O/'selection.json'),manifest_sha256=sha(O/'seed_manifest.json')))
 print('V2_COMPLETE',json.dumps(selection['stages']),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:save('failure.json',dict(error=traceback.format_exc()));raise
