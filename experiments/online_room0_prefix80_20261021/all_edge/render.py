"""Render original full-frame binary masks for review; display overlays only."""
from pathlib import Path
from collections import OrderedDict,Counter
import json,sys,hashlib,os
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
n=int(sys.argv[1]);R=Path(f'/data/chenkejun/CVPR/results/online_room0_prefix{n}_20261021')
C=Path(f'/home/chenkejun/CVPR/experiments/online_room0_prefix{n}_20261021')
O=R/'all_edge_20261021';W=O/'review';P=R/'grouping_r2'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def link(dst,src):
 if not dst.exists():dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())
W.mkdir(exist_ok=True);(W/'assets').mkdir(exist_ok=True);(W/'frames').mkdir(exist_ok=True);link(W/'masks',P/'masks')
for name,src in [('audit.json',O/'boundary_audit.json'),('complete.json',O/'complete.json'),('selection.json',O/'v2/selection.json')]:link(W/name,src)
complete=read(O/'complete.json');assert complete['status']=='PASS'
boundary=read(O/'boundary_audit.json');src=read(P/'grouping.json');active=read(O/'v1/grouping.json')
manifest=read(O/'v1/seed_manifest.json');selection=read(O/'v2/selection.json');sem=read(O/'v2/semantic/classification.json')
legacy=read(R/'v1/seed_manifest.json');legacy_sem=read(R/'v2/semantic/classification.json')
sems={r['group_id']:r for r in sem['records']};oldsem={r['group_id']:r for r in legacy_sem['records']}
old_sem_digest={g['primary']['digest']:oldsem[g['id']] for g in legacy['groups']}
audit_by_id={r['id']:r for r in boundary['masks']};kept=set(selection['retained_group_ids'])
sources={s['id']:s for s in active['seeds']+boundary['edge_excluded']+boundary['noise_excluded']}
group_by_mask={active['seeds'][i]['id']:g for g in active['groups'] for i in g['members']}
decision={r['group_id']:r for r in selection['records']}
noise={s['id']:s['noise_decision'] for s in boundary['noise_excluded']}
coverage={r['group_id']:r for r in read(O/'v2/measurements.json')['coverage']}
cfg=read(C/'config.json');raw=cfg['source'];frames=OrderedDict();rows=[]
stage_names={'semantic':'背景语义','covisible_iou_only':'跨视角去重','full_support_containment':'直接包含 >70%','skip_repair_B':'覆盖充分，无需修复','pr_filter':'p/r 筛选','noise':'原有噪声筛选'}
for no,s0 in enumerate(src['seeds']):
 s=sources[s0['id']];a=audit_by_id[s['id']];fid=s['frame'];mid=s['id'];gid=None;semrow=old_sem_digest.get(s['digest'],{})
 g=group_by_mask.get(mid);is_primary=g is not None and active['seeds'][g['primary']]['id']==mid
 if a['touches_any_edge']:
  status='edge'
  why='整帧与局部裁剪均触边，最前阶段整张排除' if a['touches_full_frame'] and a['artificial_crop_edges'] else '接触原始整帧边界，最前阶段整张排除' if a['touches_full_frame'] else '接触来源局部裁剪边界，最前阶段整张排除'
 elif mid in noise:status='noise';why=noise[mid]['reason']
 elif is_primary:
  gid=g['id'];semrow=sems[gid]
  if gid in kept:status='kept';why='通过全局 / 局部边界检查及后续筛选'
  else:status='later';why=stage_names.get(decision[gid]['terminal_stage'],decision[gid]['terminal_stage'])
 else:
  status='aux';gid=g['id'];why='同组辅助验证 mask，未单独保留为主种子'
 path=P/s0['mask'];assert sha(path)==s0['mask_sha256']
 binary=np.asarray(Image.open(path))>0
 ec={name:int(arr.sum()) for name,arr in [('left',binary[:,0]),('top',binary[0,:]),('right',binary[:,-1]),('bottom',binary[-1,:])]}
 assert ec==a['frame_edge_pixels']
 if status=='kept':assert not any(ec.values()) and not a['artificial_crop_edges']
 rgb_path=Path(raw['scene_root'])/raw['rgb_pattern'].format(frame=fid)
 if fid not in frames:
  frames[fid]=np.array(Image.open(rgb_path).convert('RGB'))
  if len(frames)>8:frames.popitem(last=False)
 frames.move_to_end(fid);rgb=frames[fid]
 assert rgb.shape[:2]==binary.shape==(680,1200)
 link(W/'frames'/f'frame{fid:06d}.jpg',rgb_path)
 tint=np.array([236,78,81] if status=='edge' else [24,208,159],dtype=float)
 overlay=rgb.copy();overlay[binary]=(rgb[binary]*.54+tint*.46).astype(np.uint8)
 contour=binary & ~ndi.binary_erosion(binary,structure=np.ones((3,3)))
 overlay[contour]=[255,229,162] if status=='edge' else [228,255,246]
 full=W/'assets'/f'{mid}_full.webp';zoom=W/'assets'/f'{mid}_zoom.webp'
 display=Image.fromarray(overlay)
 if a['crop_contacts']:
  draw=ImageDraw.Draw(display)
  for contact in a['crop_contacts']:
   bx0,by0,bx1,by1=contact['source_box'];draw.rectangle((bx0,by0,bx1-1,by1-1),outline=(255,194,65),width=2)
 old_assets=R/'full_frame_edge_20261021/review/assets'
 if not a['crop_contacts'] and (old_assets/full.name).exists():link(full,old_assets/full.name)
 else:display.save(full,quality=87,method=3)
 x0,y0,x1,y1=s['bbox'];pad=max(20,int(.1*max(x1-x0,y1-y0)))
 box=(max(0,x0-pad),max(0,y0-pad),min(1200,x1+pad),min(680,y1+pad))
 if not a['crop_contacts'] and (old_assets/zoom.name).exists():link(zoom,old_assets/zoom.name)
 else:
  im=display.crop(box);im.thumbnail((650,420));im.save(zoom,quality=90,method=3)
 cov=coverage.get(gid,{}) if is_primary else {}
 rows.append(dict(id=mid,group=gid,primary=bool(is_primary),status=status,reason=why,frame=fid,processed=fid//5+1,
  model=s['representative']['model'],area=s['area'],support_count=s['support_count'],semantic=semrow.get('top_class'),
  edge_pixels=ec,crop_edges=a['artificial_crop_edges'],crop_contacts=a['crop_contacts'],legacy_primary=a['legacy_primary_group'],legacy_member=s.get('legacy_member_group'),
  legacy_primary_retained=a['legacy_primary_retained'],mask_sha256=s['mask_sha256'],mask='masks/'+Path(s0['mask']).name,
  full=f'assets/{mid}_full.webp',zoom=f'assets/{mid}_zoom.webp',rgb=f'frames/frame{fid:06d}.jpg',
  p=cov.get('p'),r=cov.get('r'),instance_id=cov.get('instance_id'),
  supporting_frames=g['frames'] if g else [],bbox=s['bbox']))
 if (no+1)%100==0:print('RENDER',n,no+1,'/',len(src['seeds']),flush=True)
for path,h in active['source_sha256'].items():assert sha(path)==h
summary={**complete,'counts_by_status':dict(Counter(r['status'] for r in rows)),'render_pid':os.getpid(),
 'retained_ids':[r['group'] for r in rows if r['status']=='kept']}
save(W/'data.json',dict(prefix=n,summary=summary,rows=rows,
 rules=dict(frame_width=1200,frame_height=680,any_foreground_edge_pixel_excluded=True,margin_pixels=0,crop_contact_excluded=True,all_aliases_checked=True,
  grouping_recomputed=True,source_masks_unchanged=True,repair_applied=False,GT_used=False)))
save(O/'review_complete.json',dict(status='PASS',rows=len(rows),retained=summary['retained'],render_pid=os.getpid(),data_sha256=sha(W/'data.json')))
print('RENDER_COMPLETE',n,json.dumps(summary['counts_by_status']),flush=True)
