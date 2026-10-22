"""Extend E1 prerepair room0 C/U/T to top 20 each; create candidates, never human choices."""
from pathlib import Path
import copy, hashlib, json, math, shutil
import numpy as np
from PIL import Image

NAME='room0-prefix80-online-ROI-20261021'
HOME=Path('/home/chenkejun/CVPR/experiments/online_room0_prefix80_20261021/frontend')
ROOT=Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/frontend')
SOURCE=ROOT/'source'
CATALOG=Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/catalog/room0')
OLD=Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/unused_historical_cache')
FROZEN=Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/inputs/masks')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bbox(mask):
 yy,xx=np.nonzero(mask)
 return [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None
def union(a,b):return [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
def expanded(box,margin=.15,minimum=(400,226)):
 x0,y0,x1,y1=box
 w=min(1200,max(minimum[0],math.ceil((x1-x0)*(1+2*margin))))
 h=min(680,max(minimum[1],math.ceil((y1-y0)*(1+2*margin))))
 x=max(0,min(1200-w,round((x0+x1-w)/2)));y=max(0,min(680-h,round((y0+y1-h)/2)))
 return [x,y,x+w,y+h]
def model_size(box):
 w,h=box[2]-box[0],box[3]-box[1];s=min(800/min(w,h),1333/max(w,h))
 return [round(w*s),round(h*s)]
def tiles(box):
 x0,y0,x1,y1=box;w,h=min(600,x1-x0),min(380,y1-y0)
 nx=max(1,math.ceil((x1-x0-w)/(w*.75))+1);ny=max(1,math.ceil((y1-y0-h)/(h*.75))+1)
 return [[int(x),int(y),int(x+w),int(y+h)] for y in np.linspace(y0,y1-h,ny) for x in np.linspace(x0,x1-w,nx)]
def link_verified(source,target,expected):
 source=Path(source);target=Path(target);assert sha(source)==expected,str(source)
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():target.symlink_to(source.resolve())
 assert sha(target)==expected

def main():
 for d in ['inputs','cropformer','sam2_1','review','source/room0/assets','runtime_cache','cuda_cache']:(ROOT/d).mkdir(parents=True,exist_ok=True)
 protected={str(p):sha(p) for p in [CATALOG/'candidate_manifest.json',CATALOG/'membership.npz']}
 catalog=read(CATALOG/'candidate_manifest.json');assert catalog['no_GT_used'] and not catalog['repair_applied']
 for key in ['surface','evidence','graph','camera_config']:
  assert sha(catalog[key+'_file'])==catalog[key+'_sha256'],key
 assert sha(CATALOG/'membership.npz')==catalog['membership_sha256']
 entries=[e for e in catalog['entries'] if e['rank']<=20 and e['views']]
 nentries=len(entries); ncases=len({e['uid'] for e in entries}); nviews=sum(len(e['views']) for e in entries)
 assert nentries>0, 'No viewable prefix ROI'
 assert len(entries)<=60 and all(sum(e['group']==g for e in entries)<=20 for g in ['CONFLICT','U','T'])
 assert all(1<=len(e['views'])<=3 and all(v['frame'] in range(0,396,5) for v in e['views']) for e in entries)
 assert all([e['count'] for e in entries if e['group']==g]==sorted([e['count'] for e in entries if e['group']==g],reverse=True) for g in ['CONFLICT','U','T'])
 components={c['uid']:c for c in catalog['unique_components']};cores=[];contexts=[];items=[]
 with np.load(CATALOG/'membership.npz') as z:
  for e in entries:
   c=components[e['uid']];k=c['component_index']
   core=z['core_point_index'][z['core_offsets'][k]:z['core_offsets'][k+1]].copy()
   context=z['context_point_index'][z['context_offsets'][k]:z['context_offsets'][k+1]].copy()
   assert hashlib.sha256(core.astype('<i4').tobytes()).hexdigest()==e['core_sha256']
   cores.append(core);contexts.append(context)
   items.append(dict(fixed_order=e['rank'],uid=e['uid'],preferred_entry=e['id'],ROI=c,
                     entries=[copy.deepcopy(e)],all_aliases=[a['id'] for a in c['aliases']],
                     core_sha256=e['core_sha256']))
 co=np.r_[0,np.cumsum([len(x) for x in cores])];xo=np.r_[0,np.cumsum([len(x) for x in contexts])]
 np.savez_compressed(SOURCE/'room0/membership.npz',core_offsets=co,core_point_index=np.concatenate(cores).astype(np.int32),
  context_offsets=xo,context_point_index=np.concatenate(contexts).astype(np.int32),uid=np.asarray([e['uid'] for e in entries]))
 write(SOURCE/'room0/manifest.json',dict(name=NAME,scene='room0',items=items,
  unique_selected_ROIs=ncases,ranked_entries=nentries,candidate_manifest_sha256=sha(CATALOG/'candidate_manifest.json'),
  surface_sha256=catalog['surface_sha256'],evidence_sha256=catalog['evidence_sha256'],
  fixed_membership_sha256=sha(SOURCE/'room0/membership.npz'),no_GT_used=True,repair_applied=False,
  selection_basis='C/U/T each top20 absolute point count; component ID ties; up to 20 per state; actual prefix counts',
  human_mask_selections_created=False))
 old={'jobs':[],'policy':{'candidate_inputs':'prefix-only ROI; unchanged balanced/context/tile crop rules','old_sequence_cache_used':False}};old_cases={}
 old_jobs={j['key']:j for j in old['jobs']};old_by_rgb={}
 for j in old['jobs']:old_by_rgb.setdefault(j['RGB_sha256'],[]).append(j)
 frozen_meta={r['frame']:r for r in map(json.loads,(FROZEN/'frames.jsonl').read_text().splitlines())}
 jobs=[];cases=[];content={};verified={};summary=[];cache_rows=[]
 def add(key,rgb,base,box,extra):
  x0,y0,x1,y1=box;im=rgb[y0:y1,x0:x1]
  ip=ROOT/'inputs'/(key+'_rgb.png');bp=ROOT/'inputs'/(key+'_baseline.png')
  Image.fromarray(im).save(ip);Image.fromarray(base[y0:y1,x0:x1]).save(bp)
  digest=sha(ip);dup=content.get(digest);content.setdefault(digest,key)
  j=dict(key=key,input_file=str(ip),baseline_file=str(bp),crop_xyxy=box,shape=list(im.shape[:2]),
   RGB_sha256=digest,baseline_sha256=sha(bp),same_RGB_as=dup,model_expected_wh=model_size(box),
   effective_zoom=round((model_size(box)[0]/im.shape[1])/(1333/1200),4),**extra)
  jobs.append(j);return j
 for e in entries:
  c=components[e['uid']];case=dict(scene='room0',order=e['rank'],rank=e['rank'],uid=e['uid'],ROI=e['id'],
   aliases=[a['id'] for a in c['aliases']],counts=e['counts'],group=e['group'],views=[])
  summary_views=[]
  for v in e['views']:
   for field,df in [('RGB_file','RGB_sha256'),('alarm_file','alarm_sha256')]:
    rel='room0/'+v[field];link_verified(CATALOG/v[field],SOURCE/rel,v[df]);verified[rel]=v[df]
   rgb=np.asarray(Image.open(SOURCE/'room0'/v['RGB_file']).convert('RGB'))
   basepath=FROZEN/f'frame{v["frame"]:06d}.png';fm=frozen_meta[v['frame']]
   assert sha(basepath)==fm['mask_sha256']
   assert v['RGB_sha256']==fm.get('input_sha256',fm.get('RGB_sha256'))
   base=np.asarray(Image.open(basepath));assert rgb.shape==(680,1200,3) and base.shape==(680,1200)
   alarm=np.asarray(Image.open(SOURCE/'room0'/v['alarm_file']).convert('RGB'));target=alarm[:,:,[0,1,2][{'U':0,'T':1,'CONFLICT':2}[e['group']]]]>0
   assert target.any();seed=bbox(target);support=seed;included=[];skipped=[]
   common=dict(scene='room0',ROI=e['id'],uid=e['uid'],view=v['view'],frame=v['frame'],kind='crop')
   previous=old_cases.get(e['uid']);pv=next((x for x in previous['views'] if x['number']==v['view'] and x['frame']==v['frame']),None) if previous else None
   if pv:
    assert sha(OLD/'../..'/'roi_selected_fixed_20261006/selection-20261006-f49eb927'/pv['source_RGB'])==v['RGB_sha256']
    # Preserve the five previous candidates and mask IDs exactly; only the list order changes.
    for key in pv['variants']:
     oj=old_jobs[key];assert sha(oj['input_file'])==oj['RGB_sha256'] and sha(oj['baseline_file'])==oj['baseline_sha256']
     extra={k:oj[k] for k in ['strategy','target_pixel_coverage','all_target_pixels_covered']}
     for k in ['tile_number','window_limited']:
      if k in oj:extra[k]=oj[k]
     j=add(key,rgb,base,oj['crop_xyxy'],dict(**common,**extra));assert j['RGB_sha256']==oj['RGB_sha256'] and j['baseline_sha256']==oj['baseline_sha256']
    view=copy.deepcopy(pv);view['source_RGB']='room0/'+v['RGB_file'];view['source_alarm']='room0/'+v['alarm_file']
    view['previous_candidates_unchanged']=True
   else:
    ids,counts=np.unique(base[target],return_counts=True)
    for lid,n in zip(ids,counts):
     if lid==0 or n<max(5,int(target.sum()*.02)):continue
     mask=base==lid;obj=bbox(mask)
     if mask.sum()>.40*mask.size or (obj[2]-obj[0])*(obj[3]-obj[1])>max(400*226,4*(seed[2]-seed[0])*(seed[3]-seed[1])):
      skipped.append(int(lid));continue
     support=union(support,obj);included.append(int(lid))
    initial=expanded(support);x0,y0,x1,y1=initial;border=np.zeros(base.shape,bool)
    if x0>0:border[y0:y1,x0:min(x0+3,x1)]=True
    if x1<1200:border[y0:y1,max(x0,x1-3):x1]=True
    if y0>0:border[y0:min(y0+3,y1),x0:x1]=True
    if y1<680:border[max(y0,y1-3):y1,x0:x1]=True
    halo=expanded(seed,.10,(64,64));edge_ids,edge_n=np.unique(base[border],return_counts=True)
    for lid,n in zip(edge_ids,edge_n):
     if lid==0 or n<5:continue
     mask=base==lid;obj=bbox(mask);area=int(mask.sum());inside=int(mask[y0:y1,x0:x1].sum());near=int(mask[halo[1]:halo[3],halo[0]:halo[2]].sum())
     if area>mask.size*.40 or area<200 or inside<max(32,area*.015) or near<max(20,area*.005):continue
     if (obj[2]-obj[0])*(obj[3]-obj[1])>max(400*226,4*(seed[2]-seed[0])*(seed[3]-seed[1])):continue
     if sum([obj[0]==0,obj[1]==0,obj[2]==1200,obj[3]==680])>=1 and area>mask.size*.12:continue
     if obj[0]<x0-3 or obj[1]<y0-3 or obj[2]>x1+3 or obj[3]>y1+3:support=union(support,obj);included.append(int(lid))
    balanced=expanded(support);context=expanded(union(balanced,v['crop_xyxy']),.08)
    stem=f'room0_{e["id"]}_v{v["view"]}_f{v["frame"]:06d}';variants=[]
    for strategy,box in [('balanced',balanced),('context',context)]:
     coverage=float(target[box[1]:box[3],box[0]:box[2]].sum()/target.sum());assert coverage==1
     j=add(stem+'_'+strategy,rgb,base,box,dict(**common,strategy=strategy,target_pixel_coverage=coverage,
      all_target_pixels_covered=True,expanded_old_mask_ids=included,excluded_giant_old_mask_ids=skipped));variants.append(j['key'])
    if jobs[-2]['effective_zoom']<1.5:
     for no,box in enumerate(tiles(balanced),1):
      n=int(target[box[1]:box[3],box[0]:box[2]].sum())
      if n:
       j=add(stem+f'_tile{no}',rgb,base,box,dict(**common,strategy='tile',tile_number=no,
        target_pixel_coverage=float(n/target.sum()),all_target_pixels_covered=False,window_limited=True));variants.append(j['key'])
    view=dict(number=v['view'],frame=v['frame'],source_RGB='room0/'+v['RGB_file'],source_alarm='room0/'+v['alarm_file'],
     source_crop=v['crop_xyxy'],seed_bbox=seed,object_support_bbox=support,recommended=stem+'_balanced',variants=variants)
   view.update(trigger_group=e['group'],source_entry=e['id'],source_rank=e['rank'],source_view=v['view']);case['views'].append(view);summary_views.append(dict(v,saved_RGB=view['source_RGB'],saved_alarm=view['source_alarm']))
  cases.append(case);summary.append(dict(fixed_order=e['rank'],uid=e['uid'],preferred_entry=e['id'],counts=e['counts'],all_aliases=case['aliases'],views=summary_views))
  print('INPUT',e['id'],len(jobs),flush=True)
 # Same physical component appears once; retain all three views from each qualifying state ranking.
 merged={}
 for case in cases:
  uid=case['uid']
  if uid not in merged:
   merged[uid]=dict(case,order=len(merged)+1,rank=len(merged)+1,views=[],rankings=[],aliases=[])
  dest=merged[uid];dest['rankings'].append(dict(id=case['ROI'],group=case['group'],rank=case['rank'],count=case['counts'][case['group']]))
  dest['aliases'].append(case['ROI'])
  for view in case['views']:
   view['number']=len(dest['views'])+1;dest['views'].append(view)
 cases=list(merged.values())
 assert len(cases)==ncases and sum(len(c['views']) for c in cases)==nviews
 first=cases[0]['views'][0];rgb=np.asarray(Image.open(SOURCE/first['source_RGB']).convert('RGB'));base=np.asarray(Image.open(FROZEN/f'frame{first["frame"]:06d}.png'))
 add('room0_full_control',rgb,base,[0,0,1200,680],dict(scene='room0',frame=first['frame'],kind='control',strategy='full'))
 # Reuse only the same PNG input and the existing room0-compatible model pipeline.
 for j in jobs:
  matches=old_by_rgb.get(j['RGB_sha256'],[])
  for model in ['cropformer','sam2_1']:
   for oj in matches:
    recp=OLD/model/(oj['key']+'.json');png=recp.with_suffix('.png')
    if not recp.exists():continue
    r=read(recp)
    if r['RGB_sha256']!=j['RGB_sha256']:continue
    assert sha(png)==r['mask_sha256'];dst=ROOT/model/(j['key']+'.png');link_verified(png,dst,r['mask_sha256'])
    rr=copy.deepcopy(r);rr.pop('frozen_cache_comparison',None)
    rr.update(key=j['key'],mask_file=str(dst),RGB_sha256=j['RGB_sha256'],seconds=0.,reuse_from=oj['key'],reuse_source=str(OLD))
    if model=='sam2_1':
     npz=recp.with_suffix('.npz');assert sha(npz)==r['proposals_sha256'];dn=dst.with_suffix('.npz');link_verified(npz,dn,r['proposals_sha256']);rr['proposal_file']=str(dn)
    write(dst.with_suffix('.json'),rr);cache_rows.append(dict(key=j['key'],model=model,source_key=oj['key']));break
 source_summary=dict(name=NAME,selected_ROIs=ncases,ranked_entries=nentries,preferred_views=nviews,scenes=[dict(scene='room0',ROI_count=ncases,ranked_entries=nentries,items=summary)],human_mask_choices_created=False)
 write(SOURCE/'selection_summary.json',source_summary)
 write(SOURCE/'files_sha256.json',dict(verified,**{'room0/manifest.json':sha(SOURCE/'room0/manifest.json'),'room0/membership.npz':sha(SOURCE/'room0/membership.npz'),'selection_summary.json':sha(SOURCE/'selection_summary.json')}))
 manifest=dict(name=NAME,source=str(SOURCE),source_summary_sha256=sha(SOURCE/'selection_summary.json'),selected_cases=ncases,ranked_entries=nentries,preferred_views=nviews,
  cases=cases,jobs=jobs,controls=['room0_full_control'],verified_sources=verified,
  source_candidate_sha256=sha(CATALOG/'candidate_manifest.json'),source_surface_sha256=catalog['surface_sha256'],
  ranking=catalog['ranking'],ranking_group='CONFLICT/U/T',no_GT_used=True,map_modified=False,tracking_started=False,
  previous_candidates_used_as_human_choices=False,cache_reuse=cache_rows,protected_source_hashes=protected,
  policy=old['policy'])
 assert sum(len(c['views']) for c in cases)==nviews and len({j['key'] for j in jobs})==len(jobs)
 for p,h in protected.items():assert sha(p)==h
 write(ROOT/'input_manifest.json',manifest)
 print('PREPARED',json.dumps(dict(cases=ncases,ranked_entries=nentries,views=nviews,jobs=len(jobs),reused_predictions=len(cache_rows),unique_RGB=len(content))),flush=True)

if __name__=='__main__':main()
