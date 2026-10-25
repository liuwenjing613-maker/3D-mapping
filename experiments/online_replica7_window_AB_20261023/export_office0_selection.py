"""Read-only export of existing office0 selection evidence. No inference or selection rerun."""
from pathlib import Path
from collections import Counter,defaultdict
import json,gzip,hashlib,shutil,tarfile,time
import numpy as np
from PIL import Image
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023/office0')
E=Path('/data/chenkejun/CVPR/results/office0_mask_selection_20261025')
W=E/'page';W.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
files={};seen={};encoded={};evidence={};checks=[]
def asset(p):
 p=Path(p).resolve();key=str(p)
 if key in seen:return seen[key]
 h=sha(p);rel='assets/'+h+p.suffix.lower();dst=W/rel
 if not dst.exists():dst.parent.mkdir(exist_ok=True);shutil.copyfile(p,dst)
 files[rel]=dict(sha256=h,bytes=dst.stat().st_size);seen[key]=rel;return rel
def label_asset(p):
 raw=asset(p);h=files[raw]['sha256']
 if h in encoded:return raw,encoded[h]
 a=np.asarray(Image.open(p));assert a.ndim==2 and a.min()>=0 and a.max()<65536
 rgb=np.zeros(a.shape+(3,),np.uint8);rgb[:,:,0]=a.astype(np.uint16)%256;rgb[:,:,1]=a.astype(np.uint16)//256
 dst=W/'assets'/('labels_'+h+'.png');Image.fromarray(rgb).save(dst)
 b=np.asarray(Image.open(dst));assert np.array_equal(a,b[:,:,0].astype(np.uint16)+256*b[:,:,1].astype(np.uint16))
 rel=dst.relative_to(W).as_posix();files[rel]=dict(sha256=sha(dst),bytes=dst.stat().st_size);encoded[h]=rel;return raw,rel
def source(root,rel):
 p=root/rel;d=read(p);evidence[str(p)]=sha(p);return d
index=dict(scene='office0',runs=[],source_rule='window-only special SAM source; cabinet ordinary; PR exemptions unchanged',algorithm_rerun=False)
start=time.time()
for n in [50,100,200,300,400]:
 for branch in ['A','B']:
  R=O/str(n)/branch;P=R/'candidates';key=f'{n}_{branch}'
  complete=source(P,'selection_complete.json');assert complete['status']=='PASS'
  cat=source(P,'catalog/office0/candidate_manifest.json')
  over=source(P,'overmerge/web/data.json');versions=source(P,'versions/selection.json');route=source(P,'semantic_route/selection.json')
  grouping=source(P,'grouping_r2/grouping.json');boundary=source(P,'all_edge_20261021/boundary_audit.json')
  review=source(P,'all_edge_20261021/review/data.json');selection=source(P,'all_edge_20261021/v2/selection.json')
  active=source(P,'all_edge_20261021/v1/grouping.json');seedmanifest=source(P,'all_edge_20261021/v2/seed_manifest.json')
  gate=source(R,'tracking/gate_complete.json');track=source(R,'tracking/seed_manifest.json')
  global_ex=source(P,'grouping_r2/semantic_exclusions.json');window=source(P,'window_source_policy.json')
  occurrence=defaultdict(list);route_counts=Counter()
  for crop,c in route['crops'].items():
   vc=versions['crops'][crop]
   for node in c['nodes']:
    lab=str(node['label']);model=node['model'];decision=c['models'][model]['labels'][lab]
    occurrence[node['candidate_id']].append(dict(crop=crop,roi=c['ROI'],uid=c['uid'],frame=c['frame'],model=model,label=node['label'],support=node['support'],eligible=node['eligible'],semantic=node.get('semantic'),semantic_hit=node.get('semantic_hit'),version=vc['models'][model]['labels'].get(lab),route=decision,group=next((g for g in c['groups'] if g['id']==decision.get('group')),None)))
    route_counts[decision['status']]+=1
  for c in over['candidates']:c['occurrences']=occurrence[c['id']]
  for s in over['sources']:
   s['original'],s['labels']=label_asset(s['path']);s.pop('ids',None)
  frames={}
  for f in over['frames']:
   frames[f]=asset(R/'mapping/inputs/scene/results'/f'frame{int(f):06d}.jpg')
  for entry in cat['entries']:
   for view in entry['views']:
    for k in ['RGB_file','alarm_file','thumbnail']:
     if view.get(k):view[k]=asset(P/'catalog/office0'/view[k])
  bymask={s['id']:s for s in grouping['seeds']};decisions={r['group_id']:r for r in selection['records']}
  activegroups={g['id']:g for g in active['groups']}
  edgeby={s['id']:s for s in boundary['masks']};noiseby={s['id']:s for s in boundary['noise_excluded']}
  for row in review['rows']:
   original=bymask[row['id']];gid=row['group']
   for k in ['full','zoom','rgb','mask']:row[k]=asset(P/'all_edge_20261021/review'/row[k])
   assert files[row['mask']]['sha256']==row['mask_sha256']
   row['aliases']=original['aliases'];row['digest']=original['digest'];row['boundary']=edgeby[row['id']]
   row['noise']=noiseby.get(row['id'],{}).get('noise_decision')
   row['selection']=decisions.get(gid) if row['primary'] else None
   if gid:
    g=activegroups[gid];row['group_primary_mask']=active['seeds'][g['primary']]['id'];row['group_member_masks']=[active['seeds'][i]['id'] for i in g['members']]
   row['gate']=next((g for g in gate['tracks'].values() if g['mask_id']==row['id']),None)
  selected_ids={c['id'] for c in over['candidates'] if any(o['route']['status']=='selected' for o in c['occurrences'])}
  seeded={g['primary']['id'] for g in seedmanifest['groups']};kept={r['id'] for r in review['rows'] if r['status']=='kept'}
  assert seeded==kept=={g['mask_id'] for g in gate['tracks'].values()}
  assert len(kept)==complete['retained']
  selected=set(selected_ids)
  for s in grouping['seeds']:assert all(a['candidate_id'] in selected for a in s['aliases'])
  counts=dict(roi_entries=len(cat['entries']),roi_unique=len(cat['unique_components']),crops=len(over['jobs']),raw_unique=len(over['candidates']),raw_occurrences=sum(len(c['occurrences']) for c in over['candidates']),overmerge=dict(Counter(c['decision'] for c in over['candidates'])),route_occurrences=dict(route_counts),route_selected_unique=len(selected_ids),grouping_input_masks=len(grouping['seeds']),boundary=boundary['stats'],final=len(kept),post_statuses=dict(Counter(r['status'] for r in review['rows'])))
  data=dict(key=key,checkpoint=n,branch=branch,counts=counts,catalog=cat,candidates=over['candidates'],sources=over['sources'],frames=frames,jobs=over['jobs'],overmerge_rule=over['rule'],route_rule=route['rule'],rows=review['rows'],stages=selection['stages'],selection=selection,global_source_exclusions=global_ex,window_policy=window,gate=gate,tracking=track,grouping_audit=source(P,'grouping_r2/audit.json'),boundary_rule=boundary['rule'])
  dest=W/'runs'/f'{key}.json.gz';dest.parent.mkdir(exist_ok=True)
  with gzip.open(dest,'wt',encoding='utf-8',compresslevel=6) as out:json.dump(data,out,ensure_ascii=False,separators=(',',':'),allow_nan=False)
  rel=dest.relative_to(W).as_posix();files[rel]=dict(sha256=sha(dest),bytes=dest.stat().st_size)
  index['runs'].append(dict(key=key,checkpoint=n,branch=branch,path=rel,counts=counts,raw_frame_range=cat['candidate_raw_frame_range']))
  checks.append(dict(key=key,mask_sha256_verified=len(review['rows']),final_seed_equals_tracking=True,route_aliases_verified=True))
  print('EXPORT',key,json.dumps(counts),flush=True)
for p,h in evidence.items():assert sha(p)==h,('source_changed',p)
index['source_hashes']=evidence;index['validation']=dict(status='PASS',checks=checks,encoded_label_sources=len(encoded),algorithm_rerun=False,source_records_unchanged=True)
dump(W/'index.json',index);files['index.json']=dict(sha256=sha(W/'index.json'),bytes=(W/'index.json').stat().st_size)
dump(W/'file_manifest.json',dict(status='PASS',files=files))
with tarfile.open(E/'page_data.tar.gz','w:gz',compresslevel=1) as tar:
 for p in W.rglob('*'):
  if p.is_file():tar.add(p,arcname=p.relative_to(W))
receipt=dict(status='PASS',runs=10,files=len(files),archive_bytes=(E/'page_data.tar.gz').stat().st_size,archive_sha256=sha(E/'page_data.tar.gz'),seconds=time.time()-start)
dump(E/'export_complete.json',receipt);print('COMPLETE',json.dumps(receipt),flush=True)
