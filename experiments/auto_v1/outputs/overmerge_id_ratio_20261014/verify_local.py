from pathlib import Path
import collections,gzip,hashlib,json
import numpy as np
from PIL import Image
H=Path(__file__).resolve().parent
OUT=H.parents[1]/'results/固定案例_三模型对比_20261006/room2_E1_overmerge_audit_20261014'
d=json.loads((OUT/'data.json').read_text(encoding='utf-8'));cs=d['candidates'];nc=len(d['native_ids'])+1
assert len(cs)==d['stats']['unique']==12357
assert sum(len(c['occ']) for c in cs)==18303
assert sum(c['decision']=='reject' for c in cs)==194
assert len({c['id'] for c in cs})==len(cs)
groups=collections.defaultdict(list)
for sid,s in enumerate(d['sources']):groups[s['frame']].append((sid,s))
mask_checks=point_checks=0
for f,ss in sorted(groups.items()):
 p=np.frombuffer(gzip.decompress((OUT/d['frames'][str(f)]['points']).read_bytes()),dtype='<u2').reshape(-1,3)
 assert len(p)==d['frames'][str(f)]['visible_vertices']
 for sid,s in ss:
  im=np.asarray(Image.open(OUT/s['ids']),np.uint16);labels=im[:,:,0]+im[:,:,1]*256
  x0,y0,x1,y1=s['box'];assert labels.shape==(y1-y0,x1-x0)
  inside=(p[:,0]>=x0)&(p[:,0]<x1)&(p[:,1]>=y0)&(p[:,1]<y1);pp=p[inside];lid=labels[pp[:,1]-y0,pp[:,0]-x0].astype(np.int64)
  hist=np.bincount(lid*nc+pp[:,2],minlength=(int(labels.max())+1)*nc).reshape(-1,nc)
  area=np.bincount(labels.ravel())
  for lidstr,cid in s['mapping'].items():
   lid=int(lidstr);c=cs[cid];assert c['frame']==f and c['model']==s['model']
   b=c['box'];mask=labels[b[1]-y0:b[3]-y0,b[0]-x0:b[2]-x0]==lid
   digest=hashlib.sha256(json.dumps([f,s['model'],b]).encode()+np.packbits(mask).tobytes()).hexdigest()[:20]
   assert c['id']==digest,(sid,lid,'binary shape mismatch')
   assert mask.sum()==area[lid]==c['pixels']
   assert int(hist[lid].sum())==c['N'] and int(hist[lid,0])==c['unconfirmed']
   pairs=sorted([[d['native_ids'][k-1],int(n)] for k,n in enumerate(hist[lid]) if k and n],key=lambda a:(-a[1],a[0]));assert pairs==c['ids']
   n2=pairs[1][1] if len(pairs)>1 else 0
   expected='reject' if n2>=10 and n2*5>=c['N'] else 'keep' if c['N'] else 'no_support'
   assert expected==c['decision'];assert sum(c['states'])==c['N']
   if s['model']==0:assert s['box']==[0,0,1200,680]
   mask_checks+=1;point_checks+=int(hist[lid].sum())
for c in cs:
 for ji,lid in c['occ']:
  j=d['jobs'][ji];s=d['sources'][j['sources'][c['model']]]
  assert s['mapping'][str(lid)]==c['index'] and c['index'] in j['members'][c['model']]
report=dict(status='PASS',candidate_count=len(cs),unique_shape_checks=mask_checks,point_membership_checks=point_checks,
 all_viewer_label_shapes_match_server_fingerprints=True,all_counts_recomputed_from_downloaded_projected_vertices=True,
 all_occurrence_links_verified=True,original_CF_uses_full_mask=True,rule='n2 >= 10 AND 5*n2 >= N',
 image_assets_verified=True,frame_count=len(groups),stats=d['stats'])
(H/'local_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
