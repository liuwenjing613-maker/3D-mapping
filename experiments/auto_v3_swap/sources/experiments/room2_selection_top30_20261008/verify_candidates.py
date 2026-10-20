"""Check every visible crop, mask ID, projected annotation and immutable old case."""
from pathlib import Path
import base64,hashlib,io,json,zipfile
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
BASE=WORK/'results/固定案例_三模型对比_20261006';OUT=BASE/'room2_selection_top30_20261008'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 with p.open('w',encoding='utf-8',newline='\n') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
 d=read(OUT/'viewer_data.json');pr=read(OUT/'projection_annotations.json');checks=read(OUT/'files_sha256.json')
 for rel,digest in checks.items():
  if rel=='selection_setup.json':
   with zipfile.ZipFile(HERE/'room2_top30_candidates.zip') as z:raw=z.read(rel)
   assert hashlib.sha256(raw).hexdigest()==digest
   original=json.loads(raw);current=read(OUT/rel)
   assert all(current[k]==v for k,v in original.items())
  else:assert sha(OUT/rel)==digest,rel
 assert len(d['cases'])==30 and len({c['uid'] for c in d['cases']})==30
 assert [c['ROI'] for c in d['cases']]==[f'ROI-C{i:04d}' for i in range(1,31)]
 keys=set();full_counts={};npositions=0;nrows=0;nlabels=0
 for c in d['cases']:
  assert c['scene']=='room2' and c['group']=='CONFLICT' and len(c['views'])==3
  for v in c['views']:
   a=np.asarray(Image.open(OUT/v['alarm']).convert('RGBA'));assert a.shape==(680,1200,4)
   codes=np.zeros(a.shape[:2],np.uint8)
   allowed=(a[:,:,3]==0)&(~a[:,:,:3].any(2))
   for code,color in [(1,(140,145,155)),(2,(255,190,30)),(3,(245,60,70))]:
    match=np.all(a[:,:,:3]==color,2)&(a[:,:,3]==220);codes[match]=code;allowed|=match
   assert allowed.all()
   assert (codes==3).any();assert v['recommended'] in v['variants']
   assert Image.open(OUT/v['full_rgb']).size==(1200,680)
   for k in v['variants']:
    keys.add(k);j=d['crops'][k];p=pr['crops'][k];x0,y0,x1,y1=j['box'];w,h=j['wh'];assert [w,h]==[x1-x0,y1-y0]
    assert Image.open(OUT/j['RGB']).size==(w,h) and len(j['models'])==3
    local=codes[y0:y1,x0:x1]
    if j['complete']:assert int((local==3).sum())==int((codes==3).sum())
    for mi,m in enumerate(j['models']):
     asset=OUT/m['ids_file'];assert base64.b64decode(m['ids'].split(',',1)[1])==asset.read_bytes()
     a2=np.asarray(Image.open(asset).convert('RGB'));labels=a2[:,:,0].astype(np.uint16)+256*a2[:,:,1].astype(np.uint16)
     assert labels.shape==(h,w) and not a2[:,:,2].any();values,areas=np.unique(labels,return_counts=True);area_by_id=dict(zip(values,areas));ids=set(map(int,values[values>0]))
     assert ids==set(map(int,m['colors'])) and len(ids)==m['count']
     assert Image.open(OUT/m['color']).size==Image.open(OUT/m['boundary']).size==(w,h)
     for a3 in m['positions']:
      assert labels[a3['y'],a3['x']]==a3['id'] and int(area_by_id[a3['id']])==a3['area'];npositions+=1
     for code,name in [(1,'U'),(2,'T'),(3,'C')]:
      rows=p['models'][mi][name];yy,xx=np.nonzero(local==code);vv=labels[yy,xx]
      expected=dict(zip(*np.unique(vv,return_counts=True)))
      assert {r['id']:r['count'] for r in rows}==expected
      for r in rows:assert labels[r['y'],r['x']]==r['id'] and local[r['y'],r['x']]==code;nrows+=1
     nlabels+=1
 assert keys==set(d['crops'])==set(pr['crops'])
 old=read(BASE/'viewer_data.json');old_cases=[c for c in old['cases'] if c['scene']=='room2'];new={c['uid']:c for c in d['cases']};old_crop_count=0
 for c in old_cases:
  nc=new[c['uid']]
  for ov,nv in zip(c['views'],nc['views']):
   assert ov['frame']==nv['frame'] and ov['recommended']==nv['recommended'] and ov['variants']==nv['variants']
   for k in ov['variants']:
    assert old['crops'][k]['box']==d['crops'][k]['box']
    for om,nm in zip(old['crops'][k]['models'],d['crops'][k]['models']):
     oa=np.asarray(Image.open(io.BytesIO(base64.b64decode(om['ids'].split(',',1)[1]))));na=np.asarray(Image.open(io.BytesIO(base64.b64decode(nm['ids'].split(',',1)[1]))))
     assert np.array_equal(oa,na),k
    old_crop_count+=1
 assert old_crop_count==75
 audit=dict(status='PASS',cases=30,views=90,crops=len(keys),verified_label_maps=nlabels,mask_positions_verified=npositions,
  projection_histograms_and_anchors_verified=nrows,server_asset_hashes_verified=len(checks),previous_five_cases_exactly_unchanged=True,
  human_annotations_created=0,tracking_started=False,map_modified=False)
 save(HERE/'data_verification.json',audit);save(OUT/'data_verification.json',audit)
 # The locally added UI and setup have their own final checksum manifest.
 final={str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_files_sha256.json'}
 save(OUT/'final_files_sha256.json',final)
 print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
