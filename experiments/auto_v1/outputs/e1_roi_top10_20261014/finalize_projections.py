"""Preserve the inspected state's projected pixels when states share a pixel."""
from pathlib import Path
import json,zipfile
import numpy as np
from PIL import Image
from prepare_inputs import ROOT,SOURCE,read,write,sha
O=ROOT/'review';data=read(O/'viewer_data.json');projections=read(O/'projection_annotations.json');inputs=read(ROOT/'input_manifest.json');jobs={j['key']:j for j in inputs['jobs']}
names={1:'U',2:'T',3:'C'};rows=[]
def box(x,y):return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
for case in data['cases']:
 for v in case['views']:
  a=np.asarray(Image.open(SOURCE/v['source_alarm']).convert('RGB'));k={'U':0,'T':1,'CONFLICT':2}[v['trigger_group']]
  codes=np.zeros(a.shape[:2],np.uint8);rgba=np.zeros((*a.shape[:2],4),np.uint8)
  for channel,color,code in sorted([(0,(140,145,155),1),(1,(255,190,30),2),(2,(245,60,70),3)],key=lambda row:row[0]==k):
   mask=a[:,:,channel]>0;codes[mask]=code;rgba[mask]=[*color,220]
  assert np.array_equal(codes==k+1,a[:,:,k]>0)
  ap=O/v['alarm'];Image.fromarray(rgba).save(ap,compress_level=2)
  full={name:int((codes==code).sum()) for code,name in names.items()}
  pv=projections['views'][case['uid']+'_v'+str(v['number'])]
  pv.update(alarm_sha256=sha(ap),projected_pixel_counts=full,overlap_priority=v['trigger_group'])
  for key in v['variants']:
   j=jobs[key];x0,y0,x1,y1=j['crop_xyxy'];c=codes[y0:y1,x0:x1];pos={name:np.nonzero(c==code) for code,name in names.items()}
   pr=projections['crops'][key];pr.update(counts={n:int(len(y)) for n,(y,x) in pos.items()},full_counts=full,state_boxes={n:box(x,y) for n,(y,x) in pos.items()},models=[],overlap_priority=v['trigger_group'])
   for path in [Path(j['baseline_file']),ROOT/'cropformer'/(key+'.png'),ROOT/'sam2_1'/(key+'.png')]:
    labels=np.asarray(Image.open(path));states={}
    for name,(yy,xx) in pos.items():
     vals=labels[yy,xx];items=[]
     for lid,count in zip(*np.unique(vals,return_counts=True)):
      keep=vals==lid;x=xx[keep];y=yy[keep];idx=int(((x-x.mean())**2+(y-y.mean())**2).argmin())
      assert labels[y[idx],x[idx]]==lid
      items.append(dict(id=int(lid),count=int(count),x=int(x[idx]),y=int(y[idx]),projection_box=box(x,y)))
     states[name]=sorted(items,key=lambda r:(-r['count'],r['id']));assert sum(x['count'] for x in items)==pr['counts'][name]
    pr['models'].append(states)
  rows.append(dict(uid=case['uid'],view=v['number'],group=v['trigger_group'],source_target_pixels=int((a[:,:,k]>0).sum()),displayed_target_pixels=full[names[k+1]]))
write(O/'projection_annotations.json',projections)
audit=read(O/'projection_annotation_audit.json');audit.update(target_pixels_preserved=True,pixel_collision_policy='inspected state wins only for display; source multichannel projections retained',views_checked=rows);write(O/'projection_annotation_audit.json',audit)
setup=read(O/'selection_setup.json');setup['projection_overlap_priority']='current inspected state';write(O/'selection_setup.json',setup)
files={p.relative_to(O).as_posix():sha(p) for p in O.rglob('*') if p.is_file() and p.name!='files_sha256.json'};write(O/'files_sha256.json',files)
archive=ROOT/'room2_CUT_top10_candidates.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
 for rel in sorted(files)+['files_sha256.json']:z.write(O/rel,rel)
receipt=read(ROOT/'delivery_receipt.json');receipt.update(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(files)+1,target_projection_pixels_preserved=True);write(ROOT/'delivery_receipt.json',receipt)
print('PROJECTIONS_VERIFIED',len(rows),'ARCHIVE_BYTES',archive.stat().st_size,flush=True)
