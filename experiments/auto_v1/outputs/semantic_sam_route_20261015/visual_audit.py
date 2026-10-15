from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;W=H.parents[1]/'results/固定案例_三模型对比_20261006/room2_E1_prerepair_CUT_top20_20261015'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
s=read(H/'semantic.json');f=read(H.parent/'overmerge_id_ratio_top20_20261015/data.json');d=read(W/'viewer_data.json');cs={c['id']:c for c in f['candidates']}
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
def panel(rec):
 c=cs[rec['candidate_id']];job=f['jobs'][c['occ'][0][0]];v=d['crops'][job['key']];m=v['models'][2]
 rgb=np.asarray(Image.open(W/v['RGB']).convert('RGB'));rgba=np.asarray(Image.open(W/m['ids_file']).convert('RGBA'));ids=rgba[:,:,0].astype(np.int32)+rgba[:,:,1].astype(np.int32)*256;mask=ids==rec['label'];assert int(mask.sum())==rec['area'];assert hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()==rec['mask_sha256']
 yy,xx=np.nonzero(mask);x0,y0,x1,y1=xx.min(),yy.min(),xx.max()+1,yy.max()+1;crop=rgb[y0:y1,x0:x1].copy();crop[~mask[y0:y1,x0:x1]]=127
 side=max(crop.shape[:2]);square=np.full((side,side,3),127,np.uint8);left=(side-crop.shape[1])//2;top=(side-crop.shape[0])//2;square[top:top+crop.shape[0],left:left+crop.shape[1]]=crop
 key=hashlib.sha256(s['cache_namespace'].encode()+str(square.shape).encode()+square.tobytes()).hexdigest();assert key==rec['feature_cache_key'],job['key']
 im=Image.fromarray(rgb);pen=ImageDraw.Draw(im);pen.rectangle([max(0,x0-8),max(0,y0-8),min(im.width-1,x1+8),min(im.height-1,y1+8)],outline='#ffcc00',width=4)
 im.thumbnail((600,250));masked=Image.fromarray(square).resize((230,230));row=Image.new('RGB',(1000,300),'#ffffff');row.paste(im,(0,44));row.paste(masked,(610,44));pen=ImageDraw.Draw(row)
 pen.text((6,4),job['key']+' | SAM '+str(rec['label'])+' | '+str(rec['area'])+' px',font=font,fill='black')
 pen.text((6,24),rec['target_term']+' '+format(rec['target_score'],'.5f')+' vs '+rec['control_term']+' '+format(rec['control_score'],'.5f')+' | delta '+format(rec['margin'],'.5f'),font=font,fill='black')
 return row
for family in ['cabinet','window']:
 records=sorted([r for r in s['candidates'].values() if r['hit'] and r['target_family']==family],key=lambda r:r['margin'],reverse=True)
 if family=='window':records=records[:3]+records[-3:]
 for start in range(0,len(records),5):
  batch=records[start:start+5];out=Image.new('RGB',(1000,300*len(batch)),'white')
  for i,r in enumerate(batch):out.paste(panel(r),(0,i*300))
  out.save(H/(family+'_audit_'+str(start//5)+'.jpg'),quality=92)
 print(family,len(records))
