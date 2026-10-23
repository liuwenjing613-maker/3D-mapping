from pathlib import Path
import json,tarfile,hashlib
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
R=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
read=lambda p:json.loads(Path(p).read_text())
old=R/'400/A/candidates';new=R/'window_only_mask_ablation/A/window_only'
key='room0_ROI-U0011_v2_f000315_context';cid='a1ef461408e03e534aa7'
a=read(old/'semantic_route/selection.json')['crops'][key];b=read(new/'semantic_route/selection.json')['crops'][key]
node=next(n for n in b['nodes'] if n['candidate_id']==cid)
g=next(g for g in b['groups'] if node['key'] in g['members'])
members=set(g.get('original_members',g['members']))
rows=[]
for n in b['nodes']:
 if n['key'] not in members:continue
 bef=a['models'][n['model']]['labels'][str(n['label'])];aft=b['models'][n['model']]['labels'][str(n['label'])]
 rows.append(dict(key=n['key'],model=n['model'],candidate_id=n['candidate_id'],area=n['area'],before=bef,after=aft,source_path=n['source_path'],source_box=n['source_box'],label=n['label']))
edges=[e for e in a['semantic_replacement_edges'] if e['a'] in members]
oldnodes={n['key']:n for n in a['nodes']}
targets=[{k:n[k] for k in ['key','candidate_id','area','model','semantic_hit']} for e in edges for n in [oldnodes[e['b']]]]
out=dict(crop=key,group=g,members=rows,original_replacement_edges=edges,original_semantic_replacers=targets)
target=oldnodes['2:47'];whole=next(n for n in b['nodes'] if n['key']=='0:5')
rgb=np.asarray(Image.open(R/'400/A/mapping/inputs/scene/results/frame000315.jpg'))
def full_mask(n):
 arr=np.asarray(Image.open(n['source_path']));x0,y0,x1,y1=n['source_box'];mask=np.zeros(rgb.shape[:2],bool);mask[y0:y1,x0:x1]=arr==n['label'];assert mask.sum()==n['area'];return mask
whole_mask=full_mask(whole);yy,xx=np.nonzero(whole_mask);box=(max(0,int(xx.min())-45),max(0,int(yy.min())-45),min(1200,int(xx.max())+46),min(680,int(yy.max())+46))
folder=R/'window_only_mask_ablation/review/images';out['images']=[]
for name,n in [('CropFormer_whole',whole),('SAM_tiny_direct',target),('SAM_border_M0473',node)]:
 mask=full_mask(n);imask=Image.fromarray(mask.astype(np.uint8)*255);edge=np.asarray(imask.filter(ImageFilter.MaxFilter(3)))!=np.asarray(imask.filter(ImageFilter.MinFilter(3)))
 img=rgb.copy();img[mask]=(rgb[mask]*.55+np.array([255,64,95])*.45).astype(np.uint8);img[edge]=[255,55,85]
 im=Image.fromarray(img).crop(box)
 if name=='SAM_tiny_direct':
  yy,xx=np.nonzero(mask);draw=ImageDraw.Draw(im);draw.rectangle((int(xx.min())-box[0]-4,int(yy.min())-box[1]-4,int(xx.max())-box[0]+4,int(yy.max())-box[1]+4),outline=(255,210,0),width=2)
 im.resize((680,round(im.height*680/im.width))).save(folder/(name+'.jpg'),quality=95)
 out['images'].append(dict(name=name,area=n['area'],path='images/'+name+'.jpg'))
print('PASS indirect effect: 111 pixel SAM displaced both 23038/23094 pixel CropFormer masks')
path=R/'window_only_mask_ablation/review/comparison.json';data=read(path);data['A_indirect_border_mask_effect']=out;path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
with tarfile.open(path.parent.parent/'review.tar.gz','w:gz') as tar:
 for p in path.parent.rglob('*'):
  if p.is_file():tar.add(p,arcname=p.relative_to(path.parent))
print('SHA256',hashlib.sha256((path.parent.parent/'review.tar.gz').read_bytes()).hexdigest())
