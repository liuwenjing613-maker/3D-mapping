"""Render the complete fixed case review with matched coordinates and display colors."""
from pathlib import Path
import hashlib, io, json, math, shutil, zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.optimize import linear_sum_assignment
from prepare_inputs import ROOT,SOURCE,NAME,sha,write

OUT=ROOT/'review'
ASSETS=OUT/'assets'
ASSETS.mkdir(parents=True,exist_ok=True)

def healthy(path):
 if not path.exists():return False
 try:
  with Image.open(path) as image:image.load()
  return True
 except (OSError,ValueError):return False

def palette(i):
 if i==0:return (108,115,126)
 import colorsys
 h=(i*.618033988749895+.09)%1
 return tuple(round(x*255) for x in colorsys.hsv_to_rgb(h,.63,.97))

def color_map(before,after,offset):
 old=np.unique(before[before>0]);new=np.unique(after[after>0]);mapping={}
 if len(old) and len(new):
  ai=np.unique(before);bi=np.unique(after)
  joint=np.bincount(np.searchsorted(ai,before.ravel())*len(bi)+np.searchsorted(bi,after.ravel()),minlength=len(ai)*len(bi)).reshape(len(ai),len(bi))
  a=np.searchsorted(ai,old);b=np.searchsorted(bi,new);inter=joint[np.ix_(a,b)]
  union=joint.sum(1)[a,None]+joint.sum(0)[None,b]-inter
  iou=inter/np.maximum(1,union);rr,cc=linear_sum_assignment(-iou)
  for r,c in zip(rr,cc):
   if iou[r,c]>=.15:mapping[int(new[c])]=int(old[r])
 colors={int(lid):palette(mapping.get(int(lid),offset+int(lid))) for lid in new}
 return colors,mapping

def assets_for(labels,colors,name):
 col=np.zeros((*labels.shape,4),np.uint8)
 positions=[]
 for lid in np.unique(labels[labels>0]):
  col[labels==lid]=[*colors[int(lid)],255]
  yy,xx=np.nonzero(labels==lid)
  # Select a real interior pixel near the centroid, rather than a centroid outside a concave mask.
  if len(xx)>=20:
   center=np.array([xx.mean(),yy.mean()]);dist=(xx-center[0])**2+(yy-center[1])**2;k=int(np.argmin(dist))
   positions.append(dict(id=int(lid),x=int(xx[k]),y=int(yy[k]),area=int(len(xx))))
 color_path=ASSETS/(name+'_color.png')
 if not healthy(color_path):Image.fromarray(col).save(color_path,compress_level=2)
 boundary=np.zeros(labels.shape,bool)
 boundary[:,1:]|=labels[:,1:]!=labels[:,:-1];boundary[1:]|=labels[1:]!=labels[:-1]
 boundary&=labels>0
 edges=np.zeros_like(col);edges[boundary]=[250,250,250,255]
 edge_path=ASSETS/(name+'_boundary.png')
 if not healthy(edge_path):Image.fromarray(edges).save(edge_path,compress_level=2)
 encoded=np.zeros((*labels.shape,3),np.uint8);encoded[:,:,0]=labels%256;encoded[:,:,1]=labels//256
 ids_path=ASSETS/(name+'_ids.png')
 if not healthy(ids_path):Image.fromarray(encoded).save(ids_path,compress_level=2)
 return dict(color='assets/'+name+'_color.png',boundary='assets/'+name+'_boundary.png',ids='assets/'+name+'_ids.png',
  count=len(colors),positions=positions,colors={str(k):list(v) for k,v in colors.items()})

def font(size):
 for path in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
  if Path(path).exists():return ImageFont.truetype(path,size)
 return ImageFont.load_default()

def overlay(rgb,labels,colors):
 out=rgb.copy();used=labels>0;col=np.zeros_like(rgb)
 for lid,c in colors.items():col[labels==lid]=c
 out[used]=(out[used]*.48+col[used]*.52).astype(np.uint8)
 edge=np.zeros(labels.shape,bool);edge[:,1:]|=labels[:,1:]!=labels[:,:-1];edge[1:]|=labels[1:]!=labels[:-1];edge&=used
 out[edge]=[250,250,250]
 return Image.fromarray(out)

def static_card(rgb,masks,maps,job,counts):
 w=620;h=math.ceil(rgb.shape[0]*w/rgb.shape[1]);header=95;footer=45
 canvas=Image.new('RGB',(w*4+60,h+header+footer),(247,249,252));d=ImageDraw.Draw(canvas)
 d.text((14,8),f'{job["scene"]} / {job["ROI"]} / view {job["view"]} / frame {job["frame"]}  |  zoom {job["effective_zoom"]:.2f}x',font=font(23),fill=(20,32,50))
 titles=['RGB','CropFormer original',f'CropFormer local ({counts[1]})',f'SAM2.1 local AMG ({counts[2]})']
 images=[Image.fromarray(rgb)]+[overlay(rgb,m,c) for m,c in zip(masks,maps)]
 for i,(title,img) in enumerate(zip(titles,images)):
  x=12+i*(w+12);d.text((x,49),title,font=font(22),fill=(35,48,65));canvas.paste(img.resize((w,h),Image.Resampling.LANCZOS),(x,header))
 d.text((14,h+header+10),'Same native RGB crop and coordinates. Display colors matched by overlap only. No manual prompts or map writes.',font=font(19),fill=(65,76,89))
 return canvas

