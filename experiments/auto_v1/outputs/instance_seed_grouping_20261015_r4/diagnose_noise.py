from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4')
P=Path('/data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r3')
R.mkdir(parents=True,exist_ok=True)
d=json.loads((P/'review/data.json').read_text())
rows=[]
for s in d['seeds']:
    a=np.asarray(Image.open(P/s['mask']))>0
    y,x=np.nonzero(a)
    sub=np.pad(a[y.min():y.max()+1,x.min():x.max()+1],1)
    dist=ndi.distance_transform_edt(sub)[sub]
    labels,n=ndi.label(sub,structure=np.ones((3,3)))
    sizes=np.bincount(labels.ravel())[1:]
    w,h=int(x.max()-x.min()+1),int(y.max()-y.min()+1)
    vals=np.linalg.eigvalsh(np.cov(np.column_stack([x,y]),rowvar=False)) if len(x)>2 else [0,0]
    row={k:s[k] for k in ['id','frame','area','support_count','group','bbox','mask','representative','color','rgb']}
    row.update(group_id=d['groups'][s['group']]['id'],was_primary=d['groups'][s['group']]['primary']==s['index'],
       core2=float(np.mean(dist>2)),core3=float(np.mean(dist>3)),core4=float(np.mean(dist>4)),
       max_radius=float(dist.max()),median_radius=float(np.median(dist)),p95_radius=float(np.quantile(dist,.95)),
       components=int(n),largest_component=int(sizes.max()),bbox_fill=float(len(x)/(w*h)),
       bbox_long=max(w,h),bbox_short=min(w,h),pca_ratio=float(math.sqrt(max(vals)/max(min(vals),.01))))
    rows.append(row)
(R/'shape_diagnostics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
selected=sorted([s for s in rows if s['id'] in ['M0267','M0491','M0608','M0678','M0705','M0709','M0743','M0753','M0780']],key=lambda s:s['id'])
print('TOTAL',len(rows),'DIAGNOSTIC_CANDIDATES',len(selected))
print(json.dumps([{k:s[k] for k in ['id','area','support_count','core2','core3','max_radius','p95_radius','bbox_long','bbox_short','pca_ratio','components','was_primary']} for s in selected],ensure_ascii=False))
print('EXAMPLE',json.dumps(next(s for s in rows if s['id']=='M0734')))
for page in range(math.ceil(len(selected)/24)):
    subset=selected[page*24:page*24+24]
    sheet=Image.new('RGB',(1600,math.ceil(len(subset)/4)*290),'#edf2f7');draw=ImageDraw.Draw(sheet)
    for i,s in enumerate(subset):
        rgb=Image.open(P/'review'/s['rgb']).convert('RGB')
        mask=np.asarray(Image.open(P/s['mask']))>0
        x0,y0,x1,y1=s['bbox'];pad=max(30,int(max(x1-x0,y1-y0)*.15))
        box=(max(0,x0-pad),max(0,y0-pad),min(rgb.width,x1+pad),min(rgb.height,y1+pad))
        arr=np.array(rgb);arr[mask]=(.35*arr[mask]+.65*np.array(s['color'])).astype(np.uint8)
        im=Image.fromarray(arr).crop(box);im.thumbnail((270,220))
        binary=Image.fromarray(np.uint8(mask)*255).convert('RGB').crop(box);binary.thumbnail((110,220))
        x=(i%4)*400;y=(i//4)*290
        sheet.paste(im,(x+(270-im.width)//2,y+55+(220-im.height)//2))
        sheet.paste(binary,(x+280+(110-binary.width)//2,y+55+(220-binary.height)//2))
        draw.text((x+8,y+5),f"{s['id']} {s['group_id']} {'PRIMARY' if s['was_primary'] else ''}  px={s['area']} TSDF={s['support_count']}",fill='black')
        draw.text((x+8,y+21),f"core2={s['core2']:.3f} core3={s['core3']:.3f} rmax={s['max_radius']:.1f}",fill='black')
        draw.text((x+8,y+37),f"PCA={s['pca_ratio']:.1f} box={s['bbox_long']}x{s['bbox_short']} comps={s['components']}",fill='black')
    sheet.save(R/f'shape_montage_boundary_{page+1:02d}.jpg',quality=92)
