"""Render exact cached masks; never alter/resegment their pixel support."""
from pathlib import Path
import base64, hashlib, io, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/固定案例_三模型对比_20261006'
OUT = Path(__file__).resolve().parent
KEY = 'office2_ROI-C0008_v1_f000520_balanced'
data = json.loads((SOURCE / 'viewer_data.json').read_text(encoding='utf-8'))
crop = data['crops'][KEY]
rgb = np.array(Image.open(SOURCE / crop['RGB']).convert('RGB'))
font_path = 'C:/Windows/Fonts/arial.ttf'
def font(n): return ImageFont.truetype(font_path, n)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = {'scene':'office2', 'frame':520, 'ROI':'ROI-C0008', 'crop_key':KEY,
            'crop_xyxy':crop['box'], 'native_wh':crop['wh'],
            'RGB_source':str(SOURCE / crop['RGB']), 'RGB_sha256':sha(SOURCE / crop['RGB']),
            'source_server_root':'/data/chenkejun/CVPR/revisable_instance_map/roi_threeway_fixed_20261006/selection-20261006-f49eb927',
            'SAM_representation':'existing non-overlapping partition of AMG proposals; smallest area wins; not raw overlapping proposal masks',
            'models':[]}
names = [('cropformer','CropFormer (full-frame prediction, same ROI)'),
         ('cropformer_local','CropFormer (local crop)'),('sam2','SAM2.1 (local AMG partition)')]
scale=2; w,h=crop['wh']; panel_w=w*scale; panel_h=h*scale
for (slug,title),model in zip(names,crop['models']):
    source_ids=model['ids']
    encoded = np.array(Image.open(io.BytesIO(base64.b64decode(source_ids.split(',',1)[1])))) if source_ids.startswith('data:') else np.array(Image.open(SOURCE/source_ids))
    labels=encoded[:,:,0].astype(np.uint16)+encoded[:,:,1].astype(np.uint16)*256
    ids=[int(n) for n in np.unique(labels) if n>0]
    assert len(ids)==model['count'] and labels.shape==rgb.shape[:2]
    maskfile=OUT/(slug+'_mask_ids.png'); Image.fromarray(labels).save(maskfile)
    overlay=rgb.astype(float).copy()
    for mid in ids:
        mask=labels==mid; overlay[mask]=0.64*rgb[mask]+0.36*np.array(model['colors'][str(mid)])
    edges=np.zeros(labels.shape,bool)
    edges[:,1:]|=labels[:,1:]!=labels[:,:-1]; edges[1:]|=labels[1:]!=labels[:-1]
    edges&=labels>0; overlay[edges]=[255,255,255]
    canvas=Image.new('RGB',(panel_w*2+48,panel_h+160),'#f6f7fa')
    draw=ImageDraw.Draw(canvas)
    draw.text((16,10),title,font=font(29),fill='#172235')
    draw.text((16,48),'office2 / frame 520 / ROI-C0008 / native crop 400 x 354',font=font(20),fill='#4f5d70')
    draw.text((16,79),'RGB reference',font=font(23),fill='#172235')
    ox=panel_w+32; oy=112
    draw.text((ox,79),f'Numbered masks ({len(ids)})',font=font(23),fill='#172235')
    canvas.paste(Image.fromarray(rgb).resize((panel_w,panel_h),Image.Resampling.LANCZOS),(16,oy))
    canvas.paste(Image.fromarray(overlay.astype(np.uint8)).resize((panel_w,panel_h),Image.Resampling.NEAREST),(ox,oy))
    placed=[]; placements=[]
    # Place IDs on thick interior pixels; tiny masks get a nearby callout.
    for mid in sorted(ids,key=lambda n:-int((labels==n).sum())):
        mask=labels==mid; area=int(mask.sum()); distance=distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
        yy,xx=np.where(mask); order=np.argsort(distance[yy,xx])[::-1]
        chosen=None; tw=draw.textbbox((0,0),str(mid),font=font(26))[2]+14; th=34
        candidates=[(ox+int(xx[t])*scale,oy+int(yy[t])*scale) for t in order[::max(1,len(order)//500)]]
        if area<100:
            ax=ox+int(xx[order[0]])*scale; ay=oy+int(yy[order[0]])*scale
            candidates=[(ax+dx,ay+dy) for dx,dy in [(0,28),(30,28),(-30,28),(35,0),(-35,0)]]+candidates
        for cx,cy in candidates:
            cx=max(ox+tw//2+2,min(ox+panel_w-tw//2-2,cx)); cy=max(oy+th//2+2,min(oy+panel_h-th//2-2,cy))
            rect=(cx-tw/2,cy-th/2,cx+tw/2,cy+th/2)
            if all(rect[2]+5<r[0] or rect[0]-5>r[2] or rect[3]+5<r[1] or rect[1]-5>r[3] for r in placed):chosen=(cx,cy,rect); break
        if chosen is None:raise RuntimeError(f'Cannot place label {slug} {mid}')
        cx,cy,rect=chosen
        if area<100:
            draw.line((ax,ay,cx,cy),fill='black',width=5);draw.line((ax,ay,cx,cy),fill='white',width=2)
            draw.ellipse((ax-3,ay-3,ax+3,ay+3),fill='white',outline='black')
        draw.rounded_rectangle(rect,radius=5,fill='#132035',outline='white',width=2)
        draw.text((cx,cy-1),str(mid),anchor='mm',font=font(26),fill='white')
        placed.append(rect);placements.append({'id':mid,'area_px':area,'label_center':[cx,cy]})
    draw.text((16,panel_h+124),'Mask IDs: '+', '.join(map(str,ids))+'   |   0 = unassigned',font=font(20),fill='#4f5d70')
    output=OUT/(slug+'_vlm_input.png');canvas.save(output)
    manifest['models'].append({'method':slug,'title':title,'mask_ids':ids,'mask_count':len(ids),'input_image':str(output),
                               'input_image_sha256':sha(output),'mask_file':str(maskfile),'mask_sha256':sha(maskfile),'placements':placements})
(OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({m['method']:m['mask_ids'] for m in manifest['models']}))
