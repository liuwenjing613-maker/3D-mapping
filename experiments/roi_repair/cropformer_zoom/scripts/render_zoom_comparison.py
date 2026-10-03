"""Native RGB/mask comparisons plus compact lossless-label inline data."""
from pathlib import Path
import io,json,base64,csv,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.optimize import linear_sum_assignment
P=Path(__file__).parent;A=P/'inference_arrays'
D=Path('D:/Users/刘雯静/Downloads/CVPR/results/P1-A1_room0_局部CropFormer重分割_20261003');D.mkdir(exist_ok=True)
FONT='C:/Windows/Fonts/msyh.ttc';F={n:ImageFont.truetype(FONT,n) for n in [18,20,22,24,28,30]}
manifest=json.loads((A/'input_manifest.json').read_text(encoding='utf-8'));jobs={r['key']:r for r in manifest['jobs']}
old_annotations=json.loads(Path('outputs/p1a1_room0_roi_review_20261002/case_annotations.json').read_text(encoding='utf-8'))
provenance=json.loads((A/'model_provenance.json').read_text(encoding='utf-8'))
records={r['key']:r for r in map(json.loads,(A/'inference_records.jsonl').read_text(encoding='utf-8').splitlines())}

def palette(i):return [int((i*37+53)%195+60),int((i*83+11)%195+60),int((i*131+97)%195+60)] if i else [89,89,89]
def match_colors(before,after):
 old=np.unique(before[before>0]);new=np.unique(after[after>0]);n=max(1,int(after.max())+1)
 mapping={};table=[]
 if len(old) and len(new):
  oa=np.searchsorted(old,before[before>0]);counta=np.bincount(oa,minlength=len(old));countb=np.array([(after==j).sum() for j in new])
  joint=np.zeros((len(old),len(new)),np.int64)
  for oi,oid in enumerate(old):
   v,c=np.unique(after[before==oid],return_counts=True)
   good=v>0;joint[oi,np.searchsorted(new,v[good])]=c[good]
  iou=joint/np.maximum(1,counta[:,None]+countb[None,:]-joint);rr,cc=linear_sum_assignment(-iou)
  for x,y in zip(rr,cc):
   if iou[x,y]>=.1:mapping[int(new[y])]=int(old[x]);table.append({'local_id':int(new[y]),'full_frame_id':int(old[x]),'display_match_IoU':float(iou[x,y])})
 colors=[palette(mapping.get(i,1000+i)) if i else palette(0) for i in range(n)]
 return colors,table

def overlay(rgb,mask,colors,pure=False):
 col=np.asarray(colors,np.uint8)[mask]
 im=col if pure else (rgb*.3+col*.7).astype(np.uint8)
 boundary=np.zeros(mask.shape,bool);boundary[:,1:]|=mask[:,1:]!=mask[:,:-1];boundary[1:]|=mask[1:]!=mask[:-1];im[boundary]=[245,245,245]
 return Image.fromarray(im)
def fit(im,size):
 out=Image.new('RGB',size,(242,244,246));scale=min(size[0]/im.width,size[1]/im.height);p=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS);out.paste(p,((size[0]-p.width)//2,(size[1]-p.height)//2));return out
def wrap(draw,text,width,font):
 lines=[];line=''
 for c in text:
  if line and draw.textlength(line+c,font=font)>width:lines.append(line);line=c
  else:line+=c
 if line:lines.append(line)
 return lines
def uri(im,fmt='WEBP',quality=55):
 b=io.BytesIO()
 if fmt=='WEBP':im.save(b,format=fmt,quality=quality,method=4,lossless=im.mode=='L')
 else:im.save(b,format=fmt,optimize=True)
 return 'data:image/'+fmt.lower()+';base64,'+base64.b64encode(b.getvalue()).decode()

raw=[];rows=[];primary_tiles={};rootcase={}
for case in manifest['cases']:
 rank=case['rank'];roi=case['roi_id'];vn=case['view'];place=old_annotations.get(str(rank),{}).get('place','百叶窗表面')
 rootcase.setdefault(rank,{'rank':rank,'ROI':roi,'place':place,'representative':case['representative'],'views':[]})
 view={'view':vn,'frame':case['frame'],'label':case.get('view_label',f'视角 {vn}'),'crops':{}}
 for strategy,key in case['crops'].items():
  job=jobs[key];rec=records[key];folder=D/roi/f'视角{vn}';folder.mkdir(parents=True,exist_ok=True)
  rgb=np.asarray(Image.open(A/'inputs'/f'{key}_rgb.png').convert('RGB'));before=np.asarray(Image.open(A/'inputs'/f'{key}_baseline.png'),np.int32);after=np.asarray(Image.open(A/'inference'/f'{key}.png'),np.int32)
  assert after.shape==before.shape==rgb.shape[:2]
  colors,matching=match_colors(before,after);oldcolors=[palette(i) for i in range(int(before.max())+1)]
  bm=overlay(rgb,before,oldcolors);am=overlay(rgb,after,colors);pure_b=overlay(rgb,before,oldcolors,True);pure_a=overlay(rgb,after,colors,True)
  label='紧裁剪' if strategy=='tight' else '证据来源裁剪' if strategy=='witness' else '宽裁剪';countb=int(len(np.unique(before[before>0])));counta=int(len(np.unique(after[after>0])))
  fullrec=records[f'full_f{case["frame"]:06d}'];fsize=fullrec['model_resized_full_image_shape'];lsize=rec['model_resized_full_image_shape'];fullscale=fsize[1]/1200;localscale=lsize[1]/rgb.shape[1];zoom=localscale/fullscale
  panels=[('RGB 输入',Image.fromarray(rgb)),(f'整帧分割后裁剪 · {countb} masks',bm),(f'局部重新分割 · {counta} masks',am)]
  probe=None
  if strategy=='witness':
   u,v=map(int,job['alarm_projection_anchor_uv']);bf=int(before[v,u]);af=int(after[v,u]);assert bf==job['source_mask_local_id']
   probe={'uv':[u,v],'F':bf,'L':af,'source_3D_ID':job['source_instance_id']}
   for _,pic in panels:
    dd=ImageDraw.Draw(pic);dd.ellipse((u-6,v-6,u+6,v+6),outline=(255,20,35),width=2)
  card=Image.new('RGB',(1740,750),(250,251,252));draw=ImageDraw.Draw(card)
  draw.text((18,12),f'{roi} · {place} · {view["label"]} / 帧{case["frame"]} · {label}',font=F[30],fill=(26,39,55))
  draw.text((18,58),f'原始裁剪 {rgb.shape[1]}×{rgb.shape[0]} px → 模型整图分支 {lsize[1]}×{lsize[0]} px；相对整帧像素尺度 {zoom:.2f}×',font=F[24],fill=(45,58,71))
  for j,(title,img) in enumerate(panels):
   x=18+j*575;draw.text((x,110),title,font=F[24],fill=(30,43,57));card.paste(fit(img,(560,510)),(x,149))
  footer='相同RGB区域、权重和0.5阈值；颜色按像素重叠辅助配对，局部ID独立。白线仅显示mask边界。'
  if probe:footer=f'红圈为实际证据像素：原 F{probe["F"]}（3D ID{probe["source_3D_ID"]}）→ 新 '+(f'L{probe["L"]}' if probe['L'] else 'L0 未分割')+'；圈是显示标注，未作为模型提示。'
  draw.text((18,680),footer,font=F[22],fill=(40,54,68))
  card.save(folder/f'{label}_RGB与mask对照.png')
  Image.fromarray(rgb).save(folder/f'{label}_输入RGB.png');bm.save(folder/f'{label}_整帧mask裁剪.png');am.save(folder/f'{label}_局部重分割mask.png')
  pure_b.save(folder/f'{label}_整帧纯mask.png');pure_a.save(folder/f'{label}_局部纯mask.png')
  Image.fromarray(after.astype(np.uint16)).save(folder/f'{label}_局部mask_ID.png')
  info={'key':key,'ROI':roi,'rank':rank,'place':place,'view':vn,'frame':case['frame'],'strategy':strategy,'crop_xyxy':job['crop_xyxy'],'input_wh':[rgb.shape[1],rgb.shape[0]],'model_input_wh':[lsize[1],lsize[0]],'effective_zoom_vs_full':round(zoom,3),'baseline_mask_count':countb,'local_mask_count':counta,'local_scores':rec['selected_scores'],'color_matching':matching,'seconds':rec['seconds']}
  if probe:info['probe']=probe
  (folder/f'{label}_分割信息.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  idx=len(raw);raw.append({'rgb':rgb,'before':before,'after':after,'colors':colors,'oldcolors':oldcolors,'info':info});view['crops'][strategy]=idx
  rows.append({k:info[k] for k in ['ROI','rank','place','view','frame','strategy','input_wh','model_input_wh','effective_zoom_vs_full','baseline_mask_count','local_mask_count','seconds']})
  if vn==1 and strategy=='tight':primary_tiles[rank]=card
 rootcase[rank]['views'].append(view)

# All 32 ROIs, all three selected views. Grayscale 8-bit PNG encodes exact mask
# IDs after nearest-neighbour display sampling; native uint16 masks remain saved.
for display_width in [320,280,240,200,180,160]:
 entries=[]
 for item in raw:
  rgb=item['rgb'];ratio=min(1,display_width/rgb.shape[1]);w=max(1,round(rgb.shape[1]*ratio));h=max(1,round(rgb.shape[0]*ratio))
  im=Image.fromarray(rgb).resize((w,h),Image.Resampling.LANCZOS)
  assert item['after'].max()<256 and item['before'].max()<256
  mb=Image.fromarray(item['before'].astype(np.uint8)).resize((w,h),Image.Resampling.NEAREST);ma=Image.fromarray(item['after'].astype(np.uint8)).resize((w,h),Image.Resampling.NEAREST)
  info=item['info'];entries.append({'rgb':uri(im),'before':uri(mb),'after':uri(ma),'wh':[w,h],'colors':item['colors'],'oldcolors':item['oldcolors'],'info':info})
 data={'cases':[rootcase[k] for k in sorted(rootcase)],'entries':entries,'controls':manifest['full_frame_controls'],'ROI_views':manifest['selected_ROI_views'],'crop_jobs':len(raw),'display_width':display_width}
 encoded=json.dumps(data,ensure_ascii=False,separators=(',',':'))
 print('inline candidate',display_width,len(encoded.encode('utf-8')),flush=True)
 if len(encoded.encode('utf-8'))<985000:break
assert len(encoded.encode('utf-8'))<985000
(P/'zoom_inline_data.json').write_text(encoded,encoding='utf-8')

reps=[k for k in primary_tiles if rootcase[k]['representative']]
overview=Image.new('RGB',(1740,96+len(reps)*390),(230,234,238));draw=ImageDraw.Draw(overview);draw.text((20,18),'room0 · 13个代表ROI · 原整帧分割 vs 局部重新分割',font=F[30],fill=(28,42,58))
for no,rank in enumerate(reps):
 thumb=primary_tiles[rank].resize((1740,750));# keep captions readable; stacked cards crop only lower metadata
 # Recompose with compact peer panels rather than shrinking the full card.
 rec=next(x for x in raw if x['info']['rank']==rank and x['info']['view']==1 and x['info']['strategy']=='tight');info=rec['info'];y=90+no*390
 draw.text((20,y),f'{info["ROI"]} · {info["place"]} · {info["input_wh"][0]}×{info["input_wh"][1]} px · 相对整帧 {info["effective_zoom_vs_full"]:.2f}×',font=F[24],fill=(25,39,54))
 for j,(name,img) in enumerate([('RGB',Image.fromarray(rec['rgb'])),('原整帧 mask',overlay(rec['rgb'],rec['before'],rec['oldcolors'])),('局部重分割 mask',overlay(rec['rgb'],rec['after'],rec['colors']))]):
  x=20+j*575;draw.text((x,y+40),name,font=F[22],fill=(36,50,66));overview.paste(fit(img,(560,290)),(x,y+78))
overview.save(D/'13代表案例_紧裁剪重分割总览.png')
for rank,case in rootcase.items():
 # Each case contains all selected views and both crops where available.
 pages=[]
 for v in case['views']:
  for s in v['crops']:
   name='紧裁剪' if s=='tight' else '证据来源裁剪' if s=='witness' else '宽裁剪';pages.append(Image.open(D/case['ROI']/f'视角{v["view"]}'/f'{name}_RGB与mask对照.png').convert('RGB'))
 out=Image.new('RGB',(1740,sum(p.height for p in pages)),(250,251,252));y=0
 for page in pages:out.paste(page,(0,y));y+=page.height
 out.save(D/f'{case["ROI"]}_全部视角对照.jpg',quality=92)
with (D/'全部裁剪推理统计.csv').open('w',encoding='utf-8-sig',newline='') as f:
 cw=csv.DictWriter(f,fieldnames=list(rows[0]));cw.writeheader();cw.writerows(rows)
for name in ['inference_audit.json','model_provenance.json','model_config.yaml','input_manifest.json']:(D/name).write_bytes((A/name).read_bytes())
(D/'分割与配色说明.txt').write_text('每个候选ROI的三个已选历史视角均重新推理。紧裁剪为此前选中的原RGB裁剪；宽裁剪为当前展示的较宽观察窗口。\nCropFormer权重、配置、0.5阈值和重叠胜出规则均与原缓存相同。整帧也重新运行，检查原缓存复现。\n局部输入没有手工mask提示，没有深度细化，没有GT提示。由模型标准resize带来局部尺度提升；插值不会创造新图像信息。\n颜色仅用像素重叠的一对一配对辅助比较，F为整帧局部ID、L为新裁剪局部ID；它们不是3D持久实例ID。\n所有图像按同一个RGB裁剪范围并列。分割实例更多不代表质量更好，需要检查目标完整性、背景污染和过度碎分。\n这是前端可视化诊断，不是新增v3建图评估分数。\n',encoding='utf-8')
print(json.dumps({'ROIs':len(rootcase),'views':manifest['selected_ROI_views'],'crop_jobs':len(raw),'inline_bytes':len(encoded.encode('utf-8')),'display_width':display_width,'native_output':str(D)},ensure_ascii=False))
