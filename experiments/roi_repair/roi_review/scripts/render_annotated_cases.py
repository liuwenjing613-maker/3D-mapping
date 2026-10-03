"""Diagnostic figures from frozen arrays; no point cloud export or map writes."""
from pathlib import Path
import json,io,base64,csv,shutil
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).parent
DATA=HERE/'conflict_diagnosis_arrays'
DEST=Path('D:/Users/刘雯静/Downloads/CVPR/results/P1-A1_room0_ROI优先级_20261002/冲突情况标注')
DEST.mkdir(exist_ok=True)
FONT='C:/Windows/Fonts/msyh.ttc'
F={n:ImageFont.truetype(FONT,n) for n in [16,18,20,22,24,26,28,30,32]}
cases=json.loads((DATA/'conflict_cases.json').read_text(encoding='utf-8'))
annotations=json.loads((HERE/'case_annotations.json').read_text(encoding='utf-8'))
inline=json.loads((HERE/'inline_data.json').read_text(encoding='utf-8'))
NAMES={2:'墙',4:'花瓶',6:'台灯',7:'柜子',13:'地面/地毯',15:'地毯',33:'天花板',57:'百叶窗',79:'沙发片段',172:'桌',179:'百叶窗',181:'花枝',182:'椅子',185:'小片段',186:'墙边',187:'书',206:'椅子',209:'坐垫',216:'沙发',224:'落地灯',236:'未发布',490:'未发布',792:'顶灯'}
STATE_COLORS={0:(145,145,145),1:(255,210,0),3:(245,25,40)}

def text_wrap(draw,text,width,font):
 lines=[];line=''
 for c in text:
  if c=='\n':lines.append(line);line='';continue
  if line and draw.textlength(line+c,font=font)>width:lines.append(line);line=c
  else:line+=c
 if line:lines.append(line)
 return lines

def wrapped(im,text,xy,width,font=22,color=(28,38,48)):
 d=ImageDraw.Draw(im);x,y=xy
 for line in text_wrap(d,text,width,F[font]):d.text((x,y),line,font=F[font],fill=color);y+=font+9
 return y

def badge(draw,label,at,color=(32,49,65),font=18):
 x,y=map(int,at);b=draw.textbbox((0,0),label,font=F[font]);w=b[2]-b[0]+12;h=font+12
 draw.rectangle((x,y,x+w,y+h),fill=color);draw.text((x+6,y+2),label,font=F[font],fill=(255,255,255))
 return w,h

def marker(im,mask,label,edge='top'):
 # The ring and leader are annotations; the underlying colored pixels stay exact.
 y,x=np.nonzero(mask)
 if not len(x):return
 tx,ty=float(np.median(x)),float(np.median(y));d=ImageDraw.Draw(im)
 bx=8 if tx<im.width*.5 else max(8,im.width-min(im.width-16,int(d.textlength(label,font=F[18]))+12)-8)
 by=8 if edge=='top' else im.height-40
 w,h=badge(d,label,(bx,by));d.line((bx+w/2,by+h/2,tx,ty),fill=(255,255,255),width=2)
 d.ellipse((tx-6,ty-6,tx+6,ty+6),outline=(255,255,255),width=2)

def image_uri(im,width=440,quality=72):
 im=im.copy();im.thumbnail((width,420));b=io.BytesIO();im.save(b,format='WEBP',quality=quality,method=4)
 return 'data:image/webp;base64,'+base64.b64encode(b.getvalue()).decode()

def fit(im,size):
 canvas=Image.new('RGB',size,(239,241,243));copy=im.copy();copy.thumbnail(size);canvas.paste(copy,((size[0]-copy.width)//2,(size[1]-copy.height)//2));return canvas

tiles=[];csvrows=[]
for r in cases:
 rank=r['rank'];ann=annotations[str(rank)];ann['GT_names_used_only_after_ROI_selection']=True;r['annotation']=ann
 counts=r['counts'];pairs=r['competition_pairs'];primary=pairs[0] if pairs else None
 with np.load(DATA/f'{r["roi_id"]}_pixels.npz') as z:
  rgb=z['rgb'];sm=z['state_map'];imap=z['instance_map'];ref=z['reference_colors'];mapped=z['mapped']
  # State pixels are native state of depth-visible core; no mask gate.
  state=rgb.copy()
  for s,color in STATE_COLORS.items():state[sm==s]=color
  stateim=Image.fromarray(state)
  core=sm!=255
  if core.any():
   visible=np.unique(sm[core]);lab='/'.join({0:'U',1:'T',3:'C'}[int(i)] for i in visible)
   marker(stateim,core,lab+' 核心')
  else:badge(ImageDraw.Draw(stateim),'主帧未投影到核心像素',(8,8))
  neighbor=rgb.copy();ok=mapped&(imap>0);neighbor[ok]=(rgb[ok]*.4+ref[ok]*.6).astype(np.uint8)
  neighim=Image.fromarray(neighbor);d=ImageDraw.Draw(neighim)
  # Select IDs near the alarm pixels plus the actual competitors visible in the crop.
  near=binary_dilation(core,iterations=28) if core.any() else np.ones(imap.shape,bool)
  ids,n=np.unique(imap[near&(imap>0)],return_counts=True);chosen=[int(ids[j]) for j in np.argsort(-n)[:4]]
  if primary:chosen+=primary['ids']
  chosen=list(dict.fromkeys(chosen))[:6];occupied=[]
  for i in chosen:
   mask=(imap==i);y,x=np.nonzero(mask)
   if not len(x):continue
   tx,ty=int(np.median(x)),int(np.median(y));label=f'ID{i} '+NAMES.get(i,'')
   w=int(d.textlength(label,font=F[18]))+12;h=30
   lx=int(np.clip(tx-w/2,6,max(6,neighim.width-w-6)));ly=int(np.clip(ty-h/2,6,max(6,neighim.height-h-6)))
   for _ in range(12):
    if not any(lx<b[2]+5 and lx+w>b[0]-5 and ly<b[3]+5 and ly+h>b[1]-5 for b in occupied):break
    ly=(ly+33)%(max(1,neighim.height-h-12))+6
   occupied.append((lx,ly,lx+w,ly+h));d.line((tx,ty,lx+w/2,ly+h/2),fill=(255,255,255),width=2);badge(d,label,(lx,ly));d.ellipse((tx-3,ty-3,tx+3,ty+3),fill=(255,255,255))
  if core.any():
   yy,xx=np.nonzero(core);bbox=(max(0,int(xx.min())-3),max(0,int(yy.min())-3),min(imap.shape[1]-1,int(xx.max())+3),min(imap.shape[0]-1,int(yy.max())+3))
   d.rectangle(bbox,outline=(255,255,255),width=2)
 sourceims=[];support=[]
 for v in r['source_views']:
  i=v['id'];detail=next(d for d in r['instance_details'] if d['id']==i)
  with np.load(DATA/f'{r["roi_id"]}_source_ID{i}.npz') as z:
   srgb=z['rgb'];m=z['mask'];pic=srgb.copy();pic[m]=(pic[m]*.55+np.array(detail['RGB'])*.45).astype(np.uint8)
   sim=Image.fromarray(pic);sd=ImageDraw.Draw(sim);x,y=map(int,z['source_pixel'])
   sd.ellipse((max(0,x-8),max(0,y-8),min(sim.width-1,x+8),min(sim.height-1,y+8)),outline=(245,25,40),width=3)
   bx=max(6,min(sim.width-90,x-40));by=max(6,min(sim.height-34,y+13));badge(sd,'同一3D点',(bx,by),font=16)
  sim.save(DEST/f'{r["roi_id"]}_来源_ID{i}.png')
  title=f'ID{i} {NAMES.get(i,"")} · 帧{v["frame_id"]} · mask{v["mask_local_id"]}'
  sourceims.append((title,sim));support.append({'id':i,'frame':v['frame_id'],'local_mask':v['mask_local_id'],'votes':v['support_votes'],'rgb':image_uri(sim,width=400,quality=70)})
 stateim.save(DEST/f'{r["roi_id"]}_三状态定位.png');neighim.save(DEST/f'{r["roi_id"]}_周围实例标注.png')
 if primary:
  p=primary
  point_vote=f'代表红点：ID{p["sample_top1_id"]} {p["sample_top1_votes"]}票 / ID{p["sample_top2_id"]} {p["sample_top2_votes"]}票'
  point_vote+=f' / 其他 {p["sample_other_votes"]}票；合计{p["sample_total_votes"]}票，首名占比{p["sample_confidence"]:.1%}'
  traces='；'.join(f'ID{i}: {p["source_support"][str(i)]["first_frame"]}–{p["source_support"][str(i)]["last_frame"]}帧' for i in p['ids'])
  traces+=f'；同帧双ID {p["same_source_frame_both_ids_count"]}次'
 else:point_vote='无红点：不涉及双实例票数比较。';traces='4个T点均为灯 ID6 的单帧一票；U点没有投票。'
 r['representative_vote_text']=point_vote;r['source_frame_ranges_text']=traces
 pair_text='；'.join('/'.join(map(str,p['ids']))+f'（{p["conflict_points"]}点）' for p in pairs)
 other=counts['CONFLICT']-sum(p['conflict_points'] for p in pairs)
 if other:pair_text+=f'；其余组合{other}点'
 card=Image.new('RGB',(1200,1450),(250,251,252));cd=ImageDraw.Draw(card)
 cd.text((24,18),r['roi_id']+' · '+ann['place'],font=F[32],fill=(20,35,50))
 cd.text((24,66),f'U 灰 {counts["U"]} / CONFLICT 红 {counts["CONFLICT"]} / T 黄 {counts["T"]}    {ann["type"]}',font=F[24],fill=(32,49,65))
 for j,(title,pic) in enumerate([('三状态核心定位 · 状态取自冻结证据',stateim),('周围实例 · ID 与参考v2配色',neighim)]):
  x=24+j*590;cd.text((x,114),title,font=F[22],fill=(35,48,63));card.paste(fit(pic,(565,410)),(x,150))
 y=583;y=wrapped(card,'已确认：'+ann['fact'],(24,y),1150,24)
 y=wrapped(card,ann['relation'],(24,y+6),1150,22)
 y=wrapped(card,point_vote,(24,y+6),1150,22)
 y=wrapped(card,'诊断：'+ann['interpretation'],(24,y+6),1150,22)
 y=max(y+16,805)
 if sourceims:
  for j,(title,pic) in enumerate(sourceims):
   x=24+j*590;cd.text((x,y),title,font=F[22],fill=(35,48,63));card.paste(fit(pic,(565,365)),(x,y+39))
  y+=423
  y=wrapped(card,'红圈为历史mask实际投票像素，已按深度回投到同一个代表3D点。',(24,y),1150,20)
 else:y=wrapped(card,'没有竞争来源mask；黄点候选只有 ID6，单帧一票。',(24,y),1150,24)
 y=wrapped(card,traces,(24,y+4),1150,18)
 y=wrapped(card,'注：票按每点、每ID、不同源帧累计；GT仅用于事后命名与关系诊断，不参与ROI形成和排序。',(24,y+5),1150,18)
 card.crop((0,0,1200,min(1450,y+18))).save(DEST/f'{r["roi_id"]}_状态与实例关系.png')
 tile=Image.new('RGB',(800,680),(250,251,252));td=ImageDraw.Draw(tile)
 td.text((14,8),r['roi_id']+' · '+ann['place'],font=F[28],fill=(23,37,53))
 td.text((14,48),f'U {counts["U"]} / C {counts["CONFLICT"]} / T {counts["T"]}',font=F[24],fill=(23,37,53))
 tile.paste(fit(stateim,(378,270)),(14,89));tile.paste(fit(neighim,(378,270)),(410,89))
 yy=wrapped(tile,ann['type'],(14,369),768,26)
 yy=wrapped(tile,'竞争：'+(pair_text or '无红点'),(14,yy+3),768,22)
 yy=wrapped(tile,ann['relation'],(14,yy+3),768,22)
 wrapped(tile,'诊断：'+ann['interpretation'],(14,yy+3),768,22)
 tiles.append(tile)
 ri=next(x for x in inline if x['rank']==rank)
 ri['place']=ann['place'];ri['diagnosis']=ann;ri['votes']=point_vote;ri['ranges']=traces;ri['pair_details']=pair_text or '无双实例竞争';ri['supports']=support
 ri['images']['states']=image_uri(stateim,width=440,quality=76);ri['images']['current3d']=image_uri(neighim,width=440,quality=76)
 # Repack the full-frame location photo for a compact overview; native figures stay full quality.
 location=Image.open(io.BytesIO(base64.b64decode(ri['images']['location'].split(',')[1])))
 ri['images']['location']=image_uri(location,width=650,quality=48)
 for name in ['rgb','cropformer']:ri['images'].pop(name,None)
 ri.pop('aux',None)
 csvrows.append({'ROI':r['roi_id'],'位置':ann['place'],'U':counts['U'],'CONFLICT':counts['CONFLICT'],'T':counts['T'],'冲突组合':pair_text,'代表点票数':point_vote,'周围实例':ann['relation'],'已确认事实':ann['fact'],'诊断_推断':ann['interpretation']})

overview=Image.new('RGB',(2440,120+5*700),(232,236,240));ov=ImageDraw.Draw(overview)
ov.text((24,14),'room0 · 13个ROI的实际状态与实例关系',font=F[32],fill=(22,36,52))
ov.text((24,62),'灰 U=无投票  |  红 C=多ID竞争  |  黄 T=本批案例均为单帧一票  |  每格左：状态，右：周围实例',font=F[24],fill=(40,52,64))
for j,t in enumerate(tiles):overview.paste(t,(10+(j%3)*810,112+(j//3)*700))
overview.save(DEST/'13案例_状态与周围实例总览.png')
with (DEST/'13案例_逐项标注.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
(DEST/'13案例_证据与诊断.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(HERE/'inline_data_annotated.json').write_text(json.dumps(inline,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
print(json.dumps({'cases':len(cases),'annotated_files':len(list(DEST.iterdir())),'inline_data_bytes':(HERE/'inline_data_annotated.json').stat().st_size,'output':str(DEST)},ensure_ascii=False))
