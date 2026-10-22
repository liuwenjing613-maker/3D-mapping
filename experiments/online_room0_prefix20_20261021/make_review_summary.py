"""Compose review figures from actual exported mask overlays; never alter mask pixels."""
import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

p=argparse.ArgumentParser()
p.add_argument('--review',required=True)
p.add_argument('--font',required=True)
a=p.parse_args()
r=Path(a.review)
d=json.loads((r/'data.json').read_text(encoding='utf-8'))
font=lambda size:ImageFont.truetype(a.font,size)
ink='#19343d';muted='#627680';green='#086f66'
names=['原图 CropFormer','局部 CropFormer','局部 SAM2.1']
classes={'cabinet':'柜子','door':'门','windowpane':'窗户','light':'灯','floor':'地板','stool':'凳子','ottoman':'脚凳','coffee table':'茶几','sofa':'沙发','wall':'墙','cushion':'靠垫','ceiling':'天花板','vase':'花瓶','bottle':'瓶子','glass':'玻璃器皿'}
def picture(canvas,path,box):
    im=Image.open(r/path).convert('RGB')
    im.thumbnail((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
    canvas.paste(im,(box[0]+(box[2]-box[0]-im.width)//2,box[1]+(box[3]-box[1]-im.height)//2))

def board(items,filename,title,subtitle,retained):
    width=1500;gap=20;margin=30;cw=(width-margin*2-gap*2)//3;ch=292
    height=178+math.ceil(len(items)/3)*(ch+gap)+60
    im=Image.new('RGB',(width,height),'#f2f5f4');draw=ImageDraw.Draw(im)
    draw.text((margin,24),'ROOM0  /  在线验证 · 第20个处理帧',font=font(19),fill=green)
    draw.text((margin,58),title,font=font(36),fill=ink)
    draw.text((margin,114),subtitle,font=font(19),fill=muted)
    for n,x in enumerate(items):
        left=margin+n%3*(cw+gap);top=164+n//3*(ch+gap)
        draw.rounded_rectangle((left,top,left+cw,top+ch),radius=10,fill='white',outline='#d7e1df',width=1)
        draw.rounded_rectangle((left+1,top+1,left+cw-1,top+184),radius=9,fill='#e9eded')
        picture(im,x['image'],(left+8,top+7,left+cw-8,top+180))
        draw.text((left+14,top+193),x['id']+' / '+x['mask_id'],font=font(21),fill=ink)
        draw.text((left+14,top+226),f"原始帧 {x['frame']} · {names[x['model']]} · {classes.get(x['semantic']['top_class'],x['semantic']['top_class'])}",font=font(16),fill=muted)
        dec=x['decision'];last=dec['stage_decisions'][-1];e=last.get('evidence') or {}
        if retained:
            text='保留 · 第5步窗户/柜子豁免' if e.get('semantic_exemption') else '保留 · 通过全部筛选'
            color=green
        else:
            color='#9d4c40'
            if dec['terminal_stage']=='semantic':text=f"背景过滤 · 高置信背景 {100*e['high_excluded_pixels']/e['area']:.1f}%"
            elif dec['terminal_stage']=='full_support_containment':text=f"包含过滤 · 被 {e[0]['larger_group_id']} 包含 {100*e[0]['intersection']/e[0]['small_count']:.1f}%"
            elif dec['terminal_stage']=='skip_repair_B':
                text=f"跳过修复 · p={e['p']:.1%}, r={e['r']:.1%}";color='#456794'
            else:text=f"p/r 排除 · p={e['p']:.1%}, r={e['r']:.1%}"
        draw.text((left+14,top+257),text,font=font(16),fill=color)
    draw.text((margin,height-44),'仅使用已处理帧 0, 5, …, 95 的数据；原始二值 mask 未修改；尚未执行实际修复。',font=font(18),fill=muted)
    im.save(r/filename)

kept=[x for x in d['items'] if x['id'] in d['retained']]
dropped=[x for x in d['items'] if x['id'] not in d['retained']]
board(kept,'retained_overview.png',f'最终保留 {len(kept)} 张 mask','36 个 ROI → 30 张主种子 → 21 张保留；绿色为完整 mask 的显示叠图。',True)
board(dropped,'excluded_overview.png','排除 8 张 · 跳过修复 1 张','背景过滤 6 张 · 包含过滤 1 张 · 覆盖充分跳过 1 张 · p/r 筛选 1 张。',False)
roi=d['rois'][0];view=roi['views'][0];variant=view['variants'][0]
im=Image.new('RGB',(1500,1040),'#f2f5f4');draw=ImageDraw.Draw(im)
draw.text((30,24),'ROI 提示与模型对照示例',font=font(34),fill=ink)
draw.text((30,80),' / '.join(roi['aliases'])+f" · 历史原始帧 {view['frame']}（第 {view['processed']} 个处理帧）",font=font(20),fill=muted)
for i,(src,label) in enumerate([(view['rgb'],'历史原图'),(view['alarm'],'当前 ROI 提示：U 黑 / T 黄 / C 红')]):
    left=30+i*735;picture(im,src,(left,130,left+705,540));draw.text((left,554),label,font=font(21),fill=ink)
for i,(src,label) in enumerate(zip(variant['models'],names)):
    left=30+i*490;picture(im,src,(left,620,left+460,935));draw.text((left,949),label,font=font(21),fill=ink)
draw.text((30,1003),'三列使用同一裁剪范围；其余 35 个 ROI、全部视角与裁剪版本均在离线展示页中。',font=font(18),fill=muted)
im.save(r/'roi_comparison_example.png')
with zipfile.ZipFile(r/'downloads/retained_binary_masks.zip','w',zipfile.ZIP_DEFLATED) as z:
    rows=[]
    for x in kept:
        binary=r/x['mask']
        assert hashlib.sha256(binary.read_bytes()).hexdigest()==x['mask_sha256']
        z.write(binary,binary.name)
        rows.append({k:x[k] for k in ['id','mask_id','frame','processed','area','mask_sha256'] } | {'file':binary.name})
    z.writestr('manifest.json',json.dumps({'scene':'room0','processed_frames':20,'last_frame':95,'masks':rows},ensure_ascii=False,indent=2))
print(json.dumps({'retained':len(kept),'excluded_or_skipped':len(dropped),'figures':3,'binary_masks':len(rows)}))
