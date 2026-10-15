from pathlib import Path
from PIL import Image,ImageDraw
import json
p=Path('results/固定案例_三模型对比_20261006');d=json.loads((p/'viewer_data.json').read_text(encoding='utf-8'))
keys=['room0_ROI-C0010_v1_f001660_balanced','room0_ROI-C0015_v2_f000515_balanced','room0_ROI-C0021_v1_f000945_balanced','room2_ROI-C0008_v3_f000885_target_balanced','office0_ROI-C0011_v2_f000305_balanced','office2_ROI-C0008_v1_f000520_balanced']
out=Image.new('RGB',(1500,1000),'white'); draw=ImageDraw.Draw(out)
for i,k in enumerate(keys):
 im=Image.open(p/d['crops'][k]['RGB']); im.thumbnail((480,440)); x=(i%3)*500;y=(i//3)*500;out.paste(im,(x,y+30));draw.text((x+5,y+5),k,fill='black')
out.save('outputs/vlm_threeway_20261014/candidate_contact.jpg')
