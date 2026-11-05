from pathlib import Path
import json,hashlib,html
import numpy as np
from PIL import Image
from instance_repair.two_audit_sources import sha256_file
from instance_repair.two_audit_cache import write_json
R=Path('/data/chenkejun/CVPR/results/two-audit/replica_room0_office2_v1_20261104/stable_seed_filter_v1')
O=R/'review/replica8';D=O/'containment_images';D.mkdir(exist_ok=True)
sections=[];records=[]
for scene in ['office0','office1']:
 review=json.loads((R/scene/'containment_review.json').read_text())
 pool=json.loads((R/scene/'v1/grouping.json').read_text());seeds={s['id']:s for s in pool['seeds']}
 groups=review['seeds'];chain=review['chains'][0];ids=[chain[0]['small'],chain[0]['large'],chain[1]['large']]
 cfg=json.loads((Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/main')/scene/'400/A/mapping/config.json').read_text())['source']
 metadata={v['frame']:v for v in map(json.loads,(Path(cfg['mask_root'])/cfg['mask_metadata']).read_text().splitlines())}
 cards=[];entries=[]
 for tag,gid,col in zip(['小 Mask','中 Mask','大 Mask'],ids,[[255,140,25],[25,160,255],[210,70,220]]):
  rec=groups[gid];s=seeds[rec['mask_id']];mp=R/scene/'v1'/s['mask'];assert sha256_file(mp)==s['mask_sha256']
  mask=np.asarray(Image.open(mp))>0;assert mask.sum()==s['area']
  rgbp=Path(cfg['scene_root'])/cfg['rgb_pattern'].format(frame=s['frame']);assert sha256_file(rgbp)==metadata[s['frame']]['input_sha256']
  rgb=np.asarray(Image.open(rgbp).convert('RGB')).astype(float);color=np.array(col)
  rgb[mask]=rgb[mask]*.55+color*.45
  inner=mask.copy();inner[1:]&=mask[:-1];inner[:-1]&=mask[1:];inner[:,1:]&=mask[:,:-1];inner[:,:-1]&=mask[:,1:];rgb[mask&~inner]=color
  image=Image.fromarray(rgb.astype(np.uint8));name=scene+'_'+s['id'];full='containment_images/'+name+'_full.jpg';crop='containment_images/'+name+'_crop.jpg'
  image.save(O/full,quality=94)
  y,x=np.nonzero(mask);pad=max(35,int(max(np.ptp(x),np.ptp(y))*.25));box=(max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(mask.shape[1],int(x.max())+pad+1),min(mask.shape[0],int(y.max())+pad+1))
  image.crop(box).save(O/crop,quality=94)
  title=tag+' · T'+s['id'][1:]+' · '+gid
  cards.append('<article><h3>'+title+'</h3><button class="picture" data-full="'+full+'" data-crop="'+crop+'" data-title="'+title+'"><img src="'+crop+'" alt="'+title+'"></button><p>帧 '+str(s['frame'])+' · '+str(s['support_count'])+' 个支持点<br>原 Mask 面积 '+str(s['area'])+' 像素</p></article>')
  entries.append(dict(group=gid,mask=s['id'],frame=s['frame'],support=s['support_count'],mask_sha256=s['mask_sha256'],crop=crop,full=full))
 sections.append('<section><h2>'+scene+'</h2><div class="relations">小 → 中：覆盖 '+format(chain[0]['fraction']*100,'.2f')+'%　｜　中 → 大：覆盖 '+format(chain[1]['fraction']*100,'.2f')+'%</div><div class="cards">'+''.join(cards)+'</div></section>')
 records.append(dict(scene=scene,chain=chain,masks=entries))
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>链式包含 · Mask 检查</title><style>
*{box-sizing:border-box}body{margin:0;background:#eff3f4;color:#21343e;font:14px "Segoe UI","Microsoft YaHei",sans-serif}header{background:white;padding:15px;display:flex;gap:15px;align-items:center;flex-wrap:wrap}h1{font-size:19px;margin:0}a{color:#126f68}main{max-width:1500px;margin:auto;padding:15px}.note{font-size:13px;color:#617680;line-height:1.8}section{background:white;padding:15px;margin:15px 0;border-radius:9px}h2{font-size:18px;margin:0 0 10px}.relations{padding:10px;background:#eaf4f2;color:#126f68;margin-bottom:12px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}h3{font-size:13px;margin:4px 0 10px}article p{font-size:12px;line-height:1.8}button{cursor:pointer;font:inherit;border:1px solid #cbd8dc;border-radius:6px;background:white;padding:8px}.picture{padding:0;width:100%;background:#edf1f3}.picture img{width:100%;height:270px;object-fit:contain;display:block}dialog{max-width:96vw;width:1200px;border:1px solid #bdcdd1;border-radius:9px}dialog img{width:100%;height:75vh;object-fit:contain}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}#caption{flex:1}@media(max-width:700px){main{padding:8px}section{padding:10px}.cards{gap:6px}.picture img{height:180px}h3{line-height:1.5}.relations{font-size:12px}}
</style><header><h1>链式包含 · Mask 检查</h1><a href="./">返回最终种子</a></header><main><p class="note">尚未淘汰。覆盖率是较小 Mask 的三维支持点被较大 Mask 覆盖的比例，阈值为 &gt;70%。图中保留原始 Mask；各图单独裁剪放大，显示尺寸不代表物体大小。点击图片可切换完整来源帧。</p>'''+''.join(sections)+'''</main><dialog id="zoom"><div class="toolbar"><b id="caption"></b><button id="crop">局部放大</button><button id="full">完整来源帧</button><button id="close">关闭</button></div><img id="large" alt="待检查的原始 Mask"></dialog><script>
let active;const $=id=>document.getElementById(id);for(const b of document.querySelectorAll('.picture'))b.onclick=()=>{active=b.dataset;$('caption').textContent=active.title;$('large').src=active.crop;$('zoom').showModal()};$('crop').onclick=()=>{$('large').src=active.crop};$('full').onclick=()=>{$('large').src=active.full};$('close').onclick=()=>$('zoom').close();
</script></html>'''
(O/'containment.html').write_text(page)
write_json(O/'containment_review.json',dict(status='REVIEW_ONLY',no_masks_removed=True,scenes=records))
for p in [O/'index.html',Path('/home/chenkejun/CVPR/worktrees/two-audit/scripts/two_audit_replica8_final.html')]:
 s=p.read_text()
 if 'containment.html' not in s:s=s.replace('</header>','<a href="containment.html">查看待确认包含链</a></header>')
 p.write_text(s)
print('PASS: 2 chains, 6 original masks, 12 images; no selection changed')
