from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import json, html, hashlib

ROOT=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
P=ROOT/'main/scene0011_00/200/A/candidates'
OUT=ROOT/'instance_ply_comparison/warp_review_20261106'
OUT.mkdir(exist_ok=True)
cfg=json.loads(Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106/main/scene0011_00/selection_code/200/A/config.json').read_text())
scene=Path(cfg['source']['scene_root']);poses=np.loadtxt(scene/'traj.txt').reshape(-1,4,4);ks=np.loadtxt(scene/'intrinsics.txt').reshape(-1,3,3)
group=json.loads((P/'v1/grouping.json').read_text());orig_frames=json.loads(Path('/data/chenkejun/CVPR/results/scannet_ovimap_strict_dbscan_20261104/inputs/scene0011_00/selection.json').read_text())['original_frames']

def load_seed(name):
 s=next(s for s in group['seeds'] if s['id']==name);fr=s['frame'];rgb=Image.open(scene/'results'/('frame%06d.jpg'%fr)).convert('RGB');depth=np.asarray(Image.open(scene/'results'/('depth%06d.png'%fr)),dtype=np.float64)/1000.;mask=np.asarray(Image.open(P/'v1'/s['mask']))>0
 assert depth.shape==mask.shape==(480,640) and rgb.size==(640,480)
 return dict(name=name,frame=fr,processed=fr//5+1,original=orig_frames[fr//5],rgb=rgb,depth=depth,mask=mask,pose=poses[fr],K=ks[fr],mask_file=str((P/'v1'/s['mask']).resolve()))

def outline(mask):
 img=Image.fromarray(mask.astype(np.uint8)*255,'L')
 return (np.asarray(img.filter(ImageFilter.MaxFilter(3)))>np.asarray(img.filter(ImageFilter.MinFilter(3))))
def dilate(mask,size=5):
 return np.asarray(Image.fromarray(mask.astype(np.uint8)*255,'L').filter(ImageFilter.MaxFilter(size)))>0
def mask_overlay(s,color):
 a=np.asarray(s['rgb']).copy();m=s['mask'];a[m]=(a[m].astype(float)*.52+np.asarray(color)*.48).astype(np.uint8);a[outline(m)]=color;return Image.fromarray(a)

def forward(src,dst):
 da=src['depth'];K=src['K'];K2=dst['K'];y,x=np.nonzero((da>0)&(da<10));z=da[y,x]
 pc=np.column_stack(((x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z));world=pc@src['pose'][:3,:3].T+src['pose'][:3,3];qb=(world-dst['pose'][:3,3])@dst['pose'][:3,:3];zz=qb[:,2]
 front=(zz>.05)&(zz<10);uv=np.full((len(zz),2),-1,np.int32);uv[front]=np.rint(np.column_stack((K2[0,0]*qb[front,0]/zz[front]+K2[0,2],K2[1,1]*qb[front,1]/zz[front]+K2[1,2]))).astype(np.int32)
 inside=front&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480);ix=np.flatnonzero(inside);u,v=uv[ix].T;db=dst['depth'][v,u]
 back=np.column_stack(((u-K2[0,2])*db/K2[0,0],(v-K2[1,2])*db/K2[1,1],db));ok=(db>0)&(db<10)&(np.linalg.norm(back-qb[ix],axis=1)<=.02);ix=ix[ok];u,v=uv[ix].T
 common=np.zeros((480,640),bool);common[v,u]=True;warped=np.zeros_like(common);keep=src['mask'][y[ix],x[ix]];warped[v[keep],u[keep]]=True
 target=dst['mask']&common;overlap=warped&target;src_only=warped&~target;dst_only=target&~warped;excluded=dst['mask']&~common;union=warped|target
 stats=dict(source=src['name'],target=dst['name'],source_frame=src['frame'],target_frame=dst['frame'],source_processed=src['processed'],target_processed=dst['processed'],source_original=src['original'],target_original=dst['original'],source_mask_pixels=int(src['mask'].sum()),target_mask_pixels=int(dst['mask'].sum()),target_common_domain_pixels=int(common.sum()),warped_pixels=int(warped.sum()),target_mask_common_pixels=int(target.sum()),intersection=int(overlap.sum()),warped_only=int(src_only.sum()),target_only=int(dst_only.sum()),union=int(union.sum()),iou=float(overlap.sum()/union.sum()) if union.any() else None,unobserved_target_pixels=int(excluded.sum()),gt_used=False,depth_consistency_m=.02,rasterization='rounded target pixel; any valid projected source depth pixel; overlapping projected source pixels form union')
 assert stats['intersection']+stats['warped_only']==stats['warped_pixels']
 assert stats['intersection']+stats['target_only']==stats['target_mask_common_pixels']
 assert stats['intersection']+stats['warped_only']+stats['target_only']==stats['union']
 # Original masks and IoU are untouched. Dilation below is ONLY to make sparse projected pixels visible.
 a=np.asarray(dst['rgb']).copy();a[excluded]=(a[excluded].astype(float)*.55+np.array([144,152,161])*.45).astype(np.uint8);a[outline(dst['mask'])]=[245,185,42]
 a[dilate(src_only)]=[19,156,244];a[dilate(dst_only)]=[243,137,30];a[dilate(overlap)]=[33,199,108]
 overlay=Image.fromarray(a);target_pixels=warped|dst['mask'];yy,xx=np.nonzero(target_pixels)
 if len(xx):
  x0=max(0,int(xx.min())-28);x1=min(640,int(xx.max())+29);y0=max(0,int(yy.min())-28);y1=min(480,int(yy.max())+29);crop=overlay.crop((x0,y0,x1,y1));crop.thumbnail((1024,768),Image.Resampling.NEAREST)
 else:crop=overlay.copy()
 prefix=src['name']+'_to_'+dst['name'];mask_overlay(src,(35,164,255)).save(OUT/(prefix+'_source.png'));mask_overlay(dst,(243,165,34)).save(OUT/(prefix+'_target.png'));overlay.save(OUT/(prefix+'_overlap.png'));crop.save(OUT/(prefix+'_zoom.png'))
 np.savez_compressed(OUT/(prefix+'_exact_masks.npz'),common=common,warped=warped,target_in_common=target,overlap=overlap,source_only=src_only,target_only=dst_only,target_unobserved=excluded)
 return stats

def card(s,idx):
 pre=s['source']+'_to_'+s['target'];I=s['intersection'];U=s['union'];title=html.escape(s['source']+' → '+s['target']);imgs=[('源视角：蓝色是源 mask',pre+'_source.png'),('目标视角：黄色是目标 mask',pre+'_target.png'),('目标视角：投影与重合',pre+'_overlap.png'),('投影重合区域放大',pre+'_zoom.png')]
 images=''.join('<figure><figcaption>'+html.escape(t)+'</figcaption><a href="'+html.escape(f)+'" target="_blank"><img loading="lazy" src="'+html.escape(f)+'" alt="'+html.escape(t)+'"></a></figure>' for t,f in imgs)
 return '<section id="case'+str(idx)+'" class="panel"><div class="row"><h2>'+title+'</h2><span class="value">共同可见像素 IoU '+format(s['iou'],'.3f')+'</span></div><p>源：处理第 '+str(s['source_processed'])+' 帧 / 原始帧 '+str(s['source_original'])+'；目标：处理第 '+str(s['target_processed'])+' 帧 / 原始帧 '+str(s['target_original'])+'。</p><div class="metric">投影 mask <b>'+str(s['warped_pixels'])+'</b> 像素 · 目标 mask 在共同可见区域 <b>'+str(s['target_mask_common_pixels'])+'</b> 像素 · 重合 <b>'+str(I)+'</b> 像素 · 仅源 <b>'+str(s['warped_only'])+'</b> · 仅目标 <b>'+str(s['target_only'])+'</b> · 并集 <b>'+str(U)+'</b> · IoU <b>'+str(I)+' / '+str(U)+' = '+format(s['iou'],'.3f')+'</b></div><div class="grid">'+images+'</div><p class="note">目标完整 mask 共 '+str(s['target_mask_pixels'])+' 像素，其中 '+str(s['unobserved_target_pixels'])+' 像素没有被这次投影覆盖或未通过深度一致性检查；它们不计入本方向的 IoU，不等于确定不可见。</p></section>'

seeds={x:load_seed(x) for x in ['M0053','M0092','M0126']};directions=[('M0053','M0092'),('M0092','M0053'),('M0053','M0126'),('M0126','M0053')];results=[forward(seeds[a],seeds[b]) for a,b in directions]
expected=[(495,1050),(461,1095),(114,424),(130,437)]
assert [(r['intersection'],r['union']) for r in results]==expected
report=dict(status='PASS',scene='scene0011_00',branch='main',mode='offline_200',scope='read-only frame-to-frame projection, not repair or score',gt_used=False,source_config=str(scene),source_masks={k:dict(frame=v['frame'],mask=v['mask_file'],sha256=hashlib.sha256(Path(v['mask_file']).read_bytes()).hexdigest()) for k,v in seeds.items()},results=results)
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
style='''*{box-sizing:border-box}body{font:14px system-ui,-apple-system,"Microsoft YaHei",sans-serif;background:#f3f6f9;color:#26384f;margin:0}.wrap{max-width:1320px;margin:auto;padding:22px}h1{font-size:23px;color:#172d45;margin:0 0 5px}h2{font-size:18px;margin:0}.row{display:flex;justify-content:space-between;align-items:center;gap:12px}p{font-size:13px;color:#64768a;line-height:1.65;margin:5px 0}.nav{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.nav a{border:1px solid #cddbe7;color:#315f96;background:#fff;text-decoration:none;border-radius:7px;padding:8px 11px;font-size:13px}.banner,.panel{background:white;border:1px solid #dbe4ed;border-radius:9px;padding:15px;margin:12px 0}.banner{background:#edf5ff;border-color:#d6e6f8}.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}.legend span{font-size:12px;color:#425770}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.metric{font-size:13px;color:#3b5169;background:#f4f7fa;border-radius:6px;padding:9px;margin:10px 0}.value{font-size:15px;color:#1a5a95;font-weight:700}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}figure{margin:0;min-width:0}figcaption{font-size:13px;font-weight:650;margin:6px 0;color:#39526d}img{display:block;width:100%;max-height:540px;object-fit:contain;background:#e6ecf2;border:1px solid #dce4ec;border-radius:5px}.note{font-size:12px;margin-top:10px}.foot{color:#6c7d90;font-size:12px}@media(max-width:800px){.grid{grid-template-columns:1fr}.wrap{padding:14px}.row{align-items:start;flex-direction:column}}'''
nav='<a href="../covis_diagnosis.html">← 返回共同可见诊断</a>'+''.join('<a href="#case'+str(i+1)+'">'+a+' → '+b+'</a>' for i,(a,b) in enumerate(directions))
intro='<div class="banner"><p><b>怎样计算：</b>把源帧深度反投影到三维，再投到目标帧；与目标深度三维距离不超过 2 cm 的点才保留。将所有源帧有效深度点投到目标帧，得到共同可见的投影覆盖像素，再在这个范围里比较投影 mask 和目标 mask。该范围会受光栅化空洞影响。</p><p>绿色表示原始像素真正重合，蓝色表示只有源 mask 投影，橙色表示只有目标 mask，灰色表示目标 mask 中未被投影覆盖或未通过深度一致性检查的部分。黄色线是目标完整 mask 轮廓。<b>彩色点仅为方便观看加粗 5×5；数值仍按未加粗的原始像素计算。</b></p><p>两种方向的投影覆盖区域和像素采样不同，IoU 可能不同。遮挡、深度误差和像素落点会造成空洞；灰色区域不代表确定不可见，也不代表两个 mask 不重合。本页是只读诊断，未用 GT 做投影或筛选。</p><div class="legend"><span><i class="dot" style="background:#21c76c"></i>双方重合</span><span><i class="dot" style="background:#139cf4"></i>仅源 mask 投影</span><span><i class="dot" style="background:#f3891e"></i>仅目标 mask</span><span><i class="dot" style="background:#929aa3"></i>目标 mask 未被投影覆盖</span><span><i class="dot" style="background:#f5b92a"></i>目标 mask 完整轮廓</span></div></div>'
blocks='<h2>例一：M0053 与 M0092</h2><p>事后 GT 诊断显示它们主要对应同一实例。</p>'+card(results[0],1)+card(results[1],2)+'<h2>例二：M0053 与 M0126</h2><p>事后 GT 诊断显示它们主要对应不同实例；投影仍可能出现部分重合。</p>'+card(results[2],3)+card(results[3],4)
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>跨视角 mask 投影对照</title><style>'+style+'</style></head><body><main class="wrap"><h1>跨视角 mask 投影与重合</h1><p>scene0011_00 · main 离线 · 200 帧 · PPT 最新方法与原筛选规则 · 四个方向的只读诊断</p><nav class="nav">'+nav+'</nav>'+intro+blocks+'<p class="foot">源 mask、相机位姿与深度来自原实验；结果与未经加粗的二值区域保存在服务器诊断目录。</p></main></body></html>'
(OUT/'index.html').write_text(page,encoding='utf-8')
print(json.dumps({'status':'PASS','output':str(OUT),'results':results},ensure_ascii=False),flush=True)
