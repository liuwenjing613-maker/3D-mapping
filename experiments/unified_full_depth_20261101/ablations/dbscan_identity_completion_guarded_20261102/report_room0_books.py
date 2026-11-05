import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
O=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101');D=O/'diagnostics/room0_books_20261102';T=O/'ablations/dbscan_history_seeded_20261102';V=O/'instance_comparison_viewer'
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def hist(x):
 a,n=np.unique(x,return_counts=True);return dict(zip(map(int,a),map(int,n)))
original=load(O/'published/online_B/room0/final_B/instance_surface.npz');xyz=original['xyz_m'];regions=load(D/'book_regions.npz');labels=[original['instance_id'],load(T/'base/online_B/room0/merge_fill/instance_surface.npz')['instance_id'],load(T/'online_B/room0/instance_surface.npz')['instance_id']]
ix=regions['book_92'];cols={235:'#e34a36',605:'#3fa8f5',890:'#f5bd45',-1:'#77808d'}
plt.rcParams.update({'figure.facecolor':'#0b1320','axes.facecolor':'#0b1320','text.color':'#dce7f4','axes.labelcolor':'#dce7f4','xtick.color':'#91a1b5','ytick.color':'#91a1b5','font.size':11})
fig,axes=plt.subplots(1,4,figsize=(15,4.8),layout='constrained')
rgb=original['rgb'][ix];rgb=rgb/255. if rgb.max()>1 else rgb
for ax,title,lab in zip(axes,['RGB surface','Original v2-B','Previous DBSCAN','History-core DBSCAN'],[None,*labels]):
 colors=rgb if lab is None else [cols.get(int(x),'#a66cf5') for x in lab[ix]]
 ax.scatter(xyz[ix,0],xyz[ix,1],c=colors,s=12,marker='s',linewidths=0);ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('world x (m)');ax.set_ylabel('world y (m)')
 if lab is not None:ax.text(.02,.02,f'gray: {int((lab[ix]<0).sum())} / {len(ix)}',transform=ax.transAxes,color='#ffffff',bbox=dict(facecolor='#101927',alpha=.9,edgecolor='none'))
fig.suptitle('room0 side-table book: gray points recover to book ID 605 (blue), not table ID 235 (red)',fontsize=12)
fig.savefig(D/'book92_comparison.png',dpi=160);plt.close(fig)
for cp,case,fid,tid,box in [(300,'room0v2-f001150-b01',1150,20,(375,410,485,480)),(400,'room0v2-f001515-b01',1515,31,(210,560,360,680))]:
 root=O/'online/room0'/str(cp)/'B';cfg=json.loads((root/'tracking/track_config.json').read_text());raw=np.array(Image.open(Path(cfg['source']['scene_root'])/'results'/f'frame{fid:06d}.jpg').convert('RGB'));mask=np.array(Image.open(root/'repair/cases'/case/'prepared_seed.png'))==tid;over=raw.copy();over[mask]=(.45*over[mask]+.55*np.array([0,225,255])).astype(np.uint8)
 a=Image.fromarray(raw).crop(box).resize((440,280));b=Image.fromarray(over).crop(box).resize((440,280));panel=Image.new('RGB',(880,310),'#101927');panel.paste(a,(0,30));panel.paste(b,(440,30));draw=ImageDraw.Draw(panel);draw.text((12,9),f'frame {fid} RGB',fill='white');draw.text((450,9),f'accepted seed track {tid}',fill='white');panel.save(D/f'seed_{cp}.png')
audit={'diagnostic_only_GT_not_used_by_method':True,'book92_native_points':len(ix),'gray_before':int((labels[0][ix]<0).sum()),'gray_previous':int((labels[1][ix]<0).sum()),'gray_adjusted':int((labels[2][ix]<0).sum()),'labels_adjusted':hist(labels[2][ix]),'official_iou':{}}
for name,path in [('original',O/'evaluation/online_B/room0/r2/overlap_matrix.npz'),('previous',T/'base/online_B/room0/merge_fill/r2/overlap_matrix.npz'),('adjusted',T/'online_B/room0/r2/overlap_matrix.npz')]:
 z=load(path);col=int(np.flatnonzero(z['gt_ids']==92)[0]);row=int(np.flatnonzero(z['pred_uids']=='605')[0]);audit['official_iou'][name]=float(z['iou'][row,col])
(D/'adjustment_audit.json').write_text(json.dumps(audit,indent=2))
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>room0书本灰点根因审计</title><style>body{background:#0b1320;color:#dce7f4;max-width:1150px;margin:30px auto;padding:0 20px;font:16px/1.7 system-ui}h1{font-size:25px}h2{font-size:20px;margin-top:30px}p{margin:12px 0}a{color:#7edbc3}img{width:100%;border:1px solid #344256;border-radius:8px;margin:12px 0}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #344256;text-align:left}code{background:#202c3f;padding:2px 5px}.note{color:#aabbd1;font-size:14px}</style><h1>room0 桌面书本：身份先拆开、后续又被桌子覆盖</h1><p>重点对象是边桌上灰点明显的书（诊断定位 GT 92）。所有GT仅用于事后定位和评价，没有进入调整算法。</p><p><a href="../dbscan-history/?scene=room0&method=online_B&arm=history_seeded&roi=sidebook">打开书本局部3D对照</a> · <a href="../dbscan-history/">完整八场景对照</a></p><h2>根因由同一个三维点的完整票据核实</h2><table><tr><th>阶段</th><th>书面中心点1658536的有效票</th><th>后果</th></tr><tr><td>300帧修复之前</td><td>桌子235：23票</td><td>书被桌子ID包含</td></tr><tr><td>300帧修复之后</td><td>书605：19票；桌235：4票</td><td>书ID占82.61%，已确认</td></tr><tr><td>后续100帧</td><td>新增桌235：19票；书605：0票</td><td>新增观测又按桌子身份累积</td></tr><tr><td>400帧最终修复后</td><td>桌235：20票；书605：19票；混合候选890：3票</td><td>最高47.62%，变成灰点</td></tr></table><p>300帧的书种子与旧235的Dice为0.5525，已经超过0.5；桌子种子的Dice为0.7985，一对一分配保留235给桌子，书获得605。因此这里的关键不是把0.5阈值调低，而是分离后的书ID没有在后续观测中持续保持。</p><p>300帧书种子100个投影点中93个属于书、7个属于桌子；400帧候选则混合了328个桌子点和88个书点。跟踪gate限制了可回写帧数，但中心点在300帧已正确确认，之后的身份回退才是这次重新变灰的直接证据。</p><img src="seed_300.png" alt="300帧的书本独立种子"><img src="seed_400.png" alt="400帧混合书桌候选"><h2>为什么上一版DBSCAN没解决</h2><p>原方案只用当前至少4票且占比80%的确认点作锚点。书ID605只剩少量确认点，大部分支持点已经因旧桌子票竞争变灰；聚类前的邻域锚点、法向和平面条件就会挡住它们。607个剩余灰点中394个缺少8个兼容同ID锚点，122个没有法向/平面兼容锚点，40个在5厘米内没有锚点，另外51个受混合邻居、历史票或无效属性限制。把235和605整体合并，会重新把书并回桌子。</p><h2>调整与结果</h2><p>保留原≥2票且≥67%的原生投票；终端补全增加仍被当前提交保留的历史修复核心，用当前至少4票、占比至少25%且不低于最高票一半的支持点参与局部DBSCAN。要求历史核心、几何对应、法向/平面和密度支持一致，不新增ID，不覆盖已经分配的点，也不使用GT作决策。</p><img src="book92_comparison.png" alt="书本表面三版本对照"><p>972个诊断书面点：原始616个灰点 → 上一版607个 → 调整版175个；新增432个点归入书ID605。官方评价中605与这本书的IoU从2.79%升至55.05%，跨过0.5匹配门槛。仍有252个书面点归桌235、75个归890，当前方案没有把这些已分配点强制改写。</p><p>茶几上的两本书（GT12/59）还存在另一类边界混票、低观测和几何邻域不足，不能用这一例代表所有书已解决。main没有该次历史拆分的连续提交证据，所以不会借用v2-B的ID或核心。</p><p class="note">这是终端后处理开发验证，修复了部分显示与实例归属；没有修复在线新观测的身份关联机制，正式算法/阈值/GT/评价协议均未修改。</p><p><a href="summary.json">逐书统计</a> · <a href="trace.json">逐帧票据与种子审计</a> · <a href="adjustment_audit.json">调整后归属与IoU</a></p></html>'''
(D/'index.html').write_text(html,encoding='utf-8')
if not (V/'room0-books').exists():(V/'room0-books').symlink_to(D,target_is_directory=True)
print(json.dumps(audit),flush=True)
