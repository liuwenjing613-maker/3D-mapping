from pathlib import Path
import json,colorsys,numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image,ImageDraw
C=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022')
T=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
B=Path('/data/chenkejun/CVPR/results/online_room0_prefix50_20261021')
R=T/'repair50'
O=T/'review/diagnosis50';O.mkdir(parents=True,exist_ok=True)
before=np.load(B/'baseline/final/instance_surface.npz');xyz=before['xyz_m'];b=before['instance_id']
stages={}
for phase in ['before','direct','core_all','whole_to_partial']:
 stages[phase]={}
 for kind in ['native','final']:
  path=B/'baseline'/kind/'instance_surface.npz' if phase=='before' else R/'partial'/phase/'room0'/kind/'instance_surface.npz'
  z=np.load(path);np.testing.assert_array_equal(xyz,z['xyz_m']);stages[phase][kind]=z['instance_id']
a=stages['whole_to_partial']['final']
ii=np.flatnonzero(b==19);pairs=cKDTree(xyz[ii]).query_pairs(.021,output_type='ndarray')
graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(ii),len(ii)))
n,cc=connected_components(graph,directed=False);best=int(np.bincount(cc[a[ii]==91]).argmax());target=ii[cc==best]
def counts(v):
 u,c=np.unique(v,return_counts=True);return {str(x):int(y) for x,y in zip(u,c)}
summary=dict(status='PASS',frame=245,processed_frames=50,geometry_identical=True,GT_used=False,analysis_only=True,
 target=dict(original_id=19,new_id=91,wall_id=2,points=len(target),component_rule='ID19 points connected within 0.021m; select component with maximal overlap to final ID91',xyz_bounds=[xyz[target].min(0).tolist(),xyz[target].max(0).tolist()]),
 phases={phase:{kind:counts(labels[target]) for kind,labels in kinds.items()} for phase,kinds in stages.items()},
 all_old19_to_after=counts(a[b==19]),target_points_reassigned_to_wall2=int((a[target]==2).sum()),
 core_changes_in_target=int((stages['direct']['final'][target]!=stages['core_all']['final'][target]).sum()),
 E1_changes_in_target=int((stages['core_all']['final'][target]!=a[target]).sum()),
 legacy_palette_rgb={str(i):[int(x*255) for x in colorsys.hsv_to_rgb((i*.61803398875)%1,.67,.94)] for i in [2,19,91]},
 unassigned_target_evidence={})
for phase in ['before','direct','whole_to_partial']:
 path=B/'baseline/surface_evidence.npz' if phase=='before' else R/'partial'/phase/'room0/native/surface_evidence.npz'
 e=np.load(path);unassigned=target[stages[phase]['native'][target]<0]
 summary['unassigned_target_evidence'][phase]=dict(n=len(unassigned),states=counts(e['state'][unassigned]),top1=counts(e['top1_instance_id'][unassigned]),top2=counts(e['top2_instance_id'][unassigned]),mean_votes=float(e['total_frame_votes'][unassigned].mean()) if len(unassigned) else None)
config=json.loads((C/'mapping_config50.json').read_text());scene=Path(config['source']['scene_root'])
frame=245
pose=np.fromstring((scene/'traj.txt').read_text().splitlines()[frame],sep=' ').reshape(4,4)
pc=(xyz-pose[:3,3])@pose[:3,:3];depth=pc[:,2];u=config['camera']['fx']*pc[:,0]/depth+config['camera']['cx'];v=config['camera']['fy']*pc[:,1]/depth+config['camera']['cy']
height=680;width=1200
inside=(depth>0)&(u>=-2)&(u<width+2)&(v>=-2)&(v<height+2)
indexes=np.flatnonzero(inside);ux=np.rint(u[inside]).astype(int);vy=np.rint(v[inside]).astype(int)
# Nearest visible surface point per camera pixel, with the same 3D coordinates and a 3x3 point splat.
pixels=[];point_ids=[]
for dy in [-1,0,1]:
 for dx in [-1,0,1]:
  xx=ux+dx;yy=vy+dy;valid=(xx>=0)&(xx<width)&(yy>=0)&(yy<height)
  pixels.append(yy[valid]*width+xx[valid]);point_ids.append(indexes[valid])
pixels=np.concatenate(pixels);point_ids=np.concatenate(point_ids)
order=np.lexsort((point_ids,depth[point_ids],pixels));sorted_pixels=pixels[order]
first=np.r_[True,np.diff(sorted_pixels)!=0];ids_image=np.full(height*width,-1,np.int64)
ids_image[sorted_pixels[first]]=point_ids[order[first]];ids_image=ids_image.reshape(height,width)
target_mask=np.isin(ids_image,target);ys,xs=np.where(target_mask);box=[int(xs.min())-36,int(ys.min())-40,int(xs.max())+37,int(ys.max())+42]
box=[max(0,box[0]),max(0,box[1]),min(width,box[2]),min(height,box[3])]
summary['projection']=dict(camera_frame=frame,fx=600,fy=600,splat_pixels=3,nearest_depth=True,crop_xyxy=box,method='TSDF point projection, not a screen capture',visible_target_pixel_count=int(target_mask.sum()))
rgb=Image.open(scene/'results'/f'frame{frame:06d}.jpg').convert('RGB')
rgb.crop(tuple(box)).resize(((box[2]-box[0])*3,(box[3]-box[1])*3)).save(O/'rgb_crop.png')
full_rgb=rgb.copy();draw=ImageDraw.Draw(full_rgb);draw.rectangle(box,outline=(255,45,70),width=4);full_rgb.resize((900,510)).save(O/'rgb_context.jpg',quality=90)
for tag,labels,high in [('before_legacy',b,False),('after_legacy',a,False),('after_contrast',a,True),('before_contrast',b,True)]:
 palette=np.zeros((max(108,int(labels.max())+1),3),np.uint8)
 for i in range(1,len(palette)):
  palette[i]=np.array(colorsys.hsv_to_rgb((i*.61803398875)%1,.67,.94))*255
 if high:
  palette[2]=[185,185,185];palette[19]=[152,79,239];palette[91]=[0,100,255]
 e_path=B/'baseline/surface_evidence.npz' if tag.startswith('before') else R/'partial/whole_to_partial/room0/native/surface_evidence.npz'
 ev=np.load(e_path);states=ev['state']
 colors=np.zeros((len(labels),3),np.uint8);colored=labels>0;colors[colored]=palette[labels[colored]]
 colors[(labels<0)&(states==1)]=[255,255,0];colors[(labels<0)&(states==3)]=[255,0,0]
 arr=np.zeros((height,width,3),np.uint8)+245;valid=ids_image>=0;arr[valid]=colors[ids_image[valid]]
 Image.fromarray(arr).crop(tuple(box)).resize(((box[2]-box[0])*3,(box[3]-box[1])*3),Image.Resampling.NEAREST).save(O/(tag+'.png'))
ev=np.load(R/'partial/whole_to_partial/room0/native/surface_evidence.npz')
reg=np.load(R/'e1/region.npz')
unresolved=target[a[target]<0]
pair_conflict=((ev['top1_instance_id'][unresolved]==19)&(ev['top2_instance_id'][unresolved]==91))|((ev['top1_instance_id'][unresolved]==91)&(ev['top2_instance_id'][unresolved]==19))
summary['edge_evidence']=dict(unassigned_final_points=len(unresolved),old19_vs_new91_top2_conflict_points=int(pair_conflict.sum()),unassigned_confidence_max=float(ev['confidence'][unresolved].max()),confirmation_threshold=.67,unassigned_state_counts=counts(ev['state'][unresolved]),unassigned_E1_owner_counts=counts(reg['owner'][unresolved]),old19_remaining_E1_owner_counts=counts(reg['owner'][target[a[target]==19]]))
(O/'diagnosis.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
# Preserve actual point index selection for reproducible read-only audit.
np.savez_compressed(O/'target_surface_indices.npz',surface_index=target.astype(np.int32))
table=''
for phase,title in [('before','未修复'),('direct','partial 直接撤换'),('core_all','历史核心'),('whole_to_partial','E1 后')]:
 for kind,title2 in [('native','严格投票'),('final','提交后')]:
  row=summary['phases'][phase][kind]
  table+='<tr><td>'+title+'</td><td>'+title2+'</td>'+''.join('<td>'+str(row.get(str(i),0))+'</td>' for i in [19,91,2,-1])+'</tr>'
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>50 帧矩形实例核验</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#f3f5fa;color:#172239;margin:0}main{max-width:1180px;margin:0 auto;padding:28px}h1{font-size:27px;margin:0 0 12px}h2{font-size:20px}.callout{background:#e2f3ed;padding:16px 20px;border-left:5px solid #177752;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}figure{margin:0;background:white;padding:16px;border-radius:12px}figure img{display:block;width:100%;height:auto;image-rendering:pixelated}figcaption{margin:8px 0;color:#536173}table{border-collapse:collapse;background:white;width:100%}td,th{padding:9px 12px;border-bottom:1px solid #dce2ed;text-align:right}td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}a{color:#2457ad}code{background:#e6eaf2;padding:2px 5px}.note{color:#536173;font-size:14px}@media(max-width:720px){.grid{grid-template-columns:1fr}main{padding:18px}}</style>
<main><p><a href="../50/">返回 50 帧交互 3D</a> · <a href="diagnosis.json">逐阶段数值 JSON</a></p><h1>矩形还在：新实例颜色与墙面过近</h1>
<div class="callout">该矩形原有 <b>9,182</b> 个表面点。修复后 <b>8,815 点改属独立实例 ID 91</b>，<b>227 点仍为 ID 19</b>，<b>140 点未定</b>。归入墙面 ID 2 的点数为 <b>0</b>。旧配色让 ID 91 与墙面只相差 5 个 RGB 数值，产生“物体被吞掉”的观感。</div>
<p>下面使用相同 50 帧 TSDF、同一表面点索引，按原始第 245 帧相机投影，深度近的点覆盖远点。仅改变展示颜色；没有修改实例标签、追踪或修复结果。</p>
<div class="grid">
<figure><img src="rgb_context.jpg"><figcaption>原始第 245 帧 RGB；红框为下方核验区域。</figcaption></figure>
<figure><img src="rgb_crop.png"><figcaption>原始 RGB 局部，便于对照矩形的真实位置。</figcaption></figure>
<figure><img src="before_legacy.png"><figcaption>未修复 · 原配色：矩形为 ID 19，紫色。</figcaption></figure>
<figure><img src="after_legacy.png"><figcaption>修复后 · 原配色：矩形 ID 91 为 RGB (167,239,79)，墙面 ID 2 为 (172,239,79)。</figcaption></figure>
<figure><img src="before_contrast.png"><figcaption>未修复 · 诊断配色：ID 19 紫色，墙 ID 2 灰色。</figcaption></figure>
<figure><img src="after_contrast.png"><figcaption>修复后 · 诊断配色：ID 91 蓝色，ID 19 紫色，墙 ID 2 灰色。红/黄仍表示未确定点的原始状态。</figcaption></figure>
</div>
<h2>在什么阶段发生</h2><p>主体身份转换发生在 <code>partial 直接撤换</code>。其严格投票已将 8,784 点确定为 ID 91，后处理再补 31 点。之后的历史核心与 E1 对此矩形的最终标签均为逐点不变。剩余边缘 140 点未确定、227 点保留旧身份，是实际可继续检查的边缘问题。140 个未定点中，111 点的前两名选票恰为旧 ID 19 与新 ID 91；全部确认比例小于 0.67。这 367 个边缘点全部不在 E1 可信区域，因此不会被后续阶段强制重标。</p>
<table><thead><tr><th>阶段</th><th>输出</th><th>ID 19</th><th>ID 91</th><th>墙 ID 2</th><th>未定</th></tr></thead><tbody>'''+table+'''</tbody></table>
<p class="note">矩形范围的复现定义：在未修复 ID 19 中以 2.1 cm 邻接构成连通域，取与修复后 ID 91 相交最多的连通块，共 9,182 点。核验未使用 GT；本页能证明身份未并入墙面，不能独立证明分割正确率。高对比图使用固定诊断配色，与交互网页后续的统一新配色可以不同。</p></main></html>'''
(O/'index.html').write_text(html,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
