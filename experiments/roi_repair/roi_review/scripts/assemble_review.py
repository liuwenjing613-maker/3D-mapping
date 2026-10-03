"""Assemble reproducible review figures; embedded UI keeps all data local."""
from pathlib import Path
import json,hashlib,base64,io,shutil,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/P1-A1_room0_ROI优先级_20261002'
VIZ=Path('D:/codex/home/visualizations/2026/10/02/01a0fb2d-4a26-7950-bad3-be6d50b1b315')
sys.path.insert(0,str(ROOT/'scripts/visualization'))
from align_instance_ply import read_binary_ply_vertices,xyz_of

FONT='C:/Windows/Fonts/msyh.ttc'
def font(size):return ImageFont.truetype(FONT,size)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jpeg_uri(p,maxw=400,maxh=250,quality=64):
 im=Image.open(p).convert('RGB');im.thumbnail((maxw,maxh));b=io.BytesIO();im.save(b,format='JPEG',quality=quality,optimize=True)
 return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode('ascii')

def edges(lo,hi):
 corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
 return [(a,b) for i,a in enumerate(corners) for b in corners[i+1:] if np.count_nonzero(a!=b)==1]

def local_cloud_plot(directory):
 z=np.load(directory/'cloud.npz');xyz=z['xyz'];center=xyz.mean(0);ext=np.ptp(xyz,axis=0);extent=max(ext.max(),.12)*1.12
 fig=plt.figure(figsize=(12,5.4),dpi=130)
 for j,key in enumerate(['rgb','state_rgb']):
  ax=fig.add_subplot(1,2,j+1,projection='3d');ax.scatter(*xyz.T,c=z[key]/255,s=4,depthshade=False,edgecolors='none')
  for axis,c in zip('xyz',center):getattr(ax,f'set_{axis}lim')(c-extent/2,c+extent/2)
  ax.set_box_aspect((1,1,1));ax.view_init(elev=20,azim=-65)
  ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
  ax.set_title('Current instances (reference v2 colors)' if j==0 else 'Alarm core + neighbouring surface')
 fig.tight_layout();fig.savefig(directory/'local3d.png');plt.close(fig)

def figure_sheets(ranked,selection):
 sheetdir=OUT/'ROI详情';sheetdir.mkdir(exist_ok=True)
 names=['RGB 局部','三状态触发点','当前 3D 实例投影','原始 CropFormer mask']
 for rank in selection['rendered']:
  r=ranked[rank-1];directory=OUT/'review_assets'/r['roi_id'];local_cloud_plot(directory)
  width=1800;views=len(r['selected_views']);height=145+views*440+830
  sheet=Image.new('RGB',(width,height),'white');d=ImageDraw.Draw(sheet)
  cs=r['core_counts'];d.text((35,20),f'{r["roi_id"]}  优先分数 {100*r["score_priority"]:.2f}  /  异常团 {r["parent_component"]}',font=font(32),fill='black')
  d.text((35,72),f'触发 U {cs["U"]} · CONFLICT {cs["CONFLICT"]} · 证据不足 {cs["T"]}    ROI {r["context_points"]} 点，含已分配邻域 {r["stable_context_points"]} 点',font=font(23),fill='#333333')
  for j,view in enumerate(r['selected_views']):
   y=140+j*440
   angle=view['min_angle_to_previous_deg'];angletext='' if angle is None else f' · 与前视角最小夹角 {angle:.1f}°'
   d.text((35,y),f'历史帧 {view["frame_id"]} · 核心可见 {100*view["core_coverage"]:.1f}%{angletext}',font=font(23),fill='black')
   for k,name in enumerate(['rgb','states','current3d','cropformer']):
    x=25+k*445;d.text((x,y+38),names[k],font=font(23),fill='black')
    im=ImageOps.contain(Image.open(directory/f'view{j+1}_{name}.png').convert('RGB'),(420,355))
    sheet.paste(im,(x+(420-im.width)//2,y+78+(355-im.height)//2))
  y=145+views*440
  d.text((35,y),'历史主帧位置 / 局部 3D（坐标与原 PLY 一致）',font=font(25),fill='black')
  loc=ImageOps.contain(Image.open(directory/'view1_location.jpg'),(1740,370));sheet.paste(loc,(30,y+45))
  im=ImageOps.contain(Image.open(directory/'local3d.png'),(1740,360));sheet.paste(im,(30,y+430))
  sheet.save(sheetdir/f'{r["roi_id"]}_三视角对照.png')
 # Distinct groups in descending priority, with the original global ROI IDs.
 ranks=selection['distinct_component_representatives']
 sheet=Image.new('RGB',(1900,120+len(ranks)*260),'white');d=ImageDraw.Draw(sheet)
 d.text((24,10),'room0 · 不同异常团的优先代表（保留完整全局编号）',font=font(32),fill='black')
 labels=['RGB 场景定位','RGB 局部','三状态（核心）','当前 3D 投影','原始 CropFormer']
 for j,label in enumerate(labels):d.text((15+j*380,65),label,font=font(23),fill='black')
 for pos,rank in enumerate(ranks):
  r=ranked[rank-1];directory=OUT/'review_assets'/r['roi_id'];y=120+pos*260
  cs=r['core_counts'];d.text((18,y),f'{r["roi_id"]}  {r["score_priority"]*100:.2f}分  |  U {cs["U"]} / C {cs["CONFLICT"]} / T {cs["T"]}  |  ROI {r["context_points"]}点',font=font(22),fill='black')
  files=['view1_location.jpg','view1_rgb.png','view1_states.png','view1_current3d.png','view1_cropformer.png']
  for j,name in enumerate(files):
   im=ImageOps.contain(Image.open(directory/name).convert('RGB'),(360,210));sheet.paste(im,(10+j*380+(360-im.width)//2,y+40+(210-im.height)//2))
 sheet.save(OUT/'room0_ROI_不同异常团优先对照总览.png')

def overview(ranked,selection):
 v=read_binary_ply_vertices(ROOT/'可视化ply/room0_P1-A1_最终地图_参考v2配色_完整.ply')
 xyz=xyz_of(v);colors=np.column_stack([v[k] for k in ['red','green','blue']])/255
 # Display-only ceiling cut and 3cm sampling, never used in proposals or ranking.
 keep=np.flatnonzero(xyz[:,2]<.90);_,jj=np.unique(np.floor(xyz[keep]/.03).astype(np.int32),axis=0,return_index=True);jj=keep[jj]
 fig=plt.figure(figsize=(15,6.4),dpi=140)
 a=fig.add_subplot(121);order=np.argsort(xyz[jj,2]);ids=jj[order]
 a.scatter(xyz[ids,0],xyz[ids,1],c=colors[ids],s=3,edgecolors='none');a.set_aspect('equal');a.set_xlabel('X (m)');a.set_ylabel('Y (m)');a.set_title('Top view / distinct anomaly groups')
 b=fig.add_subplot(122,projection='3d');b.scatter(*xyz[jj].T,c=colors[jj],s=1.4,depthshade=False,edgecolors='none');b.view_init(elev=30,azim=-70);b.set_box_aspect(np.ptp(xyz[jj],axis=0));b.set_xlabel('X (m)');b.set_ylabel('Y (m)');b.set_zlabel('Z (m)');b.set_title('3D review / ceiling hidden for visibility')
 for rank in selection['distinct_component_representatives']:
  r=ranked[rank-1];lo=np.array(r['bbox_min']);hi=np.array(r['bbox_max']);center=np.array(r['center'])
  a.plot([lo[0],hi[0],hi[0],lo[0],lo[0]],[lo[1],lo[1],hi[1],hi[1],lo[1]],color='#00a9c9',lw=1.0)
  a.annotate(str(rank),xy=center[:2],xytext=(8,10+(rank%3)*10),textcoords='offset points',fontsize=9,color='#073947',arrowprops={'arrowstyle':'-','lw':.7,'color':'#00a9c9'})
  b.add_collection3d(Line3DCollection(edges(lo,hi),colors='#00a9c9',linewidths=1));b.text(*center,str(rank),color='#004756',fontsize=8)
 fig.tight_layout();fig.savefig(OUT/'room0_ROI_3D位置总览.png');plt.close(fig)

def create_inline_data(ranked,selection):
 # A compact review of distinct groups, plus the highest mostly-T example.
 ranks=list(selection['distinct_component_representatives'])
 for rank in selection['additional_U_T_examples']:
  if rank not in ranks:ranks.append(rank)
 records=[]
 for rank in ranks:
  r=ranked[rank-1];directory=OUT/'review_assets'/r['roi_id'];z=np.load(directory/'cloud.npz')
  xyz=z['xyz'];role=z['role'];core=np.flatnonzero(role==1);other=np.flatnonzero(role!=1)
  # Keep all small cores; cap only display samples. Exact point memberships are
  # retained in the remote membership archive.
  sample=np.unique(np.r_[core[np.linspace(0,len(core)-1,min(len(core),1000),dtype=int)],
       other[np.linspace(0,len(other)-1,min(len(other),700),dtype=int)] if len(other) else np.array([],int)])
  center=(xyz.min(0)+xyz.max(0))/2
  rel=np.rint((xyz[sample]-center)*1000).astype('<i2')
  dtype=np.dtype([('xyz','<i2',(3,)),('rgb','u1',(3,)),('state_rgb','u1',(3,)),('role','u1')])
  packed=np.empty(len(sample),dtype)
  packed['xyz']=rel;packed['rgb']=z['rgb'][sample];packed['state_rgb']=z['state_rgb'][sample];packed['role']=role[sample]
  cloud=base64.b64encode(packed.tobytes()).decode('ascii')
  images={}
  for name in ['rgb','states','current3d','cropformer']:
   images[name]=jpeg_uri(directory/f'view1_{name}.png',maxw=300,maxh=210,quality=55)
  images['location']=jpeg_uri(directory/'view1_location.jpg',maxw=680,maxh=385,quality=55)
  auxiliary=[{'frame_id':view['frame_id'],'coverage':round(view['core_coverage']*100,1),
    'angle':None if view['min_angle_to_previous_deg'] is None else round(view['min_angle_to_previous_deg'],1),
    'rgb':jpeg_uri(directory/f'view{j+1}_rgb.png',maxw=160,maxh=120,quality=48)} for j,view in enumerate(r['selected_views']) if j>0]
  records.append({'rank':rank,'id':r['roi_id'],'parent':r['parent_component'],'score':round(r['score_priority']*100,2),
    'counts':r['core_counts'],'context':r['context_points'],'stable':r['stable_context_points'],
    'extent':np.round(r['extent_m'],3).tolist(),'density':round(r['score_density']*100,2),
    'frame':r['selected_views'][0]['frame_id'],'coverage':round(r['primary_coverage']*100,1),
    'competition':r['dominant_competing_ids'],'parent_patches':sum(x['parent_component']==r['parent_component'] for x in ranked),
    'cloud':cloud,'cloud_count':len(sample),'images':images,'aux':auxiliary})
 return records

def report(ranked,selection):
 counts=[r['core_counts'] for r in ranked];viewable=sum(r['viewable'] for r in ranked)
 text=f'''P1-A1 room0 · 三状态 ROI 筛选与优先级审阅

范围：与上一份三状态 PLY 相同，触发源是最终仍未分配的 145,379 点：
U 75,821；CONFLICT 59,477；证据不足 T 10,081。
最终已补全的 20,183 个原始三状态点不新增触发，但落入 ROI 时作为上下文保留。

实际输出：{len(ranked)} 个局部 ROI；{viewable} 个有历史深度可见视图；{len(ranked)-viewable} 个没有。
所有触发点恰好分入一个核心；不设最小点数门槛；完整排序见 roi_ranking.csv。
{sum(r['core_points']<100 for r in ranked)} 个核心小于 100 点，全部保留。

形成方法：
1. 复用表面邻域构图方法，按几何距离和法向形成表面连接。这里的连接用于排查区域，不决定实例归属。
2. 三状态点先扩一圈表面邻居，用于跨过零散的稳定点，形成异常团。
3. 超过 0.60m 的异常团分成沿表面最远 0.30m 的局部检查块；保留父异常团编号。
4. 每个核心沿表面扩 10cm，并纳入 6cm 空间邻近对照面，包含已分配的稳定点；尖锐边缘不会让查看范围只剩孤立报警点。
5. 上下文重叠本身不继续串联合并，防止把半个房间变成一个 ROI。

排序方法：
冲突权重 3、证据不足权重 2、U 权重 1。先统计三圈邻域（至少 3cm）中的加权异常密度。
以 1cm 占据格做等面积近似，减少 TSDF 顶点密度差异的影响。
优先分数 = 100 × 加权异常密度 × n/(n+9)，n 为局部占据格数。
最后一项只平滑极小邻域，不删除小区域；9 格相当于约 3cm×3cm 的表面采样量。
有历史可见视图的 ROI 优先；同分再看冲突已有多少帧支持。
分数是检查启发式，不是错误概率；1cm 占据格只是面积近似。

历史帧：
只查实际建图的 400 帧（0:2000:5）。不把无 mask 票数误当成无图像证据。
使用正深度与点到深度反投影的欧氏距离 ≤2cm 判断可见；不受 mask 是否前景限制。
选 1 张主视图，再选 2 张兼顾覆盖率、角度差异和新增覆盖的辅助视图。
辅助视图角度不足 25°时保留实际角度，不宣称证据相互独立。
当前 3D 投影由测量深度逐像素回投，再匹配固定 TSDF 表面，超过 2cm 的像素留空。

请优先看不同异常团代表：{', '.join(ranked[k-1]['roi_id'] for k in selection['distinct_component_representatives'])}。
原始全局前 20 个 ROI 全来自一个百叶窗异常团。不同异常团代表按各自最高分排序，保留原始全局编号，避免一处大区域占满浏览列表。
证据不足主导的额外样例：{', '.join(ranked[k-1]['roi_id'] for k in selection['additional_U_T_examples'] if ranked[k-1]['core_counts']['T']>ranked[k-1]['core_points']*.7)}。
ROI详情/ 中提供前20个、不同异常团代表及 U/T 样例的三视角完整对照图，共 {len(selection['rendered'])} 个。

查看结果：
ROI详情/ 提供逐编号的完整三视角对照图；交互预览可切换编号、查看局部 3D。
当前实例仍用 reference v2 配色；坐标是原地图世界坐标，无身份改写。
按最新要求，不提供或生成 PLY。

我的判断：
这版能用于验证“状态报警→局部排查”的定位设计。单独按密度排序会让百叶窗这类大面冲突占据前排，因此推荐按父异常团去重查看代表，再展开子块。
10cm 只是首轮范围，不能保证已经包住完整物体；尤其仅在边缘报警时，需根据 RGB 判断是否向完整物体扩展。
三状态无法发现所有稳定错误，本轮也不能仅因红点多就认定需要拆分或合并。

核验：未读取 GT；未修改地图/关联/历史贡献；原证据与最终地图 SHA256 保持一致；原始 mask 哈希与冻结投影缓存一致。
这是 ROI 设计审阅，尚未执行任何实例修复，也未改变统一 v3 评估口径。
'''
 (OUT/'ROI设计与查看说明.txt').write_text(text,encoding='utf-8')

def main():
 ranked=json.loads((OUT/'roi_ranked.json').read_text(encoding='utf-8'))
 selection=json.loads((OUT/'review_selection.json').read_text(encoding='utf-8'))
 figure_sheets(ranked,selection);overview(ranked,selection);report(ranked,selection)
 audit=json.loads((OUT/'review_audit.json').read_text(encoding='utf-8'))
 with np.load(ROOT/'outputs/p1a1_raw_replica8_20261002/room0_final_instance_surface.npz') as f:
  for rank in selection['rendered']:
   cloud=np.load(OUT/'review_assets'/ranked[rank-1]['roi_id']/'cloud.npz')
   assert np.array_equal(cloud['xyz'],f['xyz_m'][cloud['point_index']])
 for key in ['ranked_three_states_PLY_sha256','numbered_boxes_PLY_sha256','all_145379_alarm_points_present_once_in_ranked_PLY']:audit.pop(key,None)
 audit['all_145379_alarm_points_present_once_in_ROI_cores']=True
 audit['deliverable']='ranked ROI images; PLY generation disabled at user request'
 records=create_inline_data(ranked,selection)
 (Path(__file__).parent/'inline_data.json').write_text(json.dumps(records,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
 audit['local_download_and_coordinates_verified']=True
 (OUT/'本地核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'status':'PASS','preview_ROIs':len(records),'inline_data_bytes':(Path(__file__).parent/'inline_data.json').stat().st_size,'detailed_figures':len(selection['rendered'])},ensure_ascii=False))

if __name__=='__main__':main()
