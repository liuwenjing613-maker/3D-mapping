from pathlib import Path
import numpy as np,json,hashlib,colorsys
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
ROOT=Path('/data/chenkejun/CVPR/results/trusted_dbscan_20261105')
OLD=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/ablations/dbscan_history_covisible_20261102/online_B')
OUT=ROOT/'ply_view';OUT.mkdir(exist_ok=True)
cmp=json.loads((ROOT/'comparison/comparison.json').read_text())
focus=[('room0',92),('room0',30),('room0',85),('room0',58),('room0',4),('office0',9),('office0',22),('office1',18),('room2',38)]
manifest={'status':'RUNNING','scenes':[],'cases':[],'description':'Actual final400 native TSDF surfaces. Common deterministic point sampling. GT used for display crop only. Display-only palette anchored by matching GT best predictions, then geometric overlap. One-to-one colors keep split IDs distinct. No label changes.'}
dtype=np.dtype([('xyz','<f4',(3,)),('color','u1',(3,)),('rgb','u1',(3,)),('instance_id','<i4')])
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sample(ix,n=200000):return ix if len(ix)<=n else ix[np.linspace(0,len(ix)-1,n,dtype=np.int64)]
def ply(path,xyz,rgb,ids,colors):
 a=np.empty(len(xyz),dtype=dtype);a['xyz']=xyz;a['rgb']=rgb;a['instance_id']=ids;a['color']=colors
 header=('ply\nformat binary_little_endian 1.0\ncomment native output; display colors only\nelement vertex '+str(len(a))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar rgb_red\nproperty uchar rgb_green\nproperty uchar rgb_blue\nproperty int instance_id\nend_header\n')
 with path.open('wb') as f:f.write(header.encode());a.tofile(f)
def view(x,center=None):
 lo=x.min(0);hi=x.max(0);c=(lo+hi)/2 if center is None else np.array(center)
 return {'center':c.tolist(),'extent':(hi-lo).tolist()}
def colors_for(ids,mp):return np.array([mp[int(i)] for i in ids],dtype=np.uint8)
for scene in ['room0','room1','room2','office0','office1','office2','office3','office4']:
 print('BUILD',scene,flush=True);folder=OUT/scene;folder.mkdir(exist_ok=True)
 paths=[OLD/scene/'instance_surface.npz',ROOT/'candidate'/scene/'instance_surface.npz']
 zz=[np.load(p) for p in paths];xyz=zz[0]['xyz_m'];rgb=np.uint8(np.clip(zz[0]['rgb']*255,0,255));ids=[z['instance_id'] for z in zz]
 np.testing.assert_array_equal(xyz,zz[1]['xyz_m']);np.testing.assert_array_equal(zz[0]['rgb'],zz[1]['rgb'])
 u=[np.unique(v[v>0]) for v in ids];m=(ids[0]>0)&(ids[1]>0)
 ia=np.searchsorted(u[0],ids[0][m]);ib=np.searchsorted(u[1],ids[1][m]);mat=np.bincount(ia*len(u[1])+ib,minlength=len(u[0])*len(u[1])).reshape(len(u[0]),len(u[1]))
 # Display-only anchors: best native predictions for the same evaluable GT.
 ovpair=[np.load(OLD/scene/'r2/overlap_matrix.npz'),np.load(ROOT/'candidate'/scene/'r2/overlap_matrix.npz')]
 anchors=[]
 for j,gid in enumerate(ovpair[0]['gt_ids']):
  k=int(np.where(ovpair[1]['gt_ids']==gid)[0][0]);a=int(np.argmax(ovpair[0]['iou'][:,j]));b=int(np.argmax(ovpair[1]['iou'][:,k]))
  va=float(ovpair[0]['iou'][a,j]);vb=float(ovpair[1]['iou'][b,k])
  if min(va,vb)>.02:anchors.append((min(va,vb),int(ovpair[0]['pred_uids'][a]),int(ovpair[1]['pred_uids'][b])))
 locked=[];sa=set();sb=set()
 for score,a,b in sorted(anchors,reverse=True):
  if a not in sa and b not in sb:locked.append((int(np.searchsorted(u[0],a)),int(np.searchsorted(u[1],b))));sa.add(a);sb.add(b)
 remA=np.array([i for i,v in enumerate(u[0]) if int(v) not in sa],dtype=int);remB=np.array([i for i,v in enumerate(u[1]) if int(v) not in sb],dtype=int)
 if len(remA) and len(remB):
  aa,bb=linear_sum_assignment(-mat[np.ix_(remA,remB)]);locked.extend(zip(remA[aa],remB[bb]))
 ai=np.array([a for a,b in locked]);bi=np.array([b for a,b in locked])
 tokens=[{int(i):int(i) for i in u[0]},{}];nextid=int(max(u[0]))+1;matches=[]
 for a,b in zip(ai,bi):
  if mat[a,b]>0:tokens[1][int(u[1][b])]=int(u[0][a]);matches.append([int(u[0][a]),int(u[1][b]),int(mat[a,b])])
 for i in u[1]:
  if int(i) not in tokens[1]:tokens[1][int(i)]=nextid;nextid+=1
 allkeys=sorted(set(tokens[0].values())|set(tokens[1].values()));adj={i:set() for i in allkeys}
 ix=sample(np.arange(len(xyz)),65000);dist,nn=cKDTree(xyz[ix]).query(xyz[ix],k=8)
 for arm in [0,1]:
  lab=ids[arm][ix];left=np.repeat(lab,8);right=lab[nn.ravel()];valid=(dist.ravel()<.09)&(left>0)&(right>0)&(left!=right)
  for a,b in np.unique(np.stack([left[valid],right[valid]],1),axis=0):
   a=tokens[arm][int(a)];b=tokens[arm][int(b)]
   if a!=b:adj[a].add(b);adj[b].add(a)
 palette=np.array([np.array(colorsys.hsv_to_rgb((i*.61803398875)%1,.70 if i%2 else .90,.98 if i%3 else .72))*255 for i in range(48)])
 assigned={}
 for t in sorted(allkeys,key=lambda i:(-len(adj[i]),i)):
  near=[assigned[n] for n in adj[t] if n in assigned]
  if near:
   scores=np.min(np.linalg.norm(palette[:,None,:]-np.array(near)[None,:,:],axis=2),axis=1)
   best=int(np.argmax(scores))
  else:best=(t*7)%len(palette)
  assigned[t]=palette[best].astype(np.uint8).tolist()
 maps=[{i:assigned[t] for i,t in tok.items()} for tok in tokens]
 for arm in [0,1]:
  for v in np.unique(ids[arm][ids[arm]<=0]):maps[arm][int(v)]=[115,126,143]
 center=np.median(xyz,axis=0);baseview=view(xyz)
 sceneinfo={'scene':scene,'total_points':len(xyz),'sampled_points':len(ix),'view':baseview,'palette':maps,'matches':matches,'source':[str(p) for p in paths],'sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}
 fullix=sample(np.arange(len(xyz)),240000);sceneinfo['sampled_points']=len(fullix)
 for arm,key in enumerate(['old','trusted']):
  ply(folder/(key+'.ply'),xyz[fullix],rgb[fullix],ids[arm][fullix],colors_for(ids[arm][fullix],maps[arm]))
 manifest['scenes'].append(sceneinfo)
 thisfocus=[g for s,g in focus if s==scene]
 if thisfocus:
  ms=json.loads((ROOT/'candidate'/scene/'r2/adapter_manifest.json').read_text());gt=np.load(ms['gt_file']);ov=[np.load(OLD/scene/'r2/overlap_matrix.npz'),np.load(ROOT/'candidate'/scene/'r2/overlap_matrix.npz')]
  for gid in thisfocus:
   row=next(r for r in cmp['per_gt'] if r['scene']==scene and r['raw_gt_id']==gid)
   gx=gt['xyz_ref'][(gt['raw_instance_id']==gid)&gt['valid_vertex_mask']]
   assert len(gx)>0
   lo=gx.min(0);hi=gx.max(0);c=(lo+hi)/2;extent=hi-lo
   pad=max(.09,min(.25,float(max(extent))*.10))
   local=np.where(np.all((xyz>=lo-pad)&(xyz<=hi+pad),axis=1))[0]
   picked=[];leg=[]
   for arm in [0,1]:
    z=ov[arm];j=int(np.where(z['gt_ids']==gid)[0][0]);inter=z['intersection'][:,j];order=np.argsort(-inter)
    chosen=[int(z['pred_uids'][i]) for i in order[:5] if inter[i]>=max(1,len(gx)*.03)]
    picked.append(chosen);leg.append([{'id':int(z['pred_uids'][i]),'intersection':int(inter[i]),'iou':float(z['iou'][i,j]),'color':maps[arm][int(z['pred_uids'][i])]} for i in order[:5] if inter[i]>0])
   involved=np.where(np.isin(ids[0],picked[0])|np.isin(ids[1],picked[1]))[0]
   direction=center-c
   if np.linalg.norm(direction)<.1:direction=np.array([1.,-1.,.5])
   direction/=np.linalg.norm(direction)
   key=f'{scene}_{gid}';case={'key':key,'scene':scene,'gt':gid,'name':row['semantic_class_zh'],'iou':[row['old_dbscan'],row['trusted_dbscan']],'delta':row['policy_dbscan_delta'],'picked':picked,'legend':leg,'bbox':[lo.tolist(),hi.tolist()],'direction':direction.tolist(),'view':view(gx),'modes':{},'gt_points':len(gx)}
   for mode,ind in [('local',local),('involved',involved)]:
    if len(ind)==0:ind=local
    sel=sample(ind,180000);case['modes'][mode]={'total':len(ind),'sampled':len(sel),'view':view(xyz[sel])}
    for arm,name in enumerate(['old','trusted']):ply(folder/f'{gid}_{mode}_{name}.ply',xyz[sel],rgb[sel],ids[arm][sel],colors_for(ids[arm][sel],maps[arm]))
   ply(folder/f'{gid}_gt.ply',gx,np.full((len(gx),3),255,np.uint8),np.full(len(gx),gid,np.int32),np.full((len(gx),3),255,np.uint8))
   manifest['cases'].append(case)
 dump(OUT/'manifest.json',manifest)
 for z in zz:z.close()
manifest['status']='PASS';manifest['case_count']=len(manifest['cases']);manifest['scene_count']=len(manifest['scenes'])
dump(OUT/'manifest.json',manifest);print('PASS',len(manifest['cases']),flush=True)
