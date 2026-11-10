"""Display-only, deterministic spatial palette shared across checkpoint variants."""
import colorsys
import numpy as np
from scipy.spatial import cKDTree

def rgb_to_lab(rgb):
 x=np.asarray(rgb,dtype=np.float64)/255.
 x=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
 xyz=x@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
 xyz/=np.array([.95047,1.,1.08883])
 f=np.where(xyz>(6/29)**3,np.cbrt(xyz),xyz/(3*(6/29)**2)+4/29)
 return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)

def candidate_colors():
 rgb=np.unique(np.asarray([np.rint(np.array(colorsys.hsv_to_rgb(h/180,s,v))*255)
  for h in range(180) for s in (.55,.72,.9) for v in (.6,.78,.96)],dtype=np.uint8),axis=0)
 lab=rgb_to_lab(rgb)
 reserved=rgb_to_lab([[0,0,0],[255,255,0],[255,0,0],[233,238,240]])
 d=np.linalg.norm(lab[:,None,:]-reserved[None,:,:],axis=2)
 keep=(lab[:,0]>=35)&(lab[:,0]<=80)&(np.linalg.norm(lab[:,1:],axis=1)>=35)&(d.min(1)>=22)
 return rgb[keep],lab[keep]

def build_palette(xyz,label_arrays,radius_m=.25):
 xyz=np.asarray(xyz,dtype=np.float64)
 labels=[np.asarray(a) for a in label_arrays]
 ids=np.unique(np.concatenate([a[a>0] for a in labels])).astype(int)
 clouds=[];trees=[];bounds=[];sizes=[]
 for k in ids:
  pick=np.logical_or.reduce([a==k for a in labels]);p=xyz[pick]
  clouds.append(p);trees.append(cKDTree(p));bounds.append((p.min(0),p.max(0)));sizes.append(len(p))
 edges=[]
 for i in range(len(ids)):
  for j in range(i):
   gap=np.maximum(0,np.maximum(bounds[i][0]-bounds[j][1],bounds[j][0]-bounds[i][1]))
   if np.linalg.norm(gap)>radius_m:continue
   small,big=(i,j) if sizes[i]<=sizes[j] else (j,i)
   d=trees[big].query(clouds[small],distance_upper_bound=radius_m,workers=2)[0]
   nearest=float(d.min())
   if np.isfinite(nearest):edges.append((j,i,nearest))
 rgb,lab=candidate_colors();n=len(ids);adj=[set() for _ in ids]
 for i,j,d in edges:adj[i].add(j);adj[j].add(i)
 degrees=np.asarray([len(x) for x in adj]);sizes=np.asarray(sizes)
 def score(choice):
  near=np.asarray([np.linalg.norm(lab[choice[i]]-lab[choice[j]]) for i,j,d in edges])
  all_d=np.linalg.norm(lab[choice,None,:]-lab[None,choice,:],axis=2)
  glob=all_d[np.triu_indices(n,1)]
  return (float(near.min()) if len(near) else 999.,float(np.percentile(near,10)) if len(near) else 999.,float(glob.min()) if len(glob) else 999.)
 best=None
 for trial in range(16):
  rng=np.random.default_rng(20261022+trial)
  chosen=np.full(n,-1,int)
  priority=rng.random(n)*.15
  for step in range(n):
   remaining=np.flatnonzero(chosen<0)
   i=max(remaining,key=lambda k:(sum(chosen[j]>=0 for j in adj[k]),degrees[k]+priority[k],sizes[k]))
   used=np.flatnonzero(chosen>=0);nbr=[j for j in adj[i] if chosen[j]>=0]
   global_d=np.linalg.norm(lab[:,None,:]-lab[chosen[used]][None,:,:],axis=2).min(1) if len(used) else np.full(len(lab),100.)
   near_d=np.linalg.norm(lab[:,None,:]-lab[chosen[nbr]][None,:,:],axis=2).min(1) if nbr else np.full(len(lab),100.)
   eligible=global_d>=13.
   if not eligible.any():eligible=global_d==global_d.max()
   objective=near_d+.06*global_d+.002*rng.random(len(lab))
   objective[~eligible]=-np.inf
   chosen[i]=int(np.argmax(objective))
  for iteration in range(3):
   old_score=score(chosen)
   order=sorted(range(n),key=lambda i:min([np.linalg.norm(lab[chosen[i]]-lab[chosen[j]]) for j in adj[i]] or [999]))
   for i in order:
    used=np.asarray([j for j in range(n) if j!=i],int);nbr=list(adj[i])
    global_d=np.linalg.norm(lab[:,None,:]-lab[chosen[used]][None,:,:],axis=2).min(1) if len(used) else np.full(len(lab),100.)
    near_d=np.linalg.norm(lab[:,None,:]-lab[chosen[nbr]][None,:,:],axis=2).min(1) if nbr else np.full(len(lab),100.)
    objective=near_d+.04*global_d
    objective[global_d<13.]=-np.inf
    k=int(np.argmax(objective))
    cur_min=min([np.linalg.norm(lab[chosen[i]]-lab[chosen[j]]) for j in nbr] or [999])
    if near_d[k]>cur_min+.01:chosen[i]=k
   if score(chosen)==old_score:break
  merit=score(chosen)
  if best is None or merit>best[0]:best=(merit,chosen.copy())
 merit,chosen=best
 palette={str(int(k)):rgb[v].astype(int).tolist() for k,v in zip(ids,chosen)}
 edge_rows=[dict(a=int(ids[i]),b=int(ids[j]),nearest_m=d,deltaE76=float(np.linalg.norm(lab[chosen[i]]-lab[chosen[j]]))) for i,j,d in edges]
 edge_rows.sort(key=lambda x:x['deltaE76'])
 return palette,dict(method='joint-before-after spatial adjacency; deterministic Lab maximin',radius_m=radius_m,
  instance_count=len(ids),neighbor_pairs=len(edges),minimum_neighbor_deltaE76=merit[0],
  p10_neighbor_deltaE76=merit[1],minimum_all_ID_deltaE76=merit[2],
  colors_rgb=palette,neighbor_pairs_audit=edge_rows,
  scientific_labels_unchanged=True,future_frames_used=False)
