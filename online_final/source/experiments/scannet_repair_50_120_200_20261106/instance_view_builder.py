from pathlib import Path
import colorsys, hashlib, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
OLD=Path('/data/chenkejun/CVPR/results/scannet_ovimap_strict_dbscan_20261104')
OUT=ROOT/'instance_ply_comparison'
OUT.mkdir(exist_ok=True)
SCENES=['scene0011_00','scene0050_00']
METHODS=['baseline','main','online','main_dbscan','online_dbscan']

def files(scene):
    return {
        'baseline': OLD/'published/no_repair'/scene/'final_B/instance_surface.npz',
        'main': ROOT/'published/main'/scene/'final_B/instance_surface.npz',
        'online': ROOT/'published/online_B'/scene/'final_B/instance_surface.npz',
        'main_dbscan': ROOT/'dbscan/main'/scene/'instance_surface.npz',
        'online_dbscan': ROOT/'dbscan/online_B'/scene/'instance_surface.npz',
    }

def assign_groups(reference, target, reference_groups, next_group):
    rids, rc=np.unique(reference[reference>0],return_counts=True)
    tids, tc=np.unique(target[target>0],return_counts=True)
    mask=(reference>0)&(target>0)
    pairs=np.column_stack((reference[mask],target[mask]))
    pairs, cnt=np.unique(pairs,axis=0,return_counts=True)
    mat=np.zeros((len(rids),len(tids)),dtype=np.int64)
    mat[np.searchsorted(rids,pairs[:,0]),np.searchsorted(tids,pairs[:,1])]=cnt
    rows,cols=linear_sum_assignment(mat,maximize=True)
    groups={}
    for r,c in zip(rows,cols):
        n=mat[r,c]
        if n>=20 and n/min(rc[r],tc[c])>=0.2:
            groups[int(tids[c])]=reference_groups[int(rids[r])]
    for t in tids:
        if int(t) not in groups:
            groups[int(t)]=next_group; next_group+=1
    return groups,next_group

def write_ply(path,xyz,ids,rgb):
    sel=ids>0;n=int(sel.sum())
    dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4')])
    header=('ply\nformat binary_little_endian 1.0\ncomment full-resolution final prediction; colors matched across methods by overlap\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nend_header\n'%n).encode()
    with path.open('wb') as f:
        f.write(header)
        idx=np.flatnonzero(sel)
        for start in range(0,len(idx),400000):
            j=idx[start:start+400000];a=np.empty(len(j),dtype=dt)
            for k,axis in enumerate('xyz'): a[axis]=xyz[j,k]
            for k,col in enumerate(['red','green','blue']): a[col]=rgb[j,k]
            a['instance_id']=ids[j];a.tofile(f)
    return n

def main():
  metadata={'scenes':{},'sampling':'viewer: one deterministic shared surface sample, at most 700000 points; full-resolution PLY contains every positively assigned vertex','matching':'one-to-one maximum-overlap match with at least 20 vertices and 20% of smaller instance; unmatched instances receive distinct display groups; colors do not imply ground-truth identity','version':'PPT unified_full_depth_20261101; main 76eca7b; online-v2 a15f761','evaluation':'object_observed_repair revision 2'}
  for scene in SCENES:
    paths=files(scene);labels={};xyz=None
    for method,path in paths.items():
      with np.load(path,allow_pickle=False) as z:
        x=z['xyz_m']; lab=z['instance_id'].astype(np.int32)
      if xyz is None: xyz=x
      else:
        if not np.array_equal(x,xyz): raise RuntimeError('different geometry '+scene+' '+method)
      labels[method]=lab
      print('READ',scene,method,len(lab),flush=True)
    base=labels['baseline']; ids=np.unique(base[base>0]);group={'baseline':{int(v):i for i,v in enumerate(ids)}}; nxt=len(ids)
    for method in METHODS[1:]: group[method],nxt=assign_groups(base,labels[method],group['baseline'],nxt)
    # Compare unmatched online IDs with unmatched main IDs on the same surface.
    for m,ref in [('online','main'),('main_dbscan','main'),('online_dbscan','online')]:
      target=labels[m]; r=labels[ref]; ok=(target>0)&(r>0)
      pairs,cnt=np.unique(np.column_stack((target[ok],r[ok])),axis=0,return_counts=True)
      tc=np.bincount(target[target>0]);rc=np.bincount(r[r>0])
      order=np.argsort(cnt)[::-1];claimed=set(group[m].values())
      for ix in order:
        tid,rid=map(int,pairs[ix]);g=group[ref][rid]
        if group[m][tid]>=len(ids) and g>=len(ids) and g not in claimed and cnt[ix]>=20 and cnt[ix]/min(tc[tid],rc[rid])>=0.2:
          old=group[m][tid];group[m][tid]=g;claimed.remove(old);claimed.add(g)
    rng=np.random.default_rng(20261106)
    sample=np.sort(rng.choice(len(xyz),min(len(xyz),700000),replace=False))
    centers=np.zeros((nxt,3));counts=np.zeros(nxt)
    for method in METHODS:
      lab=labels[method][sample];good=lab>0;u,inv=np.unique(lab[good],return_inverse=True);g=np.array([group[method][int(i)] for i in u])[inv];pos=xyz[sample[good]]
      np.add.at(counts,g,1)
      for axis in range(3): np.add.at(centers[:,axis],g,pos[:,axis])
    active=np.flatnonzero(counts);centers[active]/=counts[active,None]
    palette=np.array([colorsys.hsv_to_rgb((i*0.618033988749895)%1,0.65+0.22*((i%3)/2),0.72+0.24*((i%4)/3)) for i in range(120)])
    chosen={};tree=cKDTree(centers[active]); order=active[np.argsort(counts[active])[::-1]]
    for g in order:
      _,nearest=tree.query(centers[g],k=min(15,len(active)))
      nearby=[chosen[int(active[i])] for i in np.atleast_1d(nearest) if int(active[i]) in chosen]
      if nearby:
        dist=np.sum((palette[:,None,:]-np.asarray(nearby)[None,:,:])**2,axis=2).min(axis=1);color=palette[np.argmax(dist)]
      else: color=palette[int(g)%len(palette)]
      chosen[int(g)]=color
    for g in range(nxt):
      if g not in chosen: chosen[g]=palette[g%len(palette)]
    sceneout=OUT/scene;sceneout.mkdir(exist_ok=True)
    focus=xyz[base>0][::10]
    metadata['scenes'][scene]={'total_geometry_vertices':len(xyz),'methods':{},'bounds_min':xyz.min(axis=0).tolist(),'bounds_max':xyz.max(axis=0).tolist(),'focus_min':np.quantile(focus,.005,axis=0).tolist(),'focus_max':np.quantile(focus,.995,axis=0).tolist()}
    for method in METHODS:
      lab=labels[method];lookup=np.full((int(lab.max())+1,3),176,dtype=np.uint8)
      for iid,g in group[method].items(): lookup[iid]=np.rint(chosen[g]*255).astype(np.uint8)
      color=np.zeros((len(lab),3),dtype=np.uint8);pos=lab>0;color[pos]=lookup[lab[pos]]
      n=write_ply(sceneout/(method+'.ply'),xyz,lab,color)
      subset=sample[lab[sample]>0]
      with (sceneout/(method+'.bin')).open('wb') as f:
        np.asarray(xyz[subset],dtype='<f4').tofile(f);color[subset].tofile(f)
      metadata['scenes'][scene]['methods'][method]={'points':n,'display_points':len(subset),'instances':len(group[method]),'source':str(paths[method]),'sha256':hashlib.sha256(paths[method].read_bytes()).hexdigest()}
      print('DONE',scene,method,n,len(subset),flush=True)
    (OUT/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
  (OUT/'build_status.json').write_text(json.dumps({'status':'PASS','scenes':SCENES,'methods':METHODS},indent=2))
if __name__=='__main__': main()
