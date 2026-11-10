from pathlib import Path
import colorsys, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from PIL import Image, ImageFilter

ROOT=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
OLD=Path('/data/chenkejun/CVPR/results/scannet_ovimap_strict_dbscan_20261104')
OUT=ROOT/'instance_ply_comparison'
SCENES=['scene0011_00','scene0050_00']
COLORS={'added':(0,153,105),'removed':(226,49,68),'reassigned':(240,133,0)}
def paths(s):
 return {'baseline':OLD/'published/no_repair'/s/'final_B/instance_surface.npz','main':ROOT/'published/main'/s/'final_B/instance_surface.npz','online':ROOT/'published/online_B'/s/'final_B/instance_surface.npz','main_dbscan':ROOT/'dbscan/main'/s/'instance_surface.npz','online_dbscan':ROOT/'dbscan/online_B'/s/'instance_surface.npz'}
def write_bin(path,xyz,col):
 with path.open('wb') as f: np.asarray(xyz,dtype='<f4').tofile(f);np.asarray(col,dtype=np.uint8).tofile(f)
def read_palette(path):
 with path.open('rb') as f:
  while f.readline()!=b'end_header\n': pass
  off=f.tell()
 dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1'),('id','<i4')]);a=np.memmap(path,offset=off,dtype=dt,mode='r'); _,ind=np.unique(a['id'],return_index=True)
 return {int(a['id'][i]):[int(a['r'][i]),int(a['g'][i]),int(a['b'][i])] for i in ind}
def overlap_groups(base,other):
 b,bc=np.unique(base[base>0],return_counts=True); t,tc=np.unique(other[other>0],return_counts=True);both=(base>0)&(other>0)
 pairs,counts=np.unique(np.column_stack((base[both],other[both])),axis=0,return_counts=True);m=np.zeros((len(b),len(t)),np.int64)
 m[np.searchsorted(b,pairs[:,0]),np.searchsorted(t,pairs[:,1])]=counts;rr,cc=linear_sum_assignment(m,maximize=True);lookup=np.full(int(other.max())+1,-1,dtype=np.int32)
 for r,c in zip(rr,cc):
  if m[r,c]>=20 and m[r,c]/min(bc[r],tc[c])>=0.2:lookup[t[c]]=b[r]
 return lookup
def write_change_ply(path,xyz,category):
 dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('change_type','u1')]);n=int((category>0).sum());indices=np.flatnonzero(category)
 with path.open('wb') as f:
  f.write(('ply\nformat binary_little_endian 1.0\ncomment change_type 1 added 2 removed 3 reassigned; approximate matching\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar change_type\nend_header\n'%n).encode());pal=np.array([[0,0,0],COLORS['added'],COLORS['removed'],COLORS['reassigned']],dtype=np.uint8)
  for start in range(0,n,400000):
   q=indices[start:start+400000];a=np.empty(len(q),dtype=dt)
   for k,x in enumerate('xyz'):a[x]=xyz[q,k]
   for k,x in enumerate(['red','green','blue']):a[x]=pal[category[q],k]
   a['change_type']=category[q];a.tofile(f)

def build_changes_and_ovi(meta):
 rng=np.random.default_rng(20261108)
 for scene in SCENES:
  directory=OUT/scene;fs=paths(scene)
  with np.load(fs['baseline']) as z:xyz=z['xyz_m'];base=z['instance_id']
  baseidx=np.flatnonzero(base>0);context=rng.choice(baseidx,min(len(baseidx),50000),replace=False)
  write_bin(directory/'context.bin',xyz[context],np.tile(np.array([[200,207,215]],dtype=np.uint8),(len(context),1)))
  info={'context_points':len(context),'methods':{}}
  for method in ['main','online','main_dbscan','online_dbscan']:
   with np.load(fs[method]) as z:
    if not np.array_equal(xyz,z['xyz_m']):raise ValueError('geometry mismatch')
    v=z['instance_id']
   matching=overlap_groups(base,v);g=np.full(len(v),-1,dtype=np.int32);q=v>0;g[q]=matching[v[q]]
   cat=np.zeros(len(v),dtype=np.uint8);cat[(base<=0)&q]=1;cat[(base>0)&~q]=2;cat[(base>0)&q&(g!=base)]=3
   values={}
   for num,name in [(1,'added'),(2,'removed'),(3,'reassigned')]:
    idx=np.flatnonzero(cat==num);n=len(idx);s=rng.choice(idx,min(n,130000),replace=False) if n else idx
    write_bin(directory/(method+'_'+name+'.bin'),xyz[s],np.tile(np.array([COLORS[name]],dtype=np.uint8),(len(s),1)))
    values[name]={'points':n,'display_points':len(s)}
   write_change_ply(directory/(method+'_changes.ply'),xyz,cat);info['methods'][method]=values
   print('CHANGE',scene,method,values,flush=True)
  of=OLD/'ovimap'/scene/'200/ovi/f000995.npz'
  with np.load(of) as z: ox=z['xyz_m'];ol=z['instance_id']
  good=ol>0;oi=np.flatnonzero(good);take=rng.choice(oi,min(len(oi),300000),replace=False)
  tree=cKDTree(xyz[baseidx]);dist,nn=tree.query(ox[take],k=1,distance_upper_bound=.02,workers=4);ok=np.isfinite(dist);b=base[baseidx[nn[ok]]];v=ol[take[ok]]
  bi,bc=np.unique(b,return_counts=True);vi,vc=np.unique(v,return_counts=True);pp,cc=np.unique(np.column_stack((b,v)),axis=0,return_counts=True);m=np.zeros((len(bi),len(vi)),np.int64)
  if len(pp):m[np.searchsorted(bi,pp[:,0]),np.searchsorted(vi,pp[:,1])]=cc
  rr,ccc=linear_sum_assignment(m,maximize=True);basepal=read_palette(directory/'baseline.ply');ovipal={}
  for r,c in zip(rr,ccc):
   if m[r,c]>=20 and m[r,c]/min(bc[r],vc[c])>=.2:ovipal[int(vi[c])]=basepal[int(bi[r])]
  for iid in np.unique(ol[good]):
   if int(iid) not in ovipal:ovipal[int(iid)]=[round(255*x) for x in colorsys.hsv_to_rgb((int(iid)*.61803398875)%1,.8,.9)]
  lut=np.zeros((int(ol.max())+1,3),dtype=np.uint8)
  for k,c in ovipal.items():lut[k]=c
  display=rng.choice(oi,min(len(oi),250000),replace=False);write_bin(directory/'ovimap.bin',ox[display],lut[ol[display]])
  # This PLY is the full OVI-MAP evaluation surface; its original triangle mesh is linked separately.
  import importlib.util
  spec=importlib.util.spec_from_file_location('builder','/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106/instance_view_builder.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  rgb=np.zeros((len(ol),3),np.uint8);rgb[good]=lut[ol[good]];mod.write_ply(directory/'ovimap.ply',ox,ol,rgb)
  mesh=directory/'ovimap_original_mesh.ply';original=OLD/'ovimap'/scene/'200/cropformer_inst/instance_mesh_200.ply'
  if not mesh.exists():mesh.symlink_to(original)
  meta['scenes'][scene]['methods']['ovimap']={'points':len(oi),'display_points':len(display),'instances':len(ovipal),'source':str(of),'spatial_color_matching_distance_m':.02,'nearby_sample_fraction':float(ok.mean()),'original_mesh':str(original)}
  meta['scenes'][scene]['changes']=info
  print('OVIMAP',scene,'points',len(oi),'nearby',round(ok.mean(),3),flush=True)
  (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))

def build_seeds():
 out=OUT/'seeds';out.mkdir(exist_ok=True);rows=[]
 for scene in SCENES:
  mapping=json.loads((OLD/'inputs'/scene/'selection.json').read_text())['original_frames']
  for method in ['main','online']:
   for cp in ([200] if method=='main' else [50,120,200]):
    d=ROOT/method/scene/str(cp)/('A' if method=='main' else 'B')/'tracking'; manifest=json.loads((d/'seed_manifest.json').read_text());gate=json.loads((d/'gate_complete.json').read_text())['tracks']; conf=json.loads((d/'track_config.json').read_text());frame_root=Path(conf['source']['scene_root'])/'results';dest=out/scene/method/str(cp);dest.mkdir(parents=True,exist_ok=True)
    for case in manifest['seeds']:
     frame=case['source_choice']['frame'];img=Image.open(frame_root/('frame%06d.jpg'%frame)).convert('RGB');prepared=Image.open(d/case['seed_file']);arr=np.asarray(prepared);orig=dest/('frame%06d.jpg'%frame)
     if not orig.exists():img.save(orig,quality=88)
     for obj in case['objects']:
      track=int(obj['track_id']); mask=(arr==track); n=int(mask.sum())
      if n==0:raise ValueError('missing exact tracking seed '+str((scene,method,cp,case['case_uid'],track)))
      name=case['case_uid']+'-T'+str(track); overlay=dest/(name+'.jpg');binary=dest/(name+'.png')
      maskim=Image.fromarray(mask.astype(np.uint8)*255,'L');maskim.save(binary,optimize=True)
      arrgb=np.asarray(img).copy();arrgb[mask]=(arrgb[mask].astype(np.float32)*.46+np.array([0,180,255])*.54).astype(np.uint8)
      edge=np.asarray(maskim.filter(ImageFilter.MaxFilter(3)))>np.asarray(maskim.filter(ImageFilter.MinFilter(3)));arrgb[edge]=[0,255,255];Image.fromarray(arrgb).save(overlay,quality=88)
      g=gate[str(track)];rows.append({'scene':scene,'method':method,'checkpoint':cp,'processed_frame':frame//5+1,'alias_frame':frame,'original_frame':mapping[frame//5],'case':case['case_uid'],'mask_id':obj['mask_id'],'group_id':obj['group_id'],'track_id':track,'prepared_pixels':n,'accepted_frames':g['accepted_mapping_frames'],'projected_points':obj.get('projected_seed_points'),'original':str(orig.relative_to(OUT)),'overlay':str(overlay.relative_to(OUT)),'mask':str(binary.relative_to(OUT))})
    print('SEEDS',scene,method,cp,len([r for r in rows if r['scene']==scene and r['method']==method and r['checkpoint']==cp]),flush=True)
 rows.sort(key=lambda r:(r['scene'],r['method'],r['checkpoint'],-r['accepted_frames'],r['processed_frame'],r['mask_id']))
 (OUT/'seed_index.json').write_text(json.dumps({'rows':rows,'note':'actual prepared_seed mask fed into tracking; accepted_frames is gate-passed frames, not proof of changed final points; M0211 skipped'},ensure_ascii=False))

if __name__=='__main__':
 meta=json.loads((OUT/'metadata.json').read_text());build_seeds();build_changes_and_ovi(meta)
 (OUT/'changes_build_status.json').write_text(json.dumps({'status':'PASS','scenes':SCENES}))
