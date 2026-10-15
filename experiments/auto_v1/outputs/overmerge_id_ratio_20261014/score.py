"""One predeclared old-ID heuristic. Read-only source, no GT, no inference, no repair."""
from pathlib import Path
import collections,gzip,hashlib,json,sys,time,zipfile
import numpy as np
from PIL import Image
from scipy.ndimage import find_objects

SRC=Path('/data/chenkejun/CVPR/results/e1_roi_top10_20261014')
OUT=Path('/data/chenkejun/CVPR/results/overmerge_id_ratio_20261014')
WEB=OUT/'web'; (WEB/'assets').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/strict_vote_annotation_20261010')
from common import project_visible
OLD_URL='../room2_E1_prerepair_CUT_top10_20261014/'
BASE=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain')

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decision(N,n2):return 'reject' if n2>=10 and n2*5>=N else ('no_support' if N==0 else 'keep')

def main():
 start=time.time();m=read(SRC/'input_manifest.json');viewer=read(SRC/'review/viewer_data.json')
 assert m['no_GT_used'] and not m['map_modified']
 paths=[BASE/'final/instance_surface.npz',BASE/'surface_evidence.npz',SRC/'input_manifest.json']
 before={str(p):sha(p) for p in paths}
 assert before[str(paths[0])]=='3f611dfe701a68259c71b94962ae6bae3c13fbac9c24f1e3eca5b414b8d2d4f3'
 assert before[str(paths[1])]=='b106b99741a4cb115d9ceed0ea3604ed46e0cc9a48d1feb865c2a0a60661c992'
 with np.load(paths[0]) as s,np.load(paths[1]) as e:
  xyz=s['xyz_m'].copy();state=e['state'].copy();pid=e['top1_instance_id'].copy()
  assert np.array_equal(xyz,e['xyz_m']);confirmed=(state==2)&(pid>0)
  assert np.all(s['instance_id'][confirmed]>0)
  assert np.array_equal(s['instance_id'][confirmed],pid[confirmed])
 assert decision(50,10)=='reject' and decision(51,10)=='keep' and decision(40,9)=='keep' and decision(0,0)=='no_support'
 allids=sorted(np.unique(pid[confirmed]).tolist());assert len(allids)<65535
 ci=np.zeros(len(xyz),np.uint16);ci[confirmed]=np.searchsorted(allids,pid[confirmed])+1
 cfg=read('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2/configs/raw.json');cam=cfg['camera']
 scene=Path(cfg['source']['scene_root']);poses=np.loadtxt(scene/cfg['source']['trajectory']).reshape(-1,4,4)
 jobs=[j for j in m['jobs'] if j['kind']=='crop'];assert len(jobs)==412
 byframe=collections.defaultdict(list)
 for j in jobs:byframe[j['frame']].append(j)
 frame_rgb={v['frame']:OLD_URL+v['full_rgb'] for c in viewer['cases'] for v in c['views']}
 data=dict(schema='overmerge-confirmed-id-v1',models=['原图 CropFormer','局部 CropFormer','局部 SAM2.1'],native_ids=allids,
  sources=[],jobs=[],candidates=[],frames={},cases=[],rule=dict(ratio=.20,min_second_points=10,denominator='all unique visible TSDF vertices inside mask',state=2),
  source_hashes=before,old_page=OLD_URL+'index.html')
 for c in viewer['cases']:
  data['cases'].append({k:c[k] for k in ['uid','ROI','aliases','counts','rankings']})
 candidates=data['candidates'];unique={};checked=[];input_hashes={};occurrences=0
 for frame,frame_jobs in sorted(byframe.items()):
  depth=np.asarray(Image.open(scene/cfg['source']['depth_pattern'].format(frame=frame)),np.float64)/cam['depth_png_units_per_meter']
  visible,uv=project_visible(xyz,poses[frame],depth,cam);idx=np.flatnonzero(visible);xy=uv[idx];cc=ci[idx];st=state[idx]
  points=np.column_stack([xy,cc]).astype('<u2');pointfile=f'assets/frame{frame:06d}_points.bin.gz'
  with gzip.open(WEB/pointfile,'wb',compresslevel=6) as f:f.write(points.tobytes())
  np.savez_compressed(OUT/f'frame{frame:06d}_visible.npz',vertex_index=idx.astype(np.int32),uv=xy,confirmed_native_id=np.where(confirmed[idx],pid[idx],0),state=st)
  data['frames'][str(frame)]=dict(rgb=frame_rgb[frame],points=pointfile,visible_vertices=len(idx),wh=[1200,680])
  basepath=Path('/data/chenkejun/CVPR/datasets/cropformer_frozen/room2')/f'frame{frame:06d}.png'
  base=np.asarray(Image.open(basepath),np.uint16);assert base.shape==(680,1200)
  input_hashes[str(basepath)]=sha(basepath)
  encoded=np.zeros((680,1200,3),np.uint8);encoded[:,:,0]=base%256;encoded[:,:,1]=base//256
  bfile=f'assets/frame{frame:06d}_original_ids.png';Image.fromarray(encoded).save(WEB/bfile)
  cache={}
  def process(labels,box,model,url,source_path):
   # Cache exact same array and footprint. Dedup at the individual mask level below.
   sig=(model,tuple(box),hashlib.sha256(labels.tobytes()).hexdigest())
   if sig in cache:return cache[sig]
   x0,y0,x1,y1=box;inside=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
   localxy=xy[inside]-[x0,y0];lv=labels[localxy[:,1],localxy[:,0]].astype(np.int64)
   maxid=int(labels.max());num=len(allids)+1
   hist=np.bincount(lv*num+cc[inside],minlength=(maxid+1)*num).reshape(maxid+1,num)
   shist=np.bincount(lv*4+st[inside],minlength=(maxid+1)*4).reshape(maxid+1,4)
   area=np.bincount(labels.ravel(),minlength=maxid+1);slices=find_objects(labels);mapping={};source_id=len(data['sources'])
   for lid,sl in enumerate(slices,1):
    if sl is None:continue
    sy,sx=sl;b=[x0+sx.start,y0+sy.start,x0+sx.stop,y0+sy.stop];mask=labels[sl]==lid
    digest=hashlib.sha256(json.dumps([frame,model,b]).encode()+np.packbits(mask).tobytes()).hexdigest()
    N=int(hist[lid].sum());pairs=sorted([[allids[k-1],int(hist[lid,k])] for k in range(1,num) if hist[lid,k]],key=lambda a:(-a[1],a[0]))
    n2=pairs[1][1] if len(pairs)>1 else 0;d=decision(N,n2)
    if digest in unique:
     c=candidates[unique[digest]];assert c['N']==N and c['ids']==pairs and c['states']==shist[lid].tolist()
    else:
     cid=len(candidates);unique[digest]=cid
     # Border is only a visual warning, never an additional exclusion rule.
     c=dict(index=cid,id=digest[:20],frame=frame,model=model,source=source_id,label=lid,box=b,pixels=int(area[lid]),
      N=N,ids=pairs,states=shist[lid].tolist(),unconfirmed=int(hist[lid,0]),n2=n2,r2=n2/N if N else 0,decision=d,
      touches_crop=bool(model>0 and ((b[0]==x0 and x0>0) or (b[1]==y0 and y0>0) or (b[2]==x1 and x1<1200) or (b[3]==y1 and y1<680))),
      touches_frame=bool(b[0]==0 or b[1]==0 or b[2]==1200 or b[3]==680),occ=[])
     candidates.append(c)
    mapping[str(lid)]=c['index']
   data['sources'].append(dict(model=model,frame=frame,box=box,ids=url,mapping=mapping,path=str(source_path)))
   cache[sig]=source_id
   return source_id
  # Only original labels occurring in a selected crop are candidates for this batch.
  original=process(base,[0,0,1200,680],0,bfile,basepath)
  for j in frame_jobs:
   x0,y0,x1,y1=j['crop_xyxy'];crop=viewer['crops'][j['key']]
   assert np.array_equal(np.asarray(Image.open(j['baseline_file'])),base[y0:y1,x0:x1])
   js=dict(key=j['key'],uid=j['uid'],ROI=j['ROI'],view=j['view'],frame=frame,strategy=j['strategy'],box=j['crop_xyxy'],sources=[original],members=[])
   for model,folder in [(1,'cropformer'),(2,'sam2_1')]:
    path=SRC/folder/(j['key']+'.png');rec=read(path.with_suffix('.json'))
    h=sha(path);assert h==rec['mask_sha256'] and rec['RGB_sha256']==j['RGB_sha256'];input_hashes[str(path)]=h
    labels=np.asarray(Image.open(path),np.uint16);assert labels.shape==(y1-y0,x1-x0)
    sid=process(labels,j['crop_xyxy'],model,OLD_URL+crop['models'][model]['ids_file'],path);js['sources'].append(sid)
   job_index=len(data['jobs'])
   for model,sid in enumerate(js['sources']):
    mapping=data['sources'][sid]['mapping']
    lids=np.unique(base[y0:y1,x0:x1]) if model==0 else np.array(list(map(int,mapping)))
    members=[]
    for lid in lids:
     if lid==0:continue
     cid=mapping[str(lid)];members.append(cid);candidates[cid]['occ'].append([job_index,int(lid)]);occurrences+=1
    js['members'].append(members)
   data['jobs'].append(js)
  print('FRAME',frame,'visible',len(idx),'jobs',len(data['jobs']),'candidates',len(candidates),flush=True)
  # Independent direct membership recount for every 100th candidate and every threshold boundary.
  frame_candidates=[c for c in candidates if c['frame']==frame]
  for c in frame_candidates:
   if c['index']%100 and not (8<=c['n2']<=12 or (c['N'] and abs(c['r2']-.2)<.005)):continue
   source=data['sources'][c['source']];arr=np.asarray(Image.open(source['path']));x0,y0,x1,y1=source['box']
   okay=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1);ii=np.flatnonzero(okay)
   picked=ii[arr[xy[ii,1]-y0,xy[ii,0]-x0]==c['label']]
   actual=idx[picked];assert len(actual)==c['N'] and len(np.unique(actual))==c['N']
   vals=pid[actual[confirmed[actual]]];a,b=np.unique(vals,return_counts=True)
   pairs=sorted([[int(x),int(y)] for x,y in zip(a,b)],key=lambda p:(-p[1],p[0]));assert pairs==c['ids']
   checked.append(c['id'])
 # Remove whole-frame original labels never visible in any selected crop; preserve stable array indexes via remapping.
 active=[c for c in candidates if c['occ']];remap={c['index']:i for i,c in enumerate(active)}
 for i,c in enumerate(active):c['index']=i
 data['candidates']=active
 for s in data['sources']:s['mapping']={lid:remap[cid] for lid,cid in s['mapping'].items() if cid in remap}
 for j in data['jobs']:j['members']=[[remap[cid] for cid in ids] for ids in j['members']]
 assert occurrences==18303 and sum(len(c['occ']) for c in active)==occurrences
 stats={}
 for model in range(3):
  cs=[c for c in active if c['model']==model]
  stats[str(model)]=dict(unique=len(cs),occurrences=sum(len(c['occ']) for c in cs),reject=sum(c['decision']=='reject' for c in cs),
   keep=sum(c['decision']=='keep' for c in cs),no_support=sum(c['decision']=='no_support' for c in cs))
 data['stats']=dict(models=stats,unique=len(active),occurrences=occurrences,reject=sum(c['decision']=='reject' for c in active),
  keep=sum(c['decision']=='keep' for c in active),no_support=sum(c['decision']=='no_support' for c in active),frames=len(byframe),jobs=len(jobs),ROIs=19,rank_entries=30,view_entries=90)
 write(WEB/'data.json',data)
 for name,sel in [('proposed_reject',[c for c in active if c['decision']=='reject']),('retained',[c for c in active if c['decision']!='reject'])]:
  write(OUT/(name+'.json'),dict(rule=data['rule'],source_hashes=before,candidates=sel,original_masks_modified=False))
 assert {str(p):sha(p) for p in paths}==before
 for path,h in input_hashes.items():assert sha(path)==h
 audit=dict(status='PASS',stats=data['stats'],confirmed_final_native_mapping_verified=True,all_source_hashes_unchanged=True,
  no_GT_used=True,no_model_inference=True,no_map_write=True,unique_TSDF_vertices=True,denominator_includes_all_states=True,
  independent_recounts=len(checked),independent_checked_ids=checked,code_sha256=sha(__file__),seconds=round(time.time()-start,2),
  original_masks='full-frame frozen masks, only IDs occurring in the selected crops',
  local_masks='existing selectable mutually exclusive label maps; SAM2 overlapping raw proposals are outside this audit',
  dedup='same frame + same model + identical full-image binary mask; never merges different frames or models')
 write(OUT/'audit.json',audit);write(WEB/'audit.json',audit);write(OUT/'input_hashes.json',input_hashes)
 files={str(p.relative_to(WEB)):sha(p) for p in WEB.rglob('*') if p.is_file()};write(WEB/'files_sha256.json',files)
 with zipfile.ZipFile(OUT/'web_assets.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
  for p in WEB.rglob('*'):
   if p.is_file():z.write(p,str(p.relative_to(WEB)))
 print(json.dumps(data['stats']),flush=True);print('COMPLETE',round(time.time()-start,2),flush=True)

if __name__=='__main__':main()
