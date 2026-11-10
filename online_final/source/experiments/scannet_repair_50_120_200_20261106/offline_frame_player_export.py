from pathlib import Path
from collections import Counter
import colorsys,gzip,hashlib,json,os,time,traceback
import numpy as np
from PIL import Image

ROOT=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
OLD=Path('/data/chenkejun/CVPR/results/scannet_ovimap_strict_dbscan_20261104')
OUT=ROOT/'main_offline_frame_player'
VIEW=ROOT/'instance_ply_comparison'
SCENES=['scene0011_00','scene0050_00']
LIMIT=350000
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(path,x):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));os.replace(tmp,path)
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def palette(path):
 with path.open('rb') as f:
  while f.readline()!=b'end_header\n':pass
  off=f.tell()
 dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1'),('id','<i4')]);a=np.memmap(path,offset=off,dtype=dt,mode='r');_,idx=np.unique(a['id'],return_index=True)
 return {int(a['id'][i]):[int(a['r'][i]),int(a['g'][i]),int(a['b'][i])] for i in idx}
def make_lut(scene,base):
 colors=palette(VIEW/scene/'baseline.ply');colors.update(palette(VIEW/scene/'main.ply'));lut=np.zeros((base+2,3),dtype=np.uint8);lut[0]=[204,210,217]
 for i in range(1,base+1):lut[i]=colors.get(i,[round(255*x) for x in colorsys.hsv_to_rgb((i*.61803398875)%1,.78,.85)])
 return lut
def overlay(rgb,layerlist,lut):
 label=np.zeros(layerlist[0].shape,np.int32);conflict=np.zeros(label.shape,bool)
 for layer in layerlist:
  mask=layer>0;conflict |= mask&(label>0)&(label!=layer);label[mask]=layer[mask]
 if label.max()>=len(lut):raise RuntimeError('unexpected instance id')
 result=rgb.copy();on=label>0;result[on]=(result[on].astype(np.float32)*.48+lut[label[on]].astype(np.float32)*.52).astype(np.uint8);result[conflict]=[220,35,150]
 return result,int(on.sum()),int(conflict.sum())
def init_ledger(path,base,N):
 z=load(path);p=z['surface_point_index'].astype(np.int64);ids=z['instance_id'].astype(np.int64);keys=p*base+ids;order=np.argsort(keys);keys=keys[order];expected=z['frame_votes'][order].astype(np.uint16)
 return {'keys':keys,'expected':expected,'counts':np.zeros(len(keys),np.uint16),'total':np.zeros(N,np.uint16),'best':np.zeros(N,np.uint16),'top':np.full(N,-1,np.int32)}
def strict_candidates(candidate,actual,base):
 points=candidate//base;uniq,counts=np.unique(points,return_counts=True);single=uniq[counts==1];eff=candidate[np.isin(points,single,assume_unique=False)];
 if not np.array_equal(eff,actual):raise RuntimeError('strict per-frame candidates mismatch')
 return uniq,counts
def accumulate(L,framekeys,base):
 pos=np.searchsorted(L['keys'],framekeys)
 if not np.all(pos<len(L['keys'])) or not np.array_equal(L['keys'][pos],framekeys):raise RuntimeError('frame key absent from saved ledger')
 L['counts'][pos]+=1;pts=(framekeys//base).astype(np.int32);ids=(framekeys%base).astype(np.int32);L['total'][pts]+=1;v=L['counts'][pos];old=L['best'][pts];prior=L['top'][pts];win=(v>old)|((v==old)&((prior<0)|(ids<prior)));q=pts[win];L['best'][q]=v[win];L['top'][q]=ids[win]
def event_for(points,counts,inverse,n):
 idx=inverse[points];valid=idx>=0;event=np.zeros(n,np.uint8);event[idx[valid]]=np.where(counts[valid]>1,2,1).astype(np.uint8);return event
def encoded_state(L,sample,seen):
 total=L['total'][sample];top=L['top'][sample];best=L['best'][sample];ok=(best>=2)&(best>=.67*total)&(top>0)
 out=np.where(seen,1,0).astype('<u2');out[ok]=(top[ok]+1).astype('<u2');return out
def verify(L,path,base):
 if not np.array_equal(L['counts'],L['expected']):raise RuntimeError('replayed 200-frame vote ledger mismatch '+str(path))
 with np.load(path,allow_pickle=False) as z:truth=z['instance_id']
 good=(L['best']>=2)&(L['best']>=.67*L['total'])&(L['top']>0);result=np.where(good,L['top'],-1).astype(np.int32)
 if not np.array_equal(result,truth):raise RuntimeError('replayed native strict label mismatch '+str(path))
 return {'votes_exact':True,'native_labels_exact':True,'confirmed_surface_points':int(good.sum())}

def run(scene):
 start=time.time();out=OUT/scene;out.mkdir(parents=True,exist_ok=True);d=ROOT/'main'/scene/'200/A';repair=d/'repair';mapping=d/'mapping';journal=json.loads((repair/'journal/manifest.json').read_text());receipts={r['frame']:r for r in json.loads((repair/'partial/frame_receipts.json').read_text())};assoc=json.loads((d/'tracking/global_association.json').read_text());base=int(assoc['instance_base']);assert journal['processed_frames']==200 and base<65000
 finalpath=ROOT/'published/main'/scene/'final_B/instance_surface.npz';bfinalpath=OLD/'published/no_repair'/scene/'final_B/instance_surface.npz'
 with np.load(finalpath) as z:xyz=z['xyz_m'];final=z['instance_id']
 N=len(xyz);bs=mapping/'baseline/surface_instance_frame_votes.npz';rs=repair/'partial/whole_to_partial'/scene/'native/surface_instance_frame_votes.npz';bn=mapping/'baseline/native/instance_surface.npz';rn=repair/'partial/whole_to_partial'/scene/'native/instance_surface.npz'
 B=init_ledger(bs,base,N);R=init_ledger(rs,base,N);candidates=np.unique(np.r_[B['keys']//base,R['keys']//base,np.flatnonzero(final>0)]);rng=np.random.default_rng(20261109);sample=np.sort(rng.choice(candidates,min(LIMIT,len(candidates)),replace=False)).astype(np.int32);n=len(sample);inverse=np.full(N,-1,np.int32);inverse[sample]=np.arange(n,dtype=np.int32);np.asarray(xyz[sample],dtype='<f4').tofile(out/'points.bin');lut=make_lut(scene,base);(out/'palette.bin').write_bytes(lut.tobytes());np.where(final[sample]>0,final[sample]+1,0).astype('<u2').tofile(out/'final_B.bin')
 if sha(finalpath)!=json.loads((ROOT/'published/main'/scene/'final_B/scene_finalization.json').read_text())['output_sha256']['instance_surface.npz']:raise RuntimeError('final archive hash mismatch')
 amap={}
 with (mapping/'association/associations.jsonl').open() as f:
  for line in f:
   a=json.loads(line);amap.setdefault(int(a['frame_id']),{})[int(a['mask_local_id'])]=int(a['instance_id'])
 selection=json.loads((OLD/'inputs'/scene/'selection.json').read_text())['original_frames']; seedrows=json.loads((VIEW/'seed_index.json').read_text())['rows'];seedmap={x['track_id']:x for x in seedrows if x['scene']==scene and x['method']=='main' and x['checkpoint']==200}
 cfg=json.loads((d/'tracking/track_config.json').read_text());imagepath=Path(cfg['source']['scene_root'])/'results';maskpath=mapping/'inputs/masks';seen={'baseline':np.zeros(n,bool),'repair':np.zeros(n,bool)};rows=[]
 for i,entry in enumerate(journal['frames']):
  fid=entry['frame_id'];assert fid==i*5;rec=receipts[fid];tx=load(repair/'partial/transactions'/('f%06d.npz'%fid));
  with Image.open(imagepath/('frame%06d.jpg'%fid)) as im:rgb=np.array(im.convert('RGB'))
  with Image.open(maskpath/('frame%06d.png'%fid)) as im:local=np.array(im,dtype=np.int32)
  amapf=amap[fid];mappinglut=np.zeros(max(int(local.max()),max(amapf,default=0))+1,np.int32)
  for k,v in amapf.items():mappinglut[k]=v
  old=mappinglut[local];assert np.all(old[local>0]>0)
  layers=[]
  for j in entry['layer_files']:
   if sha(j['path'])!=j['sha256']:raise RuntimeError('journal layer hash mismatch')
   with Image.open(j['path']) as im:layers.append(np.array(im,dtype=np.int32))
  for variant,layerlist in [('baseline',[old]),('repair',layers)]:
   over,pixels,collision=overlay(rgb,layerlist,lut);Image.fromarray(over).save(out/('%s_%03d.jpg'%(variant,i+1)),quality=86)
   if variant=='repair':repairpixels,collisions=pixels,collision
  data=[];extra={}
  for variant,ledger,rawkey,framekey in [('baseline',B,'baseline_candidate_keys','baseline_frame_keys'),('repair',R,'whole_to_partial_candidate_keys','whole_to_partial_frame_keys')]:
   cand=tx[rawkey];effective=tx[framekey];pts,counts=strict_candidates(cand,effective,base);sampleind=inverse[pts];valid=sampleind>=0;seen[variant][sampleind[valid]]=True;ev=event_for(pts,counts,inverse,n);accumulate(ledger,effective,base);state=encoded_state(ledger,sample,seen[variant]);data.extend([state.tobytes(),ev.tobytes()]);extra[variant]={'effective_votes':len(effective),'conflicting_points':int((counts>1).sum()),'visible_sampled_points':int(seen[variant].sum()),'confirmed_sampled_points':int((state>1).sum())}
  with gzip.open(out/('state_%03d.bin.gz'%(i+1)),'wb',compresslevel=3) as f:
   for part in data:f.write(part)
  accepted=[{'track':t,'mask':seedmap[t]['mask_id'],'kind':seedmap[t]['identity_kind'],'persistent_id':seedmap[t]['persistent_id']} for t in rec['accepted_tracks']]
  rows.append({'frame':i+1,'alias':fid,'original_frame':selection[i],'baseline':extra['baseline'],'repair':extra['repair'],'accepted':accepted,'mask_pixels':repairpixels,'pixel_layer_conflicts':collisions,'relabel_pixels':sum(v['core_relabelled_pixels']+v['expanded_relabelled_pixels'] for v in rec['counts'])})
  if (i+1)%10==0:print('FRAMES',scene,i+1,'/200',round(time.time()-start,1),'s',flush=True)
 check={'baseline':verify(B,bn,base),'repair':verify(R,rn,base)}
 with np.load(bfinalpath) as z:
  if not np.array_equal(xyz,z['xyz_m']):raise RuntimeError('baseline geometry mismatch')
  np.where(z['instance_id'][sample]>0,z['instance_id'][sample]+1,0).astype('<u2').tofile(out/'baseline_final_B.bin')
 checks={'status':'PASS','input_journal_layer_sha256_verified':True,'per_frame_candidate_strict_votes_exact':True,'final_ledger_and_native_exact':check,'archived_final_B_sha256':sha(finalpath),'frame_count':len(rows),'fixed_final_surface_only':True,'baseline_source':str(bs),'repair_source':str(rs),'repair_exception':'scene0011_00: M0211 skipped by user' if scene=='scene0011_00' else None}
 dump(out/'manifest.json',{'status':'PASS','scene':scene,'method':'main offline final 200-frame repair, revision 2','frames':rows,'points':n,'palette_entries':len(lut),'sample_strategy':'fixed surface, deterministic sample of all points in baseline/repaired votes or final B','geometry':'fixed final TSDF surface; prefix is a retrospective cumulative replay, not historical TSDF geometry or online causal output','sample_bbox':{'min':xyz[sample].min(0).tolist(),'max':xyz[sample].max(0).tolist()},'focus_bbox':{'min':json.loads((VIEW/'metadata.json').read_text())['scenes'][scene]['focus_min'],'max':json.loads((VIEW/'metadata.json').read_text())['scenes'][scene]['focus_max']},'verification':checks});dump(out/'audit.json',checks);print('DONE',scene,'points',n,'seconds',round(time.time()-start,1),flush=True)

if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 try:
  for s in SCENES:run(s)
  dump(OUT/'status.json',{'status':'PASS','scenes':SCENES})
 except Exception:
  dump(OUT/'status.json',{'status':'FAILED','traceback':traceback.format_exc()});raise
