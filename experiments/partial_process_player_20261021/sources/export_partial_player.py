"""Read-only partial replay export. Saved transactions are the authority.

All source-pixel reconstructions must match saved candidates. Each stage ledger
and native map is checked exactly, followed by complete inverse replay.
"""
from pathlib import Path
import gzip,hashlib,io,json,os,sys,time,zipfile
import numpy as np
from PIL import Image

C0=Path('/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021')
sys.path.insert(0,str(C0))
import bootstrap
import run_history_core as h
from instance_repair.cli import load_script
obs=load_script('_partial_player_observation_ops',bootstrap.SOURCE_CODE/'observation_ops.py')
GROUP=bootstrap.GROUP;SCENE=bootstrap.SCENE;SRC=bootstrap.SOURCE
PARTIAL=Path(bootstrap.CFG['partial_transactions']).parent
ROOT=Path('/data/chenkejun/CVPR/results/partial_process_player_20261021')
VAR=GROUP+'_'+SCENE;OUT=ROOT/'assets'/VAR
STAGES=['direct','core_all','whole_to_partial']
LOCK={};read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False))
def load(p,expected=None):
 b=Path(p).read_bytes();s=hashlib.sha256(b).hexdigest()
 if expected:assert s==expected,str(p)
 LOCK[str(p)]=s
 with np.load(io.BytesIO(b),allow_pickle=False) as z:return dict(z)
def bundle(p,arrays):
 desc={};blocks=[];offset=0
 for k,a in arrays.items():
  a=np.ascontiguousarray(a);pad=(-offset)%4
  if pad:blocks.append(bytes(pad));offset+=pad
  desc[k]=dict(dtype=a.dtype.str,shape=a.shape,offset=offset,length=a.size)
  b=a.tobytes();blocks.append(b);offset+=len(b)
 header=json.dumps(desc,separators=(',',':')).encode();header+=b' '*((-len(header))%4)
 p.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(p,'wb',compresslevel=5) as f:
  f.write(np.array([len(header)],'<u4').tobytes());f.write(header)
  for b in blocks:f.write(b)
 return dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))
def packed(p,a):
 x=np.zeros((*a.shape,3),np.uint8);x[:,:,0]=a%256;x[:,:,1]=a//256
 p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(x).save(p,optimize=True)
def cat(xs,dtype=np.int64):return np.concatenate(xs).astype(dtype) if xs else np.empty(0,dtype)
def subset(k,pts,base):return k[np.isin(k//base,pts)]
def pair(folder,base,hashes):
 p=folder/'native/surface_instance_frame_votes.npz'
 if not p.exists():p=folder/'surface_instance_frame_votes.npz'
 z=load(p,hashes.get(str(p)));k=z['surface_point_index'].astype(np.int64)*base+z['instance_id'];o=np.argsort(k)
 return k[o],z['frame_votes'][o].astype(np.int32)
def state_at(keys,counts,points,base):
 kp=keys//base;lo=np.searchsorted(kp,points);hi=np.searchsorted(kp,points,side='right');sz=hi-lo;valid=sz>0
 labels=np.full(len(points),-1,np.int16);states=np.zeros(len(points),np.uint8)
 if not valid.any():return labels,states
 lo=lo[valid];sz=sz[valid];st=np.r_[0,np.cumsum(sz)[:-1]];ix=np.repeat(lo-st,sz)+np.arange(sz.sum())
 val=counts[ix];best=np.maximum.reduceat(val,st);total=np.add.reduceat(val,st);g=np.repeat(np.arange(len(st)),sz);pos=np.arange(len(ix))
 first=np.minimum.reduceat(np.where(val==best[g],pos,len(ix)),st);second=np.maximum.reduceat(np.where(pos!=first[g],val,0),st)
 ratio=np.divide(best,total,out=np.zeros(len(best),float),where=total>0)
 state=np.where(total==0,0,np.where((best>=2)&(ratio>=.67),2,np.where((second>0)&(ratio<.67),3,1)))
 labels[valid]=np.where(state==2,keys[ix[first]]%base,-1);states[valid]=state
 return labels,states
def main():
 started=time.time();OUT.mkdir(parents=True,exist_ok=True)
 assert not (OUT/'complete.json').exists(),'Do not overwrite a completed export'
 cases,assoc,obj,info,canonical,fixed=h.catalog(SRC);base=int(assoc['instance_base'])
 scene=h.r.Scene(SCENE);xyz=scene.xyz;N=len(xyz);assert N*base<2**32
 decisions={d['frame']:d for d in h.read(SRC/'validated/full_track/frame_decisions.json')['frames']}
 freeze=read(PARTIAL/'predictions_freeze.json');hashes=freeze['output_sha256'];assert not freeze['GT_read_by_mapper']
 baseline=read(SRC/'original_repair/predictions_freeze.json')['scenes'][SCENE]['predictions']
 B=Path(baseline['strict_baseline_final']['path']).parent.parent
 folders=[B]+[PARTIAL/s/SCENE for s in STAGES];pairs=[];evs=[];maps=[]
 for folder in folders:
  p=folder/'native/surface_evidence.npz'
  if not p.exists():p=folder/'surface_evidence.npz'
  ev=load(p,hashes.get(str(p)));np.testing.assert_array_equal(xyz,ev['xyz_m']);evs.append(ev);pairs.append(pair(folder,base,hashes))
  p=folder/'final/instance_surface.npz';m=load(p,hashes.get(str(p)));np.testing.assert_array_equal(xyz,m['xyz_m']);maps.append(m['instance_id'])
 reg=load(SRC/'e1/region.npz');prior=reg['prior_owner'];owner=reg['owner'];ap=np.flatnonzero((prior!=-1)|(owner!=-1))
 can2pid=np.zeros(max(fixed)+1,np.int64)
 for cid,pid in fixed.items():can2pid[cid]=pid
 groups={}
 for a in assoc['objects']:
  pid=int(a['persistent_id']);members=a['member_track_ids'];seedrows=[]
  for tid in members:
   cfg=info[obj[tid]]['config'];seedrows.append(dict(track=int(tid),frame=int(cfg['seed_frame']),oldFamily=cfg['old_family_ids'],group='G'+str(tid).zfill(3)))
  groups[pid]=dict(id=pid,tracks=members,seeds=seedrows,name=' / '.join(x['group'] for x in seedrows),gates=[])
 chunks=[[] for _ in range(3)];removed=[[] for _ in range(3)];added=[[] for _ in range(3)];events=[[] for _ in range(3)];cams=[]
 for fi,fid in enumerate(range(0,2000,5)):
  frame=scene.source.load_frame(fid);d=decisions[fid];tx=load(PARTIAL/'transactions'/f'f{fid:06d}.npz')
  table=np.zeros(int(frame.mask_local.max())+1,np.int64)
  for mid in np.unique(frame.mask_local):
   if mid:table[mid]=scene.lookup[(fid,int(mid))]
  p,mid,u,v=h.sourcepixels(frame,scene);old=table[mid]
  np.testing.assert_array_equal(h.keys(p,old,base),tx['baseline_candidate_keys'])
  revised=dict(d,old_observation_plans={k:'partial' for k in d.get('old_observation_plans',{})})
  residual,rp,rl,np_,nc=h.reconstruct_current(frame,table,scene,revised,obj,info,canonical,fixed)
  maskpath=SRC/'validated/full_track/accepted_masks'/f'f{fid:06d}.png';h.protect(maskpath)
  merged=np.asarray(Image.open(maskpath));mp,mc=h.project(frame,merged,scene)
  np.testing.assert_array_equal(h.keys(mp,mc,max(fixed)+1),h.keys(np_,nc,max(fixed)+1))
  adopted=can2pid[merged];available=residual[v,u]==mid;newkeys=h.keys(np_,can2pid[nc],base)
  core=old.copy();choose=available&(prior[p]>0)&(old!=prior[p]);core[choose]=prior[p[choose]]
  vis=np.zeros(N,bool);vis[ap],_,_=obs.visible_samples(xyz[ap],frame.camera_to_world,frame.camera,frame.depth_m,.02)
  expanded,_=obs.revise_all_history(p,old,available,prior,owner,vis)
  ids=[old,core,expanded];cam=frame.camera;pose=frame.camera_to_world
  cameras=dict(frame=fid,pose=pose.tolist(),camera={k:float(getattr(cam,k)) for k in ['fx','fy','cx','cy']});cams.append(cameras)
  for j,stage in enumerate(STAGES):
   prefix='baseline' if j==0 else 'partial_'+STAGES[j-1];nextp='partial_'+stage
   before=tx[prefix+'_candidate_keys'];after=tx[nextp+'_candidate_keys'];bef=tx[prefix+'_frame_keys'];aft=tx[nextp+'_frame_keys']
   np.testing.assert_array_equal(np.union1d(h.keys(p[available],ids[j][available],base),newkeys),after,err_msg=f'{VAR} {fid} {stage} source pixels')
   np.testing.assert_array_equal(obs.strict_frame(before,base),bef);np.testing.assert_array_equal(obs.strict_frame(after,base),aft)
   rem=np.setdiff1d(bef,aft);add=np.setdiff1d(aft,bef);cr=np.setdiff1d(before,after);ca=np.setdiff1d(after,before)
   touched=np.unique(np.r_[cr//base,ca//base,np_ if j==0 else []]).astype(np.int64)
   bc=subset(before,touched,base);ac=subset(after,touched,base)
   pts=xyz[touched];camera_xyz=(pts-pose[:3,3])@pose[:3,:3];positive=camera_xyz[:,2]>0
   uv=np.full((len(touched),2),-1,np.int16);ii=np.flatnonzero(positive)
   uu=np.rint(cam.fx*camera_xyz[ii,0]/camera_xyz[ii,2]+cam.cx).astype(int);vv=np.rint(cam.fy*camera_xyz[ii,1]/camera_xyz[ii,2]+cam.cy).astype(int)
   inside=(uu>=0)&(uu<1200)&(vv>=0)&(vv<680);ii=ii[inside];uu=uu[inside];vv=vv[inside]
   dd=frame.depth_m[vv,uu];ok=np.isfinite(dd)&(dd>0)&(dd<10)&(np.abs(dd-camera_xyz[ii,2])<=.02);uv[ii[ok]]=np.c_[uu[ok],vv[ok]]
   if j==0:sel=~available;oldids=old;newids=np.zeros(len(p),np.int64)
   else:oldids=ids[j-1];newids=ids[j];sel=available&(oldids!=newids)
   sp=np.column_stack((p[sel],u[sel],v[sel],oldids[sel],newids[sel])).astype('<u4')
   evindex=j*400+fi;receipt=bundle(OUT/'detail'/f'{evindex:04d}.bin.gz',{'point':touched.astype('<u4'),'uv':uv,'before':bc.astype('<u4'),'after':ac.astype('<u4'),'source':sp})
   bp,ct=np.unique(bc//base,return_counts=True);bconf=bp[ct>1];ap_,ct=np.unique(ac//base,return_counts=True);aconf=ap_[ct>1]
   byid={}
   for pid,g in groups.items():
    related=np.unique(np.r_[bc[bc%base==pid]//base,ac[ac%base==pid]//base])
    byid[str(pid)]=dict(points=len(related),removed=int(np.isin(rem//base,related).sum()),added=int(np.isin(add//base,related).sum()),conflicts=int(np.isin(aconf,related).sum()),candidateRemoved=int((cr%base==pid).sum()),candidateAdded=int((ca%base==pid).sum()))
   events[j].append(dict(stage=j,frame=fid,removed=len(rem),added=len(add),candidateRemoved=len(cr),candidateAdded=len(ca),points=len(touched),conflicts=len(aconf),newConflicts=len(np.setdiff1d(aconf,bconf)),sourcePixels=len(sp),byId=byid,detailBytes=receipt['bytes']))
   removed[j].append(rem);added[j].append(add)
  file=f'f{fid:06d}'
  rgbpath=ROOT/'assets'/SCENE/'rgb'/(file+'.jpg');rgbpath.parent.mkdir(parents=True,exist_ok=True)
  if not rgbpath.exists():Image.fromarray(frame.rgb).save(rgbpath,quality=89)
  packed(OUT/'original'/(file+'.png'),table[frame.mask_local]);packed(OUT/'adopted'/(file+'.png'),adopted)
  retired=np.where((frame.mask_local>0)&(residual==0),table[frame.mask_local],0);packed(OUT/'retired'/(file+'.png'),retired)
  for pid,g in groups.items():
   tids=g['tracks'];rows=[d['objects'][str(t)] for t in tids];best=max(rows,key=lambda x:x.get('visible_seed_samples',0))
   g['gates'].append(dict(accepted=any(t in d['accepted_track_ids'] for t in tids),acceptedTracks=[t for t in tids if t in d['accepted_track_ids']],coverage=best.get('tracked_coverage',0),visible=best.get('visible_seed_samples',0),bbox=best.get('new_surface_bbox_fraction',0),reasons=sorted({r for x in rows for r in x.get('reasons',[])})))
  if fi%25==0:print(json.dumps(dict(variant=VAR,phase='evidence',frames=fi+1,seconds=round(time.time()-started,1))),flush=True)
 for g in groups.values():
  for seed in g['seeds']:
   f=scene.source.load_frame(seed['frame']);mask=np.asarray(Image.open(info[obj[seed['track']]]['seedpath']))==seed['track'];im=f.rgb.copy();im[mask]=(im[mask]*.55+np.array([30,210,168])*.45).astype(np.uint8)
   dest=OUT/'seeds'/f"t{seed['track']}.jpg";dest.parent.mkdir(exist_ok=True);Image.fromarray(im).save(dest,quality=91)
 allrem=sum(removed,[]);alladd=sum(added,[]);allEvents=sum(events,[])
 keys=np.unique(np.r_[pairs[0][0],cat(allrem),cat(alladd)]);counts=np.zeros(len(keys),np.int32);counts[np.searchsorted(keys,pairs[0][0])]=pairs[0][1]
 label=np.where(evs[0]['state']==2,evs[0]['top1_instance_id'],-1).astype('<i2');state=evs[0]['state'].astype('u1').copy()
 arrays=dict(xyz=xyz.astype('<f4'),initialLabel=label.copy(),initialState=state.copy(),voteKey=pairs[0][0].astype('<u4'),voteCount=pairs[0][1].astype('<u2'))
 cp=[];cl0=[];cl1=[];cs0=[];cs1=[];offs={k:[0] for k in 'rap'};checks=[]
 for e,(rem,add) in enumerate(zip(allrem,alladd)):
  touched=np.unique(np.r_[rem//base,add//base]);oldlabel=label[touched].copy();oldstate=state[touched].copy()
  counts[np.searchsorted(keys,rem)]-=1;counts[np.searchsorted(keys,add)]+=1;assert (counts[np.searchsorted(keys,np.r_[rem,add])]>=0).all()
  newlabel,newstate=state_at(keys,counts,touched,base);label[touched]=newlabel;state[touched]=newstate
  changed=(oldlabel!=newlabel)|(oldstate!=newstate);cp.append(touched[changed]);cl0.append(oldlabel[changed]);cl1.append(newlabel[changed]);cs0.append(oldstate[changed]);cs1.append(newstate[changed])
  for k,value in [('r',rem),('a',add),('p',cp[-1])]:offs[k].append(offs[k][-1]+len(value))
  allEvents[e]['labelChanges']=int((oldlabel!=newlabel).sum());allEvents[e]['lost']=int(((oldstate==2)&(newstate!=2)).sum());allEvents[e]['gained']=int(((oldstate!=2)&(newstate==2)).sum())
  if (e+1)%400==0:
   stage=(e+1)//400;positive=counts>0
   np.testing.assert_array_equal(keys[positive],pairs[stage][0]);np.testing.assert_array_equal(counts[positive],pairs[stage][1])
   np.testing.assert_array_equal(state,evs[stage]['state']);np.testing.assert_array_equal(label,np.where(evs[stage]['state']==2,evs[stage]['top1_instance_id'],-1))
   arrays[f'native{stage}']=label.copy();arrays[f'state{stage}']=state.copy();checks.append(dict(stage=STAGES[stage-1],ledgerExact=True,nativeMapExact=True))
   print('LEDGER_EXACT',VAR,stage,flush=True)
 for rem,add in zip(reversed(allrem),reversed(alladd)):counts[np.searchsorted(keys,add)]-=1;counts[np.searchsorted(keys,rem)]+=1
 positive=counts>0;np.testing.assert_array_equal(keys[positive],pairs[0][0]);np.testing.assert_array_equal(counts[positive],pairs[0][1])
 arrays.update(removed=cat(allrem,'<u4'),added=cat(alladd,'<u4'),changedPoint=cat(cp,'<u4'),oldLabel=cat(cl0,'<i2'),newLabel=cat(cl1,'<i2'),oldState=cat(cs0,'u1'),newState=cat(cs1,'u1'))
 for k,v in offs.items():arrays['offset_'+k]=np.asarray(v,'<u4')
 for j,m in enumerate(maps):arrays[f'final{j}']=m.astype('<i2')
 for k,o in [('removed','r'),('added','a'),('changedPoint','p')]:
  x=arrays[k];off=arrays['offset_'+o]
  for e in range(1200):
   lo,hi=int(off[e]),int(off[e+1]);x[lo+1:hi]=np.diff(x[lo:hi])
 arrays['voteKey'][1:]=np.diff(arrays['voteKey'])
 receipt=bundle(OUT/'replay.bin.gz',arrays)
 defaults={('automatic','room0'):(2,1255),('manual','room0'):(886,465),('manual','room1'):(417,390)};pid,fid=defaults.get((GROUP,SCENE),(next(iter(groups)),0))
 bookmarks=[]
 if GROUP=='automatic' and SCENE=='room0':bookmarks=[dict(id=2,frame=f,title=t) for f,t in [(735,'插座附近 · 帧735'),(890,'G008种子 · 墙面'),(1040,'插座附近 · 帧1040'),(1255,'多个插座 · 帧1255'),(1475,'插座附近 · 帧1475')]]
 if GROUP=='manual' and SCENE=='room1':bookmarks=[dict(id=417,frame=f,title=t) for f,t in [(385,'相邻前帧 · 385'),(390,'枕头冲突 · 390'),(520,'高冲突帧 · 520'),(920,'追踪扩张 · 920'),(925,'门控拒绝 · 925'),(1145,'G035种子 · 1145')]]
 if GROUP=='manual' and SCENE=='room0':bookmarks=[dict(id=886,frame=f,title=t) for f,t in [(420,'书本旧ID132 · 420'),(450,'书本旧ID140 · 450'),(465,'书本旧ID145 · 465'),(515,'G091种子 · 515'),(1990,'书本末段 · 1990')]]
 meta=dict(variant=VAR,scene=SCENE,title=('自动196' if GROUP=='automatic' else '人工76')+' · '+SCENE+' · partial',selectedSeeds=bootstrap.COUNT,base=base,surfacePoints=N,events=allEvents,objects=list(groups.values()),cameras=cams,defaultTarget=pid,defaultFrame=fid,bookmarks=bookmarks,keyEncoding='per-event-positive-delta-v1',bundle=receipt,stages=STAGES,verification=dict(status='PASS',stageChecks=checks,exactRollback=True,sourcePixelCandidatesExact=True,frameStrictAbstentionExact=True),sources=dict(source=str(SRC),partial=str(PARTIAL),baseline=str(B)),semantics='固定400帧完整基线上的离线事务重放；阶段内按帧号重放，不是在线建图前缀。原生投票状态与终态后处理标签分别显示。')
 dump(OUT/'data.json',meta);LOCK.update(h.protected);dump(OUT/'source_manifest.json',dict(files=LOCK,exporter_sha256=sha(__file__)))
 complete=dict(variant=VAR,status='PASS',frames=400,events=1200,objects=len(groups),selectedSeeds=bootstrap.COUNT,checks=checks,rollbackExact=True,seconds=round(time.time()-started,1));dump(OUT/'complete.json',complete)
 archive=ROOT/(VAR+'.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as z:
  for p in sorted(OUT.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(ROOT))
  if GROUP=='automatic':
   for p in sorted((ROOT/'assets'/SCENE/'rgb').glob('*.jpg')):z.write(p,p.relative_to(ROOT))
 dump(ROOT/(VAR+'_delivery.json'),dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),**complete));print('COMPLETE',json.dumps(complete),flush=True)
if __name__=='__main__':main()
