from pathlib import Path
import os,sys,json,time,traceback,itertools,collections,hashlib
import numpy as np
S=sys.argv[1];start=time.time();ROOT=Path('/data/chenkejun/CVPR/results/online_final_trusted_20261110/online')/S/'50/B';G=ROOT/'candidates/v1';SEM=ROOT/'candidates/background070/result.json';VIS=ROOT/'candidates/visibility';OUT=ROOT/'candidates/fixedH';OUT.mkdir(parents=True,exist_ok=True);old=G;maps=ROOT/'candidates'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def progress(stage,**k):write(OUT/'status.json',dict(status='RUNNING',scene=S,stage=stage,elapsed_seconds=time.time()-start,**k))
try:
 sem=read(SEM);assert sem['status']=='PASS';visdone=read(VIS/'status.json');assert visdone['status']=='PASS';seedrows=read(old/'grouping.json')['seeds'];seeds={x['id']:x for x in seedrows};rows={x['mask_id']:dict(x) for x in sem['retained_candidates']};assert len(rows)==len(sem['retained_candidates'])
 with np.load(G/'surface_supports.npz') as z:indices=z['indices'];offsets=z['offsets'];counts=np.diff(offsets)
 with np.load(G/'pair_metrics.npz') as z:inter=z['intersection']
 with np.load(maps/'baseline/final/instance_surface.npz') as z:labels=z['instance_id'].copy();npoints=len(labels)
 with np.load(maps/'baseline/surface_evidence.npz') as z:assert len(z['xyz_m'])==npoints
 def support(mid):
  i=seeds[mid]['index'];return indices[offsets[i]:offsets[i+1]]
 byframe=collections.defaultdict(list)
 for mid,r in rows.items():byframe[r['frame']].append(mid)
 bitvis={}
 for fid in byframe:
  with np.load(VIS/f'f{fid:06d}.npz') as z:vis=z['vertex_index']
  b=np.zeros(npoints,bool);b[vis]=True;bitvis[fid]=b
 # Exact original fixed-primary, cross-visible IoU >= 0.8.
 order=sorted(rows,key=lambda m:(-int(counts[seeds[m]['index']]),seeds[m]['frame'],seeds[m]['representative']['model'],seeds[m]['digest']));retained=[];covis_removed={}
 while order:
  a=order[0];retained.append(a);av=support(a);group=[a]
  for b in order[1:]:
   bv=support(b);na=int(bitvis[rows[b]['frame']][av].sum());nb=int(bitvis[rows[a]['frame']][bv].sum());c=int(inter[seeds[a]['index'],seeds[b]['index']]);u=na+nb-c;assert c<=min(na,nb)
   if u>0 and 5*c>=4*u:group.append(b);covis_removed[b]=a
  rm=set(group);order=[x for x in order if x not in rm]
 progress('CONTAINMENT_AND_COVERAGE',covis_input=len(rows),covis_retained=len(retained))
 # Containment is used to form relation groups, without removing a member.
 neighbors={m:set() for m in retained}
 for a,b in itertools.combinations(sorted(retained),2):
  c=int(inter[seeds[a]['index'],seeds[b]['index']]);den=min(int(counts[seeds[a]['index']]),int(counts[seeds[b]['index']]));
  if den and 5*c>2*den:neighbors[a].add(b);neighbors[b].add(a)
 groups=[];seen=set()
 for mid in sorted(retained,key=lambda m:(-int(counts[seeds[m]['index']]),m)):
  if mid in seen:continue
  todo=[mid];members=set()
  while todo:
   m=todo.pop()
   if m in members:continue
   members.add(m);todo.extend(neighbors[m]-members)
  seen.update(members);groups.append({'id':f'H{len(groups)+1:03d}','members':sorted(members,key=lambda m:(rows[m]['group_id'],rows[m]['cluster'],m))})
 owner={m:g['id'] for g in groups for m in g['members']};fullids,fullnums=np.unique(labels[labels>0],return_counts=True);fullcount=dict(zip(map(int,fullids),map(int,fullnums)));visible_counts={}
 for fid,b in bitvis.items():
  vals,ns=np.unique(labels[b & (labels>0)],return_counts=True);visible_counts[fid]=dict(zip(map(int,vals),map(int,ns)))
 aftercov=[];afterfrag=[]
 for mid in retained:
  r=rows[mid];sp=support(mid);labs=labels[sp];ids,ns=np.unique(labs[labs>0],return_counts=True);matched=min(zip(map(int,ids),map(int,ns)),key=lambda q:(-q[1],q[0])) if len(ids) else (None,0);pid,c=matched;den=visible_counts[r['frame']].get(pid,0);n=len(sp);assert n>0
  skip=bool(pid is not None and ((10*c>=9*n and 100*c>=85*den) or (100*c>=85*n and 10*c>=9*den)));r['coverage']={'instance_id':pid,'intersection':c,'support_count':n,'instance_visible_count':den,'final_unassigned_count':int((labs<=0).sum())};r['coverage_excluded']=skip
  if skip:continue
  aftercov.append(mid);total=fullcount.get(pid,0);u=r['coverage']['final_unassigned_count'];frag=bool(pid is not None and total>0 and (100*c<total or (100*c<8*total and 100*u<30*n)));r['fragment_full_count']=total;r['fragment_excluded']=frag
  if not frag:afterfrag.append(mid)
 progress('ID_AND_COMMON_VIEW',coverage_retained=len(aftercov),fragment_retained=len(afterfrag))
 # Best-Dice old-ID selection, then one winner per target.
 contenders=collections.defaultdict(list)
 for mid in afterfrag:
  r=rows[mid];sp=support(mid);labs=labels[sp];ids,ns=np.unique(labs[labs>0],return_counts=True);vc=visible_counts[r['frame']];cands=[]
  for pid,c in zip(map(int,ids),map(int,ns)):
   den=len(sp)+vc[pid];cands.append({'id':pid,'intersection':c,'mask_points':len(sp),'visible_instance_points':vc[pid],'dice':2*c/den})
  cands.sort(key=lambda x:(-x['dice'],x['id']));r['id_candidates']=cands;r['target_id']=cands[0]['id'] if cands and cands[0]['dice']>.25 else None;r['relation_group']=owner[mid]
  if r['target_id'] is not None:contenders[r['target_id']].append(mid)
 for mid in afterfrag:
  r=rows[mid];targets=contenders.get(r['target_id'],[]);winner=min(targets,key=lambda x:(-rows[x]['id_candidates'][0]['dice'],x)) if targets else None;r['id_action']='inherit' if mid==winner else 'new';r['conflict_winner']=winner
 group_members={g['id']:[m for m in g['members'] if m in set(afterfrag)] for g in groups};group_members={k:v for k,v in group_members.items() if v};best={}
 # Recompute max-min visible shared view over exactly the archived frame set.
 for i,fid in enumerate(visdone['frames'],1):
  with np.load(VIS/f'f{fid:06d}.npz') as z:vis=z['vertex_index']
  bit=np.zeros(npoints,bool);bit[vis]=True;fractions={m:float(bit[support(m)].sum())/len(support(m)) for m in afterfrag}
  for gid,members in group_members.items():
   vals=[fractions[m] for m in members];key=(min(vals),sum(vals),-fid)
   if gid not in best or key>best[gid][0]:best[gid]=(key,fid)
  if i%20==0:progress('COMMON_VIEW',frames=i,total_frames=len(visdone['frames']),groups=len(group_members))
 selected=[];decisions=[]
 for gid,members in group_members.items():
  frame=best[gid][1]
  with np.load(VIS/f'f{frame:06d}.npz') as z:vis=z['vertex_index'];uv=z['uv']
  pixels={}
  for mid in members:
   shared=np.intersect1d(support(mid),vis,assume_unique=True);xy=uv[np.searchsorted(vis,shared)];pixels[mid]=np.unique(xy[:,1].astype(np.int64)*1200+xy[:,0]);rows[mid]['projected_area']=len(pixels[mid]);rows[mid]['reference_frame']=frame
  wins=sorted([m for m in members if rows[m]['top_class'] in ('windowpane','blind')],key=lambda x:(-len(pixels[x]),x));keep=set(wins[:1]);ordered=sorted([m for m in members if m not in wins],key=lambda x:(len(pixels[x]),x));others=[]
  for mid in ordered:
   reject=False
   for other in others:
    c=len(np.intersect1d(pixels[mid],pixels[other],assume_unique=True));den=min(len(pixels[mid]),len(pixels[other]));
    if 10*c>den:reject=True;break
   if not reject:others.append(mid);keep.add(mid)
  selected.extend(sorted(keep));decisions.append({'id':gid,'reference_frame':frame,'members':members,'retained':sorted(keep)})
 # Assign identities only among final same-view survivors. A removed candidate cannot reserve an ID.
 selected=[m for m in selected if len(support(m))>=100]
 final_contenders=collections.defaultdict(list)
 for mid in selected:
  r=rows[mid];r['pre_dedup_id_action']=r['id_action'];r['pre_dedup_conflict_winner']=r['conflict_winner']
  cands=r['id_candidates'];r['target_id']=cands[0]['id'] if cands and cands[0]['dice']>0.25 else None
  if r['target_id'] is not None:final_contenders[r['target_id']].append(mid)
 for mid in selected:
  r=rows[mid];targets=final_contenders.get(r['target_id'],[])
  winner=min(targets,key=lambda x:(-rows[x]['id_candidates'][0]['dice'],x)) if targets else None
  r['id_action']='inherit' if mid==winner else 'new';r['conflict_winner']=winner
 before_new_support=list(selected)
 selected=[m for m in selected if rows[m]['id_action']=='inherit' or len(support(m))>=3500]
 selected_set=set(selected)
 assert all(rows[m]['conflict_winner'] is None or rows[m]['conflict_winner'] in selected_set for m in selected)
 inherited=[rows[m]['target_id'] for m in selected if rows[m]['id_action']=='inherit']
 assert len(inherited)==len(set(inherited))
 for d in decisions:d['retained']=[m for m in d['retained'] if m in selected_set]
 manual=None
 result={'status':'PASS','scene':S,'stage':'PREFIX_SCREENING_ONLY_NOT_REPAIR','version':'medoids-v3/bg070/covis080/contain040/coverage9085/fragment-r01-or-r08-u30/id-best-conflict/fixed-H-sameview-v3','baseline':'Replica causal online B 300 frame native prefix baseline','source_eligibility':'causal checkpoint-specific candidates with SAM3 proposals; no offline-ID manual override','input_semantic_retained':len(rows),'covis_retained':len(retained),'containment_groups':len(groups),'coverage_retained':len(aftercov),'fragment_retained':len(afterfrag),'sameview_retained':len(selected),'manual_room0':manual,'selected_masks':selected,'selected_records':[rows[m] for m in selected],'group_decisions':decisions,'GT_used':False,'tracking_or_repair_executed':False,'elapsed_seconds':time.time()-start}
 result['version']='postid-opt-v1/dice0.25/new3500';result['threshold_tuning']={'inherit_dice_strict':0.25,'min_inherit_support':100,'min_new_support':3500,'id_assignment_after_sameview':True};result['baseline']='online-v2 B@a15f7612 adapter; certified OVI native prefix geometry; checkpoints100/300/400';result['new_support_removed']=[m for m in before_new_support if m not in selected_set];result.update(version='ppt5-full-depth-postid-uniform025-new3500-v1',baseline='online-v2 B@a15f7612; original PPT full-depth geometry; five causal checkpoints',checkpoint=50,threshold_tuning=dict(inherit_dice_strict=0.25,min_inherit_support=100,min_new_support=3500,id_assignment_after_sameview=True));write(OUT/'result.json',result);write(OUT/'status.json',{k:v for k,v in result.items() if k not in ('selected_records','group_decisions','selected_masks')})
except BaseException:write(OUT/'status.json',dict(status='FAILED',scene=S,elapsed_seconds=time.time()-start,traceback=traceback.format_exc()));raise
