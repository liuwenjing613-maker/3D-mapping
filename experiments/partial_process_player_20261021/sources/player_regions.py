from pathlib import Path
import hashlib,json,numpy as np
R=Path('/data/chenkejun/CVPR/results/onebyone_case_audit_20261021');P=Path('/data/chenkejun/CVPR/results/partial_process_player_20261021')
cfg=json.loads(Path('/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021/config.json').read_text())
for group,scene,gids,names in [('automatic','room0',[44,56,80],['插座 GT44','插座 GT56','插座 GT80']),('manual','room1',[9],['受影响枕头 GT9']),('manual','room0',[85],['书本 GT85']),('automatic','room1',[],[])]:
 rows=[]
 for gid,name in zip(gids,names):
  p=R/group/scene/f'GT{gid}_trace.npz';z=np.load(p)
  rows.append(dict(id=gid,name=name,points=z['points'].tolist(),source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),events=[None]*1200))
 if rows:
  c=cfg[group][scene];base=json.loads((Path(c['source'])/'global_association.json').read_text())['instance_base'];stages=['baseline','partial_direct','partial_core_all','partial_whole_to_partial']
  for fi,fid in enumerate(range(0,2000,5)):
   with np.load(Path(c['partial_transactions'])/f'f{fid:06d}.npz') as z:
    tx={s+suffix:z[s+suffix] for s in stages for suffix in ['_candidate_keys','_frame_keys']}
    for row in rows:
     pts=np.array(row['points']);cc=[tx[s+'_candidate_keys'][np.isin(tx[s+'_candidate_keys']//base,pts)] for s in stages];ee=[tx[s+'_frame_keys'][np.isin(tx[s+'_frame_keys']//base,pts)] for s in stages]
     for j in range(3):
      bc,ac=cc[j],cc[j+1];rem=np.setdiff1d(ee[j],ee[j+1]);add=np.setdiff1d(ee[j+1],ee[j]);cr=np.setdiff1d(bc,ac);ca=np.setdiff1d(ac,bc);by={}
      for pid in np.unique(np.r_[bc%base,ac%base]):
       rel=np.unique(np.r_[bc[bc%base==pid]//base,ac[ac%base==pid]//base]);by[str(pid)]=dict(removed=int(np.isin(rem//base,rel).sum()),added=int(np.isin(add//base,rel).sum()),candidateRemoved=int((cr%base==pid).sum()),candidateAdded=int((ca%base==pid).sum()))
      row['events'][j*400+fi]=dict(removed=len(rem),added=len(add),candidateRemoved=len(cr),candidateAdded=len(ca),byId=by)
   if fi%100==0:print(group,scene,fi,flush=True)
 out=P/'assets'/f'{group}_{scene}'/'regions.json';out.write_text(json.dumps(dict(regions=rows,purpose='Post-freeze inspection bookmarks only. No GT used by replay or algorithms.'),ensure_ascii=False))
print('INSPECTION_REGIONS_WRITTEN')
