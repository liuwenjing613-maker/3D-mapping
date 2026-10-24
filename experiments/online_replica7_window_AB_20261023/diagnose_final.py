"""Read-only final400 instance diagnostics using the existing main R2 overlaps."""
from pathlib import Path
import sys,json,hashlib,numpy as np
M=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009');sys.path.insert(0,str(M))
from unified_eval.metrics import _matching
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
read=lambda p:json.loads(Path(p).read_text())
protocol=read(M/'unified_eval/configs/current_protocol.json');summary=read(O/'review/summary.json');rows=[]
for scene in summary['scenes']:
 gtpath=Path(protocol['canonical_GT'][scene]['file']);g=dict(np.load(gtpath,allow_pickle=False));scope=read(gtpath.parent/'gt_scope.json')
 names={x['raw_gt_id']:x['semantic_class'] for x in scope['objects']}
 for branch in ['A','B']:
  E=O/scene/'evaluation_r2'/branch;P=O/scene/'400'/branch/'repair'
  pair=read(E/'pair/paired_repair_metrics.json');over={s:dict(np.load(E/s/'overlap_matrix.npz',allow_pickle=False)) for s in ['before','after']}
  matched={s:{int(c):int(r) for r,c in _matching(z['iou'],.5,True)} for s,z in over.items()}
  seeds={}
  association=read(P/'global_association.json');seedmap={x['track_id']:x for x in read(P/'seed_id_map.json')['objects']}
  for x in association['objects']:
   seeds[str(x['persistent_id'])]=[dict(track_id=t,mask_id=seedmap[t]['mask_id'],frame=seedmap[t]['frame'],
      model=seedmap[t]['source']['representative']['model']) for t in x['member_track_ids']]
  objects=[]
  for item in pair['per_gt']:
   gid=item['raw_gt_id'];xyz=g['xyz_ref'][(g['raw_instance_id']==gid)&g['valid_vertex_mask']]
   assert len(xyz)>0
   row={**item,'class':names[gid],'bounds':np.stack([xyz.min(0),xyz.max(0)]).tolist()}
   for stage,z in over.items():
    col=list(z['gt_ids']).index(gid);best=int(np.argmax(z['iou'][:,col]));top=np.argsort(z['intersection'][:,col])[-5:][::-1]
    row[stage]=dict(pid=str(z['pred_uids'][best]),best_iou=float(z['iou'][best,col]),matched50=col in matched[stage],
     matched_pid=str(z['pred_uids'][matched[stage][col]]) if col in matched[stage] else None,
     top_fragments=[dict(pid=str(z['pred_uids'][i]),intersection=int(z['intersection'][i,col]),pred_size=int(z['pred_size'][i]),
         gt_size=int(z['gt_size'][col]),iou=float(z['iou'][i,col]),new_seed=seeds.get(str(z['pred_uids'][i]),[])) for i in top if z['intersection'][i,col]>0])
    assert abs(row[stage]['best_iou']-item[stage+'_best_iou'])<1e-12
   row['lost_match50']=row['before']['matched50'] and not row['after']['matched50']
   row['gained_match50']=not row['before']['matched50'] and row['after']['matched50']
   objects.append(row)
  row=dict(scene=scene,branch=branch,objects=sorted(objects,key=lambda x:x['delta_iou']),
   TP_before=len(matched['before']),TP_after=len(matched['after']),
   lost_match_GT=[x['raw_gt_id'] for x in objects if x['lost_match50']],gained_match_GT=[x['raw_gt_id'] for x in objects if x['gained_match50']])
  result=next(x for x in summary['rows'] if x['scene']==scene and x['branch']==branch)
  assert row['TP_before']==result['before']['CA_PRF1_0_5']['TP'] and row['TP_after']==result['after']['CA_PRF1_0_5']['TP']
  rows.append(row);print(scene,branch,'lost',row['lost_match_GT'],'top loss',[(x['raw_gt_id'],x['class'],round(x['delta_iou']*100,2)) for x in row['objects'][:3]],flush=True)
dest=O/'diagnostics.json';dest.write_text(json.dumps(dict(status='PASS',profile_revision=2,rows=rows,
 interpretation='Existing final400 fixed-geometry overlap diagnostics only; best IoU and one-to-one match losses are distinct. No Gate/E1 causal isolation is claimed.'),ensure_ascii=False,indent=2)+'\n')
print('DIAGNOSTICS_PASS',dest)
