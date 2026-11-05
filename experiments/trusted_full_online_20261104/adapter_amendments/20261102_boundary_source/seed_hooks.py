"""Connect approved boundary, preflight and final ABC to each reached branch."""
from common_continue import *
import sys,copy,hashlib
import numpy as np
from PIL import Image
SC=CONT/'selection_code'/str(CHECKPOINT)/GROUP
PRE=CANDIDATE_ROOT/'final_preflight/seed_manifest.json'
CTX=CANDIDATE_ROOT/'abc_context.json'
FINAL=CANDIDATE_ROOT/'final_abc'
def policy():
 sys.path.insert(0,str(POLICY_ROOT))
def boundary():
 policy()
 import importlib.util
 spec=importlib.util.spec_from_file_location('approved_boundary_source_revision',Path(__file__).with_name('seed_boundary.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 screen=module.screen
 p=CANDIDATE_ROOT/'grouping_r2'
 stats=screen(p/'grouping.json',p/'surface_supports.npz',p/'pair_metrics.npz',SC/'v1',S)
 grouping=read(S/'grouping.json');manifest=read(S/'seed_manifest.json')
 sources={**read(p/'grouping.json')['source_sha256'],**grouping['source_sha256']}
 for path,h in sources.items():assert digest(path)==h,path
 for row in grouping['seeds']:row['mask']='masks/'+Path(row['mask']).name
 byid={x['id']:x for x in grouping['seeds']}
 for g in manifest['groups']:g['primary']=byid[g['primary']['id']]
 (S/'masks').symlink_to((p/'masks').resolve(),target_is_directory=True)
 grouping['source_sha256']=sources;dump(S/'grouping.json',grouping);dump(S/'seed_manifest.json',manifest)
 audit=read(S/'audit.json');audit['source_sha256']=sources
 audit['output_sha256']={n:digest(S/n) for n in audit['output_sha256']}
 audit['interface_adapter']='original relative masks and inherited provenance; numerical rules unchanged'
 dump(S/'audit.json',audit);print(json.dumps(stats),flush=True)
def qa():
 policy()
 from instance_repair.seed_boundary import validate_selected
 result=validate_selected(CHOSEN,S/'grouping.json',S)
 mapping=read(B/'mapping_complete.json');assert mapping['status']=='PASS' and mapping['frame_ids']==FIDS
 for path,h in read(CANDIDATE_ROOT/'mapping_input_freeze.json')['source_hashes'].items():assert digest(path)==h,path
 selected=read(CHOSEN);selection=read(CANDIDATE_ROOT/'v2/selection.json')
 assert {g['id'] for g in selected['groups']}==set(selection['retained_group_ids'])
 start=0 if GROUP=='A' else OLD_COUNT*5
 assert all(start<=g['primary']['frame']<=CURRENT_RAW for g in selected['groups'])
 cat=read(CANDIDATE_ROOT/'catalog'/SCENE/'candidate_manifest.json')
 assert cat['candidate_search_frame_ids']==list(range(start,CURRENT_RAW+1,5))
 assert cat['scoring_history_frame_ids']==FIDS
 dump(CANDIDATE_ROOT/'online_qa.json',dict(status='PASS',retained=len(selected['groups']),source_manifest=str(CHOSEN),source_manifest_sha256=digest(CHOSEN),checkpoint=CHECKPOINT,group=GROUP,candidate_raw_frame_range=[start,CURRENT_RAW],boundary=result,GT_used=False))
def preflight():
 from bootstrap_continue import r
 from dataclasses import replace
 original=read(CHOSEN);result=copy.deepcopy(original);kept=[];audit=[]
 scene=r.Scene(SCENE)
 for g in original['groups']:
  p=g['primary'];fid=p['frame'];tid=int(g['id'][1:])
  path=S/p['mask'];assert digest(path)==p['mask_sha256']
  mask=np.asarray(Image.open(path))>0;frame=scene.source.load_frame(fid)
  points,ids,stats=r.project_frame_regions(replace(frame,mask_local=mask.astype(np.uint16)*tid),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
  count=len(np.unique(points[ids==tid]))
  audit.append(dict(group_id=g['id'],mask_id=p['id'],frame=fid,mask_sha256=p['mask_sha256'],projected_seed_points=count,accepted=count>0))
  if count:kept.append(g)
 result['groups']=kept
 result['projection_preflight']={'source_manifest':str(CHOSEN),'source_sha256':digest(CHOSEN),'pixel_stride':2,'max_distance_m':.015}
 dump(PRE,result);dump(PRE.parent/'audit.json',dict(status='PASS',input=len(original['groups']),output=len(kept),candidates=audit,GT_used=False))
 scene.verify_unchanged()
 cfg=read(B/'config.json');assert Path(cfg['source']['scene_root']).resolve()==(B/'inputs/scene').resolve()
 context=dict(grouping=str(S/'grouping.json'),supports=str(S/'surface_supports.npz'),pairs=str(S/'pair_metrics.npz'),mask_root=str(S),visible_root=str(CANDIDATE_ROOT/'overmerge'),geometry=str(B/'geometry'/('geometry_prefix%d.json'%CHECKPOINT)),mapping_config=str(B/'config.json'),semantics=str(CANDIDATE_ROOT/'abc_semantics/semantics.json'))
 dump(CTX,context)
def semantics():
 policy()
 from instance_repair.seed_abc_semantics import prepare
 from instance_repair.seed_final_abc import load_context
 prepare(load_context(CTX),CANDIDATE_ROOT/'v2/semantic/classification.json',CANDIDATE_ROOT/'abc_semantics')
def abc():
 policy()
 from instance_repair.seed_final_abc import finalize
 result=finalize(PRE,CTX,FINAL);dump(FINAL/'handoff.json',result)
def guard():
 policy()
 from instance_repair.seed_final_abc import validate_selected
 result=validate_selected(FINAL/'seed_manifest.json',FINAL/'receipt.json')
 print(json.dumps(result),flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
