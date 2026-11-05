"""Export complete, per-chosen-mask filtering history for the read-only page."""
from pathlib import Path
import json,shutil
from scripts.two_audit_seed_filter import ROOT,OUT,read,write,sha
STAGES=[('chosen','每簇选一张'),('boundary','全触边排除'),('noise','噪声筛选'),('grouping','分组取主种子'),('semantic','背景语义'),
 ('covisible_iou_only','共视去重'),('full_support_containment','包含筛选'),('skip_repair_B','无需修复'),
 ('pr_filter','P/R 筛选'),('preflight','投影预检'),('abc','最终 ABC')]
def build():
 output=OUT/'review';output.mkdir(exist_ok=True);scenes={};allrows=[]
 for scene in ['room0','office2']:
  o=OUT/scene;assert read(o/'complete.json')['status']=='PASS'
  chosen=read(o/'chosen_masks.json')['records'];source=read(o/'candidates/grouping.json');pool=read(o/'v1/grouping.json')
  boundary=read(o/'v1/boundary_audit.json');manifest=read(o/'v1/seed_manifest.json')
  semantic=read(o/'semantic/classification.json');sem={r['mask_id']:r for r in semantic['records']}
  selection=read(o/'v2/selection.json');preflight=read(o/'final_preflight/audit.json');abc=read(o/'final_abc/audit.json')
  final=read(o/'final_abc/seed_manifest.json');rows={};byid={s['id']:s for s in source['seeds']}
  for x in chosen:
   s=byid[x['id']]
   rows[x['id']]=dict(scene=scene,track=x['track'],id=x['id'],frame=x['frame'],member=x['member'],mask_local_id=x['mask_local_id'],
       selection_support=x['selected_support'],filter_support=s['support_count'],area=s['area'],history=[],terminal=None,final=False,
       image='/cluster-api/image?scene='+scene+'&id='+x['track']+'&member='+str(x['member'])+'&crop=1&overlay=1',
       cluster='/clusters/#scene='+scene+'&track='+x['track'])
   if x['id'] in sem:rows[x['id']]['semantic']= {k:sem[x['id']][k] for k in ['top_class','top_class_fraction','high_excluded_pixels','high_excluded_fraction']}
  current=set(rows);summaries=[]
  def stage(key,survivors,reasons=None):
   nonlocal current
   survivors=set(survivors);assert survivors<=current
   removed=current-survivors
   summaries.append(dict(id=key,name=dict(STAGES)[key],input=len(current),retained=len(survivors),removed=len(removed),
      retained_ids=sorted(survivors),removed_ids=sorted(removed)))
   for mid in current:
    keep=mid in survivors;row=rows[mid]
    row['history'].append(dict(stage=key,kept=keep,evidence=(reasons or {}).get(mid)))
    if not keep:row['terminal']=key
   current=survivors
  stage('chosen',current)
  edge={r['id']:r for r in boundary['edge_excluded']}
  stage('boundary',current-set(edge),{k:dict(frame_edge_pixels=r['frame_edge_pixels'],crop_contacts=r['crop_contacts']) for k,r in edge.items()})
  noise={r['id']:r for r in boundary['noise_excluded']}
  stage('noise',current-set(noise),{k:r['noise_decision'] for k,r in noise.items()})
  primary={g['primary']['id'] for g in manifest['groups']}
  group_reasons={}
  for g in manifest['groups']:
   rows[g['primary']['id']]['group']=g['id']
   for m in g['auxiliary_ids']:group_reasons[m]=dict(primary_mask=g['primary']['id'],primary_track=rows[g['primary']['id']]['track'],group=g['id'])
  stage('grouping',primary,group_reasons)
  groupmap={g['id']:g['primary']['id'] for g in manifest['groups']}
  for st in selection['stages']:
   reasons={}
   for rec in selection['records']:
    for d in rec['stage_decisions']:
     if d['stage']==st['stage']:reasons[groupmap[rec['group_id']]]=d['evidence']
   stage(st['stage'],[groupmap[g] for g in st['retained_group_ids']],reasons)
  stage('preflight',[r['mask_id'] for r in preflight['candidates'] if r['accepted']],{r['mask_id']:r for r in preflight['candidates']})
  stage('abc',[g['primary']['id'] for g in final['groups']],{mid:dict(witnesses=rs) for mid,rs in abc['witnesses'].items()})
  for mid in current:rows[mid]['final']=True;rows[mid]['terminal']='final'
  assert len(rows)==(179 if scene=='room0' else 118)
  assert all(r['terminal'] for r in rows.values())
  scenes[scene]=dict(stages=summaries,chosen=len(rows),final=len(current),final_ids=sorted(current),
    policy='one maximum audit-support original mask per stable track; earlier frame tie; no fallback')
  allrows.extend(rows.values())
 write(output/'data.json',dict(status='PASS',scenes=scenes,rows=allrows,stages=STAGES,GT_used=False,map_modified=False))
 shutil.copyfile(Path(__file__).with_name('two_audit_seed_review.html'),output/'index.html')
 write(output/'validation.json',dict(status='PASS',chosen=len(allrows),final=sum(s['final'] for s in scenes.values()),
   all_chosen_masks_have_terminal_stage=True,all_stages_monotone=True,original_masks_unchanged=True,
   data_sha256=sha(output/'data.json')))
 link=ROOT/'visual_review_full_v2/seeds'
 if not link.exists():link.symlink_to(output,target_is_directory=True)
 print(json.dumps(dict(status='PASS',scenes=scenes)),flush=True)
if __name__=='__main__':build()
