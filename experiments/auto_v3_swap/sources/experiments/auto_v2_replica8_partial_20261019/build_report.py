"""Collect completed results; never calculate or change prediction labels here."""
from pathlib import Path
import csv,hashlib,json,shutil,sys,time
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/auto_v2_replica8_partial_20261019')
O=R/'review'
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(R/'complete.json')['status']=='PASS'
 O.mkdir(exist_ok=True);(O/'reports').mkdir(exist_ok=True)
 results=[];rows=[];sources={}
 for scene in SCENES:
  source=Path('/data/chenkejun/CVPR/results')/('auto_v2_room0_pr35_20261019/e1_repair' if scene=='room0' else 'e1_auto35_original_20261019' if scene=='room2' else 'auto_v2_replica6_20261019/'+scene+'/e1_repair')
  evaluation=read(R/scene/'evaluation_r2/comparison.json');protocol=read(R/scene/'evaluation_r2/protocol_freeze.json')
  original_protocol=read(source/'evaluation_r2/protocol_freeze.json')
  assert evaluation['status']=='PASS' and evaluation['baseline_and_old_E1_scores_reproduced_exactly'] and evaluation['same_geometry_GT_scope_and_correspondence']
  for k in ['current_protocol','registry_sha256','fixed_correspondence_sha256']:assert protocol[k]==original_protocol[k],(scene,k)
  selection=read(source/'confirmed_selection.json');count=len(selection['included'])
  value=dict(scene=scene,seed_count=count,GT_count=evaluation['GT_count'],metrics=evaluation['rows'],per_GT=read(R/scene/'evaluation_r2/per_GT.json'),GT_changes=evaluation['GT_changes'],source=str(source),partial_result=str(R/scene),mapping_reused=(R/scene/'reuse_receipt.json').exists(),selection_user_request=selection.get('user_request'),user_retained_exceptions=selection.get('user_retained_exceptions',[]),selection_policy='此前确认的 room2 35 张（含用户保留项）' if scene=='room2' else '窗户 / 柜子主语义豁免第 5 步')
  frontend=source.parent/'frontend/input_manifest.json'
  if frontend.exists():
   inputs=read(frontend)
   value['input_views']=dict(ranked_entries=inputs['ranked_entries'],unique_ROIs=inputs['selected_cases'],selected_views=inputs['preferred_views'],unviewable_entries=inputs.get('unviewable_entries',[]))
   sources[str(frontend)]=sha(frontend)
   shutil.copy2(frontend,O/'reports'/(scene+'__frontend_input_manifest.json'))
  results.append(value)
  for row in evaluation['rows']:
   rows.append(dict(scene=scene,seed_count=0 if row['variant']=='baseline' else count,GT_count=evaluation['GT_count'],**{k:v for k,v in row.items() if not isinstance(v,dict)}))
  for root,names in [(R/scene,['evaluation_r2/comparison.json','evaluation_r2/per_GT.json','evaluation_r2/protocol_freeze.json','predictions_freeze.json','policy.json','reuse_receipt.json']),(source,['confirmed_selection.json','evaluation_r2/comparison.json','evaluation_r2/protocol_freeze.json'])]:
   for name in names:
    p=root/name
    if not p.exists():continue
    sources[str(p)]=sha(p);tag='partial' if root==R/scene else 'whole'
    shutil.copy2(p,O/'reports'/(scene+'__'+tag+'__'+name.replace('/','__')))
 assert sum(x['GT_count'] for x in results)==350
 configs={read(R/s/'evaluation_r2/protocol_freeze.json')['current_protocol']['config_sha256'] for s in SCENES};assert len(configs)==1
 report=dict(status='PASS',scenes=results,GT_count=350,protocol='object_observed_repair / revision 2',frozen=False,source_sha256=sources,current_config_sha256=configs.pop(),generated=time.time(),partial_change='old_observation_plans: whole → partial; all other frozen inputs and numerical rules held fixed',aggregate_policy='Scene metrics are shown separately; the overview mean is an unweighted mean over eight scenes, not pooled AP/PQ.')
 (O/'data.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':')))
 with (O/'reports/all_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 for name in ['index.html','style.css','app.js']:shutil.copy2(C/'review_web'/name,O/name)
 manifest=dict(status='PASS',files={p.relative_to(O).as_posix():sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='review_manifest.json'},scenes=8,GT_count=350)
 (O/'review_manifest.json').write_text(json.dumps(manifest,indent=2));print('EIGHT_SCENE_REPORT_COMPLETE',len(manifest['files']))
if __name__=='__main__':main()
