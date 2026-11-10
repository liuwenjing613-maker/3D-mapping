from pathlib import Path
import json,hashlib,collections,html
T=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
V=T/'review'/'compare100'
def read(p): return json.loads(Path(p).read_text())
first=read(T/'repair50/global_association.json')
prior_ids={int(x['persistent_id']) for x in first['objects']}
out={'status':'PARTIAL','same_scoring_geometry':None,'first_repair_target_ids':sorted(prior_ids),'GT_used':False,'repair100_applied':False,'arms':{},'notes':['Matching a prior repair target ID is a diagnostic flag, not proof the candidate is unnecessary.','A retained mask is a proposed repair input, not verified ground truth.','Exact seed equality requires the same source frame and PNG SHA256; same persistent ID alone is not seed equality.']}
for arm,folder in [('A','prefix100'),('B','prefix100_new50')]:
 root=T/folder; v=root/'all_edge_20261021'/'v2'
 if not (v/'seed_manifest.json').exists(): continue
 seed=read(v/'seed_manifest.json');selection=read(v/'selection.json')
 records={r['group_id']:r for r in selection['records']}
 rows=[]
 for g in seed['groups']:
  r=records[g['id']];p=g['primary']
  cov=next((s.get('evidence') for s in r['stage_decisions'] if s['stage']=='skip_repair_B'),{})
  if not cov: cov={}
  mask=root/'all_edge_20261021'/'v1'/p['mask']
  assert hashlib.sha256(mask.read_bytes()).hexdigest()==p['mask_sha256']
  rep=p.get('representative',{})
  rows.append({'group':g['id'],'mask':p['id'],'raw_frame':p['frame'],'processed_frame':p['frame']//5+1,
   'mask_sha256':p['mask_sha256'],'source_model':rep.get('model'),'source_semantic_family':rep.get('semantic_family'),
   'support_count':p['support_count'],'dominant_persistent_id':cov.get('instance_id'),'p':cov.get('p'),'r':cov.get('r'),
   'shares_prior_repair_target_id':cov.get('instance_id') in prior_ids,'stage5_semantic_exemption':cov.get('semantic_exemption'),
   'both_global_and_crop_edge_free':not p['touches_any_edge']})
 views=read(root/'catalog/room0/candidate_manifest.json')['entries']
 vi=[z['frame'] for x in views if x['rank']<=20 for z in x['views']]
 out['arms'][arm]={'retained':len(rows),'historical_seed_count':sum(r['raw_frame']<=245 for r in rows),
  'new50_seed_count':sum(r['raw_frame']>=250 for r in rows),'source_raw_frames':sorted(set(r['raw_frame'] for r in rows)),
  'source_model_counts':dict(collections.Counter({0:'original_cropformer',1:'roi_cropformer',2:'SAM'}.get(r['source_model'],'unknown') for r in rows)),
  'shares_prior_repair_target_id_count':sum(r['shares_prior_repair_target_id'] for r in rows),
  'ranked_view_records':len(vi),'ranked_historical_view_records':sum(f<=245 for f in vi),'ranked_new50_view_records':sum(f>=250 for f in vi),'rows':rows}
if len(out['arms'])==2:
 out['status']='CANDIDATES_COMPLETE'
 a={(r['raw_frame'],r['mask_sha256']) for r in out['arms']['A']['rows']}
 b={(r['raw_frame'],r['mask_sha256']) for r in out['arms']['B']['rows']}
 out['exact_shared_seed_count']=len(a&b)
 out['A_only_exact_seed_count']=len(a-b);out['B_only_exact_seed_count']=len(b-a)
 ca=read(T/'prefix100/catalog/room0/candidate_manifest.json')
 cb=read(T/'prefix100_new50/catalog/room0/candidate_manifest.json')
 out['shared_scoring_hashes']={key:ca[key]==cb[key] for key in ['surface_sha256','evidence_sha256','graph_sha256']}
 out['same_scoring_geometry']=all(out['shared_scoring_hashes'].values())
 # File hashes are separately audited by each variant's input and QA manifests.
def value(arm,key):
 d=out['arms'].get(arm)
 if d is None:return '处理中'
 return str(d[key])
rows=[('最终保留种子','retained'),('种子来自第1～50帧','historical_seed_count'),('种子来自第51～100帧','new50_seed_count'),('ROI视角记录','ranked_view_records'),('与第50帧修复目标ID重合的候选','shares_prior_repair_target_id_count')]
table='<h3>候选统计</h3><table><tr><th>项目</th><th>A · 1～100</th><th>B · 51～100</th></tr>'+''.join('<tr><td>'+label+'</td><td>'+value('A',key)+'</td><td>'+value('B',key)+'</td></tr>' for label,key in rows)+'</table><p class="muted">与历史修复目标ID重合不等于无需修复；本次先核查候选和追踪，未执行第二次改票。</p>'
if len(out['arms'])==2:table+='<p>两组来源帧和PNG完全一致的种子：'+str(out['exact_shared_seed_count'])+' 个。其余种子的不同不直接等同于不同物体。</p>'
table+='<p><a href="selection_comparison.json">逐种子来源、p/r与历史修复目标对应</a></p>'
(V/'selection_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
p=V/'summary.json';s=read(p);s['candidate_comparison_status']=out['status'];s['final_comparison_html']=table;p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:{x:d[x] for x in ['retained','historical_seed_count','new50_seed_count','shares_prior_repair_target_id_count']} for k,d in out['arms'].items()},ensure_ascii=False))
