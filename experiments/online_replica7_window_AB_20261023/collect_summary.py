"""Collect truthful final-only R2 metrics and small review evidence, not prefix scores."""
from pathlib import Path
import json,hashlib,shutil,tarfile,time
C=Path(__file__).resolve().parent
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
SCENES=['room1','room2','office0','office1','office2','office3','office4']
GT=dict(room1=45,room2=52,office0=31,office1=23,office2=41,office3=58,office4=28)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 review=O/'review';review.mkdir(exist_ok=True);rows=[];exports=[];files=[]
 def collect(p):
  p=Path(p)
  if not p.exists():return
  rel=p.relative_to(O);dest=review/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(p,dest);files.append(dict(path=str(rel),sha256=sha(dest),bytes=dest.stat().st_size))
 for scene in SCENES:
  for g in ['A','B']:
   assert read(O/scene/f'post_{g}_complete.json')['status']=='PASS'
   audit=read(O/scene/'verification'/f'{g}_causal_audit.json');assert audit['status']=='PASS'
   ev=O/scene/'evaluation_r2'/g
   before=read(ev/'before/metrics.json')['summary'];after=read(ev/'after/metrics.json')['summary'];pair=read(ev/'pair/paired_repair_metrics.json')
   assert all(s['profile_revision']==2 and s['scene_count']==1 for s in [before,after])
   assert sum(pair[k] for k in ['corrected_instances','degraded_instances','unchanged_instances'])==GT[scene]
   timing=O/scene/f'timings_{g}.jsonl'
   attempts=[json.loads(s) for s in timing.read_text().splitlines()]
   origins=[]
   for n in [50,100,200,300,400]:
    tracking=O/scene/str(n)/g/'tracking'
    cases=read(tracking/'seed_diagnostics.json')['cases'];reports=[read(tracking/'cases'/case['case_uid']/'tracking/complete.json') for case in cases]
    assert all(report['status']=='PASS' for report in reports)
    origins.append(dict(checkpoint=n,cases=len(reports),actual_inferred_cases=sum(not r.get('inference_reused',False) for r in reports),
      exact_cache_reused_cases=sum(r.get('inference_reused',False) for r in reports),basis='Final per-case inference origin, including successful work before a queue retry'))
    for rel in ['tracking_case_timings.jsonl','process_timing_events.jsonl','tracking_queue_failure.json']:collect(tracking/rel)
   rows.append(dict(scene=scene,branch=g,evaluable_GT=GT[scene],before=before,after=after,repair_pair=pair,tracking_origins=origins,
       checkpoints=audit['checkpoints'],stage_attempts=attempts,
       AP50_delta=after['CA_AP50_uniform']-before['CA_AP50_uniform']))
   collect(timing);collect(O/scene/f'post_{g}_complete.json');collect(O/scene/'verification'/f'{g}_causal_audit.json')
   for p in ev.rglob('*.json'):collect(p)
   for n in [50,100,200,300,400]:
    root=O/scene/str(n)/g
    for rel in ['mapping/mapping_complete.json','mapping/mapping_timing.json','mapping/restore_audit.json',
                'candidates/selection_complete.json','candidates/online_qa.json','candidates/window_source_policy.json',
                'tracking/complete.json','tracking/gate_complete.json','repair/repair_complete.json','repair/commit_complete.json']:
     collect(root/rel)
    for stage in ['before','after']:
     p=O/'deliverables'/scene/f'{scene}_{g}_{n}_{stage}_export.json';e=read(p)
     assert e['status']=='PASS' and sha(e['path'])==e['sha256']
     exports.append(e);collect(p)
 for name in ['run_plan.json','adapter_audit.json','catalog_path_adapter_fix.json','frozen_kernel_audit.json','status_write_race_fix.json','recovery_launch.json','recovery_status.json','launch.json','batch_status.json','batch_mapping_repair_complete.json']:
  collect(O/name)
 result=dict(status='PASS',scenes=SCENES,branches=['A','B'],scene_GT=GT,seven_scene_GT_total=sum(GT.values()),
   protocol='main object_observed_repair revision 2',protocol_total_GT=350,frozen=False,rows=rows,exports=exports,
   scope='14 fresh online runs; checkpoints 50,100,200,300,400; window-only SAM source rule at every checkpoint; final400 before/after R2 only',
   limitations=['A/B are candidate-search strategies, not old/new efficiency modes.',
    'room0 was a window-only400 rerun with earlier original-rule repair history; it is not a uniform eighth fresh run.',
    'Earlier room0 original/optimized real closed-loop comparison had a Gate threshold flip and 2 label differences; this batch does not remove that limitation.',
    'Execution timings include recorded cache reuse and scheduling; they are not cold-model-inference timings.'],generated_unix=time.time())
 (review/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['七场景完整在线 A/B：仅窗户 SAM 特殊来源规则','',
  '七个场景、14条分支均从空地图开始，运行400个建图帧，并完成50/100/200/300/400帧修复提交。A从全部已到达帧选候选，B从上次检查点后的新帧选候选，两者都使用完整历史投票。柜子按普通来源筛选；P/R豁免、SAM2、Gate、partial＋E1规则保持不变。','',
  '下表比较各分支第400帧本次修复前后，修复前已含此前修复。评分为main当前R2、frozen=false；全协议350 GT，本批七景278 GT。未将400帧GT套用到在线前缀。','',
  '| 场景 | 分支 | AP50前 | AP50后 | 变化(百分点) | GT改善/下降/不变 |','|---|---|---:|---:|---:|---|']
 for r in rows:
  p=r['repair_pair'];lines.append(f"| {r['scene']} | {r['branch']} | {100*r['before']['CA_AP50_uniform']:.2f}% | {100*r['after']['CA_AP50_uniform']:.2f}% | {100*r['AP50_delta']:+.2f} | {p['corrected_instances']}/{p['degraded_instances']}/{p['unchanged_instances']} |")
 lines+=['','每次修复均导出全部点、原始坐标与真实持久ID的PLY，共140份，哈希及逐元素回读核验见summary.json。各检查点后的51、101、201、301帧核验是否继承已提交状态。','',
  '耗时、模型实际推理及严格缓存复用分开记录在逐检查点证据中。失败重试也保留，不把缓存耗时当作从头推理速度。','',
  'room0此前仅在400帧应用新规则重修，前序修复历史不同，因此不能将它直接合并为同条件八景结果。前次优化等价验证存在Gate阈值翻转及2点标签差异的限制仍然保留。','']
 (review/'REPORT.txt').write_text('\n'.join(lines))
 for name in ['summary.json','REPORT.txt']:
  p=review/name;files.append(dict(path=name,sha256=sha(p),bytes=p.stat().st_size))
 (review/'evidence_manifest.json').write_text(json.dumps(dict(status='PASS',files=files),indent=2)+'\n')
 with tarfile.open(O/'review.tar.gz','w:gz') as tar:tar.add(review,arcname='review')
 print(json.dumps(dict(status='PASS',branches=len(rows),exports=len(exports),review=str(O/'review.tar.gz'))))
if __name__=='__main__':main()
