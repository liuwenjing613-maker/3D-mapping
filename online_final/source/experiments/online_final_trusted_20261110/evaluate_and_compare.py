from pathlib import Path
import sys,os,json,time,hashlib,subprocess,traceback
import numpy as np
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/online_final_trusted_20261110')
BASE=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')
REF=Path('/data/chenkejun/CVPR/results/ppt5_staged_postid_full_depth_20261110/evaluation')
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def rd(p):return json.loads(p.read_text())
def wr(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(1024*1024),b''):h.update(a)
 return h.hexdigest()
def run(args,log):
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'};env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 with log.open('ab') as f:subprocess.run([sys.executable,'-B','-m','unified_eval.cli',*map(str,args)],cwd=MAIN,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
def protocol():
 d=rd(MAIN/'unified_eval/configs/current_protocol.json')
 assert (d['evaluation_profile'],d['profile_revision'],d['evaluable_GT_count'],d['frozen'])==('object_observed_repair',2,350,False)
 for p,h in d['evaluator_code_sha256'].items():assert sha(MAIN/'unified_eval'/p)==h
 assert sha(MAIN/'unified_eval/configs'/d['config_file'])==d['config_sha256']
 return d
def metric(d):return dict(F1=d['CA_PRF1_0_5']['F1'],AP50=d['CA_AP50_uniform'],AP=d['CA_AP_uniform'],PQ=d['CA_PQ']['PQ'],mCov=d.get('CA_mCov'),**{k:d['CA_PRF1_0_5'][k] for k in ['TP','FP','FN']})
def scene(s):
 ev=R/'evaluation'/s;ev.mkdir(parents=True,exist_ok=True)
 if (ev/'status.json').exists() and rd(ev/'status.json')['status']=='PASS':return
 lock=protocol();root=R/'online'/s
 for n in [50,100,200,300,400]:
  assert rd(root/str(n)/'B/geometry_parity_audit.json')['status']=='PASS'
  assert rd(root/str(n)/'B/repair/commit_complete.json')['processed_frames']==n
 receipt=R/'control_replay'/s/'complete.json';assert rd(receipt)['status']=='PASS'
 surface=R/'control_replay'/s/'instance_surface.npz'
 old=rd(REF/s/'r2/adapter_manifest.json')
 run(['adapt-surface','--gt',old['gt_file'],'--surface',surface,'--fixed-correspondence',old['correspondence_cache'],'--source-provenance',receipt,'--method-name','online_final_trusted','--method-commit','b7de6eea-trusted-616f3a6','--out',ev/'r2'],ev/'adapt.log')
 new=rd(ev/'r2/adapter_manifest.json')
 for k in ['profile_config_sha256','evaluator_code_sha256','gt_file_sha256','gt_scope_sha256','gt_observed_support_sha256','geometry_xyz_sha256','correspondence_sha256','correspondence_file_sha256']:
  assert new[k]==old[k],(s,k)
 run(['eval-repair-pair','--gt',old['gt_file'],'--before',REF/s/'r2/canonical_prediction.npz','--after',ev/'r2/canonical_prediction.npz','--out',ev/'paired'],ev/'paired.log')
 wr(ev/'status.json',dict(status='PASS',scene=s,stage='ALL5_ACTUAL_REPAIR_FINALB_PPT_HISTORY_DBSCAN_MAIN_R2',geometry_scope_correspondence_identical=True,prediction_sha256=sha(surface),metrics=metric(rd(ev/'r2/metrics.json')['summary']),GT_used_in_repair=False))
def report():
 out=R/'comparison';out.mkdir(exist_ok=True)
 protocol();results={};rows=[]
 for mode in ['online_final','online_final_trusted']:
  entries=[]
  for s in SCENES:
   root=REF/s/'r2' if mode=='online_final' else R/'evaluation'/s/'r2'
   a=rd(root/'adapter_manifest.json')
   entries.append(dict(gt=a['gt_file'],prediction=str(root/'canonical_prediction.npz'),diagnostic_prediction=str(root/'diagnostic_support_prediction.npz')))
  dest=out/mode;dest.mkdir(exist_ok=True);wr(dest/'batch.json',dict(scenes=entries));run(['eval-batch','--scenes',dest/'batch.json','--out',dest],dest/'eval.log');summary=rd(dest/'summary.json');results[mode]=metric(summary);assert results[mode]['TP']+results[mode]['FN']==350
 expected=rd(Path('/data/chenkejun/CVPR/results/ppt5_staged_postid_full_depth_20261110/comparison/comparison.json'))['results']['uniform_postid']
 for k in expected:
  if expected[k] is not None:assert abs(expected[k]-results['online_final'][k])<1e-12,(k,expected[k],results['online_final'][k])
 pairs=[];severe=[]
 for s in SCENES:
  old=metric(rd(REF/s/'r2/metrics.json')['summary']);new=rd(R/'evaluation'/s/'status.json')['metrics']
  pair=rd(R/'evaluation'/s/'paired/paired_repair_metrics.json')
  rows.append(dict(scene=s,before=old,after=new,delta={k:new[k]-old[k] for k in old if old[k] is not None and new[k] is not None},**{k:pair[k] for k in ['corrected_instances','degraded_instances','unchanged_instances','correct_owner_vertex_gain']}))
  for x in pair['per_gt']:
   q=dict(scene=s,**x);q['lost_iou50_qualification']=x['before_best_iou']>.5 and x['after_best_iou']<=.5;q['drop_at_least_010']=x['delta_iou']<=-.1;pairs.append(q)
   if q['lost_iou50_qualification'] or q['drop_at_least_010']:severe.append(q)
 assert len(pairs)==350
 wr(out/'comparison.json',dict(status='PASS',baseline='online-final@b7de6eea',filter='ppt5-full-depth-postid-uniform025-new3500-v1',mode='actual five-round online repair',schedule=[50,100,200,300,400],full_depth_geometry_exact_PPT=True,terminal_algorithm_exact_PPT=True,GT=350,results=results,delta={k:results['online_final_trusted'][k]-results['online_final'][k] for k in expected},per_scene=rows,per_GT=pairs,severe_degradations=severe,default_method_updated=False))
 wr(out/'status.json',dict(status='PASS',stage='FAIR_ALL8_PPT_FULL_DEPTH_5ROUND_COMPARISON_COMPLETE',results=results))
def main():
 out=R/'comparison';out.mkdir(exist_ok=True)
 while True:
  completed=[]
  for s in SCENES:
   p=R/'online'/s/'optimization_status.json'
   if not p.exists():continue
   d=rd(p)
   if d['status']=='FAILED':continue # Keep evaluating completed scenes; retain explicit failed scene status.
   if d['status']=='PASS_REPAIR_AND_PPT_POSTPROCESS':
    scene(s);completed.append(s)
  wr(out/'status.json',dict(status='WAITING_ACTUAL_ONLINE',completed=completed,total=8))
  if len(completed)==8:break
  time.sleep(30)
 for s in ['room0','office4']:
  p=Path('/data/chenkejun/CVPR/results/ppt5_staged_postid_full_depth_20261110')/'baseline_terminal_control/control_replay'/s/'complete.json'
  assert p.exists() and rd(p)['control_exact'],str(p)
 report()
if __name__=='__main__':
 try:
  if len(sys.argv)>1:scene(sys.argv[1])
  else:main()
 except BaseException:wr(R/'comparison/failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
