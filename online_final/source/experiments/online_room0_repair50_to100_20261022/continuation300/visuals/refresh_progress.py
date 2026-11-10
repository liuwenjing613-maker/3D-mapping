"""Read-only status bridge for the independent A/B 200/300 runs."""
from pathlib import Path
import json,time,argparse,traceback
T=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
ROOT=T/'continuation300';OUT=T/'review/compare200_300'
def read(p):
 try:return json.loads(Path(p).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def dump(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def state(root,name,done,active,extra=None):
 donepath=root/done;q=read(donepath);activepath=root/active;p=read(activepath)
 complete=q is not None and q.get('status')=='PASS'
 if complete and extra:
  other=read(root/extra);complete=other is not None and other.get('status')=='PASS'
 status='PASS' if complete else 'PENDING'
 if not complete and p:
  v=p.get('status')
  if v in ['RUNNING','PREPARING','READY','STARTED']:status='RUNNING'
  elif v in ['FAILED','FAIL','ERROR']:status='FAILED'
  elif q is not None and q.get('status')=='PASS':status='RUNNING'
 if not complete and q and q.get('status') in ['FAILED','FAIL','ERROR']:status='FAILED'
 failure_paths={
  '主建图':['mapping/mapping_failure.json'],
  '候选筛选与核验':['selection_tracking_failure.json'],
  '稠密双向追踪':['tracking/tracking_queue_failure.json','selection_tracking_failure.json'],
  'Gate 门控':['tracking/gate_failure.json','selection_tracking_failure.json'],
  'partial＋核心＋E1 修复':['repair/repair_failure.json'],
  '状态提交':['repair/commit_failure.json'],
  '独立最终核验':['final_audit_failure.json']}
 failure=None
 if not complete:
  for rel in failure_paths.get(name,[]):
   failed=read(root/rel)
   if failed:
    failure=str(root/rel);status='FAILED';break
 detail=''
 if p:
  if name=='主建图' and p.get('processed_frames') is not None:
   detail=f"{p.get('processed_frames')} / {p.get('target_processed_frames','?')} 个处理帧；raw {p.get('last_frame_id','?')}；"+str(p.get('stage',''))
  else:detail=' · '.join(str(p[k]) for k in ['stage','message','completed_cases','total_cases','completed','total','frames'] if k in p and not isinstance(p[k],(dict,list)))
 if failure:detail='执行记录报告异常，等待处理；查看 '+failure.rsplit('/',1)[-1]
 return dict(name=name,status=status,detail=detail,completion_source=str(donepath),active_source=str(activepath),completion_status=q.get('status') if q else None,failure_source=failure)
def refresh():
 runs=[]
 definitions=[
 ('主建图','mapping/mapping_complete.json','mapping/mapping_progress.json',None),
 ('候选筛选与核验','candidates/selection_complete.json','candidates/selection_progress.json','candidates/online_qa.json'),
 ('稠密双向追踪','tracking/tracking_queue_complete.json','tracking/tracking_queue_status.json',None),
 ('Gate 门控','tracking/gate_complete.json','tracking/progress.json',None),
 ('partial＋核心＋E1 修复','repair/repair_complete.json','repair/progress.json',None),
 ('状态提交','repair/commit_complete.json','repair/commit_progress.json',None),
 ('独立最终核验','final_audit.json','final_audit_progress.json',None)]
 for checkpoint in [200,300]:
  for group in ['A','B']:
   root=ROOT/str(checkpoint)/group
   stages=[state(root,*d) for d in definitions]
   pending=[s for s in stages if s['status']!='PASS'];current=pending[0] if pending else None
   message=(current['name']+'：'+(current['detail'] or {'PENDING':'等待前序步骤','RUNNING':'执行中','FAILED':'异常待处理'}[current['status']])) if current else '全部阶段完成；等待八份实例 PLY 导出与发布核验。'
   runs.append(dict(checkpoint=checkpoint,group=group,raw_cutoff=(checkpoint-1)*5,stages=stages,message=message))
 all_done=all(s['status']=='PASS' for r in runs for s in r['stages'])
 published=read(OUT/'manifest.json')
 published=published is not None and published.get('status')=='PASS'
 if published and all_done:
  for r in runs:r['message']='全部阶段、状态提交和实例 PLY 发布核验均已完成。'
 result=dict(status='PASS' if all_done else 'FAILED' if any(s['status']=='FAILED' for r in runs for s in r['stages']) else 'RUNNING' if any(s['status']=='RUNNING' for r in runs for s in r['stages']) else 'PENDING',
  generated_unix=time.time(),runs=runs,all_algorithms_completed=all_done,visualization_published=published,
  note='仅依据各阶段实际完成/进度记录。目录存在不代表运行；仅生成修复结果不代表状态已提交。')
 dump(OUT/'progress.json',result)
 if not published and read(OUT/'export_ready.json') is not None:
  dump(OUT/'status.json',dict(status='PREPARED_WAITING_AUDIT',message='八份实际实例 PLY 已准备；全部状态提交和四份独立最终核验通过后才正式发布。'))
 elif not published:
  dump(OUT/'status.json',dict(status='PREPARING' if all_done else result['status'],message='全部算法与提交记录已完成，正在等待八份真实实例PLY的导出和发布核验。' if all_done else '200/300 帧的 A/B 独立续建与修复正在按阶段执行，当前还未发布完整四图。'))
 return published and all_done
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
 while True:
  try:finished=refresh()
  except Exception:traceback.print_exc();finished=False
  if not args.watch or finished:break
  time.sleep(15)
