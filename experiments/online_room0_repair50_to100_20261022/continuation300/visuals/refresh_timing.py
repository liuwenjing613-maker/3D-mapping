"""Summarize measured timings with explicit parent/child inclusion; no inferred end-to-end total."""
from pathlib import Path
import json,time
TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
ROOT=TOP/'continuation300';OUT=TOP/'review/compare200_300'
def read(p):
 try:return json.loads(Path(p).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
 rows=[];records={};attempts=[];case_timings={}
 def add(prefix,name,path,key,note,record=None):
  q=read(path) if record is None else record
  value=q.get(key) if q else None
  available=q is not None and q.get('status')=='PASS' and isinstance(value,(float,int))
  rows.append(dict(stage=prefix+' · '+name,seconds=value if available else '待完成 / 核对',
   note=note,source=str(path),field=key,status='PASS' if available else 'PENDING'))
  if q is not None:records[str(path)]=q
 for checkpoint in [200,300]:
  for group in ['A','B']:
   base=ROOT/str(checkpoint)/group;prefix=f'{checkpoint} 帧 {group}';interval=f'{checkpoint-99}–{checkpoint}'
   p=base/'mapping/mapping_timing.json';q=read(p)
   add(prefix,'新增 '+interval+' 帧主建图',p,'main_mapping_seconds','本轮真实续建；已扣除单独计时的检查点导出。包含逐帧读取、实例关联、TSDF、表面提取与严格票据。候选、追踪、修复不在此项。')
   add(prefix,'恢复前序已提交状态',p,'restore_seconds','主建图之前的状态/几何/历史层恢复；主建图入口总时间的子项。')
   add(prefix,'建图检查点导出',p,'checkpoint_export_seconds','包括几何、严格表面结果和最后的关联导出；已从上一行主建图计算项分离。')
   add(prefix,'建图入口总时间',p,'total_seconds','父项，包含上面的恢复、主建图、检查点导出和其他入口核验，不再与它们相加；不含之后的候选及修复。')
   p=base/'candidates/selection_complete.json';q=read(p)
   wait=q.get('gpu_wait_seconds') if q else None
   note='候选控制器本次调用的真实墙钟，包含本次 GPU 等待、前端、筛选、QA 及候选展示；不含之前的输入准备或历史失败调用。'
   if q and q.get('skipped_previously_complete_stages'):note+=' 本次复用了 '+str(len(q['skipped_previously_complete_stages']))+' 个已完成阶段；这些历史计算不在本次入口墙钟内。'
   if isinstance(wait,(float,int)):note+=f' 其中 GPU 等待 {wait:.3f} 秒。'
   add(prefix,'候选入口墙钟',p,'seconds',note)
   add(prefix,'候选必要子进程合计',p,'production_seconds','已记录成功必要子进程合计，排除 GPU 等待、QA 与展示；如有恢复运行，可含先前成功阶段。不是连续墙钟，不与本次候选入口相加。')
   p=base/'tracking/tracking_queue_complete.json';q=read(p);note='追踪队列墙钟，包含调度和 GPU 等待；案例可并行，不能把各案例推理或子进程时间求和当墙钟。'
   if q:
    for key,label in [('inferred_cases','实际推理案例'),('reused_cases','精确复用案例'),('gpu_wait_seconds','GPU等待秒')]:
     if key in q and isinstance(q[key],(int,float)):note+=f' {label}：{q[key]}.'
    cases=q.get('cases',[])
    if isinstance(cases,list):case_timings[f'{checkpoint}_{group}']=[{k:v for k,v in x.items() if k in ['case_uid','seconds','inference_seconds','initialization_seconds','direction_seconds','gpu_wait_seconds','inference_reused','previously_complete','wall_seconds_including_wait','status']} for x in cases if isinstance(x,dict)]
   add(prefix,'追踪队列墙钟',p,'seconds',note)
   p=base/'tracking/gate_complete.json';q=read(p);note='Gate 入口墙钟，包含关联冻结、追踪完整性验证和 Gate 主体。'
   if q:
    detail=[]
    for key,label in [('association_seconds','关联'),('validation_seconds','验证'),('gate_seconds','Gate主体')]:
     if isinstance(q.get(key),(int,float)):detail.append(f'{label} {q[key]:.3f} 秒')
    if detail:note+=' 内含：'+'；'.join(detail)+'；这些子项不另加。'
   add(prefix,'Gate 入口墙钟',p,'seconds',note)
   p=base/'tracking/complete.json'
   add(prefix,'追踪控制器总墙钟',p,'seconds','本次追踪控制器调用墙钟：prepare＋queue＋Gate 执行与等待；其中本次实际执行的 queue/Gate 已包含，不重复相加；如有历史完成缓存，遵循原始 timing_scope。')
   repair=read(base/'repair/repair_complete.json')
   for key,label in [('core_region','核心区域'),('expand','E1 扩展'),('partial_core_E1','partial / 核心 / E1 历史改票与后处理')]:
    p=base/'repair'/('stage_'+key+'_complete.json');q=read(p)
    if repair:q=repair.get('timings',{}).get(key,q)
    cached=q is not None and q.get('reused_stage',False)
    field='seconds_this_invocation' if cached else 'seconds'
    note='本次修复入口的内含子阶段；不与入口墙钟重复相加。'
    if cached:note+=' 此调用复用先前已完成的阶段，当前新增耗时单列为 0。'
    add(prefix,label,p,field,note,q)
   p=base/'repair/repair_complete.json'
   add(prefix,'修复入口墙钟',p,'seconds','父项，含上述核心/E1/历史改票、后处理、入口准备与验证。不含之前候选、追踪和 Gate，也不含后续状态提交或新四图导出。')
   p=base/'repair/commit_complete.json'
   add(prefix,'状态提交',p,'seconds','保存本分支实例标签、像素历史、票据和持久身份；300 帧完成后不继续第 301 帧。')
   for sub in ['mapping','candidates','tracking','repair']:
    log=base/sub/'process_timing_events.jsonl'
    if not log.exists():continue
    for line in log.read_text().splitlines():
     try:v=json.loads(line)
     except json.JSONDecodeError:continue
     if v.get('status') in ['FAILED','FAILED_OR_INTERRUPTED','ERROR']:
      attempts.append(dict(group=f'{checkpoint}/{group}',attempt=str(v.get('stage','unknown')),
       title=sub+' 失败/重试记录',seconds=v.get('seconds',0),status=v.get('status'),source=str(log),
       note='独立列出失败过程；不将其包装成成功算法计算时间。是否已含于控制器总墙钟，以原始记录边界为准。'))
   attempt_root=base/'repair/attempts'
   if attempt_root.exists():
    for p in attempt_root.glob('*/process_status.json'):
     v=read(p)
     if v and isinstance(v.get('seconds'),(int,float)):
      attempts.append(dict(group=f'{checkpoint}/{group}',attempt=p.parent.name,title='修复启动尝试',
       seconds=v['seconds'],status=v.get('status'),source=str(p),note='独立保留启动尝试原始耗时；成功入口另计。'))
 p=OUT/'export_ready.json'
 add('展示','八份实例 PLY / 联合色盘 / 逐字段审计',p,'seconds','服务器端展示导出独立计时；不计入修复算法。页面开发、浏览器传输与交互渲染未并入此数字。')
 result=dict(status='PASS' if all(x['status']=='PASS' for x in rows) else 'PARTIAL',generated_unix=time.time(),
  rows=rows,attempt_rows=attempts,tracking_case_timings=case_timings,
  note='父阶段已经包含其子阶段，不能重复相加。A/B 共享服务器 CPU/GPU 并可能并行，耗时差不能直接证明算法速度优劣。尚未记录的连续全过程总墙钟保持为空；不从文件时间推算，也不把未完成阶段写成完成。',
  continuous_end_to_end_seconds=None,record_paths=list(records))
 dump(OUT/'timing.json',result);dump(OUT/'attempts_timings.json',dict(attempt_rows=attempts,scope='recorded_failed_processes_only'))
if __name__=='__main__':main()
