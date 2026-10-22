"""Read measured records; never infer wall time from mtimes or add overlapping totals."""
from pathlib import Path
import json,time
T=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
O=T/'review/compare50_100'
def read(p):return json.loads(Path(p).read_text())
def main():
 rows=[];sources=[]
 def row(stage,seconds,note,p,field):
  rows.append(dict(stage=stage,seconds=seconds,note=note,source=str(p),field=field))
  if str(p) not in sources:sources.append(str(p))
 p=T/'review/timing_summary.json';q=read(p)
 row('主建图 1–50 帧',q['main_mapping_first50']['main_loop_seconds'],'先前实测，本次请求复用历史结果、未重新建图；读取已有 2D mask，不含其模型推理、ROI 候选、修复和完整进程初始化。TSDF 积分 4.299 秒已包含在内。',p,'main_mapping_first50.main_loop_seconds')
 row('主建图 51–100 帧',q['main_mapping_51_to100']['main_loop_seconds'],'先前从已修复 50 帧真实续建，本次 100 帧修复直接复用此检查点；该历史计时在恢复检查点与载入前 50 帧历史之后开始，不含修复。TSDF 积分 5.822 秒已包含在内。',p,'main_mapping_51_to100.main_loop_seconds')
 p=T/'review/report50.json';q=read(p)
 row('50 帧修复：已测组件合计',q['total_measured_compute_seconds'],'历史结果复用。追踪 231.040 + 修复入口 141.557 + 提交 17.474；不含已缓存候选与主建图，不是完整暂停端到端时间。',p,'total_measured_compute_seconds')
 p=T/'review/compare100/execution_timings.json';q=read(p)
 for g in ['A','B']:
  v=q['tracking'][g]
  row('100 帧 '+g+'：已完成追踪',v['tracking_seconds'],'本次实际修复直接复用；此前 '+str(v['inferred_cases'])+' 个案例推理、'+str(v['reused_cases'])+' 个完整案例复用。此时间不计入本次修复入口。',p,'tracking.'+g+'.tracking_seconds')
  row('100 帧 '+g+'：已完成 Gate 入口',v['gate_entry_wall_seconds'],'本次复用；内含 Gate 主体 '+f"{v['gate_seconds']:.3f}"+' 秒，不再重复相加。',p,'tracking.'+g+'.gate_entry_wall_seconds')
 for g in ['A','B']:
  r=T/'repair100'/g
  for name,title in [('core_region','核心区域'),('expand','E1 扩展'),('partial_core_E1','partial / 历史核心 / E1 改票与后处理')]:
   p=r/('stage_'+name+'_complete.json')
   if p.exists():
    v=read(p)
    completed=r/'repair_complete.json'
    if completed.exists():v=read(completed).get('timings',{}).get(name,v)
    cached=v.get('reused_stage',False);sec=v.get('seconds_this_invocation',v.get('seconds')) if cached else v['seconds']
    row('100 帧 '+g+'：'+title,sec,('阶段缓存复用；原记录 '+str(v.get('seconds'))+' 秒。' if cached else '本次实测；属于下方修复入口墙钟的内含子阶段。'),p,'seconds_this_invocation' if cached else 'seconds')
   else:row('100 帧 '+g+'：'+title,'等待完成','以实际完成记录为准。',p,'seconds')
  p=r/'repair_complete.json'
  if p.exists():
   v=read(p);row('100 帧 '+g+'：修复入口墙钟',v['seconds'],'本次实测，包含上述三阶段、入口准备与哈希核验；不含复用的候选、追踪、Gate，不含提交和网页导出。',p,'seconds')
  else:row('100 帧 '+g+'：修复入口墙钟','执行中','以实际完成记录为准。',p,'seconds')
  p=r/'commit_complete.json'
  if p.exists():
   v=read(p);row('100 帧 '+g+'：状态提交',v['seconds'],'本次实测，保存独立分支的标签、历史修订、票据与身份状态；未开始第 101 帧。',p,'seconds')
  else:row('100 帧 '+g+'：状态提交','等待完成','以实际完成记录为准。',p,'seconds')
 p=O/'export_ready.json'
 if p.exists():
  v=read(p);row('五份实例 PLY / 色盘导出及审计',v['seconds'],'独立展示耗时，含色盘与文件逐字段校验，不计入修复算法。页面模板编写与浏览器传输/渲染时间不在此计时内。',p,'seconds')
 else:row('五份实例 PLY / 色盘导出及审计','等待完成','独立记录，不计入算法暂停时延。',p,'seconds')
 attempt_rows=[]
 for g in ['A','B']:
  for name,title in [('import_fix_01','首次导入适配联调'),('settings_adapter_02','第二次配置接口联调')]:
   p=T/'repair100'/g/'attempts'/name/'process_status.json'
   if p.exists():
    v=read(p);attempt_rows.append(dict(group=g,attempt=name,title=title,seconds=v['seconds'],status=v['status'],source=str(p),scope=v['scope']))
 attempts=dict(status='RECORDED',attempt_rows=attempt_rows,excluded_from_successful_pipeline=True,
  note='启动联调失败耗时单独记录；成功执行从 core/E1 全量重算，没有把失败过程或等待混入成功修复计时。A/B 同时执行，不能把两组失败耗时相加为墙钟。')
 tmp=O/'attempts_timings.json.tmp';tmp.write_text(json.dumps(attempts,ensure_ascii=False,indent=2)+'\n');tmp.replace(O/'attempts_timings.json')
 result=dict(attempt_rows=attempt_rows,status='PASS' if all(isinstance(r['seconds'],(int,float)) for r in rows) else 'PARTIAL',generated_unix=time.time(),rows=rows,sources=sources,
  continuous_0_to100_end_to_end_seconds=None,
  note='表中父阶段与其子阶段不可重复相加。A/B 本次并行运行并共享服务器 CPU，差值不能证明算法速度优劣，耗时不可直接相加为墙钟，也不与候选数及帧数不同的 50 帧实验作同条件速度比较。主建图两段均为先前实测，本次请求未重新建图；未记录连续 0–100 的完整端到端时间。')
 tmp=O/'timing.json.tmp';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(O/'timing.json');print(result['status'],len(rows))
if __name__=='__main__':main()
