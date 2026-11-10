from pathlib import Path
import json, time
T=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
D=T/'review/compare100'
def read(p): return json.loads(p.read_text()) if p.exists() else {}
def write(p,d):
 q=p.with_name(p.name+'.publishing');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
events=[json.loads(s) for s in (T/'prefix100_new50/process_timing_events.jsonl').read_text().splitlines() if s.strip()]
success={e['stage']:e for e in events if e.get('status')=='PASS'}
production=success['all_edge_select']['ended_unix']-success['catalog']['started_unix']
times={'scope':'Measured stage times; not an A/B fresh-inference speed benchmark.','main_mapping':read(T/'review/timing_summary.json'),'candidate_A':{'production_wall_seconds':None,'unavailable_reason':'Original controller did not persist complete process timing before interrupted selection. Resume time is not full pipeline time.','cropformer_internal_seconds':466.2812990900129,'sam_internal_seconds':1016.6138776969165,'model_jobs_parallel':True},'candidate_B':{'production_event_span_seconds':production,'start_event':'catalog.started_unix','end_event':'all_edge_select.ended_unix','excluded':'web rendering, QA, manual audit repair and tracking','event_source':'prefix100_new50/process_timing_events.jsonl','render_process_seconds':success['all_edge_render']['seconds'],'qa_success_process_seconds':success['qa']['seconds'],'events':events},'tracking':{},'cache':{'candidate_B':read(T/'prefix100_new50/online_qa.json').get('model_execution_counts',{}),'openclip':'Recomputed all 8623 features after exactness check failed; final reused A features = 0. 16 validation forwards and 40 cabinet-context forwards also recorded.','segformer':'Initial stage: 6 cached and 15 new frames. Edge-stage recheck: 13 cached frames.','tracking_rule':'Reuse only complete matching joint-case fingerprints; identical individual seed alone does not permit reuse.'},'repair100_applied':False}
for arm in ['A','B']:
 comp=read(T/f'tracking100/{arm}/complete.json')
 if comp.get('status')=='PASS':
  times['tracking'][arm]={k:comp.get(k) for k in ['status','cases','objects','timings','tracking_seconds','inferred_cases','reused_cases','gate_seconds','page_url']}
paired_metrics=read(T/'tracking100/comparison.json').get('groups',{})
for arm,v in times['tracking'].items():
 for field in ['gate_entry_wall_seconds','render_seconds_separate']:
  v[field]=paired_metrics.get(arm,{}).get(field)
 v['controller_timing_note']='Outer controller child durations include subprocess exit polling; entry/internal timers below measure the computation boundary directly.'
def sec(v): return '待完成' if v is None else f'{v:.2f} 秒'
A=times['tracking'].get('A',{});B=times['tracking'].get('B',{})
def row(n,a,b):return f'<tr><td>{n}</td><td>{a}</td><td>{b}</td></tr>'
rows=[
row('候选生成至筛选完成','全流程计时未完整保存',sec(production)+'（阶段起止事件区间）'),
row('追踪队列墙钟',sec(A.get('tracking_seconds')),sec(B.get('tracking_seconds'))),
row('案例数：新推理 / 精确复用',f"{A.get('inferred_cases','—')} / {A.get('reused_cases','—')}",f"{B.get('inferred_cases','—')} / {B.get('reused_cases','—')}"),
row('门控入口（含输入核验）',sec(A.get('gate_entry_wall_seconds')),sec(B.get('gate_entry_wall_seconds'))),
row('其中：门控计算',sec(A.get('gate_seconds')),sec(B.get('gate_seconds'))),
row('追踪网页导出（另计）',sec(A.get('render_seconds_separate')),sec(B.get('render_seconds_separate')))
]
html='<h3>本轮实际耗时</h3><table><tr><th>阶段</th><th>A · 1～100</th><th>B · 51～100</th></tr>'+''.join(rows)+'</table><p class="muted">“其中”已包含在上一行，不再相加。B使用了A同输入缓存，A/B模型任务调度也不同，因此这些数值不能证明哪种候选策略推理更快。B候选区间不包含网页导出、QA、等待修正审计和后续追踪；A后续恢复用时也不能充当全流程耗时。</p><p><a href="execution_timings.json">完整阶段事件、缓存次数与计时边界</a></p>'
summary=read(D/'summary.json');summary['execution_timing_html']=html
summary['tracking_ready']=all(a in times['tracking'] for a in ['A','B'])
if summary['tracking_ready']:summary['status']='CANDIDATES_AND_TRACKING_COMPLETE'
summary['second_repair_applied']=False
write(D/'execution_timings.json',times);write(D/'summary.json',summary)
p=D/'index.html';s=p.read_text()
s=s.replace('s.candidate_count??s.retained_count??s.retained_seed_count??s.retained','s.retained??s.retained_count??s.retained_seed_count??s.candidate_count')
if 'id="execution-timing"' not in s:
 s=s.replace('<section><h2>计时口径</h2>','<section><h2>计时口径</h2><div id="execution-timing"></div>')
 s=s.replace("if(d.final_comparison_html)document.querySelector('#comparison').innerHTML=d.final_comparison_html;","if(d.final_comparison_html)document.querySelector('#comparison').innerHTML=d.final_comparison_html;if(d.execution_timing_html)document.querySelector('#execution-timing').innerHTML=d.execution_timing_html;")
if 'id="implementation-changes"' not in s:
 changes='''<section id="implementation-changes"><h2>本轮具体改动与复用</h2><ol><li>B组在ROI选视角前就限制到51～100帧；没有合格新视角的ROI不回退到旧帧。候选评分仍读取完整100帧状态，筛选阈值保持原值。</li><li>B组CropFormer和SAM各有385条输入：复用A中相同图像、模型和配置的原预测160条，B内部相同输入复用19条，新推理206次。复用的是冻结的预测文件，并未声称重新推理一定逐位一致。</li><li>OpenCLIP缓存抽样出现微小数值差异，最终全部重算，跨A特征实际复用0；SegFormer按相同帧与模型配置复用已有预测。SAM2追踪仅在整个联合案例一致时复用，原种子相同但联合对象不同的案例重新计算。</li><li>100帧门控读取第50帧已提交的像素级修订历史及续建状态，保持原门控和交叠弃权规则。页面分别显示初始门控与冲突后门控，拟关联ID尚未提交。</li><li>每个子进程结束即保存耗时，保留失败与恢复记录；网页显示和诊断另计。已补主建图逐帧耗时；第50帧PLY采用空间邻近实例高对比配色，实例ID与几何保持原值。</li></ol><p>本轮没有改动筛选阈值、追踪传播频率、门控门槛或partial＋E1修复规则；第100帧尚未执行实际修复。</p></section>'''
 s=s.replace('<p><a href="../50/">第50帧修复前后实例PLY</a>',changes+'<p><a href="../50/">第50帧修复前后实例PLY</a>')
p.write_text(s)
print(json.dumps({'candidate_B_production_event_span_seconds':production,'tracking_complete_groups':list(times['tracking']),'comparison_count_display':'retained first','repair100_applied':False},ensure_ascii=False))
