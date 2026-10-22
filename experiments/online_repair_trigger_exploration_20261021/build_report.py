from pathlib import Path
import json,html,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C=Path('/home/chenkejun/CVPR/experiments/online_repair_trigger_exploration_20261021')
O=Path('/data/chenkejun/CVPR/results/online_repair_trigger_exploration_20261021');W=O/'review';W.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
d=read(O/'comparison_with_viewbank.json')
timeline=read('/data/chenkejun/CVPR/results/online_room0_prefix50_20261021/mapping_timeline.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(11.6,2.7))
t=np.arange(1,51);N=np.array([x['surface_points'] for x in timeline]);conf=np.array([x['state_counts']['conflict'] for x in timeline]);unknown=np.array([1-x['state_counts']['confirmed']/max(1,x['surface_points']) for x in timeline])
axs[0].plot(t,N/1000,color='#16766b',lw=2);axs[0].set_title('Current TSDF surface');axs[0].set_ylabel('points (thousands)')
growth=100*(N[5:]-N[:-5])/np.maximum(1,N[:-5]);growth[0]=np.nan
axs[1].plot(t[5:],growth,color='#ae7142',lw=2);axs[1].axhline(5,color='#aaaaaa',ls=':',lw=1);axs[1].set_title('Growth over previous 5 frames');axs[1].set_ylabel('%');axs[1].set_ylim(0,35)
axs[2].plot(t,conf,color='#b84840',lw=2);axs[2].set_title('Native conflict points');axs[2].set_ylabel('points')
for ax in axs:
 ax.axvline(20,color='#78888b',ls='--',lw=1);ax.axvline(50,color='#78888b',ls='--',lw=1);ax.set_xlabel('Processed frame');ax.grid(alpha=.15);ax.set_xlim(1,51)
fig.tight_layout();fig.savefig(W/'timeline.svg',bbox_inches='tight');plt.close(fig)
for n in [20,50]:
 target=W/f'prefix{n}'
 if not target.exists():target.symlink_to(Path(f'/data/chenkejun/CVPR/results/online_room0_prefix{n}_20261021/review/assets'),target_is_directory=True)
def number(x):return f'{x:,}'
def candidate(n,gid):return next(r for r in d['prefixes'][str(n)]['rows'] if r['gid']==gid)
cases=[
 (20,'G004','柜子：应先补历史验证','当前组只有 1 个支持帧；但已有 20 个历史帧符合已知表面可见条件，最大跨度 12.9°。先补历史分割与一致性验证，不直接等到 50 帧。'),
 (50,'G007','柜子：3 帧不等于 3 个独立视角','3 个支持帧的角度跨度仅 1.6°，与主 mask 的 IoU 中位数约 0.42；历史上却有 32 个几何候选视角、跨度 18.8°。叠图中央还有大块未覆盖区域，需要检查整体 / 部件关系。'),
 (50,'G004','新区域：确实应继续等待','已有分组的 4 个支持帧跨度仅 1.6°，25.4% 的组支持表面最近 5 帧才首次深度可见；历史库仅 1 帧满足当前可见性条件。'),
 (50,'G005','已有高覆盖：成熟但修复收益低','已有 5 个支持帧，主 mask 支持点中 99.9% 已属于对应实例；稳定未分配点仅 18 / 22,209。没有明确分裂、粘连证据时应监控，避免重修。')]
cards=''.join(f'<article class="case"><img loading="lazy" src="prefix{n}/seed_{gid}_zoom.webp" alt="{n} 帧 {gid} 原始 mask 叠图"><div><span class="tag">{n} 帧 · {gid}</span><h3>{title}</h3><p>{text}</p></div></article>' for n,gid,title,text in cases)
policy_names=[('fixed_frame_all','固定帧 + 现有筛选','不判断视角独立性和修复收益'),('count_3_views','至少 3 个支持帧','仍会放行近重复视角与贴边 mask'),('quorum_maturity','跨视角成熟度','只看成熟度仍会保留低收益或身份有歧义的候选'),('maturity_need_unambiguous','成熟度 + 需求 + 身份消歧','推荐结构；先补历史证据，成熟问题再进入影子测试')]
policy_rows=''.join(f'<tr><td><b>{name}</b><small>{note}</small></td><td>{d["prefixes"]["20"]["policies"][key]["count"]}</td><td>{d["prefixes"]["50"]["policies"][key]["count"]}</td></tr>' for key,name,note in policy_names)
routes=''
for n in [20,50]:
 p=d['prefixes'][str(n)]
 routes+=f'<div class="panel"><h3>{n} 帧 · 当前候选 {p["summary"]["kept"]} 张</h3><p><b>{len(p["actions"]["SHADOW_CANDIDATE"])}</b> 张可作为影子测试候选 · <b>{len(p["wait_routes"]["REVERIFY_PAST_VIEWS"])}</b> 张先补历史验证 · <b>{len(p["wait_routes"]["WAIT_NEW_VIEW"])}</b> 张等待更多视角'
 if p['actions']['REVIEW_IDENTITY']:routes+=f' · <b>{len(p["actions"]["REVIEW_IDENTITY"])}</b> 张先消除身份歧义'
 if p['actions']['LOW_GAIN_MONITOR']:routes+=f' · <b>{len(p["actions"]["LOW_GAIN_MONITOR"])}</b> 张低收益监控'
 routes+='。</p><p class="muted">这是当前证据上的诊断分流，不是最终最优 mask 数量。补历史验证后，候选数量可能增加或减少。</p></div>'
action_names={'SHADOW_CANDIDATE':'影子测试候选','REVIEW_IDENTITY':'身份歧义待查','LOW_GAIN_MONITOR':'低收益监控','REVERIFY_PAST_VIEWS':'先补历史验证','WAIT_NEW_VIEW':'等待更多视角'}
details=''
for n in [20,50]:
 body=''
 for r in d['prefixes'][str(n)]['rows']:
  route=r.get('wait_reason_proxy',r['recommend_action_proxy']);v=r['past_view_bank'];iou='—' if r['median_primary_iou'] is None else f'{r["median_primary_iou"]:.2f}'
  body+=f'<tr><td><a href="mask_{n}_{r["gid"]}.html" target="_blank">{r["gid"]}</a><small>{html.escape(r["semantic"])}（模型标签）</small></td><td>{r["supporting_views"]} / {r["max_angle_deg"]:.1f}°</td><td>{v["geometrically_eligible_past_views"]} / {v["max_angle_deg"]:.1f}°</td><td>{iou}</td><td>{r["stable_pending"]} / {r["points"]}</td><td>{action_names[route]}</td></tr>'
 details+=f'<details><summary>{n} 帧：逐候选证据与诊断分流</summary><div class="scroll"><table><thead><tr><th>候选</th><th>分组支持帧 / 跨度</th><th>几何历史视角 / 跨度</th><th>主 mask IoU 中位数</th><th>稳定待定点 / 支持点</th><th>诊断去向</th></tr></thead><tbody>{body}</tbody></table></div></details>'
sens=''
for row20,row50 in zip(d['prefixes']['20']['sensitivity'],d['prefixes']['50']['sensitivity']):
 sens+=f'<tr><td>{row20["iou"]:.1f}</td><td>{row20["angle"]}°</td><td>{row20["count"]}</td><td>{row50["count"]}</td></tr>'
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>在线修复时机探索 · room0</title><link rel="stylesheet" href="style.css"></head><body>
<header><div class="eyebrow">ROOM0 · 因果前缀诊断 · 2026-10-21</div><h1>先补历史视角，再按物体成熟度启动修复</h1><p class="lead">推荐：持续发现问题 → 优先验证已有历史视角 → 成熟且有收益的对象进入影子测试 → 验证通过后小批提交。不要把“第 20 / 50 帧触发 ROI”直接等同于“应该开始修复”。</p><nav><a href="../3d/">20 / 50 帧实际 3D 地图</a><a href="../">50 帧 ROI / mask</a><a href="comparison.json">完整诊断数据</a></nav></header>
<div class="notice"><b>本次做到哪一步：</b>只读比较 4 种候选门控、1 种全局增长触发反例、9 组参数，并检查全部已处理历史帧的可见性。原地图、原 mask 与原筛选清单未改动，未执行修复，未使用 GT。5 / 6 张只是当前保守代理判据下的影子测试候选，不能当成已证明最优数量。</div>
<section><h2>实际证据改变了判断</h2><div class="cases">{cards}</div><p class="muted">“几何可选视角”要求当前已知主 mask 支持表面至少 80% 通过深度可见性检查，投影离图像边界至少 5 像素。它只表示值得回去取证，不能证明完整物体已可见，更不能替代真实 mask 验证。</p></section>
<section><h2>几种方案的对照</h2><div class="scroll"><table><thead><tr><th>方案</th><th>20 帧候选数</th><th>50 帧候选数</th></tr></thead><tbody>{policy_rows}</tbody></table></div><div class="two">{routes}</div><p>单独使用全局地图增长也不够：每 5 个处理帧检查一次，连续 3 次新增表面点少于 5%，会在第 <b>40</b> 帧触发。但之后仍可能遇到新物体或不完整视角。成熟度必须跟着具体对象或问题区域走。</p><img class="chart" src="timeline.svg" alt="只使用历史时间线计算的地图增长和原生冲突点趋势"><p class="muted">曲线来自各时刻的原生严格一票状态，未混用最终邻域补齐后的 U/T/C 统计。增长低不能证明物体完整；未确认比例下降也不能证明所有错误已经消失。</p></section>
<section class="recommended"><h2>推荐流程</h2>
<ol><li><b>问题持续存在，才生成修复任务。</b>区分刚出现、少观测的 U/T 与反复可见却仍无法获得可靠身份的区域；同时检查稳定多视角 mask 与旧实例的分裂、粘连矛盾。只看 U/T/C 会漏掉“高置信但合错对象”的错误。</li>
<li><b>先补过去的证据，再决定是否等新帧。</b>从所有已处理帧中选一个轮廓完整、清晰的主视角，再选几何覆盖互补、方向不同的验证视角。若过去已有可用视角，优先重新分割 / 验证；过去也不足，进入等待队列。不能把同帧多裁剪、多模型算作独立观测。</li>
<li><b>成熟度与必要性必须同时通过。</b>成熟度看独立视角、轮廓是否被截断、内部大块未覆盖区域是否有合理解释、深度与共同可见部分的一致性、近期是否仍明显扩张。对内部缺口，应检查深度连续性、互补分割和跨视角的整体 / 部件关系；不能直接填洞或拼接，因为也可能是真实孔洞或遮挡。必要性看仍未解决的问题和预计改动收益。成熟但已经基本正确的对象可以继续监控；问题很多但证据不足的对象先等待。</li>
<li><b>按独立问题选择尽量少的 mask。</b>每张 mask 必须新增有效覆盖或解决独立的身份矛盾；同一物体的近重复候选只保留一个主种子，其余作为验证证据。检查每个成熟问题是否被覆盖，不能只用全局覆盖率或固定 Top-K。无合格 mask 的问题标记为待补候选，不能从台账里消失。若计算预算有限，通过门槛后再按预计修复收益、持续时间与耗时排队；按对象规模归一化并考虑等待时长，避免大物体长期占满预算。</li>
<li><b>先试算，再提交。</b>候选进入影子地图；用未参与选主种子的独立已处理视角验证，证据不足则等新帧实际到达后再验证。只提交已验证的局部改动，保存撤销记录。新增修复观测替换对应来源，严格保持同帧同点最多一票，禁止把传播标签当新增观测票。</li></ol>
<p><b>开始修复的时刻：</b>某个对象第一次同时满足“问题持续存在、独立视角验证通过、范围稳定、预计收益为正”时，就允许该对象试修；其他对象继续等待。无需等待整间房同时成熟，也不因为达到某个帧数而强行修复。</p>
</section>
<section><h2>怎样避免“不多也不少”变成口号</h2><div class="two"><div class="panel"><h3>避免多修</h3><p>用历史视角共识消除重复和局部碎片；成熟度门槛挡住截断候选；收益门槛挡住已经基本正确的区域。SAM 来源优先不代表每一个 SAM 片段都应单独修复。</p><p>50 帧 G007 与 G024 大部分都来自旧实例 7，但支持区域仅相交 99 点。G007 叠图中央有大块未覆盖区域，G024 在另一视角中覆盖中央柜门的一部分，视觉证据支持存在整体 / 部件歧义；旧实例也可能存在粘连，需要进一步消歧。稳定的局部 mask 也可能一直不完整，不能只靠多帧一致性放行；在身份关系未明确前，不能把二者机械当作两个独立修复对象。</p></div><div class="panel"><h3>避免少修</h3><p>维护“已覆盖、待历史验证、待新视角、身份待消歧、无合格候选”台账。检查每个独立问题，尤其是小物体，而非只检查总覆盖点数。等待不是排除。</p><p>实际反例：全局覆盖达到 90% 就停止，20 帧会省掉 G021；50 帧会省掉 G033 和 G035。它们各自仍有独立的稳定待定点，因此不能仅因总点数少就被整体漏掉。</p></div></div>
<div class="cases"><article class="case"><img src="prefix50/seed_G007_full.webp" alt="50 帧 G007 柜子 SAM mask 全图"><div><h3>G007：主候选完整叠图</h3><p>查看中央未覆盖区域与完整柜体的关系。</p></div></article><article class="case"><img src="prefix50/seed_G024_full.webp" alt="50 帧 G024 柜子 SAM mask 全图"><div><h3>G024：同一旧实例上的另一候选</h3><p>该候选覆盖中央柜门的一部分。这是另一时刻的画面，不能直接比较二维像素；需要依靠三维对应和历史视角核对整体 / 部件关系。</p></div></article></div><p class="notice">有限的在线观测无法严格证明未见背面已经完整，也无法在无 GT 条件下保证 mask 数量“恰好最优”。可落地的保证是：不把未知当成完整、不丢弃待验证问题、只对有证据的局部改动负责，并且可撤销。</p></section>
<section><h2>诊断参数与敏感性</h2><p>本次“成熟度”代理要求：主 mask 与两张来自不同帧的、不贴图像或裁剪边界的 mask，两两共同可见 IoU ≥ 0.7；其中视角跨度 ≥ 5°；最近 5 个处理帧才首次深度可见的组表面比例 ≤ 10%。没有跨时刻对象快照，因此尚未验证真正的对象范围连续稳定。内部缺口与整体 / 部件检查目前仅做了视觉核查，尚未量化为门控；表中的候选数未包含这一新增检查。</p><p>“稳定待定点”代理指至少在 3 个已处理深度帧中可见、最早可见早于当前 5 个处理帧、当前仍未分配实例 ID 的点。本次需求下限是 max(20 点, 主 mask 支持点的 2%)。这些都只是诊断参数，未校准；低于下限的目标保留监控，不永久排除。结构性分裂 / 粘连需要独立证据，不能仅靠这些点数发现。</p>
<details><summary>查看 9 组参数对成熟候选数量的影响</summary><table><thead><tr><th>两两 IoU 下限</th><th>角度下限</th><th>20 帧</th><th>50 帧</th></tr></thead><tbody>{sens}</tbody></table><p>候选数在 20 帧为 1–7、50 帧为 1–10，说明阈值尚未稳定，不能宣称已经找到最优参数。</p></details>
{details}</section>
<section><h2>下一轮怎样验证推荐方案</h2><p>先锁定参数与预算，再从空地图逐帧回放，对比固定 20、固定 50、全局稳定触发、按对象成熟度触发、以及“补历史 + 成熟度 + 需求”方案。维护对象的历史状态时，使用 3D 几何对应；G 编号和 surface_index 不跨前缀直接对齐。</p><p>每次修复前后使用同一时刻的 TSDF 几何、GT 范围和对应索引；评价仍从 main 的 object_observed_repair / revision 2 当前入口运行。在线前缀必须使用该时刻可观察范围，不能套用最终 400 帧 GT。GT 只用于事后评价，不进入触发器。</p><p>重点量化：修复后实例质量、提前修复与撤销次数、成熟问题遗漏数、小目标覆盖、修复延迟、模型调用与耗时。未来帧只在其真正到达后参与新决定；用于检查旧决定后果时，明确标注为事后分析。</p></section>
<footer><h2>核验与参考</h2><p>20 帧、50 帧诊断分别在自身输入读取限制下执行：75 / 135 条受审计读取事件，越界读取均为 0。原地图、mask、分组、筛选文件哈希复核通过。跨前缀匹配保存在数据文件的 retrospective 字段，只用于事后回看，不进入任何前缀门控。</p><p><a href="comparison.json">逐候选诊断数据</a> · <a href="global_trigger_example.json">全局触发反例</a> · <a href="audit.json">本轮分析审计</a></p><p>方法参考：<a href="https://arxiv.org/abs/1903.01177" target="_blank">PanopticFusion</a> 强调依据当时的 3D 地图维护实例身份；<a href="https://arxiv.org/abs/2303.16739" target="_blank">Active Implicit Object Reconstruction</a> 用不确定性评价视角信息。这些是设计参考，不构成本项目触发阈值或修复收益的验证。</p></footer>
</body></html>'''
(W/'index.html').write_text(page)
for src,name in [(O/'comparison_with_viewbank.json','comparison.json'),(O/'global_trigger_example.json','global_trigger_example.json')]:
 dest=W/name
 if not dest.exists():dest.symlink_to(src)
entry=Path('/data/chenkejun/CVPR/results/online_room0_prefix50_20261021/review/analysis')
if not entry.exists():entry.symlink_to(W,target_is_directory=True)
audits=[]
for n,pids in [(20,[3066265,3160241]),(50,[3066266,3160242])]:
 R=Path(f'/data/chenkejun/CVPR/results/online_room0_prefix{n}_20261021')
 locked=read(R/'diagnostics/trigger_exploration_20261021/metrics.json')['source_sha256']
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in locked.items())
 files=[p for p in (R/'io_audit').glob('*.jsonl') if any(str(pid) in p.name for pid in pids)]
 logs=[json.loads(line) for p in files for line in p.read_text().splitlines()]
 assert not any(r.get('blocked') for r in logs)
 audits.append(dict(prefix=n,cutoff=n*5-5,events=len(logs),blocked=0,inputs_unchanged=True,feature_sha256=hashlib.sha256((R/'diagnostics/trigger_exploration_20261021/metrics.json').read_bytes()).hexdigest()))
(W/'audit.json').write_text(json.dumps(dict(status='PASS',prefix_diagnostics=audits,retrospective_analysis_separate=True,GT_used=False,repair_applied=False,original_results_changed=False,production_algorithm_changed=False,parameters_validated=False),ensure_ascii=False,indent=2)+'\n')
for n in [20,50]:
 for r in d['prefixes'][str(n)]['rows']:
  gid=r['gid']
  preview=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{n} 帧 {gid} · 原始 mask</title><link rel="stylesheet" href="style.css"></head><body><header><h1>{n} 帧 · {gid}</h1><p>原始候选 mask 叠图；仅作证据展示，未修改或重新修复。</p><nav><a href="./">返回探索报告</a><a href="../3d/">查看实际 3D 地图</a></nav></header><section><h2>完整视野</h2><img style="max-width:100%;height:auto" src="prefix{n}/seed_{gid}_full.webp" alt="{gid} 完整视野"><h2>局部放大</h2><img style="max-width:100%;height:auto" src="prefix{n}/seed_{gid}_zoom.webp" alt="{gid} 局部放大"></section></body></html>'''
  (W/f'mask_{n}_{gid}.html').write_text(preview)
print('REPORT_READY',len(page),str(W))
