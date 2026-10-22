"""Build an honest intermediate review without choosing a containment-chain policy."""
from online_common import *
import sys,copy,subprocess,os
from itertools import combinations
sys.path.insert(0,str(C/'v2'))
from filter_rules import semantic_excluded,iou_duplicate,fixed_primary_groups,containment_small
d=read(R/'v2/measurements.json')
seeds={s['group_id']:s for s in d['seeds']}
pairs={tuple(sorted((p['a'],p['b']))):p for p in d['pairs']}
records={g:dict(**{k:s[k] for k in ['group_id','original_track_id','mask_id','frame','mask_sha256']},stage_decisions=[],decision='pending',terminal_stage='pending_containment') for g,s in seeds.items()}
stages=[]
current=sorted(seeds)
def finish(name,removed,reasons):
 global current
 incoming=current[:];removed=set(removed);current=sorted(set(incoming)-removed)
 stages.append(dict(stage=name,input=len(incoming),removed=len(removed),retained=len(current),removed_group_ids=sorted(removed),retained_group_ids=current[:]))
 for g in incoming:
  records[g]['stage_decisions'].append(dict(stage=name,decision='exclude' if g in removed else 'retain',evidence=reasons[g]))
  if g in removed:records[g].update(decision='exclude',terminal_stage=name)
finish('semantic',[g for g in current if semantic_excluded(seeds[g]['high_excluded_pixels'],seeds[g]['area'])],{g:{k:seeds[g][k] for k in ['high_excluded_pixels','area']} for g in current})
ordered=sorted(current,key=lambda g:(-seeds[g]['support_count'],seeds[g]['frame'],seeds[g]['source_model'],seeds[g]['mask_digest']))
groups=fixed_primary_groups(ordered,lambda a,b:iou_duplicate(pairs[tuple(sorted((a,b)))]['intersection'],pairs[tuple(sorted((a,b)))]['union']))
reps={g:group[0] for group in groups for g in group}
reasons={}
for g in current:
 rep=reps[g];reasons[g]=dict(primary_group_id=rep)
 if rep!=g:reasons[g].update({k:pairs[tuple(sorted((rep,g)))][k] for k in ['intersection','union']})
finish('covisible_iou_only',[g for g in current if reps[g]!=g],reasons)
edges=[]
for a,b in combinations(current,2):
 p=pairs[a,b];sl=containment_small(a,seeds[a]['support_count'],b,seeds[b]['support_count'],p['original_intersection'])
 if sl:edges.append(dict(small=sl[0],large=sl[1],intersection=p['original_intersection'],small_count=seeds[sl[0]]['support_count'],large_count=seeds[sl[1]]['support_count']))
chains=[dict(small=a['small'],middle=a['large'],large=b['large']) for a in edges for b in edges if a['large']==b['small']]
assert chains==[dict(small='G042',middle='G017',large='G007')]
selection=dict(status='REVIEW_REQUIRED',selection_complete=False,retained_group_ids=[],pending_group_ids=current,stages=stages,records=list(records.values()),containment_edges=edges,containment_chains=chains,measurements_sha256=sha(R/'v2/measurements.json'),policy_applied=False)
write(R/'review_pending/selection.json',selection)
source=(C/'build_review.py').read_text()
source=source.replace("assert read(R/'selection_complete.json')['status']=='PASS'","assert read(R/'v2/review_required.json')['reason']")
source=source.replace("selected=read(R/'v2/selection.json')","selected=read(R/'review_pending/selection.json')")
source=source.replace("kept=read(R/'v2/seed_manifest.json')","kept={'groups':[]}")
source=source.replace("retained=selected['retained_group_ids'],","retained=[],pending=selected['pending_group_ids'],selection_complete=False,containment_edges=selected['containment_edges'],containment_chains=selected['containment_chains'],")
source=source.replace("(R/'v2/seed_manifest.json','retained_masks.json'),","(R/'v1/seed_manifest.json','all_primary_masks.json'),")
source=source.replace("(R/'v2/selection.json','selection.json')","(R/'review_pending/selection.json','selection.json')")
start=source.index("    with zipfile.ZipFile(W/'downloads/retained_binary_masks.zip'")
end=source.index("    write(R/'review_complete.json'",start)
source=source[:start]+source[end:]
source=source.replace("dict(status='PASS',ROIs=", "dict(status='REVIEW_REQUIRED',selection_complete=False,ROIs=")
exec(compile(source,str(C/'build_pending_review.py')+':render','exec'),dict(__name__='__main__'))
app=(C/'review_web/app.js').read_text()
app=app.replace("function badge(dec){","function badge(dec){if(dec==='pending')return '<span class=\"badge skip\">待完成筛选</span>';")
app=app.replace("const d=item.decision;if(d.decision==='retain')","const d=item.decision;if(d.decision==='pending')return '已通过前两步；包含关系待确认，后续尚未执行。';if(d.decision==='retain')")
a=app.index('function retained(){');b=app.index('\nfunction filters()',a)
app=app[:a]+"""function retained(){const chainIds=['G007','G017','G042'];const chain=chainIds.map(id=>D.items.find(x=>x.id===id));return `<h2>包含关系待确认</h2><div class="notice">50 帧建图和模型预测已完成。43 个主种子经过背景过滤剩 36 个、可见区域去重后剩 32 个；包含过滤遇到链式关系，尚未生成最终保留清单，也未执行修复。<br>G042 被 G017 包含 99.17%；G017 被 G007 包含 91.06%；G042 同时被 G007 直接包含 100%。现有规则要求确认此情况，尚未自动选择处理方式。</div><h3>需要确认的三张 SAM mask</h3>${cards(chain)}<div class="panel section-gap"><h3>全部达到包含阈值的关系</h3><table><tr><th>较小 mask</th><th>更大 mask</th><th>交集 / 小 mask 支持点</th><th>包含比例</th></tr>${D.containment_edges.map(e=>`<tr><td>${e.small}</td><td>${e.large}</td><td>${fmt(e.intersection)} / ${fmt(e.small_count)}</td><td>${pct(e.intersection,e.small_count)}</td></tr>`).join('')}</table><p>可选处理：按直接包含关系排除小 mask；本例会排除 G017、G023、G039、G042，并让 G007、G018 进入后续筛选。最终是否保留还需执行覆盖率和 p/r 筛选。</p></div><h3 class="section-gap">32 个待完成筛选的候选</h3>${toolbar()}${cards(matching(D.items.filter(x=>D.pending.includes(x.id))))}`;}"""+app[b:]
app=app.replace('以下五步发生在主种子形成之后','后续五步发生在主种子形成之后；当前只完成前两步')
app=app.replace('全部最终结果','全部当前结果').replace('卡片标签显示最终去向','卡片标签显示当前状态')
app=app.replace('自动筛选后的保留项见“最终保留”','当前候选及包含关系见“待确认候选”')
app=app.replace('<a class="download" href="downloads/retained_masks.json" download>最终保留清单</a>','<a class="download" href="downloads/all_primary_masks.json" download>43 个原始主种子清单</a>')
app=app.replace('downloads/causality_audit.json','downloads/prefix_audit.json')
app=app.replace('<b>${D.retained.length}</b><span>最终保留 mask · 另有 ${skipped} 张跳过修复</span>','<b>${D.pending.length}</b><span>候选待完成筛选 · 包含关系需确认</span>')
(R/'review/app.js').write_text(app)
html=(C/'review_web/index.html').read_text().replace('当前完成候选筛选，尚未执行修复','包含关系待确认，尚未执行修复').replace('>最终保留<','>待确认候选<')
(R/'review/index.html').write_text(html)
audit=(C/'verify_online.py').read_text()
audit=audit.replace("selection=read(R/'v2/selection.json')","selection=read(R/'review_pending/selection.json')")
audit=audit.replace("final=read(R/'v2/seed_manifest.json')","final=source")
audit=audit.replace("==set(selection['retained_group_ids'])","==set(seeds)")
audit=audit.replace("final_masks_bit_identical_to_source=True","candidate_masks_bit_identical_to_source=True,selection_complete=False,audit_scope='Prefix inputs, votes, reads and all primary candidate masks only'")
audit=audit.replace("retained_masks=len(final['groups'])","pending_masks=len(selection['pending_group_ids'])")
audit=audit.replace("selection_sha256=sha(R/'v2/selection.json')","partial_selection_sha256=sha(R/'review_pending/selection.json')")
audit=audit.replace("'causality_audit.json'","'prefix_audit.json'").replace("'review/downloads/causality_audit.json'","'review/downloads/prefix_audit.json'")
exec(compile(audit,str(C/'build_pending_review.py')+':audit','exec'),dict(__name__='__main__'))
write(R/'pipeline_progress.json',dict(stage='awaiting_containment_review',processed_frames=50,selection_complete=False,review_url='http://127.0.0.1:8798/',chains=chains))
print('PENDING_REVIEW_READY',flush=True)
