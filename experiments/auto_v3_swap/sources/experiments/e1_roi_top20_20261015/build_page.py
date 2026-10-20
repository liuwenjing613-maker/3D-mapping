from pathlib import Path
import json,re,hashlib
H=Path(__file__).resolve().parent;W=H.parents[1]
WEB=W/'results/固定案例_三模型对比_20261006/room2_E1_prerepair_CUT_top20_20261015'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,t):p.write_text(t,encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
D=read(WEB/'viewer_data.json');P=read(WEB/'projection_annotations.json');S=read(WEB/'selection_setup.json');A=read(H/'ranking_audit.json')
assert len(D['cases'])==32 and sum(len(c['views']) for c in D['cases'])==180
assert sum(len(c['rankings']) for c in D['cases'])==60
for g in ['CONFLICT','U','T']:assert sorted(r['rank'] for c in D['cases'] for r in c['rankings'] if r['group']==g)==list(range(1,21))
assert set(D['crops'])==set(P['crops'])
for c in D['cases']:
 for v in c['views']:
  assert v['recommended'] in v['variants']
  for key in v['variants']:assert P['crops'][key]['group']==v['trigger_group']
t=(W/'outputs/room0_selection_20261007/room0_selection_template.html').read_text(encoding='utf-8').replace('room0','room2')
t=t.replace('room2 · 人工修复选择','room2 · E1 修复前种子质量审核')
t=t.replace('14 个固定问题区域 · 42 个视角 · 三种模型并排比较','冲突 / 未分配 / 证据不足各前 20 · 60 个名额去重为 32 个 ROI · 三路分割')
t=t.replace('14 个区域','32 个区域')
t=t.replace('<option value="all">room2 · 全部 32 个区域</option>','<option value="all">全部 · 32 个 ROI</option><option value="CONFLICT">冲突前 20</option><option value="U">未分配前 20</option><option value="T">证据不足前 20</option>')
t=t.replace('按区域查看，换视角确认物体完整。已保存的区域可随时修改。','按各类点数降序。重复 ROI 仅展示一次，保留各榜排名与历史视角。')
t=re.sub(r'<div class="room2Workflow">.*?</div>','<div class="room2Workflow">E1 严格投票修复前地图。先判断候选质量：选一个视角和模型，点选合格 Mask 后保存；没有合格候选就选“都不可靠”。保存仅记录你的判断，导出 JSON 后再决定下一步。</div>',t,count=1)
t=t.replace('例如：餐桌左侧两把椅子；与上一区域是同一把椅子','例如：只选书本；桌子误合并；与 ROI-C0002 的 Mask 9 是同一物体')
a=t.index('const previousReferences=');b=t.index('\nfunction refreshPreviousReference',a)
t=t[:a]+'const previousReferences={};'+t[b:]
t=re.sub(r'const pilotUIDs=new Set\(\[.*?\]\);','const pilotUIDs=new Set('+json.dumps([c['uid'] for c in D['cases']])+');',t,count=1)
t=t.replace('cvpr-room2-selection-f49eb927-20261007-v1','cvpr-room2-e1-prerepair-cut-top20-20261015-v1')
t=t.replace("source_selection:'selection-20261006-f49eb927'","source_selection:'room2-e1-prerepair-CUT-top20-20261015'")
t=t.replace("source_total_fixed_cases:37,scene_scope:'room2',asset_base_path:'../',source_data_directory:'roi_threeway_fixed_20261006/selection-20261006-f49eb927/review',", "source_total_fixed_cases:32,ranked_entries:60,scene_scope:'room2',asset_base_path:'',source_data_directory:"+json.dumps(S['source_data_directory']+'/review')+",baseline_method:'P1-A1-strict-vote / E1 prerepair',baseline_repaired:false,source_candidate_sha256:"+json.dumps(S['source_candidate_sha256'])+",source_surface_sha256:"+json.dumps(S['source_surface_sha256'])+",annotation_complete:D.cases.every(c=>!!savedSelections[c.uid]),case_order:D.cases.map(c=>({uid:c.uid,rankings:c.rankings})),")
t=t.replace("const record={case_uid:c.uid,scene:c.scene,ROI:c.ROI,aliases:c.aliases,", "const record={case_uid:c.uid,scene:c.scene,ROI:c.ROI,aliases:c.aliases,rankings:c.rankings,counts:c.counts,quality_status:chosenModel<0?'FAIL':'PASS',source_entry:v.source_entry,trigger_group:v.trigger_group,")
t=t.replace("'room2 区域 '+c.order+' · 同区域编号 '+c.aliases.join(' / ')","'ROI '+c.order+' / 32 · '+c.rankings.map(r=>({CONFLICT:'冲突',U:'未分配',T:'证据不足'}[r.group])+'第 '+r.rank+'：'+r.count+' 点').join(' ｜ ')")
t=t.replace("'视角 '+v.number+' · 帧 '+v.frame", "v.source_entry+' / 视角 '+v.source_view+' · 帧 '+v.frame")
t=t.replace("b.querySelector('b').textContent=c.scene+' / '+c.ROI;", "b.querySelector('b').textContent=c.aliases.join(' / ');")
t=t.replace("el('scene').append(...Object.entries(D.scene_counts).map(([s,n])=>option(s,s+' · '+n)));",'')
t=t.replace("c.scene===el('scene').value", "c.rankings.some(r=>r.group===el('scene').value)")
# Respect the selected state ranking when choosing its view and list order.
t=t.replace("const saved=savedSelections[c.uid],resume=saved?c.views.findIndex(v=>v.frame===saved.frame&&v.number===saved.view_number):0;", "const saved=savedSelections[c.uid],groupView=c.views.findIndex(v=>v.trigger_group===el('scene').value),resume=el('scene').value!=='all'?Math.max(0,groupView):(saved?c.views.findIndex(v=>v.frame===saved.frame&&v.number===saved.view_number):0);")
t=t.replace("chooseView(Math.max(0,resume),saved?.crop_key)","chooseView(Math.max(0,resume),el('scene').value==='all'?saved?.crop_key:undefined)")
t=t.replace("chooseCase(filtered.includes(activeCase)?activeCase:filtered[0]);", "if(el('scene').value!=='all')filtered.sort((a,b)=>D.cases[a].rankings.find(r=>r.group===el('scene').value).rank-D.cases[b].rankings.find(r=>r.group===el('scene').value).rank);chooseCase(filtered[0]);")
t=t.replace("if(c.uid!==lastProblemCase){el('otherStates').checked=c.group!=='CONFLICT';lastProblemCase=c.uid;}","const marker=c.uid+'|'+meta.group;if(marker!==lastProblemCase){el('otherStates').checked=meta.group!=='CONFLICT';lastProblemCase=marker;}")
t=t.replace("T:'T 状态'", "T:'证据不足'")
t=t.replace('需要修复的完整目标','质量合格的完整目标').replace('要修复的完整目标','质量合格的完整目标')
t=t.replace('选冲突处需要修复的完整物体','选当前 ROI 附近质量合格的完整物体').replace('红点用于定位待检查区域','状态点用于定位待检查区域')
t=t.replace('先选一套方案，再在该栏点击要修复的完整目标。','先选一套方案，再点选质量合格的 Mask。')
t=t.replace('保存本例','保存质量判断').replace('都不可靠','都不可靠 / 不通过')
t=t.replace('room2_修复候选选择_20261007.json','room2_E1修复前三状态前二十_质量判定_20261015.json')
t=t.replace('href="../mask_counts.json"','href="mask_counts.json"').replace('href="../projection_annotation_audit.json"','href="projection_annotation_audit.json"')
t=t.replace('完整目标 + 15% 上下文（默认）','目标区域 + 15% 上下文（默认）')
t=t.replace('保存用四栏对比图（完整目标）','保存四栏对比图（当前默认范围）')
t=t.replace('__DATA__',json.dumps(D,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')).replace('__PROJECTIONS__',json.dumps(P,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
assert 'f49eb927' not in t and '统一 v3' not in t and '20261007' not in t and '__DATA__' not in t
runtime=re.findall(r'<script>\s*(.*?)</script>',t,re.S);assert len(runtime)==1
save(WEB/'index.html',t);save(H/'runtime.js',runtime[0])
plan='room2 / E1 修复前 ROI 种子质量检查\n冲突 C、未分配 U、证据不足 T 各按绝对点数前20；原连通分量编号打破平局。60 个入榜名额，32 个不同 ROI。同一 ROI 跨榜只展示一次，各组保留3个历史视角，共180个视角条目。\n沿用严格投票未修复基线，最终未归属点中的原生 C/U/T，经一环邻接形成完整连通区域；2厘米深度一致性。原图 CropFormer 使用冻结整帧缓存；局部 CropFormer 与 SAM2.1 使用相同裁剪RGB。原始 SAM2 提案保留在服务器，网页沿用旧格式的互斥分区结果。\n网页选择仅记录质量判断，PASS 表示你选中的 Mask 合格，不表示整张分割图或整个 ROI 全部正确；FAIL 表示本次没有可选的可靠 Mask。没有自动追踪、修复或评分。GT 未用于候选生成。标注保存在当前浏览器，请导出JSON。\n'
save(WEB/'plan.txt',plan)
save(WEB/'ranking_audit.json',json.dumps(A,ensure_ascii=False,indent=2))
S.update(status='READY_FOR_HUMAN_QUALITY_REVIEW',baseline_method='P1-A1-strict-vote / E1 prerepair',baseline_repaired=False,case_count=32,ranked_entries=60,view_count=180,annotation_complete=False,ranking_groups={g:20 for g in ['CONFLICT','U','T']},page_sha256=sha(WEB/'index.html'))
save(WEB/'selection_setup.json',json.dumps(S,ensure_ascii=False,indent=2))
print('PAGE_READY',WEB,'cases',32,'views',180,'crops',len(D['crops']))
