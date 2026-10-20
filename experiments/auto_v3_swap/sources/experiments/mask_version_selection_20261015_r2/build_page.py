from pathlib import Path
import json,re,os,hashlib,functools,shutil
H=Path(__file__).resolve().parent;ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006';OLD=ROOT/'room2_E1_intersection_ROI_20261015';OUT=ROOT/'room2_E1_selected_ROI_20261015_r2';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@functools.lru_cache(None)
def reb(s):
 p=(OLD/s).resolve();assert p.is_file(),s;return os.path.relpath(p,OUT).replace('\\','/')
def paths(x):
 if isinstance(x,dict):return {k:paths(v) for k,v in x.items()}
 if isinstance(x,list):return [paths(v) for v in x]
 if isinstance(x,str) and (x.startswith('assets/') or x.startswith('../')):return reb(x)
 return x
before={n:sha(OLD/n) for n in ['index.html','viewer_data.json','selection_setup.json']}
d=paths(read(OLD/'viewer_data.json'));p=paths(read(OLD/'projection_annotations.json'));s=read(H/'selection.json')
d['name']='room2-e1-selected-roi-20261015-r2';d['version_selection']=dict(rule=s['rule'],stats=s['stats'],sha256=sha(H/'selection.json'))
for k,c in d['crops'].items():
 c['selection']=s['crops'][k]
 for i,m in enumerate(c['models']):
  m['selection']=s['crops'][k]['models'][i]
  assert set(m['colors'])==set(m['selection']['labels'])
  assert set(m['selection']['selected_ids']+m['selection']['deferred_ids']+m['selection']['alternative_ids']) | (set(m['selection']['noise_ids']) & set(m['relevance']['eligible_ids'])) == set(m['relevance']['eligible_ids'])
for c in d['cases']:
 keys={k for v in c['views'] for k in v['variants']};c['selected_count']=sum(len(s['crops'][k]['models'][i]['selected_ids']) for k in keys for i in range(3))
save(OUT/'viewer_data.json',d);save(OUT/'projection_annotations.json',p)
html=(OLD/'index.html').read_text(encoding='utf-8')
for tag,value in [('reviewData',d),('projectionData',p)]:
 encoded=json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
 html,n=re.subn(r'(<script id="'+tag+r'" type="application/json">).*?(</script>)',lambda m:m[1]+encoded+m[2],html,flags=re.S);assert n==1
def change(a,b):
 global html
 assert a in html,a[:100];html=html.replace(a,b)
change('room2 · 异常点扩展与相交候选检查','room2 · 自动选择后的 Mask 检查')
change('三类各前 20 · 32 个 ROI · 三维扩展 2 cm · 默认显示相交候选','三类各前 20 · 32 个 ROI · 默认仅显示最终选中 Mask')
html=re.sub(r'<div class="room2Workflow">.*?</div>','<div class="room2Workflow">裁剪触边仅提示；整体／局部等分歧仍暂缓。排除面积≤25 px，或面积≤100 px、宽≤2 px且长宽比≥5的噪声。三者一致的主体单独选中，额外片段待查。默认只显示选中 Mask；下方逐 ID 列出暂缓原因。按原图像素计算，每个视角、裁剪独立选择。</div>',html,count=1)
bar='<div class="finalToolbar"><label><input id="showDeferred" type="checkbox">显示暂缓 Mask</label><label><input id="showAlternatives" type="checkbox">显示未选版本（对照）</label><label><input id="showNoise" type="checkbox">显示已排除噪声</label><a href="#deferredReasons">查看逐 ID 暂缓原因 ↓</a><b id="finalSummary"></b></div><div id="noFinal" role="status"></div><details id="decisions"><summary>查看选择／暂缓依据</summary><div id="decisionGroups"></div></details>'
change('<div class="intersectionToolbar">',bar+'<div class="intersectionToolbar">')
for i in range(3):
 anchor='<div class="intersectionPanel"><label><input type="checkbox" id="showUnrelated'+str(i)+'"'
 repl='<div class="finalPanel"><label><input id="showDeferred'+str(i)+'" type="checkbox">显示本栏暂缓</label><label><input id="showAlternatives'+str(i)+'" type="checkbox">显示本栏未选版本</label><label><input id="showNoise'+str(i)+'" type="checkbox">显示本栏噪声</label></div>'+anchor
 change(anchor,repl)
html=re.sub(r'<button class="modelPick".*?</button>','',html)
manual='''<section class="selection" aria-label="自动结果质量检查"><div class="selectionHead"><span>检查当前自动结果</span><span id="selectionProgress"></span></div><small>当前帧、当前裁剪的自动结果已固定。检查选中的 Mask 是否可用；暂缓项可单独打开核查。</small><div class="selectionActions"><button data-quality="PASS">当前选中结果通过</button><button data-quality="FAIL">存在误选</button><button data-quality="REVIEW">需要进一步检查</button></div><label class="targetNote">检查说明<input id="targetNote" maxlength="500" placeholder="可记录模型、Mask ID、误选或暂缓原因"></label><div class="selectionActions"><button id="saveSelection">保存质量判断</button><button id="saveAndNext">保存并下一个</button><button id="exportSelections">导出检查记录</button></div><div id="savedSelection"></div><div id="selectionMessage" role="status"></div><details><summary>查看检查记录</summary><textarea id="selectionJson" readonly aria-label="检查记录 JSON"></textarea></details></section>'''
reasons='''<section id="deferredReasons" class="reasonSection" aria-label="逐 ID 暂缓原因"><h3 id="deferredHeading">逐 ID 暂缓原因</h3><p id="deferredContext"></p><p id="deferredEmpty" hidden>当前范围没有暂缓 Mask。</p><div class="reasonScroll"><table id="deferredTable"><thead><tr><th>来源 / Mask ID</th><th>完整面积</th><th>暂缓原因</th><th>具体依据</th><th>检查</th></tr></thead><tbody id="deferredRows"></tbody></table></div><details id="noiseReasons"><summary id="noiseHeading">查看已排除噪声</summary><div class="reasonScroll"><table><thead><tr><th>来源 / Mask ID</th><th>完整面积</th><th>排除原因</th><th>判定数值</th><th>检查</th></tr></thead><tbody id="noiseRows"></tbody></table></div></details></section>'''
html,n=re.subn(r'<section class="selection".*?</section>',reasons+manual,html,count=1,flags=re.S);assert n==1
html=re.sub(r'<div class="maskHelp">.*?</div>','<div class="maskHelp">点击 Mask 可高亮，滚轮同步放大。暂缓和未选版本打开后仍用原色，以文字标记区分；每个 ID 的暂缓原因直接列在下方，点击“定位此 ID”可显示并放大检查。</div>',html,count=1)
change('const label=intersectsROI(model,p.id)?String(p.id):\'无交集 \'+p.id,tw=', 'const label=selectionLabel(model,p.id),tw=')
change("const label=row.id===0?'未分割':'ID '+row.id,tw=","const label=row.id===0?'未分割':selectionLabel(model,row.id),tw=")
change("if(!m.hiddenMasks?.size&&!m.data.relevance.no_intersection_ids.length)return {color:m.color,boundary:m.boundary};", "if(!m.hiddenMasks?.size&&maskIDs(model).every(id=>selectionAllowed(model,id)))return {color:m.color,boundary:m.boundary};")
change('prepareRejectedComparison();refreshSelection();','prepareRejectedComparison();refreshSelection();refreshFinalControls();')
change("img.onerror=()=>reject(new Error('图像无法加载：'+src));img.src=src;", "let retried=false;img.onerror=()=>{if(!retried){retried=true;img.src=src+(src.includes('?')?'&':'?')+'retry=1';return;}imageCache.delete(src);reject(new Error('图像无法加载：'+src));};img.src=src;")
change("loadToken;activeKey=key;const j=D.crops[key]", "loadToken;activeKey=key;loaded=null;canvases.forEach(c=>c.getContext('2d').clearRect(0,0,c.width,c.height));el('finalSummary').textContent='正在读取当前范围的选择结果…';el('noFinal').textContent='';el('decisionGroups').replaceChildren();el('deferredRows').replaceChildren();el('noiseRows').replaceChildren();el('deferredHeading').textContent='正在读取暂缓原因…';el('deferredContext').textContent='';for(let i=0;i<3;i++)el('maskLegend'+i).replaceChildren();const j=D.crops[key]")
change("cvpr-room2-e1-intersection-roi-20261015-v1","cvpr-room2-e1-selected-roi-20261015-r2-v1")
change("(savedSelections[c.uid]?'已选 · ':'')","(savedSelections[c.uid]?'已检查 · ':'')+(c.selected_count?'':'全部暂缓 · ')")
change("filtered=D.cases.flatMap((c,i)=>pilotUIDs.has(c.uid)?[i]:[]);\nrefreshSelection();","filtered=D.cases.flatMap((c,i)=>pilotUIDs.has(c.uid)?[i]:[]);")
change('refreshIntersectionControls();refreshProblemSummary();drawAll();','refreshIntersectionControls();refreshFinalControls();refreshProblemSummary();drawAll();')
change("el('problemLabels').checked", "el('problemLabels').checked")
change('<input id="numbers" type="checkbox">','<input id="numbers" type="checkbox" checked>')
change('r.no_intersection_ids.length;total+=n;',"r.no_intersection_ids.filter(id=>selectedStatus(i,id)==='no_intersection').length;total+=n;")
change('相交候选 '+"'+r.eligible_ids.length+'",'初筛相交 '+"'+r.eligible_ids.length+'")
change('window.threewayReview=',(H/'selection_ui.js').read_text(encoding='utf-8')+'\nwindow.threewayReview=')
css='.finalToolbar{display:flex;flex-wrap:wrap;gap:12px;padding:12px;background:#ecf9f2;border:1px solid #9bd3b2;border-radius:8px;margin:12px 0 6px}.finalToolbar label{font-weight:600;font-size:13px}.finalToolbar b{font-size:12px;flex-basis:100%;color:#286449}.finalPanel{padding:8px 10px;background:#f1f8f4;display:flex;gap:10px;flex-wrap:wrap;font-size:12px}.intersectionBadge.selected{color:#147644}.intersectionBadge.deferred{color:#976100}.intersectionBadge.alternative{color:#5c6b86}#noFinal{color:#956213;font-size:13px;padding:4px 0}#decisions{font-size:13px;margin:7px 0}.decisionGroup{border:1px solid #dae4df;border-radius:6px;padding:8px;margin:6px 0;background:#fff}.decisionGroup.deferred{border-color:#e2c796}.decisionGroup p{font-size:12px;margin:6px 0}.decisionGroup table{width:100%;border-collapse:collapse;font-size:11px}.decisionGroup td,.decisionGroup th{padding:6px;border-bottom:1px solid #eee;text-align:left}.groupActions{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}.groupActions button{font-size:11px}.problemInfo{display:none}.maskRow[hidden]{display:none!important}.intersectionToolbar{font-size:12px}.selectionActions button.chosen{background:#1a694f;color:white}'
css+='.reasonSection{background:#fff;border:1px solid #dce3e9;border-radius:9px;margin:16px 0;padding:15px}.reasonSection h3{margin:0 0 7px;font-size:17px}.reasonSection p{font-size:12px;color:#566578;margin:6px 0 12px}.reasonScroll{overflow:auto}.reasonSection table{width:100%;border-collapse:collapse;font-size:12px;line-height:1.6;min-width:750px}.reasonSection th{background:#f4f6f8;font-weight:600;white-space:nowrap}.reasonSection td,.reasonSection th{text-align:left;padding:10px;border-bottom:1px solid #e5e9ee;vertical-align:top}.reasonSection th:first-child,.reasonSection td:first-child{min-width:120px;white-space:nowrap}.reasonSection th:nth-child(3){min-width:180px}.reasonSection td:nth-child(4){min-width:300px}.reasonSection td:nth-child(2){white-space:nowrap}.reasonSection button{white-space:nowrap;font-size:12px}.reasonSwatch{display:inline-block;width:11px;height:11px;border:1px solid #566578;border-radius:2px;margin-right:5px}.reasonSection details{margin-top:15px}.reasonSection summary{font-size:13px;cursor:pointer}.intersectionBadge.noise{color:#9c3d79}.finalToolbar a{font-size:13px;align-self:center}'
change('</style>',css+'</style>')
(OUT/'index.html').write_text(html,encoding='utf-8');runtime=re.findall(r'<script>\s*(.*?)</script>',html,re.S);assert len(runtime)==1;(H/'runtime.js').write_text(runtime[0],encoding='utf-8')
for name in ['selection.json','audit.json','revision_changes.json']:shutil.copyfile(H/name,OUT/name)
projection_audit=read(OLD/'projection_annotation_audit.json');projection_audit['intersection_audit_file']='../room2_E1_intersection_ROI_20261015/audit.json';projection_audit['selection_audit_file']='audit.json';save(OUT/'projection_annotation_audit.json',projection_audit)
save(OUT/'mask_counts.json',dict(stats=s['stats'],crops=[dict(key=k,selected=[len(m['selected_ids']) for m in v['models']],deferred=[len(m['deferred_ids']) for m in v['models']],alternative=[len(m['alternative_ids']) for m in v['models']],noise=[len(m['noise_ids']) for m in v['models']]) for k,v in s['crops'].items()]))
setup=read(OLD/'selection_setup.json');setup.update(status='READY_FOR_AUTO_SELECTION_REVIEW',name=d['name'],version_selection=d['version_selection'],isolated_storage_key='cvpr-room2-e1-selected-roi-20261015-r2-v1',page_sha256=sha(OUT/'index.html'),export_filename='room2_E1自动Mask选择_质量检查_20261015.json');save(OUT/'selection_setup.json',setup)
(OUT/'plan.txt').write_text('同一帧、同一裁剪独立比较三种来源的完整2D Mask，原图候选使用全帧完整范围。通过过合并和异常扩展交集筛选的候选之间，交集/较小面积>=0.50连接为比较组。IoU>=0.90且互为唯一最佳匹配为近似一致；交集/较小面积>=0.95且IoU<0.90为包含冲突。无交集且未被过合并拒绝的Mask作为关系参考，不作为种子，也不把不同组连接起来。单候选按用户要求直接保留，边界和组外包含冲突仅提示。多候选优先检查整体—部件冲突；无冲突且三者两两一致保留原始；仅两个CropFormer一致也按用户确认保留原始；唯一CropFormer与SAM一致则保留该CropFormer，其余分歧暂缓。裁剪触边仅为提示，不再作为暂缓条件；其他包含和范围分歧仍按原规则暂缓。三者两两一致的主体从带有额外片段的比较组中单独选出，额外非噪声片段单独待查。面积<=25像素，或面积<=100像素且旋转包围框宽<=2像素、长宽比>=5，作为噪声排除；噪声不参与匹配和分组。宽度为像素中心最小旋转包围框短边加1，长边同理。无语义打分，无异常点数量排序，无跨视角物体融合，无地图修复。当前结果是自动提名，等待人工质量检查。数字均注明候选或展示次数，不等于物体数。页面默认只显示选中项；暂缓、未选版本、噪声、无交集和过合并对照可分别打开；下方无条件列出当前范围全部暂缓ID、原因和具体数值，所有非过合并对照保持原色及统一透明度。',encoding='utf-8')
assert before=={n:sha(OLD/n) for n in before};report=dict(status='PASS',ROIs=len(d['cases']),crops=len(d['crops']),unique_asset_paths_verified=reb.cache_info().currsize,old_page_unchanged=True,stats=s['stats']);save(H/'page_verification.json',report);print(json.dumps(report))
