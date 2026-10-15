const selectionPreferences={deferred:[false,false,false],alternative:[false,false,false],noise:[false,false,false]};
const visibilityPrefixes={deferred:'showDeferred',alternative:'showAlternatives',noise:'showNoise'};
const reasonNames={SINGLE_REMAINING:'仅剩一个相交候选，直接保留',THREE_AGREE:'三种结果两两一致，保留原始',CROPFORMERS_AGREE:'两个 CropFormer 一致，保留原始',CROSS_MODEL_AGREE:'CropFormer 与 SAM 一致，保留 CropFormer',ONE_CROPFORMER_WITH_SAM:'选择与 SAM 一致的 CropFormer',WHOLE_PART_CONFLICT:'整体与子部分的范围冲突',CROP_TRUNCATION_UNCERTAIN:'局部裁剪截断，范围无法可靠比较',MULTIPLE_SUPPORTED_VERSIONS:'多个版本获得支持，尚不能唯一选择',UNRESOLVED_DISAGREEMENT:'候选范围不一致，缺少一致支持'};
Object.assign(reasonNames,{AGREED_MAIN_KEEP:'三者一致的主体已选中，额外片段单独待查',EXTRA_FRAGMENT_REVIEW:'一致主体之外的额外片段，单独待查',TINY_AREA_NOISE:'面积不超过 25 像素，按噪声排除',THIN_LINE_NOISE:'面积不超过 100 像素且细长，按噪声排除'});
Object.assign(reasonNames,{SEMANTIC_SAM_DIRECT:'窗户／柜子识别命中，SAM 直接选中',SEMANTIC_CROPFORMER_REPLACED:'已由对应 SAM 替代，退出后续比较',OVERMERGE:'过合并拟剔除'});
const semanticTerms={window:'窗户','window blinds':'百叶窗',cabinet:'柜子',cupboard:'橱柜',wardrobe:'衣柜',mirror:'镜子',painting:'画',door:'门',curtain:'窗帘',wall:'墙',vent:'通风口',bookshelf:'书架','open shelving unit':'开放书架','storage rack':'置物架',table:'桌子',desk:'书桌',chair:'椅子',sofa:'沙发',bed:'床',ceiling:'天花板',floor:'地板',plant:'植物',flowerpot:'花盆',vase:'花瓶',box:'盒子',book:'书','electrical outlet':'插座','small object':'小物件','indistinct fragment':'不明确碎片'};
const warningNames={TOUCHES_ARTIFICIAL_CROP:'触及人工裁剪边界（仅提示，不作为暂缓理由）',TOUCHES_IMAGE_EDGE:'触及图像边界',SINGLE_HAS_NONINTERSECTING_CONTAINMENT_REFERENCE:'无交集对照存在包含冲突；按单候选规则保留'};
const statusNames={selected:'选中',deferred:'暂缓',alternative:'未选版本',noise:'噪声',no_intersection:'无交集'};
let reviewQuality=null;
let initialSemanticInspectionDone=false;
function selectedStatus(model,id){return loaded?.models[model]?.data.selection.labels[id]?.status||'no_intersection';}
function selectionAllowed(model,id){const s=selectedStatus(model,id);return s==='selected'||(['deferred','alternative','noise'].includes(s)&&selectionPreferences[s][model])||(s==='no_intersection'&&loaded?.models[model]?.showUnrelated);}
function selectionLabel(model,id){const row=loaded?.models[model]?.data.selection.labels[id];if(row?.reason==='SEMANTIC_SAM_DIRECT'){const n=loaded.j.selection.nodes.find(n=>n.key===model+':'+id);return (n.semantic.target_family==='window'?'窗户':'柜子')+' '+id;}if(row?.reason==='SEMANTIC_CROPFORMER_REPLACED')return '已替代 '+id;return (statusNames[selectedStatus(model,id)]||'')+' '+id;}
function maskIsVisible(model,id){return id===0||(selectionAllowed(model,id)&&!loaded?.models[model]?.hiddenMasks?.has(id));}
function decorateIntersectionLegend(model){
 if(!loaded)return;const m=loaded.models[model];
 el('maskLegend'+model).querySelectorAll('.maskRow').forEach(row=>{
  const id=Number(row.dataset.maskID),r=m.data.relevance.labels[id],s=m.data.selection.labels[id];
  row.hidden=!selectionAllowed(model,id);
  let badge=row.querySelector('.intersectionBadge');if(!badge){badge=document.createElement('small');badge.className='intersectionBadge';row.querySelector('label').appendChild(badge);}
  badge.textContent=s.reason==='SEMANTIC_SAM_DIRECT'?'SAM 直接选中':s.reason==='SEMANTIC_CROPFORMER_REPLACED'?'已由 SAM 替代':statusNames[s.status];badge.className='intersectionBadge '+s.status;
  row.title=(s.reason?(s.group?s.group+' · ':'')+reasonNames[s.reason]+'；':'')+'有效表面点 '+r.N+'；扩展后交集 '+r.expanded+(s.warnings.length?'；'+s.warnings.map(x=>warningNames[x]).join('；'):'');
  row.querySelectorAll('input,button').forEach(b=>b.disabled=!selectionAllowed(model,id));
  row.querySelector('.maskSwatch').style.background='rgb('+m.data.colors[id].join(',')+')';
 });
}
function nodeName(k){const [m,id]=k.split(':');return ['原始','局部','SAM'][Number(m)]+' ID '+id;}
function percent(x){return (100*x).toFixed(2)+'%';}
function pairEvidence(e){return nodeName(e.a)+' ↔ '+nodeName(e.b)+'：IoU '+percent(e.iou)+'；覆盖前者 '+percent(e.coverage_a)+' / 后者 '+percent(e.coverage_b);}
function reasonForNode(s,n,g){
 const all=[...g.relations,...g.references],direct=all.filter(e=>e.a===n.key||e.b===n.key);
 if(g.reason==='WHOLE_PART_CONFLICT'){
  const conflicts=all.filter(e=>e.containment),own=conflicts.filter(e=>e.a===n.key||e.b===n.key),e=(own.length?own:conflicts).slice().sort((a,b)=>b.iou-a.iou)[0];
  return {reason:own.length?'本 ID 与另一候选有整体／局部包含冲突':'本 ID 所在比较组存在包含冲突，尚不能单独确定',evidence:(own.length?'':'组内冲突：')+pairEvidence(e)+(conflicts.length>1?'；另有 '+(conflicts.length-1)+' 对包含关系':'')+'。包含判据：小 Mask 覆盖≥95%，IoU<90%。'};
 }
 if(g.reason==='EXTRA_FRAGMENT_REVIEW'){
  const mains=s.groups.filter(x=>x.parent_group===g.parent_group&&x.selected);
  const e=direct.slice().sort((a,b)=>b.iou-a.iou)[0];
  return {reason:'一致主体已选中；这个额外片段单独待查',evidence:'已选主体：'+mains.map(x=>nodeName(x.selected)).join('、')+(e?'。'+pairEvidence(e):'')+'。该片段未满足噪声排除条件。'};
 }
 const e=direct.slice().sort((a,b)=>b.iou-a.iou)[0];
 return {reason:reasonNames[g.reason],evidence:e?pairEvidence(e)+'；'+(e.near?'这一对一致，但整组仍不能唯一确定。':e.mutual_best?'互为最佳匹配，但未达到 IoU≥90% 的一致阈值。':'还需满足互为唯一最佳匹配。'):'没有满足一致选择条件的对应版本。'};
}
function inspectFromReason(n){
 const type=selectedStatus(n.model,n.label);
 if(visibilityPrefixes[type])setFinalVisibility(type,n.model,true);
 if(type==='no_intersection')setUnrelated(n.model,true);
 if(type==='overmerge')setRejectedVisible(n.model,true);
 setMaskVisible(n.model,n.label,true);inspectMask(n.model,n.label,true);
 el('models').scrollIntoView({behavior:'smooth',block:'start'});
}
function appendReasonRow(body,n,reason,evidence){
 const tr=document.createElement('tr');tr.dataset.maskKey=n.key;
 const id=document.createElement('td'),swatch=document.createElement('span');swatch.className='reasonSwatch';swatch.style.background='rgb('+loaded.models[n.model].data.colors[n.label].join(',')+')';id.append(swatch,document.createTextNode(nodeName(n.key)));tr.appendChild(id);
 for(const t of [n.area.toLocaleString()+' px',reason,evidence]){const td=document.createElement('td');td.textContent=t;tr.appendChild(td);}
 const td=document.createElement('td'),button=document.createElement('button');button.textContent='定位此 ID';button.onclick=()=>inspectFromReason(n);td.appendChild(button);tr.appendChild(td);body.appendChild(tr);
}
function refreshDeferredTable(s){
 const body=el('deferredRows'),noiseBody=el('noiseRows');body.replaceChildren();noiseBody.replaceChildren();
 const groups=new Map(s.groups.map(g=>[g.id,g]));
 for(const n of s.nodes.slice().sort((a,b)=>a.model-b.model||a.label-b.label)){
  const row=s.models[n.model].labels[n.label];
  if(row.status==='deferred'){const r=reasonForNode(s,n,groups.get(row.group));appendReasonRow(body,n,r.reason,r.evidence);}
  if(row.status==='noise'){
   const z=n.geometry,why=row.reason==='TINY_AREA_NOISE'?'面积 '+n.area+'≤25 像素':'面积 '+n.area+'≤100 像素；最小旋转包围框宽 '+z.width.toFixed(2)+' px≤2，长宽比 '+z.aspect.toFixed(2)+'≥5';
   appendReasonRow(noiseBody,n,reasonNames[row.reason],why+'；'+(n.eligible?'与扩展异常点有交集':'无交集对照'));
  }
 }
 el('deferredHeading').textContent='逐 ID 暂缓原因 · '+body.children.length+' 个 Mask';
 el('deferredContext').textContent=s.ROI+' · 帧 '+s.frame+' · '+cropLabel(loaded.j)+'。下表始终列出当前范围的全部暂缓 ID；无需先打开图上的暂缓开关。';
 el('deferredEmpty').hidden=body.children.length>0;el('deferredTable').hidden=!body.children.length;
 el('noiseHeading').textContent='查看本范围已排除的噪声 · '+noiseBody.children.length+' 个 Mask';
}
function refreshFinalControls(){
 if(!loaded||loaded.j.key!==activeKey)return;
 const s=loaded.j.selection;let totals={selected:0,deferred:0,alternative:0,noise:0};
 for(let i=0;i<3;i++){
  const m=s.models[i];for(const type of Object.keys(totals))totals[type]+=m[type+'_ids'].length;
  for(const type of Object.keys(visibilityPrefixes)){const t=el(visibilityPrefixes[type]+i);t.disabled=!m[type+'_ids'].length;t.checked=!!m[type+'_ids'].length&&selectionPreferences[type][i];}
  el('meta'+i).textContent='选中 '+m.selected_ids.length+(i===2?'（语义直接选中 '+m.semantic_ids.length+'）':' · 已由 SAM 替代 '+m.semantic_replaced_ids.length)+' · 暂缓 '+m.deferred_ids.length+' · 噪声 '+m.noise_ids.length;
 }
 for(const type of Object.keys(visibilityPrefixes)){
  const t=el(visibilityPrefixes[type]);const shown=s.models.reduce((n,m,i)=>n+(selectionPreferences[type][i]?m[type+'_ids'].length:0),0);
  t.disabled=!totals[type];t.checked=!!totals[type]&&shown===totals[type];t.indeterminate=shown>0&&shown<totals[type];
 }
 el('finalSummary').textContent='当前范围选中 '+totals.selected+' 个 Mask · '+s.groups.filter(g=>g.status==='deferred').length+' 组暂缓（'+totals.deferred+' 个 Mask） · '+totals.alternative+' 个未选版本 · '+totals.noise+' 个噪声';
 refreshDeferredTable(s);
 refreshSemanticTable(s);
 el('noFinal').textContent=totals.selected?'':(totals.deferred?'本范围没有自动选中的 Mask。下方已列出暂缓原因，也可打开“显示暂缓 Mask”检查。':'本范围没有自动选中的 Mask；可检查已排除噪声或切换其他视角／裁剪。');
 const host=el('decisionGroups');host.replaceChildren();
 for(const g of s.groups){
  if(g.status==='deferred'&&!g.members.some(k=>selectionPreferences.deferred[Number(k.split(':')[0])]))continue;
  const details=document.createElement('details');details.className='decisionGroup '+g.status;const summary=document.createElement('summary');
  summary.textContent=g.id+' · '+(g.selected?'选中 '+nodeName(g.selected):'暂缓 '+g.members.length+' 个候选')+' · '+reasonNames[g.reason];details.appendChild(summary);
  const p=document.createElement('p');p.textContent='本组：'+g.members.map(nodeName).join(' / ')+(g.warnings.length?'。提示：'+g.warnings.map(x=>warningNames[x]).join('；'):'');details.appendChild(p);
  const actions=document.createElement('div');actions.className='groupActions';
  for(const k of g.members){const [m,id]=k.split(':').map(Number),b=document.createElement('button');b.textContent='查看 '+nodeName(k);b.onclick=()=>{const t=selectedStatus(m,id);if(t==='deferred'||t==='alternative')setFinalVisibility(t,m,true);setMaskVisible(m,id,true);inspectMask(m,id,false);};actions.appendChild(b);}details.appendChild(actions);
  const table=document.createElement('table');table.innerHTML='<thead><tr><th>完整 Mask 对比</th><th>IoU</th><th>覆盖前者／后者</th><th>关系</th></tr></thead>';const body=document.createElement('tbody');
  for(const e of [...g.relations,...g.references]){const tr=document.createElement('tr');const text=[nodeName(e.a)+' ↔ '+nodeName(e.b)+(e.semantic_replacement?'（CF 已退出）':g.references.includes(e)?'（组外对照）':''),(e.iou*100).toFixed(2)+'%',(e.coverage_a*100).toFixed(2)+'% / '+(e.coverage_b*100).toFixed(2)+'%',e.semantic_replacement?'对应版本，SAM 直接替代':e.containment?'包含冲突':e.near?'互为最佳且近似一致':'存在范围差异'];text.forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.appendChild(td);});body.appendChild(tr);}table.appendChild(body);details.appendChild(table);host.appendChild(details);
 }
 if(!host.childElementCount){const p=document.createElement('p');p.textContent='打开“显示暂缓 Mask”可查看暂缓组及具体关系。';host.appendChild(p);}
 if(!initialSemanticInspectionDone){const wanted=new URLSearchParams(location.search).get('semanticID');initialSemanticInspectionDone=true;if(wanted){const n=s.nodes.find(n=>n.model===2&&n.label===Number(wanted));if(n)setTimeout(()=>{if(loaded?.j.selection===s)inspectFromReason(n);},0);}}
}
function refreshSemanticTable(s){
 const body=el('semanticRows');body.replaceChildren();const nodes=s.nodes.filter(n=>n.model===2),hits=nodes.filter(n=>n.semantic.hit),all=el('showAllSemanticScores').checked;
 el('semanticHeading').textContent='窗户／柜子识别检查 · 本范围直接选中 '+hits.length+' 个 SAM';
 const stats=D.version_selection.stats;el('semanticOverview').textContent='全页直接选中：窗户 '+(stats.semantic_family_occurrences.window||0)+'、柜子 '+(stats.semantic_family_occurrences.cabinet||0)+'；均为视角／裁剪展示次数，不是物体数。本范围共检查 '+nodes.length+' 个 SAM，撤销 '+s.models.reduce((v,m)=>v+m.semantic_replaced_ids.length,0)+' 个 CropFormer。';
 el('semanticEmpty').textContent=hits.length?'':all?'本范围没有命中；下列未命中分数供检查漏认。':'本范围没有命中，可勾选上方选项检查未命中 SAM。';
 for(const n of nodes.filter(n=>all||n.semantic.hit).sort((a,b)=>Number(b.semantic.hit)-Number(a.semantic.hit)||b.semantic.margin-a.semantic.margin)){
  const r=n.semantic,tr=document.createElement('tr');tr.dataset.maskKey=n.key;const id=document.createElement('td'),swatch=document.createElement('span');swatch.className='reasonSwatch';swatch.style.background='rgb('+loaded.models[2].data.colors[n.label].join(',')+')';id.append(swatch,document.createTextNode('SAM ID '+n.label+' · '+n.area.toLocaleString()+' px'));tr.appendChild(id);
  const context=r.cabinet_context,contextText=context?'；周边：'+(semanticTerms[context.target_term]||context.target_term)+' '+context.target_score.toFixed(5)+' / '+(semanticTerms[context.control_term]||context.control_term)+' '+context.control_score.toFixed(5)+' / 差 '+context.margin.toFixed(5):'';
  const edges=s.semantic_replacement_edges.filter(e=>e.b===n.key),details=[r.hit?(r.target_family==='window'?'窗户':'柜子')+' · 直接选中':(context?'柜子周边语义未确认':'未命中')+' · '+(statusNames[s.models[2].labels[n.label].status]||'过合并拟剔除'),(semanticTerms[r.target_term]||r.target_term)+' '+r.target_score.toFixed(5)+' / '+(semanticTerms[r.control_term]||r.control_term)+' '+r.control_score.toFixed(5)+' / 差 '+r.margin.toFixed(5)+contextText,edges.length?edges.map(e=>nodeName(e.a)+'（较小 Mask 覆盖 '+percent(Math.max(e.coverage_a,e.coverage_b))+'）').join('；'):'无'];
  for(const text of details){const td=document.createElement('td');td.textContent=text;tr.appendChild(td);}const actions=document.createElement('td'),b=document.createElement('button');b.textContent='定位 SAM '+n.label;b.onclick=()=>inspectFromReason(n);actions.appendChild(b);
  for(const e of edges){const cf=s.nodes.find(x=>x.key===e.a),b=document.createElement('button');b.textContent='对照 '+nodeName(e.a);b.onclick=()=>inspectFromReason(cf);actions.appendChild(b);}tr.appendChild(actions);body.appendChild(tr);
 }
 const records=D.cabinet_semantic_audit;el('cabinetAuditHeading').textContent='柜子识别复核：'+records.filter(x=>x.semantic.hit).length+' 个通过，'+records.filter(x=>!x.semantic.hit).length+' 个初步命中未获周边支持';
 const host=el('cabinetAuditRows');host.replaceChildren();
 for(const record of records){const r=record.semantic.cabinet_context,tr=document.createElement('tr');tr.dataset.candidateID=record.semantic.candidate_id;
  for(const t of [record.ROI+' / SAM '+record.label,record.area+' px',record.semantic.margin.toFixed(5),(semanticTerms[r.target_term]||r.target_term)+' '+r.target_score.toFixed(5)+' vs '+(semanticTerms[r.control_term]||r.control_term)+' '+r.control_score.toFixed(5)+'；差 '+r.margin.toFixed(5)+(r.passed?'，通过':'，不足 0.03')]){const td=document.createElement('td');td.textContent=t;tr.appendChild(td);}
  const td=document.createElement('td'),link=document.createElement('a');link.textContent='打开对应原图';link.href='?roi='+encodeURIComponent(record.uid)+'&crop='+encodeURIComponent(record.crop_key)+'&semanticID='+record.label+(qaMode?'&qa=1':'')+'#semanticAudit';td.appendChild(link);tr.appendChild(td);host.appendChild(tr);
 }
}
function setFinalVisibility(type,model,value){
 if(!loaded||loaded.j.key!==activeKey)return;
 for(const i of model===null?[0,1,2]:[model]){selectionPreferences[type][i]=value;invalidateVisibleImages(i);refreshMaskVisibility(i);}
 refreshFinalControls();refreshProblemSummary();drawAll();
}
function refreshSelection(){
 if(!D.cases[activeCase])return;const record=savedSelections[D.cases[activeCase].uid];
 el('selectionProgress').textContent=(qaMode?'功能测试 · ':'')+'已检查 '+Object.keys(savedSelections).length+' / '+D.cases.length+' 个 ROI';
 el('savedSelection').textContent=record?'已保存：'+record.quality_status+' · 帧 '+record.frame+' · '+record.auto_selected_masks.length+' 个自动选中 Mask':'本 ROI 尚未保存质量判断';
 document.querySelectorAll('[data-quality]').forEach(b=>{b.classList.toggle('chosen',b.dataset.quality===reviewQuality);b.setAttribute('aria-pressed',String(b.dataset.quality===reviewQuality));});
 el('selectionJson').value=JSON.stringify(exportPayload(),null,2);
 if(loaded&&loaded.j.key===activeKey)refreshFinalControls();
}
function selectionContextChanged(){
 const context=D.cases[activeCase].uid+'|'+activeKey;
 if(context!==selectionContext){selectionContext=context;reviewQuality=null;chosenModel=null;chosenIDs.clear();el('targetNote').value='';const r=savedSelections[D.cases[activeCase].uid];if(r?.crop_key===activeKey){reviewQuality=r.quality_status;el('targetNote').value=r.target_notes||'';}selectionNotice('');}
 refreshSelection();
}
function exportPayload(){return {schema:'cvpr_auto_mask_selection_review_v1',source_selection:D.name,is_qa:qaMode,exported_at:new Date().toISOString(),selection_rule:D.version_selection.rule,source_selection_sha256:D.version_selection.sha256,intersection_filter:D.intersection_filter,overmerge_filter:D.overmerge_filter,baseline_repaired:false,selection_scope:'independent per frame and crop; no cross-view object fusion',selections:D.cases.flatMap(c=>savedSelections[c.uid]?[savedSelections[c.uid]]:[])};}
function saveSelection(moveNext){
 if(!reviewQuality){selectionNotice('请先选择当前自动结果的质量判断。');return;}
 if(!loaded||loaded.j.key!==activeKey){selectionNotice('图像尚未加载完成。');return;}
 const c=D.cases[activeCase],v=c.views[activeView],s=loaded.j.selection;
 savedSelections[c.uid]={case_uid:c.uid,ROI:c.ROI,aliases:c.aliases,frame:v.frame,view_number:v.number,crop_key:activeKey,crop_xyxy:loaded.j.box,quality_status:reviewQuality,auto_selected_masks:s.groups.filter(g=>g.selected).map(g=>({group:g.id,...s.nodes.find(n=>n.key===g.selected),reason:g.reason,warnings:g.warnings})),deferred_groups:s.groups.filter(g=>!g.selected).map(g=>({id:g.id,members:g.members,reason:g.reason})),target_notes:el('targetNote').value.trim(),is_qa:qaMode,selected_at:new Date().toISOString()};
 try{localStorage.setItem(storageKey,JSON.stringify(savedSelections));selectionNotice('已保存当前自动结果的质量判断。');}catch(e){selectionNotice('浏览器存储失败，请立即导出记录。');}
 refreshSelection();buildCases();if(moveNext){const p=filtered.indexOf(activeCase);if(p<filtered.length-1)chooseCase(filtered[p+1]);}
}
function selectFromCanvas(){} // This page reviews automatic choices; clicks only inspect masks.
for(const type of Object.keys(visibilityPrefixes)){
 const prefix=visibilityPrefixes[type];el(prefix).onchange=e=>setFinalVisibility(type,null,e.target.checked);
 for(let i=0;i<3;i++)el(prefix+i).onchange=e=>setFinalVisibility(type,i,e.target.checked);
}
document.querySelectorAll('[data-quality]').forEach(b=>b.onclick=()=>{reviewQuality=b.dataset.quality;refreshSelection();});
el('showAllSemanticScores').onchange=()=>{if(loaded)refreshSemanticTable(loaded.j.selection);};
if(new URLSearchParams(location.search).has('semanticID'))el('showAllSemanticScores').checked=true;
el('exportSelections').onclick=()=>{const blob=new Blob([JSON.stringify(exportPayload(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=(qaMode?'QA_':'')+'room2_E1自动Mask选择_质量检查_20261015.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);selectionNotice('已导出当前质量检查记录。');};
