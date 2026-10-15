const selectionPreferences={deferred:[false,false,false],alternative:[false,false,false]};
const reasonNames={SINGLE_REMAINING:'仅剩一个相交候选，直接保留',THREE_AGREE:'三种结果两两一致，保留原始',CROPFORMERS_AGREE:'两个 CropFormer 一致，保留原始',CROSS_MODEL_AGREE:'CropFormer 与 SAM 一致，保留 CropFormer',ONE_CROPFORMER_WITH_SAM:'选择与 SAM 一致的 CropFormer',WHOLE_PART_CONFLICT:'整体与子部分的范围冲突',CROP_TRUNCATION_UNCERTAIN:'局部裁剪截断，范围无法可靠比较',MULTIPLE_SUPPORTED_VERSIONS:'多个版本获得支持，尚不能唯一选择',UNRESOLVED_DISAGREEMENT:'候选范围不一致，缺少一致支持'};
const warningNames={TOUCHES_ARTIFICIAL_CROP:'触及人工裁剪边界',TOUCHES_IMAGE_EDGE:'触及图像边界',SINGLE_HAS_NONINTERSECTING_CONTAINMENT_REFERENCE:'无交集对照存在包含冲突；按单候选规则保留'};
const statusNames={selected:'选中',deferred:'暂缓',alternative:'未选版本',no_intersection:'无交集'};
let reviewQuality=null;
function selectedStatus(model,id){return loaded?.models[model]?.data.selection.labels[id]?.status||'no_intersection';}
function selectionAllowed(model,id){const s=selectedStatus(model,id);return s==='selected'||(s==='deferred'&&selectionPreferences.deferred[model])||(s==='alternative'&&selectionPreferences.alternative[model])||(s==='no_intersection'&&loaded?.models[model]?.showUnrelated);}
function selectionLabel(model,id){return (statusNames[selectedStatus(model,id)]||'')+' '+id;}
function maskIsVisible(model,id){return id===0||(selectionAllowed(model,id)&&!loaded?.models[model]?.hiddenMasks?.has(id));}
function decorateIntersectionLegend(model){
 if(!loaded)return;const m=loaded.models[model];
 el('maskLegend'+model).querySelectorAll('.maskRow').forEach(row=>{
  const id=Number(row.dataset.maskID),r=m.data.relevance.labels[id],s=m.data.selection.labels[id];
  row.hidden=!selectionAllowed(model,id);
  let badge=row.querySelector('.intersectionBadge');if(!badge){badge=document.createElement('small');badge.className='intersectionBadge';row.querySelector('label').appendChild(badge);}
  badge.textContent=statusNames[s.status];badge.className='intersectionBadge '+s.status;
  row.title=(s.group?s.group+' · '+reasonNames[s.reason]+'；':'')+'有效表面点 '+r.N+'；扩展后交集 '+r.expanded+(s.warnings.length?'；'+s.warnings.map(x=>warningNames[x]).join('；'):'');
  row.querySelectorAll('input,button').forEach(b=>b.disabled=!selectionAllowed(model,id));
  row.querySelector('.maskSwatch').style.background='rgb('+m.data.colors[id].join(',')+')';
 });
}
function nodeName(k){const [m,id]=k.split(':');return ['原始','局部','SAM'][Number(m)]+' ID '+id;}
function refreshFinalControls(){
 if(!loaded||loaded.j.key!==activeKey)return;
 const s=loaded.j.selection;let totals={selected:0,deferred:0,alternative:0};
 for(let i=0;i<3;i++){
  const m=s.models[i];for(const type of Object.keys(totals))totals[type]+=m[type+'_ids'].length;
  for(const type of ['deferred','alternative']){const t=el('show'+(type==='deferred'?'Deferred':'Alternatives')+i);t.disabled=!m[type+'_ids'].length;t.checked=!!m[type+'_ids'].length&&selectionPreferences[type][i];}
  el('meta'+i).textContent='最终选中 '+m.selected_ids.length+' · 暂缓 '+m.deferred_ids.length+' · 未选版本 '+m.alternative_ids.length;
 }
 for(const type of ['deferred','alternative']){
  const t=el(type==='deferred'?'showDeferred':'showAlternatives');const shown=s.models.reduce((n,m,i)=>n+(selectionPreferences[type][i]?m[type+'_ids'].length:0),0);
  t.disabled=!totals[type];t.checked=!!totals[type]&&shown===totals[type];t.indeterminate=shown>0&&shown<totals[type];
 }
 el('finalSummary').textContent='当前范围最终选中 '+totals.selected+' 个 Mask · '+s.groups.filter(g=>g.status==='deferred').length+' 组暂缓（'+totals.deferred+' 个 Mask） · '+totals.alternative+' 个未选版本';
 el('noFinal').textContent=totals.selected?'':'本范围没有自动选中的 Mask。可打开“显示暂缓 Mask”检查原因，或切换其他视角／裁剪。';
 const host=el('decisionGroups');host.replaceChildren();
 for(const g of s.groups){
  if(g.status==='deferred'&&!g.members.some(k=>selectionPreferences.deferred[Number(k.split(':')[0])]))continue;
  const details=document.createElement('details');details.className='decisionGroup '+g.status;const summary=document.createElement('summary');
  summary.textContent=g.id+' · '+(g.selected?'选中 '+nodeName(g.selected):'暂缓 '+g.members.length+' 个候选')+' · '+reasonNames[g.reason];details.appendChild(summary);
  const p=document.createElement('p');p.textContent='本组：'+g.members.map(nodeName).join(' / ')+(g.warnings.length?'。提示：'+g.warnings.map(x=>warningNames[x]).join('；'):'');details.appendChild(p);
  const actions=document.createElement('div');actions.className='groupActions';
  for(const k of g.members){const [m,id]=k.split(':').map(Number),b=document.createElement('button');b.textContent='查看 '+nodeName(k);b.onclick=()=>{const t=selectedStatus(m,id);if(t==='deferred'||t==='alternative')setFinalVisibility(t,m,true);setMaskVisible(m,id,true);inspectMask(m,id,false);};actions.appendChild(b);}details.appendChild(actions);
  const table=document.createElement('table');table.innerHTML='<thead><tr><th>完整 Mask 对比</th><th>IoU</th><th>覆盖前者／后者</th><th>关系</th></tr></thead>';const body=document.createElement('tbody');
  for(const e of [...g.relations,...g.references]){const tr=document.createElement('tr');const text=[nodeName(e.a)+' ↔ '+nodeName(e.b)+(g.references.includes(e)?'（组外对照）':''),(e.iou*100).toFixed(2)+'%',(e.coverage_a*100).toFixed(2)+'% / '+(e.coverage_b*100).toFixed(2)+'%',e.containment?'包含冲突':e.near?'互为最佳且近似一致':'存在范围差异'];text.forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.appendChild(td);});body.appendChild(tr);}table.appendChild(body);details.appendChild(table);host.appendChild(details);
 }
 if(!host.childElementCount){const p=document.createElement('p');p.textContent='打开“显示暂缓 Mask”可查看暂缓组及具体关系。';host.appendChild(p);}
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
for(const type of ['deferred','alternative']){
 const prefix=type==='deferred'?'showDeferred':'showAlternatives';el(prefix).onchange=e=>setFinalVisibility(type,null,e.target.checked);
 for(let i=0;i<3;i++)el(prefix+i).onchange=e=>setFinalVisibility(type,i,e.target.checked);
}
document.querySelectorAll('[data-quality]').forEach(b=>b.onclick=()=>{reviewQuality=b.dataset.quality;refreshSelection();});
el('exportSelections').onclick=()=>{const blob=new Blob([JSON.stringify(exportPayload(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=(qaMode?'QA_':'')+'room2_E1自动Mask选择_质量检查_20261015.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);selectionNotice('已导出当前质量检查记录。');};
