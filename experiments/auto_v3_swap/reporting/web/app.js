'use strict';
let D,S,oldRows,newRows,ids;
const $=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n=x=>Number(x).toLocaleString('zh-CN'),pct=x=>x==null?'—':(x*100).toFixed(2)+'%',pp=x=>(x>0?'+':'')+(x*100).toFixed(2);
const V={baseline:'strict 无修复',old_E1:'原顺序 · E1',new_E1:'新顺序 · E1',old_partial:'原顺序 · partial E1',new_partial:'新顺序 · partial E1'};
const stages={semantic:'语义背景排除',covisible_iou_only:'共同可见 IoU 去重',full_support_containment:'完整支撑包含',skip_repair_B:'覆盖充分，不修复',pr_filter:'p/r 筛选'};
function terminal(r){return r.decision==='retain'?'保留修复':stages[r.terminal_stage]||r.terminal_stage;}
function reason(r){
 const d=r.stage_decisions.find(s=>s.stage===r.terminal_stage)||r.stage_decisions.at(-1),e=d?.evidence;
 if(r.terminal_stage==='full_support_containment')return e.map(a=>`由 ${a.larger_group_id} 包含：${n(a.intersection)} / ${n(a.small_count)} = ${pct(a.intersection/a.small_count)}`).join('；');
 if(r.terminal_stage==='covisible_iou_only')return `主种子 ${e.primary_group_id}，共同可见 IoU ${pct(e.intersection/e.union)}`;
 if(r.terminal_stage==='semantic')return `高置信背景比例 ${pct(e.high_excluded_pixels/e.area)}`;
 if(['skip_repair_B','pr_filter'].includes(r.terminal_stage))return `p = ${pct(e.p)}，r = ${pct(e.r)}`;
 if(r.terminal_stage==='manual_room2_override')return '沿用历史人工保留；不是窗户 / 柜子语义豁免';
 return S.selection.semantic_pr_exempt_group_ids.includes(r.group_id)?'窗户 / 柜子语义豁免第 5 步':'五步筛选后保留';
}
function identity(g,old=false){
 const r=oldRows.get(g),a=old?r.identity:ids.get(r.original_track_id);
 if(!a)return newRows.get(g).decision==='retain'&&!S.association?'ID 关联待完成':'本方案不进入修复';
 const inherited=old?a.decision==='inherited':a.association==='visible_surface_dice';
 return `${inherited?'沿用旧 ID':'创建新 ID'} ${a.persistent_id}${a.accepted_dice!=null?' · Dice '+a.accepted_dice.toFixed(4):''}`;
}
function card(g,old=false){
 const o=oldRows.get(g),r=old?o:newRows.get(g);
 const previous=o.actually_entered_E1&&r.decision!=='retain'?`<p class="note">原方案：${esc(identity(g,true))}</p>`:'';
 return `<article class="card"><a href="${esc(o.images.overlay)}" target="_blank"><img src="${esc(o.images.overlay)}" alt="${g} ${o.mask_id} 原图 Mask 叠加" loading="lazy"></a><div class="body"><strong>${g} / ${esc(o.mask_id)}</strong> <span class="note">帧 ${o.frame}</span><p><span class="badge ${r.decision==='retain'?'':r.decision==='skip_repair'?'neutral':'bad'}">${terminal(r)}</span><span class="badge neutral">${esc(o.semantic.top_class)}</span></p><p>${esc(reason(r))}</p><p><b>${esc(identity(g,old))}</b></p>${previous}<p class="note">原：${terminal(o)} → 新：${terminal(newRows.get(g))}</p><button data-detail="${g}">查看原图、Mask 与排除依据</button></div></article>`;
}
function table(rows,aggregate=false){
 const order=Object.keys(V),rr=[...rows].sort((a,b)=>order.indexOf(a.variant)-order.indexOf(b.variant));
 return `<table><thead><tr><th>方案</th><th>F1 ↑</th><th>PQ ↑</th><th>AP ↑</th><th>AP50 ↑</th><th>mCov ↑</th><th>TP / FP / FN</th><th>未分配点 ↓</th><th>U / T / C</th></tr></thead><tbody>${rr.map(r=>`<tr class="${r.variant.startsWith('new')?'new':''}"><td>${V[r.variant]}</td><td>${pct(r.F1)}</td><td>${pct(r.PQ)}</td><td>${pct(r.AP)}</td><td>${pct(r.AP50)}</td><td>${pct(r.mCov)}</td><td>${r.TP} / ${r.FP} / ${r.FN}</td><td>${n(r.gray_total)}</td><td>${n(r.gray_states.U)} / ${n(r.gray_states.T)} / ${n(r.gray_states.C)}</td></tr>`).join('')}</tbody></table>`;
}
function renderOverview(){
 const layer=$('layer').value,m=$('metric').value;
 $('aggregate').innerHTML=table(D.summary.filter(r=>r.stage===layer),true);
 const fmt=v=>v==null?'计算中':m==='gray_total'?n(v):pct(v);
 const diff=(a,b)=>a==null||b==null?'—':m==='gray_total'?(b-a>0?'+':'')+n(b-a):pp(b-a)+' pp';
 $('overview').innerHTML='<table><thead><tr><th>场景 / 新种子</th><th>无修复</th><th>原 E1</th><th>新 E1</th><th>E1 Δ</th><th>原 partial</th><th>新 partial</th><th>partial Δ</th><th>执行</th></tr></thead><tbody>'+D.scenes.map(s=>{
  const get=v=>s.rows.find(r=>r.stage===layer&&r.variant===v)?.[m],a=get('old_E1'),b=get('new_E1'),c=get('old_partial'),d=get('new_partial');
  return `<tr><td><button data-scene="${s.scene}">${s.scene} · ${s.old_count}→${s.new_count}</button></td><td>${fmt(get('baseline'))}</td><td>${fmt(a)}</td><td class="new">${fmt(b)}</td><td>${diff(a,b)}</td><td>${fmt(c)}</td><td class="new">${fmt(d)}</td><td>${diff(c,d)}</td><td>${s.status==='PASS'?(s.action==='rerun'?'重算完成':s.action==='reuse_completed_swap'?'复用已完成新顺序':'种子一致，复用'):'运行中'}</td></tr>`;
 }).join('')+'</tbody></table>';
}
function renderMetrics(){
 const rows=S.rows.filter(r=>r.stage===$('layer').value);
 $('sceneMetrics').innerHTML=table(rows);
 $('sceneDelta').innerHTML=['E1','partial'].map(v=>{
  const a=rows.find(r=>r.variant==='old_'+v),b=rows.find(r=>r.variant==='new_'+v);
  return b?`<p>${v==='E1'?'原版 E1':'partial E1'} 换序后：F1 ${pp(b.F1-a.F1)}、PQ ${pp(b.PQ-a.PQ)}、AP ${pp(b.AP-a.AP)} 个百分点；未分配点 ${b.gray_total-a.gray_total>0?'+':''}${n(b.gray_total-a.gray_total)}。</p>`:`<p>${v==='E1'?'原版 E1':'partial E1'} 正在计算。</p>`;
 }).join('');
 const worst=S.per_GT.filter(r=>r.partial_delta<-.01).sort((a,b)=>a.partial_delta-b.partial_delta);
 if(worst.length)$('sceneDelta').innerHTML+=`<p class="worse">partial 对照中下降超过 1 个百分点的对象：${worst.map(r=>`GT ${r.GT_ID}（${esc(r.semantic_class)}） ${pp(r.partial_delta)} pp`).join('；')}。</p>`;
 const tracking=S.tracking_difference_audit;
 if(tracking?.cases?.length)$('sceneDelta').innerHTML+=`<details><summary>同帧组成员变化后的追踪差异</summary><p>移除种子后，原有同帧组整体重跑。剩余种子的源 Mask 与 ID 不变，但传播出的逐帧 Mask 会变化；这些变化包含在本次完整流程的效果中。</p><table><thead><tr><th>新追踪组</th><th>保留的追踪 ID</th><th>变化帧 / 2000</th><th>累计差异像素</th></tr></thead><tbody>${tracking.cases.flatMap(c=>c.tracks.map(t=>`<tr><td>${c.track_ids.join('、')}</td><td>${t.track_id}</td><td>${t.changed_frames}</td><td>${n(t.total_xor_pixels)}</td></tr>`)).join('')}</tbody></table><a href="swap34/reports/${S.scene}/tracking_difference_audit.json" download>下载追踪差异核验</a></details>`;
 if(S.scene==='room1')$('sceneDelta').innerHTML+='<p class="note">room1 原实验有 1 组因成员变化重跑，G067 的 340 / 2000 帧累计变化 1021 个二值像素。<a href="room1_swap34.html#metrics">查看 room1 独立实验详情</a>。</p>';
 $('gt').innerHTML='<table><thead><tr><th>GT / 类别</th><th>无修复</th><th>原 E1</th><th>新 E1</th><th>E1 Δ</th><th>原 partial</th><th>新 partial</th><th>partial Δ</th></tr></thead><tbody>'+[...S.per_GT].sort((a,b)=>Math.abs(b.partial_delta||b.E1_delta||0)-Math.abs(a.partial_delta||a.E1_delta||0)).map(r=>`<tr><td>${r.GT_ID} / ${esc(r.semantic_class)}</td><td>${pct(r.baseline)}</td><td>${pct(r.old_E1)}</td><td>${pct(r.new_E1)}</td><td>${r.E1_delta==null?'—':pp(r.E1_delta)}</td><td>${pct(r.old_partial)}</td><td>${pct(r.new_partial)}</td><td>${r.partial_delta==null?'—':pp(r.partial_delta)}</td></tr>`).join('')+'</tbody></table>';
 $('points').innerHTML='<table><thead><tr><th>比较方向</th><th>已分配→未分配</th><th>未分配→已分配</th><th>已分配 ID 改变</th></tr></thead><tbody>'+Object.entries(S.point_changes).map(([k,r])=>{const [a,b]=k.split('_to_');return `<tr><td>${V[a]} → ${V[b]}</td><td>${n(r.assigned_to_unassigned)}</td><td>${n(r.unassigned_to_assigned)}</td><td>${n(r.assigned_id_changed)}</td></tr>`;}).join('')+'</tbody></table><p class="note">点级变化按 final 计算。净未分配点下降不代表每个点都不会新变成未分配。</p>';
}
function renderFlow(){
 const v=S.old.v1,sourceCount=v.source_candidates??v.input_unique_masks,unique=v.unique_masks;
 const v1line=`<p><b>auto-v1 来源：</b>${sourceCount??'—'} 张候选 → ${v.before_crop_filter_masks??'—'} 张初筛后 → ${v.before_noise_filter_masks??'—'} 张裁剪筛选后 → ${unique??'—'} 张去噪后 → ${S.old.rows.length} 组主种子。${v.auxiliary_masks??'—'} 张辅助 Mask；原始来源和哈希已核验。</p><div class="links"><a href="filtering/${S.scene}/reports/v1_manifest.json" download>auto-v1 种子与辅助 Mask 清单</a><a href="filtering/${S.scene}/reports/v1_audit.json" download>auto-v1 筛选审核</a></div>`;
 $('flow').innerHTML=v1line+[['原顺序',S.order.old_stages],['新顺序',S.order.new_stages]].map(([label,ss])=>`<div class="flowrow"><div><strong>${label}</strong></div>${ss.map((s,i)=>{
  const manual=S.scene==='room2'&&s.stage==='pr_filter',kept=manual?35:s.retained;
  return `<div class="step ${i===2||i===3?'highlight':''}"><div class="small">0${i+1}</div>${stages[s.stage]}<strong>${s.input} → ${kept}</strong><span class="small">减少 ${s.input-kept}${manual?'；含历史人工保留 2 张':''}</span></div>`;
 }).join('')}</div>`).join('')+(S.scene==='room2'?'<p class="note">room2 两种顺序均为自动保留 33 张，加历史人工恢复 G017、G025，实际进入 E1 为 35 张；原始日志可下载核查。</p>':'');
}
function renderGroups(){
 const mode=$('groupMode').value,groups=new Map();
 if(mode==='containment'){
  S.selection.records.filter(r=>r.terminal_stage==='full_support_containment').forEach(r=>r.stage_decisions.find(d=>d.stage==='full_support_containment').evidence.forEach(e=>{if(!groups.has(e.larger_group_id))groups.set(e.larger_group_id,new Set());groups.get(e.larger_group_id).add(r.group_id);}));
  $('groups').innerHTML=[...groups].map(([g,children])=>`<div class="pair"><h3>参照 ${g} · 包含 ${children.size} 张</h3><p>参照 Mask 最终状态：${terminal(newRows.get(g))}。具体包含比例见每张卡片。</p><div class="cards">${[g,...children].map(x=>card(x)).join('')}</div></div>`).join('')||'<p>没有符合条件的包含组。</p>';
 }else if(mode==='coverage'){
  S.selection.records.filter(r=>r.terminal_stage==='skip_repair_B').forEach(r=>{const id=oldRows.get(r.group_id).coverage.instance_id;if(!groups.has(id))groups.set(id,[]);groups.get(id).push(r.group_id);});
  $('groups').innerHTML=[...groups].map(([id,gg])=>`<div class="pair"><h3>同一原实例 ID ${id} · ${gg.length} 张 Mask</h3><p>p 与 r 均 ≥ 95%，覆盖已经充分，不进入修复。</p><div class="cards">${gg.map(g=>card(g)).join('')}</div></div>`).join('')||'<p>没有覆盖充分的 Mask。</p>';
 }else $('groups').innerHTML=S.selection.dedup_groups.filter(g=>g.length>1).map(g=>`<div class="pair"><h3>主种子 ${g[0]} · 共同可见 IoU ≥ 0.8</h3><div class="cards">${g.map(x=>card(x)).join('')}</div></div>`).join('')||'<p>没有重复组。</p>';
}
function renderChanged(){
 $('changedMasks').innerHTML=S.order.removed.map(g=>{
  const r=newRows.get(g),e=r.stage_decisions.find(d=>d.stage==='full_support_containment')?.evidence||[];
  const refs=[...new Set(e.map(x=>x.larger_group_id))];
  return `<div class="pair"><h3>${g} / ${oldRows.get(g).mask_id} 提前排除</h3><p>${esc(reason(r))}</p><div class="cards">${[g,...refs].map(x=>card(x)).join('')}</div></div>`;
 }).join('')||'<p>本场景最终保留种子不变；完整筛选过程已按新顺序重新计算。</p>';
}
function renderCards(){
 const f=$('maskFilter').value,q=$('search').value.trim().toLowerCase().replace(/^id\s*/,'');
 const rr=S.old.rows.filter(o=>{
  const r=newRows.get(o.group_id),a=ids.get(o.original_track_id);
  const ok=f==='all'||f==='retained'&&r.decision==='retain'||f==='old'&&o.actually_entered_E1||f==='removed'&&S.order.removed.includes(o.group_id)||r.terminal_stage===f;
  return ok&&(!q||[o.group_id,o.mask_id,o.frame,a?.persistent_id,o.identity?.persistent_id].join(' ').toLowerCase().includes(q));
 });
 $('maskCount').textContent=`${S.scene}：显示 ${rr.length} / ${S.old.rows.length} 张 v1 主种子；新顺序保留 ${S.new_count}，原顺序 ${S.old_count}。`;
 $('cards').innerHTML=rr.map(o=>card(o.group_id,f==='old')).join('')||'<p>没有符合条件的 Mask。</p>';
}
function renderDownloads(){
 $('ply').innerHTML=Object.keys(V).map(v=>{const files=S.ply.filter(x=>x.variant===v);return `<div class="downloadrow"><b>${V[v]}</b><div class="links">${files.map(x=>`<a href="${esc(x.url)}" download>${x.kind==='instances'?'实例 PLY':'仅三状态点 PLY'} · ${n(x.vertices)} 点</a>`).join('')||'<span class="note">待计算完成后生成。</span>'}</div></div>`;}).join('');
 $('reports').innerHTML=['selection','order','association','source_locks'].map((k,i)=>`<a href="swap34/reports/${S.scene}/${k}.json" download>${['逐 Mask 决策','新旧筛选顺序','实际 ID 关联','评价来源锁'][i]}</a>`).join('')+`<a href="filtering.html?scene=${S.scene}#maskPanel">原顺序完整页面</a>`;
}
function selectScene(name){
 S=D.scenes.find(s=>s.scene===name);$('scene').value=name;oldRows=new Map(S.old.rows.map(r=>[r.group_id,r]));newRows=new Map(S.selection.records.map(r=>[r.group_id,r]));ids=new Map();
 if(S.association)S.association.objects.forEach(r=>r.member_track_ids.forEach(t=>ids.set(t,r)));
 let extra=S.tracking?`追踪整组复用 ${S.tracking.reused_cases} 组，重新推理 ${S.tracking.new_inference_cases} 组。`:'';
 if(S.action==='reuse_identical_selection')extra='最终种子、源 Mask、帧和追踪 ID 相同，核验后复用原版 E1 与 partial 结果。';
 if(S.action==='reuse_completed_swap')extra='复用已完成并交付的 room1 新顺序结果。';
 $('sceneNote').textContent=`${name}：${S.old_count} → ${S.new_count} 张；${S.GT_count} 个 GT。${extra} ${S.association?'保留种子的 ID 关联变化 '+S.identity_changes.length+' 张。':'ID 关联待完成。'}`;
 $('search').value='';renderMetrics();renderFlow();renderGroups();renderChanged();renderCards();renderDownloads();
 const url=new URL(location.href);url.searchParams.set('scene',name);history.replaceState(null,'',url);
}
function showDetail(g){
 const o=oldRows.get(g),r=newRows.get(g);
 const labels={rgb:'原图',mask:'二值 Mask',overlay:'原图叠加',source_segmentation:'来源分割',semantic:'Mask 内语义',whole_semantic:'整帧语义',confidence:'语义置信度'};
 $('detailContent').innerHTML=`<h2>${S.scene} · ${g} / ${esc(o.mask_id)}</h2><p>原：${terminal(o)}；新：${terminal(r)}。${esc(identity(g))}</p><p>${esc(reason(r))}</p><p class="note">${esc(o.model)}；帧 ${o.frame}；原始追踪 ID ${o.original_track_id}。${o.source_correction?esc(JSON.stringify(o.source_correction)):''}</p><div class="pairgrid">${Object.entries(o.images).map(([k,url])=>`<div><h3>${labels[k]||k}</h3><a href="${esc(url)}" target="_blank"><img src="${esc(url)}" alt="${labels[k]||k}"></a></div>`).join('')}</div><details open><summary>新顺序每步判断依据</summary><pre>${esc(JSON.stringify(r.stage_decisions,null,2))}</pre></details><details><summary>原顺序每步判断依据</summary><pre>${esc(JSON.stringify(o.stage_decisions,null,2))}</pre></details><details><summary>实际身份关联与源 Mask 哈希</summary><pre>${esc(JSON.stringify({old:o.identity,new:ids.get(o.original_track_id),mask_sha256:o.mask_sha256},null,2))}</pre></details>`;
 $('detail').showModal();
}
async function main(){
 const response=await fetch('swap34/data.json',{cache:'no-store'});if(!response.ok)throw new Error(response.status);D=await response.json();
 const complete=D.scenes.filter(s=>s.status==='PASS').length;
 $('summary').innerHTML=`<div class="numbers"><div><strong>${complete} / 8</strong><span>结果已完成</span></div><div><strong>201 → 196</strong><span>最终保留种子</span></div><div><strong>637</strong><span>全部 v1 主种子可查</span></div><div><strong>350</strong><span>固定可评价 GT</span></div></div><p>减少：room1 G015、G051；office1 G033；office3 G039；office4 G057。room0、room2、office0、office2 最终保留不变。</p><p class="note">room2 继续保留历史人工例外 G017、G025；本实验不将其算作新增自动语义豁免。</p>`;
 $('scene').innerHTML=D.scenes.map(s=>`<option>${s.scene}</option>`).join('');
 const initial=new URL(location.href).searchParams.get('scene');
 selectScene(D.scenes.some(s=>s.scene===initial)?initial:'office1');renderOverview();
 $('scene').addEventListener('change',()=>selectScene($('scene').value));
 $('layer').addEventListener('change',()=>{renderOverview();renderMetrics();});
 $('metric').addEventListener('change',renderOverview);$('groupMode').addEventListener('change',renderGroups);
 $('maskFilter').addEventListener('change',renderCards);$('search').addEventListener('input',renderCards);
 $('closeDetail').addEventListener('click',()=>$('detail').close());
 document.addEventListener('click',e=>{const detail=e.target.closest('[data-detail]');if(detail)showDetail(detail.dataset.detail);const btn=e.target.closest('[data-scene]');if(btn){selectScene(btn.dataset.scene);$('scenePanel').scrollIntoView();}});
}
main().catch(e=>{$('summary').textContent='数据加载失败：'+e.message;console.error(e);});
