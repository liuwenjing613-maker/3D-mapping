'use strict';
const $=x=>document.getElementById(x),esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),pct=x=>(100*x).toFixed(1)+'%';
const names={wall:'墙',floor:'地板',ceiling:'天花板',rug:'地毯',windowpane:'窗户',cabinet:'柜子',door:'门',painting:'挂画',table:'桌子',chair:'椅子',sofa:'沙发',bed:'床',shelf:'置物架',bookcase:'书柜',curtain:'窗帘',mirror:'镜子',lamp:'灯',light:'灯具',box:'盒子',pillow:'枕头',cushion:'靠垫'};
const decisionNames={exclude:'排除',retain:'保留'};
let data,visible=[],selected,decisionScope='all';
function label(n){return names[n]?names[n]+' / '+n:n;}
function imageBox(path,title){return `<figure class="imagebox"><img src="${esc(path)}" alt="${esc(title)}"><figcaption>${esc(title)}</figcaption></figure>`;}
function chooseDecisionGroup(scope){
 decisionScope=scope;$('filter').value=scope==='excluded'?'exclude':scope==='retained'?'retain':'all';$('search').value='';renderList();
}
function renderList(){
 const f=$('filter').value,q=$('search').value.trim().toLowerCase();
 const inScope=r=>decisionScope==='all'||r.exclude===(decisionScope==='excluded');
 visible=data.records.filter(r=>inScope(r)&&(f==='all'||r.decision===f)&&(!q||[r.group_id,r.frame,r.top_class,label(r.top_class)].join(' ').toLowerCase().includes(q)));
 $('count').textContent=visible.length+' / '+data.records.filter(inScope).length+' 个种子'+(decisionScope==='all'?'':' · 本轮'+(decisionScope==='excluded'?'排除组':'保留组'));
 document.querySelectorAll('[data-decision-scope]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.decisionScope===decisionScope)));
 $('decision-all').setAttribute('aria-pressed',String(decisionScope==='all'));
 const scopeUrl=new URL(location.href);if(decisionScope==='all')scopeUrl.searchParams.delete('decision');else scopeUrl.searchParams.set('decision',decisionScope);history.replaceState(null,'',scopeUrl);
 if(!visible.some(r=>r.group_id===selected))selected=visible[0]?.group_id;
 $('seedlist').innerHTML=[true,false].filter(excluded=>decisionScope==='all'||excluded===(decisionScope==='excluded')).map(excluded=>{
  const rows=visible.filter(r=>r.exclude===excluded);
  return `<section class="clip-seed-section"><h3>本轮${excluded?'排除':'保留'} <span>${rows.length}</span></h3>${rows.map(r=>`<button class="seed ${r.group_id===selected?'active':''}" data-group="${esc(r.group_id)}"><b>${esc(r.group_id)}</b><span>${pct(r.high_excluded_fraction)}</span><small>${esc(label(r.top_class))} · ${decisionNames[r.decision]}</small></button>`).join('')||'<p class="muted">无符合当前筛选条件的种子</p>'}</section>`;
 }).join('');
 document.querySelectorAll('[data-group]').forEach(b=>b.onclick=()=>{selected=b.dataset.group;renderList();});renderDetail();
}
function renderDetail(){
 const r=visible.find(r=>r.group_id===selected);if(!r){$('detail').innerHTML='<p class="empty">没有符合筛选条件的种子。</p>';return;}
 const frame=data.frames.find(f=>f.frame===r.frame),cats=r.categories.filter(c=>c.pixels).sort((a,b)=>b.pixels-a.pixels),source=['原图 CropFormer','局部 CropFormer','SAM'][r.model]||r.model;
 const url=new URL(location.href);url.searchParams.set('group',r.group_id);history.replaceState(null,'',url);
 $('detail').innerHTML=`<div class="detail-head"><h2>${esc(r.group_id)}</h2><span class="badge ${r.decision}">${decisionNames[r.decision]}</span><span class="badge">人工：${r.human_decision==='excluded'?'排除':'保留'}</span><span class="badge">CLIP：${r.clip_exclude?'排除':'保留'}</span></div>
 <p class="muted">帧 ${r.frame} · ${esc(source)} · ${r.area.toLocaleString()} 像素 · ${esc(r.mask_id)} · 主类别：${esc(label(r.top_class))}</p>
 <div class="metrics"><div><strong>${pct(r.high_excluded_fraction)}</strong><small>高置信排除类 / Mask</small></div><div><strong>${pct(r.excluded_fraction)}</strong><small>全部排除类 / Mask</small></div><div><strong>${pct(r.low_confidence_fraction)}</strong><small>低置信像素 / Mask</small></div><div><strong>${pct(r.high_retained_fraction)}</strong><small>高置信其他类 / Mask</small></div></div>
 <div class="images">${imageBox(r.images.context,'原图 + 原 Mask 范围')}${imageBox(r.images.masked,'原 Mask 内真实 RGB 像素')}${imageBox(r.images.semantic,'原 Mask 内模型预测类别')}${imageBox(r.images.confidence,'原 Mask 内筛选证据')}</div>
 <div class="legend"><span><i class="swatch" style="background:rgb(240,75,80)"></i>高置信排除类</span><span><i class="swatch" style="background:rgb(27,188,158)"></i>高置信其他类</span><span><i class="swatch" style="background:rgb(246,195,71)"></i>低置信，不计入排除比例</span></div>
 <table class="class-table"><thead><tr><th>预测类别与图例</th><th>全部像素占 Mask</th><th>其中高置信像素占 Mask</th></tr></thead><tbody>${cats.slice(0,10).map(c=>`<tr><td><i class="swatch" style="background:rgb(${data.palette[c.id].join(',')})"></i>${esc(label(c.name))}</td><td>${pct(c.fraction)}</td><td>${pct(c.high_confidence_fraction)}</td></tr>`).join('')}</tbody></table>
 <p class="muted">所有占比分母均为原 Mask 的 ${r.area.toLocaleString()} 个像素。类别表显示占比最高的 10 类；完整记录保存在 JSON 中。</p>
 <details class="fullframes"><summary>查看这一帧的完整 RGB 与语义图</summary><div class="images">${imageBox(frame.rgb_preview,'完整原图 · 同帧所有种子共享一次推理')}${imageBox(frame.semantic_preview,'完整语义分类图 · 150 类')}</div></details>`;
}
async function init(){
 const response=await fetch('binary_comparison.json');if(!response.ok)throw new Error('结果读取失败：'+response.status);data=await response.json();
 const s=data.stats;
 $('decision-groups').innerHTML=[['excluded',true,'本轮排除'],['retained',false,'本轮保留']].map(([scope,excluded,title])=>{
  const rows=data.records.filter(r=>r.exclude===excluded);
  return `<button type="button" class="clip-card" data-decision-scope="${scope}" aria-pressed="false"><span>${title}</span><strong>${rows.length}<small> 个种子</small></strong><p>${excluded?'墙／地／顶／地毯的高置信像素占比达到排除门槛':'未达到排除门槛，全部归入保留组'}</p></button>`;
 }).join('');
 document.querySelectorAll('[data-decision-scope]').forEach(b=>b.onclick=()=>chooseDecisionGroup(b.dataset.decisionScope));$('decision-all').onclick=()=>chooseDecisionGroup('all');
 $('stats').innerHTML=[['all','最终种子',s.total,s.unique_frames+' 张原图'],['exclude','本轮排除',s.excluded,'达到固定排除门槛'],['retain','本轮保留',s.retained,'未达到排除门槛']].map(([f,t,v,n])=>`<button class="stat" data-filter="${f}">${t}<strong>${v}</strong><small>${n}</small></button>`).join('');
 document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>chooseDecisionGroup(b.dataset.filter==='exclude'?'excluded':b.dataset.filter==='retain'?'retained':'all'));
 const c=data.counts;
 $('summary').innerHTML=`<table><thead><tr><th>人工 / 程序</th><th>本轮排除</th><th>本轮保留</th></tr></thead><tbody><tr><td>人工排除 62</td><td>${c.both_excluded}</td><td>${c.human_only_excluded}</td></tr><tr><td>人工保留 28</td><td>${c.auto_only_excluded}</td><td>${c.both_retained}</td></tr></tbody></table><p>CLIP 排除 ${data.clip_stats.excluded} 个 → 本轮排除 ${s.excluded} 个；新增 ${data.versus_clip.semantic_only_excluded.length} 个，撤回 ${data.versus_clip.clip_only_excluded.length} 个。</p>`;
 $('technical').innerHTML=`<p>本轮模型：SegFormer-B5 / ADE20K。像素置信度门槛 0.7，排除类占原 Mask 门槛 90%。达到门槛归入排除，其余全部保留。</p><p>${s.unique_frames} 张图推理合计 ${s.inference_seconds.toFixed(2)} 秒；含加载、图像导出和校验共 ${s.total_seconds.toFixed(2)} 秒。</p><p><a href="selection_binary.json">下载排除／保留两组清单</a> · <a href="binary_comparison.json">完整记录</a></p>`;
 selected=new URLSearchParams(location.search).get('group')||data.records[0].group_id;
 const initialScope=new URLSearchParams(location.search).get('decision');if(['excluded','retained'].includes(initialScope))decisionScope=initialScope;
 $('filter').value=decisionScope==='excluded'?'exclude':decisionScope==='retained'?'retain':'all';$('filter').onchange=()=>{decisionScope=$('filter').value==='exclude'?'excluded':$('filter').value==='retain'?'retained':'all';renderList();};$('search').oninput=renderList;renderList();
}
init().catch(e=>{$('detail').textContent=e.message;console.error(e);});
