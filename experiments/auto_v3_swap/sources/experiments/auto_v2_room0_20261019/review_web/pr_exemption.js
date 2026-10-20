'use strict';
const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n=v=>Number(v).toLocaleString('zh-CN'),fmt=v=>Number(v).toFixed(4);
let report,scope=location.hash==='#extra'?'extra':location.hash==='#confirmed'?'confirmed':'all';
function crop(r){
  const [x0,y0,x1,y1]=r.bbox,[w,h]=r.image_size;
  const cw=Math.min(w,Math.max(120,(x1-x0)+48,((y1-y0)+48)*16/9));
  const ch=Math.min(h,cw*9/16);
  return [Math.max(0,Math.min(w-cw,(x0+x1-cw)/2)),Math.max(0,Math.min(h-ch,(y0+y1-ch)/2)),cw,ch];
}
function imageView(r,type,zoom){
  const [w,h]=r.image_size,box=zoom?crop(r):[0,0,w,h];
  const [x0,y0,x1,y1]=r.bbox;
  return `<svg class="evidence-image" viewBox="${box.join(' ')}" role="img" aria-label="${r.group_id} ${type==='rgb'?'原始 RGB':type==='mask'?'二值 Mask':'Mask 叠加'}${zoom?' · 局部放大':' · 完整原图'}"><image href="${esc(r.images[type])}" width="${w}" height="${h}"/>${!zoom&&type!=='mask'?`<rect x="${Math.max(0,x0-4)}" y="${Math.max(0,y0-4)}" width="${x1-x0+8}" height="${y1-y0+8}" fill="none" stroke="#ffc24a" stroke-width="2" vector-effect="non-scaling-stroke"/>`:''}</svg>`;
}
function reason(r){return `原先 p = ${fmt(r.coverage.p)} > 0.1，r = ${fmt(r.coverage.r)} < 0.2，触发第 5 步排除。`;}
function render(){
  const extra=scope==='extra',all=scope==='all';
  const rows=report.rows.filter(r=>all||(extra?r.eligibility==='broad_only':r.eligibility==='confirmed'));
  document.querySelectorAll('[data-scope]').forEach(b=>b.classList.toggle('active',b.dataset.scope===scope));
  const original=report.original_retained_group_ids.length;
  const total=(scope==='confirmed'?report.confirmed:report.broad).retained_group_ids.length;
  $('#scenarioSummary').innerHTML=`<div><span>最终保留种子</span><strong>${original} <small>→</small> ${total}</strong></div><div><span>本页展示新增</span><strong>${rows.length} <small>张</small></strong></div><div><span>前 4 步筛选</span><strong class="text-stat">照常执行</strong></div>`;
  $('#scopeExplanation').textContent=scope==='confirmed'
    ?'本页 4 张均已通过上游窗户 / 柜子语义确认。仅豁免第 5 步后，由排除变为保留，合计 31 张。'
    :extra?'这 4 张的后续 SegFormer 主类别为窗户 / 柜子，未通过上游确认门槛。按语义类别豁免时，它们在上游确认的 4 张之外额外加入，合计 35 张。G058 的柜子像素占比为 98.39%。'
    :'按后续 SegFormer 主类别为窗户 / 柜子豁免第 5 步，新增 8 张，合计保留 35 张；包括 G058。卡片分别标明语义类别占比和上游确认状态。';
  const type=$('#imageView').value,zoom=$('#zoom').checked;
  $('#viewLegend').textContent=type==='mask'?'白色为原始 mask，黑色为其余区域；点击查看原图对照。':type==='rgb'?'原始 RGB；关闭局部放大可查看黄色框标示的位置。':'青色为原始 mask；点击图像查看原图对照。';
  $('#addedGallery').innerHTML=['window','cabinet'].map(family=>{
    const items=rows.filter(r=>r.family===family).sort((a,b)=>a.group_id.localeCompare(b.group_id));
    return `<section class="family-section"><div class="section-head"><h2>${family==='window'?'窗户':'柜子'} <span class="family-count">${items.length} 张新增</span></h2></div><div class="addition-grid">${items.map(r=>`<article class="addition-card" data-seed="${r.group_id}"><div class="addition-title"><h3>${r.group_id} <span>${r.mask_id}</span></h3><span class="pill ${r.upstream_confirmed?'retain':'skip_repair'}">${r.upstream_confirmed?'上游也已确认':'按后续语义类别新增'}</span></div><button class="preview-image-button" data-open="${r.group_id}" aria-label="查看 ${r.group_id} 原图与 Mask">${imageView(r,type,zoom)}</button><div class="image-caption">${zoom?'局部放大':type==='mask'?'完整二值 Mask':'完整原图 · 黄色框为 mask 位置'} · 帧 ${r.frame} · ${esc(r.source_model)}</div><div class="addition-body"><div class="decision-change"><span class="old-decision">原方案：排除</span><span>→</span><b>豁免后：保留</b></div><p>${esc(reason(r))}</p><p class="semantic-source-note">后续语义：${r.family==='window'?'窗户':'柜子'}（${(r.semantic_top_fraction*100).toFixed(2)}%）；上游确认：${r.upstream_confirmed?'通过':'未通过'}。</p><div class="addition-counts"><span>${n(r.area)} 像素</span><span>${n(r.support_count)} 支撑点</span><span>原始实例 ${r.coverage.instance_id}</span></div>${r.area<300?'<p class="tiny-note">面积很小；可用局部放大检查形状，原始 mask 像素未改变。</p>':''}</div></article>`).join('')}</div></section>`;
  }).join('');
  document.querySelectorAll('[data-open]').forEach(b=>b.onclick=()=>openDetail(b.dataset.open));
}
function figure(r,type,zoom,caption){return `<figure class="figure">${imageView(r,type,zoom)}<figcaption>${caption}</figcaption></figure>`;}
function openDetail(gid){
  const r=report.rows.find(r=>r.group_id===gid),c=r.coverage;
  $('#previewDetailTitle').textContent=`${r.group_id} · ${r.mask_id} · ${r.family==='window'?'窗户':'柜子'}`;
  $('#previewDetailSubtitle').textContent=`帧 ${r.frame} · ${n(r.area)} 像素 · ${n(r.support_count)} 个支撑点`;
  $('#previewDetailBody').innerHTML=`<div class="evidence">${r.upstream_confirmed?'上游语义已确认，严格口径下新增保留。':'未通过上游确认门槛；按后续语义类别豁免第 5 步后保留。'} ${esc(reason(r))}</div><h3>局部放大 · 同一区域对照</h3><div class="pair">${figure(r,'rgb',true,'原始 RGB · 局部')}${figure(r,'overlay',true,'原始 Mask 叠加 · 局部')}</div><h3>完整原图 · 查看位置和上下文</h3><div class="pair">${figure(r,'rgb',false,'原始 RGB · 黄色框为 mask 位置')}${figure(r,'overlay',false,'原始 Mask 叠加 · 像素不变')}</div><h3>原第 5 步依据</h3><p>匹配原始实例 ${c.instance_id}；N = ${n(c.support_count)}，I = ${n(c.instance_visible_count)}，C = ${n(c.intersection)}。p = C/N = ${fmt(c.p)}，r = C/I = ${fmt(c.r)}。</p><p class="note">上游语义确认：${r.upstream_confirmed?'是':'否'}；后续 SegFormer 主类别：${esc(r.semantic_top_class)}，占 mask ${(r.semantic_top_fraction*100).toFixed(2)}%。</p><div class="links"><a href="${esc(r.images.mask)}" download>下载原始二值 Mask</a><a href="${esc(r.images.overlay)}" target="_blank" rel="noopener">打开原尺寸叠加图</a><a href="${esc(r.images.matched_instance)}" target="_blank" rel="noopener">查看与原始实例的覆盖叠图</a></div><p class="note">Mask SHA-256：${r.mask_sha256}</p>`;
  $('#previewDetail').showModal();$('#previewDetail').scrollTop=0;
}
document.querySelectorAll('[data-scope]').forEach(b=>b.onclick=()=>{scope=b.dataset.scope;history.replaceState(null,'','#'+scope);render();});
$('#imageView').onchange=render;$('#zoom').onchange=render;
$('#closePreview').onclick=()=>$('#previewDetail').close();
$('#previewDetail').onclick=e=>{if(e.target===$('#previewDetail'))$('#previewDetail').close();};
fetch('pr_exemption.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(d=>{if(d.status!=='PASS')throw Error('清单校验未通过');report=d;render();}).catch(e=>{$('#addedGallery').innerHTML=`<p class="error">加载失败：${esc(e.message)}</p>`;});
