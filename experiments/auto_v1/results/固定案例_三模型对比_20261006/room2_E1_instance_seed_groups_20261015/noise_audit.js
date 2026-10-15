'use strict';
let auditMode='noise', auditPage=0, noiseReviews={}, controlReviews={};
const pointCache=new Map();
const reasonNames={AREA_LT_100:'小于 100 像素',THIN_EDGE_BAND:'细线／碎点',SPARSE_EDGE_OUTLINE:'稀疏边框',LARGE_SPARSE_EDGE_OUTLINE:'较大稀疏边框'};
function auditSeed(kind,index){return kind==='noise'?D.excluded_noise_masks[index]:D.seeds[index];}
function percent(n,total){return (100*n/total).toFixed(1)+'%';}
function auditReason(s){
  const f=s.noise_features,decision=s.noise_decision;
  if(!decision.exclude)return `保留：未命中本轮规则；面积 ${fmt(f.area)} 像素，距边界超过 3 像素的内部占 ${percent(f.core_pixels['3'],f.area)}。`;
  if(decision.reason==='AREA_LT_100')return `面积 ${fmt(f.area)} 像素 < 100。按你指定的面积规则直接排除。`;
  const r=decision.reason==='THIN_EDGE_BAND'?3:decision.reason==='SPARSE_EDGE_OUTLINE'?8:12;
  return `跨度 ${f.extent} 像素；${percent(f.area-f.core_pixels[String(r)],f.area)} 的像素距自身边界 ≤ ${r} 像素。${r===3?'':`Mask 仅占包围框 ${percent(f.area,f.bbox_area)}。`}`;
}
function setAuditView(view){
  auditMode=['noise','controls','seeds'].includes(view)?view:'noise';
  document.body.classList.toggle('auditView',auditMode!=='seeds');
  $('#noiseAudit').hidden=auditMode==='seeds';$('#seedPanel').hidden=auditMode!=='seeds';
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===auditMode));
  const u=new URL(location.href);u.searchParams.set('view',auditMode);history.replaceState(null,'',u);
  if(auditMode!=='seeds')renderNoiseAudit();else refreshPictures();
}
function reviewHTML(s,kind){
  const store=kind==='noise'?noiseReviews:controlReviews,r=store[s.id]||{};
  const labels=kind==='noise'?{pass:'排除正确',restore:'误删，需恢复',later:'待查'}:{pass:'保留合适',remove:'仍是噪声',later:'待查'};
  return `<div class="maskReview"><div>${Object.entries(labels).map(([key,label])=>`<button class="${r.status===key?'active':''}" data-audit-review="${key}" data-id="${s.id}" data-kind="${kind}">${label}</button>`).join('')}</div><input aria-label="${s.id} 检查备注" placeholder="检查备注（只保存意见）" data-audit-note="${s.id}" data-kind="${kind}" value="${esc(r.note||'')}"></div>`;
}
function auditCardHTML(index,kind,dialog=false){
  const s=auditSeed(kind,index),f=s.noise_features;
  return `<article class="card auditCard ${kind==='noise'?'noiseCard':'controlCard'}"><div class="cardHead"><b>${s.id}</b><span class="tag ${kind==='noise'?'warning':''}">${kind==='noise'?'已排除 · '+reasonNames[s.reason]:'保留对照 · 未确认质量'}</span></div><button class="picture auditPicture" data-audit-kind="${kind}" data-audit-index="${index}" data-dialog="${dialog?'1':'0'}" aria-label="放大核查 ${s.id}"><canvas></canvas></button><div class="cardBody"><div class="metrics"><div><b>${fmt(s.area)}</b><span>原图像素面积</span></div><div><b>${fmt(s.support_count)}</b><span>TSDF 点</span></div><div><b>${f.components}</b><span>连通片段</span></div></div><div class="reasonText">${esc(auditReason(s))}</div><div class="source">frame ${s.frame} · ${sourceDescription(s)}<br>上版 ${s.previous_group}${s.previous_primary?' · 上版主种子':''}${kind==='control'?` → 当前 ${D.groups[s.group].id}`:''}</div><details><summary>测量值与全部来源</summary><p>包围框 ${f.bbox_width} × ${f.bbox_height} 像素，填充率 ${percent(f.area,f.bbox_area)}；最厚处到背景距离 ${f.max_radius.toFixed(2)} 像素。${s.noise_decision.matched_rules.length?`命中：${s.noise_decision.matched_rules.map(k=>reasonNames[k]).join('、')}。`:''}</p><ul class="aliasList">${s.aliases.map(a=>`<li>${esc(a.ROI)} · ${models[a.model]} ID ${a.label} · ${esc(a.crop_key)}</li>`).join('')}</ul></details>${dialog?'':reviewHTML(s,kind)}<div class="cardLinks"><a href="${esc(s.source_link)}" target="_blank">原 ROI 与原 ID ↗</a>${kind==='control'?`<button data-audit-group="${s.group}">查看当前组</button>`:'<span>不参与种子选择</span>'}</div></div></article>`;
}
function renderNoiseAudit(){
  const kind=auditMode==='controls'?'control':'noise', controls=kind==='control';
  const all=controls?D.seeds:D.excluded_noise_masks;
  const q=$('#auditSearch').value.trim().toUpperCase(), reason=$('#auditReason').value;
  const ids=all.map((s,i)=>i).filter(i=>{
    const s=all[i];if(controls&&!s.noise_review_control)return false;
    if(!controls&&reason==='small'&&s.reason!=='AREA_LT_100')return false;
    if(!controls&&reason==='thin'&&s.reason!=='THIN_EDGE_BAND')return false;
    if(!controls&&reason==='outline'&&!s.reason.includes('OUTLINE'))return false;
    return !q||[s.id,s.previous_group,s.frame,...s.roi_names].join(' ').toUpperCase().includes(q);
  });
  const total=controls?D.stats.review_control_masks:D.stats.excluded_noise_masks;
  const pages=Math.max(1,Math.ceil(ids.length/12));auditPage=Math.max(0,Math.min(auditPage,pages-1));
  $('#auditTitle').textContent=controls?`保留对照 · ${total} 张`:`本轮排除 · ${total} 张全部可查`;
  $('#auditIntro').textContent=controls?'这些候选未命中噪声规则，仍参加分组。请检查是否有漏筛；保留不等于质量已通过。':'全部排除项均在下方，可筛选原因、标记误删。此前人工裁剪触边记录在“重选后的种子”页下方。';
  const c=D.stats.noise_reason_counts;
  $('#auditSummary').innerHTML=controls?'<span>有内部实体区域的小目标保留</span><span>细杆、中空书架等不能仅凭形状认定为噪声</span>':`<span><b>${c.AREA_LT_100||0}</b> 小于 100 像素</span><span><b>${c.THIN_EDGE_BAND||0}</b> 细线／碎点</span><span><b>${(c.SPARSE_EDGE_OUTLINE||0)+(c.LARGE_SPARSE_EDGE_OUTLINE||0)}</b> 稀疏边框</span><span><b>${D.stats.previous_groups_emptied}</b> 上版组全部被排除</span>`;
  $('#auditReason').hidden=controls;$('#showExample').hidden=controls;
  $('#auditCount').textContent=`${ids.length} / ${total} 张 · 第 ${auditPage+1}/${pages} 页`;
  $('#auditPrevious').disabled=auditPage===0;$('#auditNext').disabled=auditPage===pages-1;
  $('#auditCards').innerHTML=ids.slice(auditPage*12,auditPage*12+12).map(i=>auditCardHTML(i,kind)).join('')||'<p class="empty">没有匹配的 Mask。点击“显示全部”恢复列表。</p>';
  wireCards($('#auditCards'));wireAuditReviews($('#auditCards'));
}
function wireAuditReviews(root){
  root.querySelectorAll('[data-audit-review]').forEach(b=>b.onclick=()=>{
    const store=b.dataset.kind==='noise'?noiseReviews:controlReviews;
    const note=b.closest('.maskReview').querySelector('input').value;
    store[b.dataset.id]={...store[b.dataset.id],note,status:b.dataset.auditReview,updated_at:new Date().toISOString()};saveAuditReviews();
    b.parentElement.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));
  });
  root.querySelectorAll('[data-audit-note]').forEach(el=>el.oninput=()=>{
    const store=el.dataset.kind==='noise'?noiseReviews:controlReviews;
    store[el.dataset.auditNote]={...store[el.dataset.auditNote],note:el.value,updated_at:new Date().toISOString()};saveAuditReviews(true);
  });
  root.querySelectorAll('[data-audit-group]').forEach(b=>b.onclick=()=>{selectGroup(+b.dataset.auditGroup);setAuditView('seeds');});
}
function saveAuditReviews(silent=false){
  try{localStorage.setItem(storageKey+'-mask-audit',JSON.stringify({noiseReviews,controlReviews}));if(!silent)toast('已保存检查意见；本轮结果未改动');}
  catch{toast('浏览器保存失败，请导出检查记录');}
}
async function maskPoints(s){
  if(pointCache.has(s.mask))return pointCache.get(s.mask);
  const im=await loadImage(s.mask),c=document.createElement('canvas');c.width=im.width;c.height=im.height;
  const ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(im,0,0);
  const a=ctx.getImageData(0,0,c.width,c.height).data,points=[];
  for(let k=0;k<a.length;k+=4)if(a[k]>127){const p=k/4;points.push(p%c.width,Math.floor(p/c.width));}
  const value=new Int32Array(points);pointCache.set(s.mask,value);if(pointCache.size>32)pointCache.delete(pointCache.keys().next().value);return value;
}
async function drawAudit(el,version){
  const s=auditSeed(el.dataset.auditKind,+el.dataset.auditIndex),dialog=el.dataset.dialog==='1';
  const [rgb,points]=await Promise.all([loadImage(s.rgb,true),maskPoints(s)]);
  if(!el.isConnected||el.renderVersion!==version||!el.clientWidth||!el.clientHeight)return;
  const c=el.querySelector('canvas');c.width=Math.round(el.clientWidth*1.5);c.height=Math.round(el.clientHeight*1.5);
  const ctx=c.getContext('2d'),full=dialog?$('#dialogFull').checked:$('#auditFull').checked;
  const overlay=dialog?$('#dialogMask').checked:$('#auditOverlay').checked;
  const enhance=$('#auditEnhance').checked;
  let box=viewBox(s,full,true);
  if(!full){const b=s.bbox,pad=Math.max(40,Math.max(b[2]-b[0],b[3]-b[1])*.18);box=[Math.max(0,b[0]-pad),Math.max(0,b[1]-pad),Math.min(rgb.width,b[2]+pad),Math.min(rgb.height,b[3]+pad)];}
  const [x0,y0,x1,y1]=box,bw=x1-x0,bh=y1-y0;
  const modes=dialog?['original','overlay','pure']:['overlay','pure'];
  const gap=6,pw=(c.width-gap*(modes.length-1))/modes.length,top=34,ph=c.height-top;
  ctx.fillStyle='#e8edf3';ctx.fillRect(0,0,c.width,c.height);
  modes.forEach((mode,j)=>{
    const ox=j*(pw+gap),scale=Math.min(pw/bw,ph/bh),dw=bw*scale,dh=bh*scale,dx=ox+(pw-dw)/2,dy=top+(ph-dh)/2;
    ctx.fillStyle=mode==='pure'?'#101c2b':'#e8edf3';ctx.fillRect(ox,top,pw,ph);
    if(mode!=='pure')ctx.drawImage(rgb,x0,y0,bw,bh,dx,dy,dw,dh);
    ctx.fillStyle='#29435e';ctx.font='bold 16px "Microsoft YaHei",sans-serif';
    ctx.fillText(mode==='original'?'原图':mode==='pure'?'纯 Mask · 原色':overlay?'原图 + 原色 Mask':'原图（叠加已关闭）',ox+9,23);
    if(mode==='pure'||mode==='overlay'&&overlay){
      ctx.save();ctx.beginPath();ctx.rect(dx,dy,dw,dh);ctx.clip();
      ctx.fillStyle=`rgb(${s.color.join(',')})`;
      // Forward-map every true source pixel, so isolated pixels survive downsampling.
      // An optional screen-space minimum mark improves visibility only.
      const size=Math.max(enhance?2.4:1,scale),extra=(size-scale)/2;
      ctx.globalAlpha=mode==='pure'||enhance?1:+$('#opacity').value/100;
      for(let k=0;k<points.length;k+=2){const x=points[k],y=points[k+1];if(x<x0||x>=x1||y<y0||y>=y1)continue;ctx.fillRect(dx+(x-x0)*scale-extra,dy+(y-y0)*scale-extra,size,size);}
      ctx.restore();
    }
  });
  el.dataset.loaded='true';
}
function openAudit(index,kind){
  const s=auditSeed(kind,index);
  $('#compareTitle').textContent=`${s.id} · ${kind==='noise'?'已排除：'+reasonNames[s.reason]:'保留对照'} · frame ${s.frame}`;
  $('#pairNumbers').textContent=auditReason(s)+' 三幅使用相同显示范围。细线增强只影响显示。';
  $('#compareCards').innerHTML=auditCardHTML(index,kind,true);$('#compareCards').style.gridTemplateColumns='1fr';
  if(!$('#compare').open)$('#compare').showModal();wireCards($('#compareCards'));wireAuditReviews($('#compareCards'));
}
function initNoiseAudit(){
  try{const r=JSON.parse(localStorage.getItem(storageKey+'-mask-audit')||'{}');noiseReviews=r.noiseReviews||{};controlReviews=r.controlReviews||{};}catch{toast('检查意见读取失败，页面仍可查看');}
  document.querySelector('[data-view="noise"]').textContent=`本轮排除 ${D.stats.excluded_noise_masks} 张`;
  document.querySelector('[data-view="controls"]').textContent=`保留对照 ${D.stats.review_control_masks} 张`;
  document.querySelector('[data-view="seeds"]').textContent=`重选后的 ${D.stats.groups} 个种子`;
  document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{auditPage=0;$('#auditSearch').value='';setAuditView(b.dataset.view);});
  $('#auditSearch').oninput=()=>{auditPage=0;renderNoiseAudit();};
  $('#auditReason').onchange=()=>{auditPage=0;renderNoiseAudit();};
  $('#auditPrevious').onclick=()=>{auditPage--;renderNoiseAudit();$('#auditTitle').scrollIntoView({block:'start'});};
  $('#auditNext').onclick=()=>{auditPage++;renderNoiseAudit();$('#auditTitle').scrollIntoView({block:'start'});};
  $('#auditReset').onclick=()=>{$('#auditSearch').value='';$('#auditReason').value='all';auditPage=0;renderNoiseAudit();};
  $('#showExample').onclick=()=>{$('#auditSearch').value='M0734';$('#auditReason').value='all';auditPage=0;renderNoiseAudit();};
  ['auditFull','auditOverlay','auditEnhance'].forEach(id=>$('#'+id).onchange=()=>refreshPictures());
  setAuditView(new URLSearchParams(location.search).get('view')||'noise');
}
