'use strict';
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = n => Number(n).toLocaleString('en-US');
const models = ['原始 CropFormer','局部 CropFormer','SAM'];
const reviewLabels = {pass:'分组和主种子通过',merge:'疑似误合并',seed:'主种子不合适',later:'待查'};
let D, selectedGroup = 0, filtered = [], reviews = {}, storageKey, pairMember = null;
let observer, activeLoads = 0, loadQueue = [], imageCache = new Map();
const qa = new URLSearchParams(location.search).has('qa');

function metric(a,b) {
  const intersection = D.intersection[a][b];
  const union = D.seeds[a].support_count + D.seeds[b].support_count - intersection;
  return {intersection,union,iou:union ? intersection / union : 0, pass:union > 0 && 4 * intersection > union};
}
function sourceDescription(s) {
  const a = s.representative;
  return `${models[a.model]} · 原 ID ${a.label}`;
}
function toast(t) { $('#toast').textContent=t;$('#toast').style.display='block';setTimeout(()=>$('#toast').style.display='none',2300); }
function readReviews() {
  try { reviews = JSON.parse(localStorage.getItem(storageKey) || '{}'); }
  catch { reviews = {};toast('检查记录读取失败；当前结果仍可查看。'); }
}
function saveReview(status) {
  const id = D.groups[selectedGroup].id;
  reviews[id] = {...reviews[id],status:status ?? reviews[id]?.status ?? 'later',note:$('#reviewNote').value,updated_at:new Date().toISOString()};
  try { localStorage.setItem(storageKey,JSON.stringify(reviews));$('#saveState').textContent='已保存到当前浏览器'; }
  catch { $('#saveState').textContent='浏览器保存失败，请导出记录'; }
  if(status) renderReview();
  renderList();
}
function renderReview() {
  const id = D.groups[selectedGroup].id, r = reviews[id] || {};
  $('#review').innerHTML = `<b>本组检查</b><div class="reviewButtons">${Object.entries(reviewLabels).map(([key,name])=>`<button data-review="${key}" class="${r.status===key?'active':''}">${name}</button>`).join('')}</div><input id="reviewNote" aria-label="本组检查备注" placeholder="可填备注，例如：M0123 属于旁边的椅子" value="${esc(r.note||'')}"><span id="saveState" class="saved">${r.updated_at?'已保存到当前浏览器':'记录仅保存在当前浏览器'}</span>`;
  $('#review').querySelectorAll('[data-review]').forEach(b=>b.onclick=()=>saveReview(b.dataset.review));
  $('#reviewNote').onchange=()=>saveReview();
}
function renderList() {
  const q = $('#search').value.trim().toUpperCase(), filter = $('#filter').value;
  filtered = D.groups.filter(g=> {
    const r = reviews[g.id]?.status;
    if(filter==='multi' && g.members.length<2) return false;
    if(filter==='single' && g.members.length!==1) return false;
    if(filter==='pending' && r) return false;
    if(filter==='flagged' && !['merge','seed','later'].includes(r)) return false;
    if(!q) return true;
    return [g.id,...g.roi_names,...g.members.map(i=>`${D.seeds[i].id} ${D.seeds[i].frame}`)].join(' ').toUpperCase().includes(q);
  });
  $('#listMeta').textContent=`${filtered.length} / ${D.groups.length} 组 · 已检查 ${Object.keys(reviews).length} 组`;
  $('#groupList').innerHTML=filtered.map(g=>{
    const s=D.seeds[g.primary],r=reviews[g.id]?.status;
    return `<button class="groupBtn ${g.index===selectedGroup?'active':''}" data-group="${g.index}" aria-label="${g.id}，${g.members.length} 张 Mask"><div class="line1"><b>${g.id}</b><span class="dot">${g.members.length} 张 · ${g.frames.length} 帧</span></div><small>主种子 ${s.id} · ${fmt(s.support_count)} 点<br>${r?reviewLabels[r]:'尚未检查'}</small></button>`;
  }).join('') || '<p class="hint">没有符合条件的组</p>';
  $('#groupList').querySelectorAll('[data-group]').forEach(b=>b.onclick=()=>selectGroup(+b.dataset.group));
  updateNav();
}
function updateNav() {
  const pos=filtered.findIndex(g=>g.index===selectedGroup);
  $('#previous').disabled=pos<=0;$('#next').disabled=pos<0||pos===filtered.length-1;
}
function navigate(step) {
  const pos=filtered.findIndex(g=>g.index===selectedGroup),g=filtered[pos+step];if(g) selectGroup(g.index);
}
function selectGroup(index, initial=false) {
  selectedGroup=index;const g=D.groups[index],s=D.seeds[g.primary];
  if(observer) observer.disconnect();loadQueue=[];
  $('#matrixDetails').open=false;$('#outsideDetails').open=false;$('#matrix').innerHTML='';$('#outside').innerHTML='';
  let warnings='';
  if(!g.validation_frames.length) warnings+='<div class="notice">本组没有其他帧的候选。当前可检查分组和种子外观，尚未做追踪验证。</div>';
  if(g.semantic_window_present) warnings+='<div class="notice">窗户／柜子专用规则：本组全部采用 SAM，主种子是其中 TSDF 点数最多的一张。对应的 CropFormer 已撤销，不参与本组分组或排序。</div>';
  $('#groupHeader').innerHTML=`<span class="eyebrow">自动建议的实例组 · 待人工核查</span><h2>${g.id} <span class="groupMeta">${g.members.length} 张 Mask / ${g.frames.length} 帧</span></h2><div class="groupMeta">唯一主种子 <b>${s.id}</b> · <b>${fmt(s.support_count)} TSDF 点</b> · frame ${s.frame} · ${sourceDescription(s)}<br>可供后续验证的其他帧：${g.validation_frames.length} 个；同帧对照：${g.same_frame_references.length} 张。${g.min_primary_iou===null?'单张独立成组。':`成员与主种子的最低 3D IoU：<b>${g.min_primary_iou.toFixed(3)}</b>。`}</div>${warnings}`;
  renderReview();
  $('#cards').innerHTML=g.members.map(i=>cardHTML(i,g.primary)).join('');
  wireCards($('#cards'));
  $('#footer').innerHTML=`输入：上一版语义筛选 r3 的入选 Mask。已全局撤销 ${D.stats.excluded_cf_masks} 张窗户／柜子对应的 CropFormer（${D.stats.excluded_cf_occurrences} 条重复展示），现有 ${D.stats.unique_masks} 张 Mask、${fmt(D.stats.selected_occurrences)} 条有效展示记录。<br>已核对 ${D.stats.semantic_groups} 个窗户／柜子组：CropFormer 成员为 0，CropFormer 主种子为 0。已核对全部 ${D.visibility_rule.verified_frames} 帧的 TSDF 投影及主种子配对；尚未追踪或写回地图。<br>结果版本：${D.display.grouping_sha256.slice(0,16)} · <a href="seed_manifest.json" download>下载种子清单</a> · <a href="semantic_exclusions.json" target="_blank">来源排除记录</a> · <a href="audit.json" target="_blank">计算核验记录</a>`;
  renderList();
  const u=new URL(location.href);u.searchParams.set('group',g.id);history.replaceState(null,'',u);
  if(!initial) $('#groupHeader').scrollIntoView({block:'start',behavior:'instant'});
}
function cardHTML(index, primary, outside=false, dialog=false) {
  const s=D.seeds[index],a=s.representative,isPrimary=index===primary,m=metric(primary,index);
  const sameFrame=s.frame===D.seeds[primary].frame;
  const role=isPrimary?'唯一主种子':outside?`其他组 ${D.groups[s.group].id}`:sameFrame?'同帧对照':'其他帧成员';
  const crop=s.aliases.some(a=>a.touches_crop), semantic=s.aliases.some(a=>a.semantic_hit);
  const aliases=s.aliases.map(a=>`<li>${esc(a.ROI)} · frame ${a.frame} · ${models[a.model]} ID ${a.label}<br>${esc(a.crop_key)}${a.touches_crop?' · 触边提示':''}</li>`).join('');
  return `<article class="card ${isPrimary?'primary':''}" data-index="${index}"><div class="cardHead"><b>${s.id}</b><span class="tag ${isPrimary?'primaryTag':''}">${role}</span></div><button class="picture" data-mask="${index}" data-dialog="${dialog?'1':'0'}" aria-label="放大 ${s.id} 与主种子对照"><canvas></canvas><span class="frameLabel">frame ${s.frame} · ${dialog?'点击上方切换视图':'点击放大对照'}</span></button><div class="cardBody"><div class="metrics"><div><b>${fmt(s.support_count)}</b><span>TSDF 支持点</span></div><div><b>${isPrimary?'主种子':m.iou.toFixed(3)}</b><span>${isPrimary?'本组点数最多':'与主种子的 3D IoU'}</span></div></div><div class="source">${sourceDescription(s)} · ${fmt(s.area)} 像素</div>${semantic?'<div class="source">此前窗户 / 百叶窗语义入选</div>':''}${crop?'<div class="warningLine">触及人工裁剪边缘，仅提示，不影响本次分组</div>':''}${s.aliases.some(a=>a.touches_frame)?'<div class="source">触及原图边缘</div>':''}<details><summary>来源 ${s.aliases.length} 条 · ${s.rois.length} 个 ROI</summary><ul class="aliasList">${aliases}</ul></details><div class="cardLinks"><a href="${esc(s.source_link)}" target="_blank">原 ROI 与原 ID ↗</a>${outside?`<button data-jump="${s.group}">前往 ${D.groups[s.group].id}</button>`:`<span>${isPrimary?'按 TSDF 点数自动选择':sameFrame?'不计为独立验证帧':'保留作后续验证'}</span>`}</div></div></article>`;
}
function wireCards(root) {
  root.querySelectorAll('.picture').forEach(el=>{
    el.onclick=()=>{if(el.dataset.dialog!=='1')openCompare(+el.dataset.mask);};
    if(el.dataset.dialog==='1') enqueue(el);else observer.observe(el);
  });
  root.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>selectGroup(+b.dataset.jump));
}
function newObserver() {
  observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.dataset.visible='1';enqueue(e.target);}else e.target.dataset.visible='0';}),{rootMargin:'180px'});
}
function enqueue(el) { if(!el.isConnected)return;el.renderVersion=(el.renderVersion||0)+1;loadQueue.push([el,el.renderVersion]);drain(); }
function drain() {
  while(activeLoads<3 && loadQueue.length){const [el,v]=loadQueue.shift();if(!el.isConnected || el.renderVersion!==v)continue;activeLoads++;draw(el,v).catch(err=>{if(el.isConnected&&!el.querySelector('.error'))el.insertAdjacentHTML('beforeend',`<span class="error">图片读取失败：${esc(err.message)}</span>`);}).finally(()=>{activeLoads--;drain();});}
}
function loadImage(url,cache=false) {
  if(cache&&imageCache.has(url)){const p=imageCache.get(url);imageCache.delete(url);imageCache.set(url,p);return p;}
  const p=new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error(url));im.src=url;});
  if(cache){imageCache.set(url,p);if(imageCache.size>16)imageCache.delete(imageCache.keys().next().value);}return p;
}
function viewBox(s,full) {
  const W=D.visibility_rule.image_width,H=D.visibility_rule.image_height;
  if(full)return [0,0,W,H];
  const members=D.groups[selectedGroup].members.filter(i=>D.seeds[i].frame===s.frame).map(i=>D.seeds[i]);
  members.push(s);
  let x0=Math.min(...members.map(s=>s.bbox[0])),y0=Math.min(...members.map(s=>s.bbox[1])),x1=Math.max(...members.map(s=>s.bbox[2])),y1=Math.max(...members.map(s=>s.bbox[3]));
  const pad=Math.max(20,Math.max(x1-x0,y1-y0)*.18);return [Math.max(0,x0-pad),Math.max(0,y0-pad),Math.min(W,x1+pad),Math.min(H,y1+pad)];
}
async function draw(el,version) {
  const s=D.seeds[+el.dataset.mask],dialog=el.dataset.dialog==='1';
  const full=dialog?$('#dialogFull').checked:$('#fullFrame').checked,show=dialog?$('#dialogMask').checked:$('#showMask').checked;
  const [rgb,mask]=await Promise.all([loadImage(s.rgb,true),show?loadImage(s.mask):Promise.resolve(null)]);
  if(!el.isConnected||el.renderVersion!==version)return;
  const canvas=el.querySelector('canvas');canvas.width=Math.round(el.clientWidth*1.5);canvas.height=Math.round(el.clientHeight*1.5);
  const ctx=canvas.getContext('2d'),[x0,y0,x1,y1]=viewBox(s,full),bw=x1-x0,bh=y1-y0;
  const scale=Math.min(canvas.width/bw,canvas.height/bh),dw=bw*scale,dh=bh*scale,dx=(canvas.width-dw)/2,dy=(canvas.height-dh)/2;
  ctx.fillStyle='#e8edf3';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(rgb,x0,y0,bw,bh,dx,dy,dw,dh);
  if(mask){
    const layer=document.createElement('canvas');layer.width=canvas.width;layer.height=canvas.height;const lc=layer.getContext('2d',{willReadFrequently:true});
    lc.imageSmoothingEnabled=false;lc.drawImage(mask,x0,y0,bw,bh,dx,dy,dw,dh);const pixels=lc.getImageData(0,0,layer.width,layer.height),raw=new Uint8Array(pixels.width*pixels.height);
    for(let k=0;k<raw.length;k++)raw[k]=pixels.data[k*4]>127?1:0;
    const alpha=+$('#opacity').value/100*255;
    for(let k=0;k<raw.length;k++){
      const at=k*4;if(!raw[k]){pixels.data[at+3]=0;continue;}
      const edge=!raw[k-1]||!raw[k+1]||!raw[k-pixels.width]||!raw[k+pixels.width];
      pixels.data[at]=s.color[0];pixels.data[at+1]=s.color[1];pixels.data[at+2]=s.color[2];pixels.data[at+3]=edge?255:alpha;
    }
    lc.putImageData(pixels,0,0);ctx.drawImage(layer,0,0);
  }
  el.dataset.loaded='true';
}
function refreshPictures(dialogOnly=false) {
  document.querySelectorAll('.picture').forEach(el=>{if(dialogOnly&&el.dataset.dialog!=='1')return;if(el.dataset.visible==='1'||el.dataset.dialog==='1')enqueue(el);});
}
function openCompare(member) {
  pairMember=member;const g=D.groups[selectedGroup],s=D.seeds[member],p=D.seeds[g.primary],m=metric(p.index,member);
  $('#compareTitle').textContent=`${g.id} · ${p.id}${member===p.index?' 主种子放大':` 与 ${s.id} 对照`}`;
  $('#pairNumbers').innerHTML=member===p.index?`主种子 ${p.id}：${fmt(p.support_count)} 个 TSDF 点，${fmt(p.area)} 像素。`:`3D IoU = <b>${fmt(m.intersection)}</b> 个交集点 ÷ <b>${fmt(m.union)}</b> 个并集点 = <b>${m.iou.toFixed(5)}</b>。 ${m.pass?'严格大于 0.25':'未超过 0.25'}。${s.group!==selectedGroup?`该 Mask 已属于 ${D.groups[s.group].id}。`:''}`;
  $('#compareCards').innerHTML=cardHTML(p.index,p.index,false,true)+(member===p.index?'':cardHTML(member,p.index,s.group!==selectedGroup,true));
  $('#compareCards').style.gridTemplateColumns=member===p.index?'1fr':'';
  if(!$('#compare').open)$('#compare').showModal();
  wireCards($('#compareCards'));
}
function renderMatrix() {
  const g=D.groups[selectedGroup];$('#matrix').innerHTML=`<table><thead><tr><th>3D IoU</th>${g.members.map(i=>`<th>${D.seeds[i].id}${i===g.primary?' ★':''}</th>`).join('')}</tr></thead><tbody>${g.members.map(a=>`<tr><th>${D.seeds[a].id}${a===g.primary?' ★':''}</th>${g.members.map(b=>{const m=metric(a,b);return `<td class="${a===b?'diagonal':m.pass?'pass':'low'}" title="交集 ${m.intersection}；并集 ${m.union}">${a===b?'—':m.iou.toFixed(3)}</td>`;}).join('')}</tr>`).join('')}</tbody></table>`;
}
async function init() {
  const response=await fetch('data.json');if(!response.ok)throw new Error(`数据读取失败 ${response.status}`);D=await response.json();
  storageKey=`room2-instance-seeds-${D.display.grouping_sha256}${qa?'-qa':''}`;readReviews();
  const st=D.stats;$('#overview').innerHTML=`<div class="stat"><b>${st.unique_masks}</b>有效 Mask</div><div class="stat"><b>${st.groups}</b>组 / 唯一主种子</div><div class="stat"><b>${st.multi_groups}</b>多成员组</div><div class="stat"><b>${st.singleton_groups}</b>单张组</div><div class="stat"><b>SAM</b>窗户／柜子唯一来源</div><div class="stat"><b>&gt; 0.25</b>普通 3D IoU</div>`;
  newObserver();
  const requested=new URLSearchParams(location.search).get('group');selectedGroup=Math.max(0,D.groups.findIndex(g=>g.id===requested));
  $('#search').oninput=renderList;$('#filter').onchange=renderList;$('#previous').onclick=()=>navigate(-1);$('#next').onclick=()=>navigate(1);
  $('#fullFrame').onchange=()=>refreshPictures();$('#showMask').onchange=()=>refreshPictures();$('#opacity').oninput=()=>refreshPictures();
  $('#dialogFull').onchange=()=>refreshPictures(true);$('#dialogMask').onchange=()=>refreshPictures(true);
  $('#closeCompare').onclick=()=>$('#compare').close();
  $('#compare').addEventListener('click',e=>{if(e.target===$('#compare')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
  $('#matrixDetails').ontoggle=()=>{if($('#matrixDetails').open)renderMatrix();};
  $('#outsideDetails').ontoggle=()=>{if($('#outsideDetails').open){const g=D.groups[selectedGroup];$('#outside').innerHTML=g.nearest_outside.length?g.nearest_outside.map(n=>cardHTML(n.member,g.primary,true)).join(''):'<p class="hint">没有与本组主种子共享 TSDF 点的其他组 Mask。</p>';wireCards($('#outside'));}};
  $('#export').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'room2-instance-seed-human-review-v1',grouping_sha256:D.display.grouping_sha256,exported_at:new Date().toISOString(),reviews},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='room2_instance_seed_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);toast(`已导出 ${Object.keys(reviews).length} 组检查记录`);};
  selectGroup(selectedGroup,true);
  window.addEventListener('resize',()=>refreshPictures());
}
init().catch(e=>{$('#overview').textContent=e.message;console.error(e);});
