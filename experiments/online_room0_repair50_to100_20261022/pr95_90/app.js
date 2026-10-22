'use strict';
let data=null,current=null;
const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>v==null?'—':Number(v).toLocaleString('en-US');
const pct=v=>v==null?'—':(100*v).toFixed(3)+'%';
const models=['CropFormer · 原图','CropFormer · 局部','SAM'];
const stageNames={semantic:'1 背景语义',covisible_iou_only:'2 共视去重',full_support_containment:'3 直接包含',skip_repair_B:'4 新 95/90 覆盖率',pr_filter:'5 原 p/r 与语义豁免'};
function exemption(r){
 if(r.stage5_restored_by_semantics)return '第5步窗/柜豁免实际恢复此候选。';
 if(r.stage5_semantic_eligible&&!r.stage5_semantic_exemption_in_scope)return '背景预测属于窗/柜，但在第5步之前已排除；原豁免不能恢复。';
 if(r.stage5_semantic_exemption_in_scope)return '具备第5步窗/柜豁免资格；本例未依靠豁免恢复。';
 return '没有第5步窗/柜语义豁免。';
}
function route(r){return r.frontend_semantic_hit?'前端窗/柜路由命中：'+r.frontend_target_families.join(' / ')+'；来源为 SAM。':'前端窗/柜路由未命中；来源按普通一致性规则选择。';}
function status(r){return r.new_retained?'新版保留':r.newly_excluded?'本次新增排除':'原已排除';}
function render(){
 if(!data)return;
 const f=$('#filter').value,q=$('#search').value.trim().toLowerCase();
 const rows=data.rows.filter(r=>(f==='all'||f==='kept'&&r.new_retained||f==='newly_excluded'&&r.newly_excluded||f==='old_kept'&&r.old_retained)&&
  (!q||[r.group_id,r.mask_id,r.frame,r.processed_frame,r.instance_id,r.semantic,models[r.model],r.frontend_target_families.join(' ')].join(' ').toLowerCase().includes(q)));
 $('#resultline').textContent=data.arm+' 组 · 当前显示 '+rows.length+' 张 · 旧版 '+data.summary.old_retained+' → 新版 '+data.summary.retained+' 张';
 $('#empty').hidden=rows.length>0;
 $('#cards').innerHTML=rows.map(r=>{
  const title=r.group_id+' · '+r.mask_id;
  const repeat=r.new_retained&&r.instance_id===91?'<p class="emphasis">第50帧修复后的方窗 ID91 仍入选；是否需要第二次修复待检查。</p>':'';
  return '<article class="card '+(r.new_retained?'keep':r.newly_excluded?'changed':'')+'"><div class="meta"><h3>'+esc(title)+'</h3><p class="sub">'+esc(models[r.model])+' · 原始帧 '+r.frame+' / 第 '+r.processed_frame+' 个处理帧</p><div class="tags"><span class="tag '+(r.new_retained?'keep':r.newly_excluded?'remove':'')+'">'+status(r)+'</span><span class="tag">'+esc(r.semantic)+' · 背景模型预测</span><span class="tag">匹配实例 '+fmt(r.instance_id)+'</span></div></div><button type="button" class="picture" data-detail="'+r.mask_id+'" data-mode="full"><img loading="lazy" src="'+esc(r.full)+'" alt="'+esc(title)+' 整帧 mask 叠图"><span>原始整帧 · 点击检查</span></button><button type="button" class="picture zoom" data-detail="'+r.mask_id+'" data-mode="zoom"><img loading="lazy" src="'+esc(r.zoom)+'" alt="'+esc(title)+' 局部放大"><span>局部放大</span></button><div class="meta"><div class="rates"><span>p <b>'+pct(r.p)+'</b></span><span>r <b>'+pct(r.r)+'</b></span></div><p class="counts">交集 '+fmt(r.intersection)+' / mask 支持 '+fmt(r.support_count)+' / 实例可见 '+fmt(r.instance_visible_count)+'</p><p class="decision">'+esc(r.reason)+'</p><p class="exemption">'+esc(exemption(r))+'</p>'+repeat+'<p class="note">'+esc(route(r))+'</p><p class="boundary">整帧触边 0 · 局部触边 0 · 原 mask 未改动</p></div></article>';
 }).join('');
}
function showView(mode){
 $('#detailimage').src=current[mode];$('#detailimage').alt=current.group_id+' · '+current.mask_id+' · '+mode;
 $('.detailimage').classList.toggle('binary',mode==='mask');
 document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('active',b.dataset.view===mode);b.setAttribute('aria-pressed',String(b.dataset.view===mode));});
}
function openDetail(id,mode){
 current=data.rows.find(r=>r.mask_id===id);if(!current)return;
 $('#detailtitle').textContent=data.arm+' 组 · '+current.group_id+' · '+current.mask_id+' · '+status(current);
 $('#detailinfo').textContent=models[current.model]+' · 原始帧 '+current.frame+' / 第 '+current.processed_frame+' 帧 · p '+pct(current.p)+' · r '+pct(current.r);
 $('#detailbottom').textContent=current.reason+'。'+exemption(current)+' '+route(current)+' 原始 mask SHA-256：'+current.mask_sha256;
 showView(mode||'full');$('#detail').showModal();
}
async function load(){
 try{
  const response=await fetch('data.json?v=pr95-90-1');if(!response.ok)throw Error('HTTP '+response.status);
  data=await response.json();const s=data.summary;
  document.title='room0 · '+data.arm+' 组 · 新规则保留 '+s.retained+' 张';
  $('#title').textContent=data.arm+' 组 · 100 帧 · 95/90 新规则';
  $('#scope').textContent='候选来源：第 '+data.candidate_processed_frame_range[0]+'–100 个处理帧（原始编号 '+data.candidate_raw_frame_range[0]+'–495）；评分使用共同的第 1–100 帧完整历史地图。';
  $('#stats').innerHTML=[[s.old_retained,'旧版保留'],[s.retained,'新版保留'],[s.newly_excluded,'本次新增排除'],[s.newly_retained,'本次新增保留'],[0,'留存主辅触边']].map((x,i)=>'<div class="stat '+(i===1?'final':'')+'"><b>'+fmt(x[0])+'</b><span>'+x[1]+'</span></div>').join('');
  $('#stages').innerHTML=s.stages.map(x=>'<span class="stage">'+stageNames[x.stage]+' '+x.input+' → '+x.retained+'</span>').join('');
  $('#edgeaudit').textContent='最前层触边过滤和噪声 / 分组结果完全复用：'+s.initial.input_masks+' 张来源合格候选 → 排除触边 '+s.initial.total_edge_removed+' 张 → '+s.initial.groups+' 个主 mask。新版留存连同辅助共 '+s.retained_primary_and_auxiliary_masks+' 张，实际二值像素复查全局 / 局部触边均为 0。';
  $('#common').textContent='原始组号与 mask 不变。A/B 新版共同的原始种子仅剩方窗：A005 / B组A006，原始帧260；旧版另一个共有种子（原始帧325）被新阈值排除。';
  $('#links').innerHTML=[['selection.json','真实筛选结果'],['seed_manifest.json','新版留存种子'],['comparison.json','逐组新旧判据'],['audit.json','一致性审计'],['boundary_audit.json','主辅mask边界核验']].map(([href,label])=>'<a href="'+href+'" target="_blank">'+label+'</a>').join(' · ');
  const requested=new URLSearchParams(location.search).get('view');if(['kept','newly_excluded','old_kept','all'].includes(requested))$('#filter').value=requested;
  render();
 }catch(e){$('#resultline').textContent='页面读取失败：'+e.message;}
}
document.addEventListener('click',e=>{
 const d=e.target.closest('[data-detail]');if(d){openDetail(d.dataset.detail,d.dataset.mode);return;}
 const v=e.target.closest('[data-view]');if(v&&current)showView(v.dataset.view);
});
$('#close').addEventListener('click',()=>$('#detail').close());
$('#detail').addEventListener('click',e=>{if(e.target===$('#detail'))$('#detail').close();});
$('#filter').addEventListener('change',()=>{history.replaceState(null,'','?view='+$('#filter').value);render();});
$('#search').addEventListener('input',render);
load();
