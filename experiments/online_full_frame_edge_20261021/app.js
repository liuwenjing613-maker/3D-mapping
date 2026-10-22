'use strict';
let prefix=Number(new URLSearchParams(location.search).get('prefix'))===20?20:50,data=null,current=null,requestNo=0;
const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const model=['CropFormer · 原图','CropFormer · 局部','SAM'];
const statusText={kept:'最终留存',edge:'整帧触边排除',later:'后续筛除',noise:'噪声筛除',aux:'辅助验证'};
const stageText={semantic:'背景语义',covisible_iou_only:'去重',full_support_containment:'直接包含',skip_repair_B:'覆盖率',pr_filter:'p/r'};
const edgeText={left:'左',top:'上',right:'右',bottom:'下'};
const fmt=v=>v==null?'—':Number(v).toLocaleString('en-US');
const rate=v=>v==null?'—':(100*v).toFixed(1)+'%';
const asset=r=>'prefix'+prefix+'/'+r;
async function loadPrefix(n){
 const ticket=++requestNo;prefix=n;$('#resultline').textContent='正在读取服务器结果…';$('#cards').innerHTML='';
 document.querySelectorAll('[data-prefix]').forEach(b=>{b.classList.toggle('active',Number(b.dataset.prefix)===n);b.setAttribute('aria-pressed',String(Number(b.dataset.prefix)===n));});
 try{
 const response=await fetch('prefix'+n+'/data.json');if(!response.ok)throw Error('HTTP '+response.status);
 const loaded=await response.json();if(ticket!==requestNo)return;data=loaded;
 const s=data.summary,i=s.initial;
 $('#stats').innerHTML=[[i.input_masks,'初期来源合格候选'],[i.full_frame_removed,'整帧触边排除'],[i.after_full_frame_filter,'通过整帧边界'],[s.retained,'后续筛选最终留存'],[s.retained_crop_only.length,'留存中仅局部触边']].map((x,j)=>'<div class="stat '+(j===3?'final':'')+'"><b>'+fmt(x[0])+'</b><span>'+x[1]+'</span></div>').join('');
 $('#stages').innerHTML='<span class="stage">整帧边界 '+i.input_masks+' → '+i.after_full_frame_filter+'</span><span class="stage">噪声 '+i.after_full_frame_filter+' → '+i.after_noise+'</span><span class="stage">分组 '+i.after_noise+' → '+i.groups+' 个主 mask</span>'+s.stages.map(x=>'<span class="stage">'+stageText[x.stage]+' '+x.input+' → '+x.retained+'</span>').join('');
 $('#datalinks').innerHTML='<a href="'+asset('data.json')+'" target="_blank">完整留存与候选数据</a> · <a href="'+asset('audit.json')+'" target="_blank">逐 mask 四边像素检查</a> · <a href="'+asset('selection.json')+'" target="_blank">后续筛选证据</a>';
 history.replaceState(null,'','?prefix='+n);render();
 }catch(e){$('#resultline').textContent='页面数据读取失败：'+e.message;}
}
function render(){
 if(!data)return;const filter=$('#status').value,crop=$('#crop').checked,q=$('#search').value.trim().toLowerCase();
 const rows=data.rows.filter(r=>{
  const match=filter==='all'||(filter==='boundary'?r.status!=='edge':r.status===filter);
  const cropMatch=!crop||(r.crop_edges.length>0&&!Object.values(r.edge_pixels).some(Boolean));
  const searchMatch=!q||[r.id,r.group,r.legacy_primary,r.legacy_member,r.semantic,model[r.model],r.frame].join(' ').toLowerCase().includes(q);
  return match&&cropMatch&&searchMatch;
 });
 $('#resultline').textContent=prefix+' 帧 · 当前显示 '+rows.length+' 张 · 最终留存 '+data.summary.retained+' 张 · 留存触整帧边界 0 张';
 $('#empty').hidden=rows.length>0;
 $('#cards').innerHTML=rows.map(r=>{
  const bad=Object.entries(r.edge_pixels).filter(([,v])=>v).map(([k,v])=>edgeText[k]+' '+v+' 像素').join(' · ');
  const crop=r.crop_edges.length>0?'<span class="tag crop">局部裁剪触边'+(bad?'':' · 允许进入后续筛选')+'</span>':'';
  const title=r.primary?r.group+' · '+r.id:r.id;
  const old=r.legacy_primary?' · 原 '+r.legacy_primary+(r.legacy_primary_retained?'（原留存）':''):'';
  return '<article class="card" data-mask="'+r.id+'"><div class="meta"><h3>'+esc(title)+'</h3><p class="sub">'+esc(model[r.model])+' · 原始帧 '+r.frame+' / 第 '+r.processed+' 个处理帧'+esc(old)+'</p><div class="tags"><span class="tag '+(r.status==='edge'?'edge':r.status==='kept'?'keep':'')+'">'+statusText[r.status]+'</span>'+crop+(r.semantic?'<span class="tag">'+esc(r.semantic)+' · 模型标签</span>':'')+'</div></div><button type="button" class="picture" data-detail="'+r.id+'" aria-label="查看 '+esc(title)+' 整帧"><img loading="lazy" src="'+asset(r.full)+'" alt="'+esc(title)+' 整帧 mask 叠图"><span>原始整帧 · 点击检查</span></button><button type="button" class="picture zoom" data-detail="'+r.id+'" data-mode="zoom" aria-label="放大 '+esc(title)+'"><img loading="lazy" src="'+asset(r.zoom)+'" alt="'+esc(title)+' 局部放大"><span>局部放大</span></button><div class="meta"><div class="numbers"><span>面积 '+fmt(r.area)+' px</span><span>支持点 '+fmt(r.support_count)+'</span><span>p '+rate(r.p)+'</span><span>r '+rate(r.r)+'</span></div><div class="borderline '+(bad?'bad':'')+'">'+(bad?'整帧触边：'+bad:'整帧四边：左 0 · 上 0 · 右 0 · 下 0')+'</div><div class="reason">'+esc(r.reason)+'</div></div></article>';
 }).join('');
}
function showView(mode){
 $('#detailimage').src=asset(current[mode]);$('#detailimage').alt=prefix+' 帧 '+current.id+' '+mode;
 $('.detailimage').classList.toggle('binary',mode==='mask');
 document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('active',b.dataset.view===mode);b.setAttribute('aria-pressed',String(b.dataset.view===mode));});
}
function openDetail(id,mode){
 current=data.rows.find(r=>r.id===id);if(!current)return;
 $('#detailtitle').textContent=prefix+' 帧 · '+(current.primary?current.group+' · ':'')+current.id+' · '+statusText[current.status];
 const edges=Object.entries(current.edge_pixels).map(([k,v])=>edgeText[k]+': '+v).join(' / ');
 $('#detailinfo').textContent=model[current.model]+' · 原始帧 '+current.frame+' · 1200×680 · 整帧边界像素数 '+edges;
 $('#detailbottom').textContent=current.reason+'。原始 mask SHA-256：'+current.mask_sha256;
 showView(mode||'full');$('#detail').showModal();
}
document.addEventListener('click',e=>{
 const p=e.target.closest('[data-prefix]');if(p){loadPrefix(Number(p.dataset.prefix));return;}
 const detail=e.target.closest('[data-detail]');if(detail){openDetail(detail.dataset.detail,detail.dataset.mode);return;}
 const v=e.target.closest('[data-view]');if(v)showView(v.dataset.view);
});
$('#close').addEventListener('click',()=>$('#detail').close());
$('#detail').addEventListener('click',e=>{if(e.target===$('#detail'))$('#detail').close();});
$('#status').addEventListener('change',render);$('#crop').addEventListener('change',render);$('#search').addEventListener('input',render);
document.addEventListener('error',e=>{
 const img=e.target;
 if(!(img instanceof HTMLImageElement)||!img.getAttribute('src'))return;
 const count=Number(img.dataset.retry||0);
 if(count>=2)return;
 img.dataset.retry=String(count+1);
 const original=img.dataset.originalSrc||img.getAttribute('src');
 img.dataset.originalSrc=original;
 setTimeout(()=>{if(img.isConnected)img.src=original+(original.includes('?')?'&':'?')+'image_retry='+(count+1);},250*(count+1));
},true);
loadPrefix(prefix);
