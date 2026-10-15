// Relevance is a separate display/selection condition; input masks are immutable.
const unrelatedPreferences=[false,false,false];
function intersectsROI(model,id){return id>0&&loaded?.models[model]?.data.relevance.labels[id]?.decision==='intersects';}
function prepareIntersection(){
 for(let i=0;i<3;i++)loaded.models[i].showUnrelated=unrelatedPreferences[i];
 refreshIntersectionControls();
}
function refreshIntersectionControls(){
 if(!loaded||loaded.j.key!==activeKey)return;
 let total=0,shown=0;
 for(let i=0;i<3;i++){
  const m=loaded.models[i],r=m.data.relevance,n=r.no_intersection_ids.length;total+=n;if(m.showUnrelated)shown+=n;
  const input=el('showUnrelated'+i);input.disabled=!n;input.checked=!!n&&m.showUnrelated;
  const newly=r.eligible_ids.filter(id=>r.labels[id].core===0).length;
  const outside=r.eligible_ids.filter(id=>r.labels[id].expanded_in_crop===0).length;
  el('intersectionCount'+i).textContent='相交候选 '+r.eligible_ids.length+' · 扩展新增 '+newly+' · 无交集 '+n+(outside?' · '+outside+' 个交集在当前裁剪外':'');
  el('meta'+i).textContent=r.eligible_ids.length+' 个相交候选 · 过合并拟剔除 '+m.data.removed_ids.length+' 个';
 }
 const all=el('showUnrelated');all.disabled=!total;all.checked=!!total&&shown===total;all.indeterminate=shown>0&&shown<total;
 el('intersectionStatus').textContent='显示 '+shown+' / '+total+' 个无交集 Mask（原色，仅作对照）';
 const f=loaded.j.intersection_frame,c=D.cases[activeCase].expansion;
 el('expansionSummary').textContent='本 ROI 原异常点 '+c.core_points.toLocaleString()+' → 2 cm 扩展后 '+c.expanded_points.toLocaleString()+' 个表面点；当前帧可靠可见 '+f.core_visible.toLocaleString()+' → '+f.expanded_visible.toLocaleString()+'。青色只表示新增表面点，不代表质量。';
}
function decorateIntersectionLegend(model){
 if(!loaded)return;const m=loaded.models[model];
 el('maskLegend'+model).querySelectorAll('.maskRow').forEach(row=>{
  const id=Number(row.dataset.maskID),r=m.data.relevance.labels[id],yes=intersectsROI(model,id),label=row.querySelector('label');
  let badge=row.querySelector('.intersectionBadge');if(!badge){badge=document.createElement('small');badge.className='intersectionBadge';label.appendChild(badge);}
  badge.textContent=yes?(r.core===0?'扩展新增':'相交'):'无交集';badge.classList.toggle('outside',!yes);
  row.title='有效表面点 '+r.N+'；原异常交集 '+r.core+'；扩展后交集 '+r.expanded+'；当前裁剪内交集 '+r.expanded_in_crop;
  row.querySelectorAll('input,button').forEach(b=>b.disabled=!yes&&!m.showUnrelated);
  row.querySelector('.maskSwatch').style.background='rgb('+m.data.colors[id].join(',')+')';
 });
}
function setUnrelated(model,value){
 if(!loaded||loaded.j.key!==activeKey)return;
 for(const i of model===null?[0,1,2]:[model]){
  unrelatedPreferences[i]=value;loaded.models[i].showUnrelated=value;
  invalidateVisibleImages(i);refreshMaskVisibility(i);
 }
 refreshIntersectionControls();refreshProblemSummary();drawAll();
}
function drawExpansion(ctx,full=false){
 if(!loaded?.expansion||!el('showExpansion').checked)return;
 ctx.save();ctx.globalAlpha=.9;ctx.imageSmoothingEnabled=false;
 if(full)ctx.drawImage(loaded.expansion,0,0);else{const [x0,y0,x1,y1]=loaded.j.box;ctx.drawImage(loaded.expansion,x0,y0,x1-x0,y1-y0,0,0,x1-x0,y1-y0);}
 ctx.restore();
}
el('showUnrelated').onchange=e=>setUnrelated(null,e.target.checked);
for(let i=0;i<3;i++)el('showUnrelated'+i).onchange=e=>setUnrelated(i,e.target.checked);
el('showExpansion').onchange=drawAll;
