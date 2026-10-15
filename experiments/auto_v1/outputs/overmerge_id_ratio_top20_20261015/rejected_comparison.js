// Display-only comparison: filtered IDs and saved selections stay unchanged.
const rejectedDisplayPreferences=[false,false,false];
function rejectedName(model){return ['原始 CropFormer','局部 CropFormer','SAM2.1'][model];}
function rejectedPixel(model,x,y){
 const m=loaded?.models[model];if(!m?.rejectedVisible||!m.rejectedLayer||loaded.j.key!==activeKey)return 0;
 const [w,h]=loaded.j.wh;if(x<0||y<0||x>=w||y>=h)return 0;
 const id=m.rejectedLayer.labels[y*w+x];return m.rejectedLayer.ids.has(id)?id:0;
}
function rejectedNotice(model,id){
 el('maskHover'+model).textContent='拟剔除 ID '+id+'（对照显示）';
 selectionNotice(rejectedName(model)+'：ID '+id+' 为拟剔除项，当前仅用于检查，没有加入合格目标。');
}
function refreshRejectedControls(){
 if(!loaded||loaded.j.key!==activeKey)return;
 let total=0,shown=0,busy=false;
 for(let i=0;i<3;i++){
  const m=loaded.models[i],n=m.data.removed_ids.length;total+=n;if(m.rejectedVisible)shown+=n;busy||=!!m.rejectedLoading;
  const check=el('showRejected'+i);check.disabled=n===0;check.checked=n>0&&m.rejectedVisible;
  el('rejectedCount'+i).textContent=n?'拟剔除 '+n+' 个 · ID '+m.data.removed_ids.join('、'):'本栏没有拟剔除项';
  el('rejectedPanel'+i).classList.toggle('showingRejected',n>0&&m.rejectedVisible);
 }
 const check=el('showRejected');check.disabled=total===0;check.checked=total>0&&shown===total;check.indeterminate=shown>0&&shown<total;
 el('rejectedStatus').textContent=total?(busy?'正在加载拟剔除区域…':'本范围显示 '+shown+' / '+total+' 个拟剔除 Mask · 红色边界和“拟剔除”编号用于区分'):'本范围没有拟剔除 Mask';
}
async function loadRejectedLayer(model){
 const owner=loaded,m=owner.models[model],source=m.data.removed_source;
 if(!source||m.rejectedLayer||m.rejectedLoading)return;
 m.rejectedLoading=true;refreshRejectedControls();
 try{
  const img=await getImage(source.ids_file);if(loaded!==owner||loaded.j.key!==activeKey)return;
  const bytes=loadIDs(img);if(!bytes)throw new Error('拟剔除编号图无法读取');
  const [w,h]=owner.j.wh,labels=new Uint16Array(w*h),rgba=new Uint8ClampedArray(w*h*4),edge=new Uint8ClampedArray(w*h*4),ids=new Set(m.data.removed_ids),sums=new Map([...ids].map(id=>[id,{id,x:0,y:0,n:0,best:Infinity}]));
  for(let i=0;i<labels.length;i++){
   const id=bytes[i*4]+bytes[i*4+1]*256;labels[i]=id;if(!ids.has(id))continue;
   const col=source.colors[id],r=sums.get(id);rgba.set([...col,255],i*4);r.x+=i%w;r.y+=Math.floor(i/w);r.n++;
  }
  for(const r of sums.values()){r.x/=r.n;r.y/=r.n;}
  for(let i=0;i<labels.length;i++){
   const id=labels[i];if(!ids.has(id))continue;const x=i%w,y=Math.floor(i/w),r=sums.get(id),dist=(x-r.x)**2+(y-r.y)**2;
   if(dist<r.best){r.best=dist;r.px=x;r.py=y;}
   if(x===0||y===0||x===w-1||y===h-1||labels[i-1]!==id||labels[i+1]!==id||labels[i-w]!==id||labels[i+w]!==id)edge.set([232,30,73,255],i*4);
  }
  m.rejectedLayer={ids,labels,color:canvasFromPixels(w,h,rgba),edge:canvasFromPixels(w,h,edge),positions:[...sums.values()].filter(r=>r.n>0)};
 }catch(error){if(loaded===owner)el('error').textContent=error.message;}
 finally{m.rejectedLoading=false;if(loaded===owner){refreshRejectedControls();drawAll();}}
}
function prepareRejectedComparison(){
 for(let i=0;i<3;i++){
  const m=loaded.models[i];m.rejectedVisible=rejectedDisplayPreferences[i];m.rejectedLayer=null;m.rejectedLoading=false;
  if(m.rejectedVisible&&m.data.removed_ids.length)loadRejectedLayer(i);
 }
 refreshRejectedControls();
}
function setRejectedVisible(model,visible){
 if(!loaded||loaded.j.key!==activeKey)return;
 const models=model===null?[0,1,2]:[model];
 for(const i of models){rejectedDisplayPreferences[i]=visible;loaded.models[i].rejectedVisible=visible;if(visible&&loaded.models[i].data.removed_ids.length)loadRejectedLayer(i);}
 refreshRejectedControls();drawAll();
}
function drawRejectedFill(ctx,model){
 const m=loaded.models[model];if(!m.rejectedVisible||!m.rejectedLayer)return;
 ctx.save();ctx.globalAlpha=el('pure').checked?1:Number(el('opacity').value)/100;ctx.drawImage(m.rejectedLayer.color,0,0);ctx.globalAlpha=1;ctx.drawImage(m.rejectedLayer.edge,0,0);ctx.restore();
}
function drawRejectedLabels(canvas,ctx,model){
 const m=loaded.models[model];if(!m.rejectedVisible||!m.rejectedLayer)return;
 const [w,h]=loaded.j.wh,factor=scale*Math.max(.01,(canvas.clientWidth||w)/w),left=(centerX-.5/scale)*w,right=(centerX+.5/scale)*w,top=(centerY-.5/scale)*h,bottom=(centerY+.5/scale)*h;
 ctx.save();ctx.font=(11/factor)+'px Segoe UI';ctx.textAlign='left';ctx.textBaseline='middle';
 for(const r of m.rejectedLayer.positions){
  if(r.px<left||r.px>right||r.py<top||r.py>bottom)continue;
  const text='拟剔除 ID '+r.id,tw=ctx.measureText(text).width,x=Math.max(left+2/factor,Math.min(right-tw-8/factor,r.px-tw/2)),y=Math.max(top+10/factor,Math.min(bottom-10/factor,r.py));
  ctx.fillStyle='#b01836';ctx.fillRect(x-3/factor,y-9/factor,tw+6/factor,18/factor);ctx.fillStyle='white';ctx.fillText(text,x,y);
 }
 ctx.restore();
}
el('showRejected').onchange=e=>setRejectedVisible(null,e.target.checked);
for(let i=0;i<3;i++)el('showRejected'+i).onchange=e=>setRejectedVisible(i,e.target.checked);
