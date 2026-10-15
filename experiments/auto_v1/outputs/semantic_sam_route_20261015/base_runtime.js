(()=>{
'use strict';
const D=JSON.parse(document.getElementById('reviewData').textContent),el=id=>document.getElementById(id);
let activeCase=0,activeView=0,activeKey='',filtered=D.cases.map((_,i)=>i),loadToken=0,loaded=null;
let scale=1,centerX=.5,centerY=.5,drag=null;
const imageCache=new Map(),canvases=[el('canvas0'),el('canvas1'),el('canvas2')];
function getImage(src){if(!imageCache.has(src)){imageCache.set(src,new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('图像无法加载：'+src));img.src=src;}));while(imageCache.size>90)imageCache.delete(imageCache.keys().next().value);}return imageCache.get(src);}
function loadIDs(img){const c=document.createElement('canvas');c.width=img.naturalWidth;c.height=img.naturalHeight;c.getContext('2d').drawImage(img,0,0);try{return c.getContext('2d').getImageData(0,0,c.width,c.height).data;}catch(e){if(e.name==='SecurityError')return null;throw e;}}
function buildCases(){el('cases').replaceChildren();filtered.forEach(i=>{const c=D.cases[i],b=document.createElement('button');b.className='case'+(i===activeCase?' active':'');b.dataset.caseIndex=i;b.innerHTML='<img alt=""><span><b></b><small></small></span>';b.querySelector('img').src=c.views[0].full_rgb;b.querySelector('b').textContent=c.aliases.join(' / ');b.querySelector('small').textContent=(savedSelections[c.uid]?'已选 · ':'')+'C '+c.counts.CONFLICT+' · U '+c.counts.U+' · T '+c.counts.T;b.onclick=()=>chooseCase(i);el('cases').appendChild(b);});}
function option(value,label){const o=document.createElement('option');o.value=value;o.textContent=label;return o;}
function chooseCase(i){activeCase=i;activeView=0;const c=D.cases[i];el('caseTitle').textContent=c.scene+' / '+c.ROI;el('caseInfo').textContent='ROI '+c.order+' / 32 · '+c.rankings.map(r=>({CONFLICT:'冲突',U:'未分配',T:'证据不足'}[r.group])+'第 '+r.rank+'：'+r.count+' 点').join(' ｜ ');el('view').replaceChildren(...c.views.map((v,j)=>option(j,v.source_entry+' / 视角 '+v.source_view+' · 帧 '+v.frame)));el('view').value='0';buildCases();updateNav();const saved=savedSelections[c.uid],groupView=c.views.findIndex(v=>v.trigger_group===el('scene').value),resume=el('scene').value!=='all'?Math.max(0,groupView):(saved?c.views.findIndex(v=>v.frame===saved.frame&&v.number===saved.view_number):0);chooseView(Math.max(0,resume),el('scene').value==='all'?saved?.crop_key:undefined);el('view').value=String(activeView);}
function updateNav(){const p=filtered.indexOf(activeCase);el('counter').textContent=(p+1)+' / '+filtered.length;el('previous').disabled=p<=0;el('next').disabled=p>=filtered.length-1;}
function cropLabel(j){if(j.strategy==='balanced')return '目标区域 + 15% 上下文（默认） · '+j.zoom.toFixed(2)+'×';if(j.strategy==='context')return '宽上下文 · '+j.zoom.toFixed(2)+'×';return '局部窗口 '+activeTileIndex(j.key)+' · '+j.zoom.toFixed(2)+'× · 目标点 '+Math.round(j.coverage*100)+'%';}
function activeTileIndex(key){const m=key.match(/_tile(\d+)$/);return m?m[1]:'';}
function chooseView(j,preferred){activeView=j;const v=D.cases[activeCase].views[j];el('crop').replaceChildren(...v.variants.map(k=>option(k,cropLabel(D.crops[k]))));const key=v.variants.includes(preferred)?preferred:v.recommended;el('crop').value=key;chooseCrop(key);}
async function chooseCrop(key){const token=++loadToken;activeKey=key;const j=D.crops[key],v=D.cases[activeCase].views[activeView];el('models').classList.add('loading');el('status').textContent='加载对比…';el('error').textContent='';scale=1;centerX=centerY=.5;
 const coverage=j.complete?'覆盖该视角全部目标状态点':'局部窗口覆盖 '+Math.round(j.coverage*100)+'% 目标状态点，物体可能延伸到窗口之外';
 const gain=j.zoom<1.05?'该区域接近整帧，完整目标输入没有明显放大；可切换局部窗口。':'相对原整帧像素尺度 '+j.zoom.toFixed(2)+' 倍。';
 selectionContextChanged();clearProblemFocus();el('projectionSummary').textContent='读取异常投影…';el('focusProblem').disabled=true;
 el('cropInfo').textContent='原始输入 '+j.wh.join(' × ')+' px → CropFormer 网络整图分支 '+j.network_input_wh.join(' × ')+' px；'+gain+' '+coverage+'。';
 el('meta0').textContent=j.models[0].count+' 个保留 mask · 已排除 '+j.models[0].removed_ids.length+' 个';el('meta1').textContent=j.models[1].count+' 个保留 mask · 已排除 '+j.models[1].removed_ids.length+' 个';el('meta2').textContent=j.models[2].count+' 个保留 mask · 已排除 '+j.models[2].removed_ids.length+' 个';
 const balanced=D.crops[v.recommended];el('staticLink').href=balanced.comparison;el('staticLink').textContent='原全量对照图（未应用交集筛选）';
 try{const [rgb,full,alarm,expansion,models]=await Promise.all([getImage(j.RGB),getImage(v.full_rgb),getImage(v.alarm),getImage(v.expansion),Promise.all(j.models.map(async m=>{const [color,boundary,ids]=await Promise.all([getImage(m.color),getImage(m.boundary),getImage(m.ids)]);return {color,boundary,idData:loadIDs(ids),data:m};}))]);if(token!==loadToken)return;loaded={j,rgb,full,alarm,expansion,models};prepareProblemAnnotations();prepareIntersection();prepareMaskDisplay();prepareRejectedComparison();refreshSelection();[el('rgbRef'),...canvases].forEach(c=>{c.width=j.wh[0];c.height=j.wh[1];});el('largeRGB').src=v.full_rgb;drawAll();el('models').classList.remove('loading');el('status').textContent='';}catch(err){if(token===loadToken){el('error').textContent=err.message;el('status').textContent='加载失败';el('models').classList.remove('loading');}}
}
function transformed(ctx,w,h,callback){ctx.save();ctx.translate(w/2-centerX*w*scale,h/2-centerY*h*scale);ctx.scale(scale,scale);callback();ctx.restore();}
function drawCanvas(c,model){if(!loaded)return;const ctx=c.getContext('2d'),[w,h]=loaded.j.wh;ctx.clearRect(0,0,w,h);ctx.fillStyle='#e6ebf1';ctx.fillRect(0,0,w,h);transformed(ctx,w,h,()=>{ctx.imageSmoothingEnabled=true;if(model===null||!el('pure').checked)ctx.drawImage(loaded.rgb,0,0);if(model!==null){const m=loaded.models[model];ctx.globalAlpha=el('pure').checked?1:Number(el('opacity').value)/100;ctx.drawImage(maskImages(model).color,0,0);ctx.globalAlpha=1;if(el('boundary').checked)ctx.drawImage(maskImages(model).boundary,0,0);if(el('numbers').checked){ctx.font=(Math.max(11,w/45)/scale)+'px Segoe UI';ctx.textAlign='center';ctx.textBaseline='middle';m.data.positions.forEach(p=>{if(p.area<60||!maskIsVisible(model,p.id)||hasProjectedLabel(model,p.id))return;const label=intersectsROI(model,p.id)?String(p.id):'无交集 '+p.id,tw=ctx.measureText(label).width;ctx.fillStyle='rgba(18,26,38,.78)';ctx.fillRect(p.x-tw/2-3/scale,p.y-8/scale,tw+6/scale,16/scale);ctx.fillStyle='white';ctx.fillText(label,p.x,p.y);});}}if(model!==null){drawRejectedFill(ctx,model);drawExpansion(ctx);drawProblemOverlay(c,ctx,model);drawRejectedLabels(c,ctx,model);}});}
function drawFull(){if(!loaded)return;const c=el('fullRef');c.width=1200;c.height=680;const ctx=c.getContext('2d');ctx.drawImage(loaded.full,0,0,1200,680);drawExpansion(ctx,true);drawFullProblemOverlay(c,ctx);const b=loaded.j.box;ctx.strokeStyle='#20d796';ctx.lineWidth=6;ctx.strokeRect(b[0]+3,b[1]+3,b[2]-b[0]-6,b[3]-b[1]-6);}
function drawAll(){if(!loaded)return;drawCanvas(el('rgbRef'),null);canvases.forEach((c,i)=>drawCanvas(c,i));drawFull();el('opacityValue').value=el('opacity').value+'%';el('zoomValue').textContent='查看 '+scale.toFixed(1)+'×';}
function clampCenter(){const t=.5/scale;centerX=Math.max(t,Math.min(1-t,centerX));centerY=Math.max(t,Math.min(1-t,centerY));}
function nativePoint(e,c){const r=c.getBoundingClientRect();return {x:centerX+(e.clientX-r.left-r.width/2)/(r.width*scale),y:centerY+(e.clientY-r.top-r.height/2)/(r.height*scale)};}
function showIDs(p){if(!loaded)return;if(loaded.models.some(m=>!m.idData)){el('hover').textContent='开启“实例编号”可查看各模型的 mask ID。';return;}const [w,h]=loaded.j.wh,x=Math.floor(p.x*w),y=Math.floor(p.y*h);if(x<0||y<0||x>=w||y>=h)return;const labels=loaded.models.map(m=>{const off=(y*w+x)*4;return m.idData[off]+256*m.idData[off+1];});showMaskIDsAtPoint(labels);el('hover').textContent='局部像素 ('+x+', '+y+') · 原始 ID '+labels[0]+' / 局部 ID '+labels[1]+' / SAM ID '+labels[2]+' · '+currentProjectedState(x,y)+' · 0 表示未分割';}
[el('rgbRef'),...canvases].forEach(c=>{
 c.addEventListener('wheel',e=>{e.preventDefault();if(!loaded)return;const before=nativePoint(e,c),old=scale;scale=Math.max(1,Math.min(12,scale*Math.exp(-e.deltaY*.0018)));centerX=before.x-(before.x-centerX)*old/scale;centerY=before.y-(before.y-centerY)*old/scale;clampCenter();drawAll();},{passive:false});
 c.addEventListener('pointerdown',e=>{if(!loaded)return;const r=c.getBoundingClientRect();drag={x:e.clientX,y:e.clientY,centerX,centerY,w:r.width,h:r.height};c.setPointerCapture(e.pointerId);});
 c.addEventListener('pointermove',e=>{if(drag){centerX=drag.centerX-(e.clientX-drag.x)/(drag.w*scale);centerY=drag.centerY-(e.clientY-drag.y)/(drag.h*scale);clampCenter();drawAll();}else showIDs(nativePoint(e,c));});
 c.addEventListener('pointerup',()=>drag=null);c.addEventListener('pointercancel',()=>drag=null);
});

el('scene').onchange=()=>{filtered=D.cases.flatMap((c,i)=>el('scene').value==='all'||c.rankings.some(r=>r.group===el('scene').value)?[i]:[]);chooseCase(filtered[0]);};
el('view').onchange=()=>chooseView(Number(el('view').value));el('crop').onchange=()=>chooseCrop(el('crop').value);
el('previous').onclick=()=>{const p=filtered.indexOf(activeCase);if(p>0)chooseCase(filtered[p-1]);};el('next').onclick=()=>{const p=filtered.indexOf(activeCase);if(p<filtered.length-1)chooseCase(filtered[p+1]);};
['opacity','pure','boundary','numbers','alarms'].forEach(id=>el(id).oninput=drawAll);el('reset').onclick=()=>{clearProblemFocus();scale=1;centerX=centerY=.5;drawAll();};
el('fullRef').onclick=()=>el('imageDialog').showModal();el('closeDialog').onclick=()=>el('imageDialog').close();
document.addEventListener('keydown',e=>{if(['SELECT','INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight')el('next').click();if(e.key==='ArrowLeft')el('previous').click();});
// Annotation only: fixed state projections and model IDs remain immutable.
const PROJECTIONS=JSON.parse(el('projectionData').textContent);
const projectionImageCache=new WeakMap();
const problemNames={C:'冲突',U:'未分配',T:'证据不足'};
const problemColors={C:'#ed173d',U:'#718094',T:'#e99709'};
let lastProblemCase=null;
function canvasFromPixels(w,h,pixels){const c=document.createElement('canvas');c.width=w;c.height=h;c.getContext('2d').putImageData(new ImageData(pixels,w,h),0,0);return c;}
function decodeProjection(image){
 if(projectionImageCache.has(image))return projectionImageCache.get(image);
 const w=image.naturalWidth,h=image.naturalHeight,p=loadIDs(image);
 if(!p)throw new Error('冲突投影图像不可读');
 const codes=new Uint8Array(w*h),red=new Uint8ClampedArray(p.length),others=new Uint8ClampedArray(p.length),indices={C:[],U:[],T:[]};
 for(let i=0;i<w*h;i++){
  const off=i*4;if(p[off+3]===0)continue;
  let state=0;if(p[off]>200&&p[off+1]<120)state=3;else if(p[off]>200&&p[off+1]>120)state=2;else state=1;
  codes[i]=state;const name=['','U','T','C'][state];indices[name].push(i);
  const layer=state===3?red:others,color=state===3?[237,23,61]:state===2?[233,151,9]:[113,128,148];layer.set([...color,255],off);
 }
 const value={w,h,codes,indices,red:canvasFromPixels(w,h,red),others:canvasFromPixels(w,h,others),sampleCache:new Map()};
 projectionImageCache.set(image,value);return value;
}
function prepareProblemAnnotations(){
 const meta=PROJECTIONS.crops[activeKey];if(!meta)throw new Error('当前范围缺少投影标注');
 loaded.problem={meta,source:decodeProjection(loaded.alarm),focus:null,tagRects:[[],[],[]]};
 const c=D.cases[activeCase];const marker=c.uid+'|'+meta.group;if(marker!==lastProblemCase){el('otherStates').checked=true;lastProblemCase=marker;}
 refreshProblemSummary();
}
function triggerState(){return loaded?.problem?.meta.group==='U'?'U':loaded?.problem?.meta.group==='T'?'T':'C';}
function refreshProblemSummary(){
 if(!loaded?.problem||loaded.j.key!==activeKey)return;
 const {meta}=loaded.problem,name=triggerState(),n=meta.counts[name];
 el('focusProblem').textContent='定位'+problemNames[name];
 el('focusProblem').disabled=n===0;
 el('projectionSummary').textContent='本范围投影：C '+meta.counts.C.toLocaleString()+' · U '+meta.counts.U.toLocaleString()+' · T '+meta.counts.T.toLocaleString()+' 像素。红 C 为冲突；灰 U、黄 T 可单独显示。编号为当前二维 mask，比例仅用于定位。';
 for(let model=0;model<3;model++){
  const root=el('problemInfo'+model);root.replaceChildren();
  const summary=document.createElement('div');summary.className='problemCount';
  const entries=meta.models[model][name],nonzero=entries.filter(row=>row.id>0),unassigned=entries.find(row=>row.id===0);
  const unassignedPercent=unassigned?unassigned.count/Math.max(1,n)*100:0;
  summary.textContent=n?problemNames[name]+'投影落入 '+nonzero.length+' 个 mask'+(unassigned?' · 未落入保留 mask '+(unassignedPercent<1?'＜1':Math.round(unassignedPercent))+'%':''): '当前范围没有'+problemNames[name]+'投影';
  root.appendChild(summary);
  const addEntry=(row,parent)=>{
   const percent=row.count/Math.max(1,n)*100,b=document.createElement('button');b.className='problemID';b.textContent=(row.id===0?'未落入保留 mask':'ID '+row.id)+' · '+(percent<1?'＜1':Math.round(percent))+'%';
   b.setAttribute('aria-label','定位'+['原始 CropFormer','局部 CropFormer','SAM2.1'][model]+' '+(row.id===0?'未分割区域':'mask ID '+row.id));
   b.title=row.count+' 个'+problemNames[name]+'投影像素；点击只定位，候选选择仍由你完成';b.classList.toggle('hiddenProblemMask',row.id>0&&!maskIsVisible(model,row.id));
   b.onclick=()=>focusProblemMask(model,row.id);parent.appendChild(b);
  };
  const list=document.createElement('div');list.className='problemList';entries.slice(0,6).forEach(row=>addEntry(row,list));root.appendChild(list);
  if(entries.length>6){const details=document.createElement('details'),title=document.createElement('summary');title.textContent='其余 '+(entries.length-6)+' 个编号';details.appendChild(title);const remaining=document.createElement('div');remaining.className='problemList';entries.slice(6).forEach(row=>addEntry(row,remaining));details.appendChild(remaining);root.appendChild(details);}
 }
}
function projectionSamples(source,kind,box,step){
 const key=kind+'|'+box.join(',')+'|'+step;
 if(source.sampleCache.has(key))return source.sampleCache.get(key);
 const [x0,y0,x1,y1]=box,cells=new Set(),points=[],cols=Math.ceil((x1-x0)/step);
 for(const index of source.indices[kind]){const x=index%source.w,y=Math.floor(index/source.w);if(x<x0||y<y0||x>=x1||y>=y1)continue;const cell=Math.floor((y-y0)/step)*cols+Math.floor((x-x0)/step);if(cells.has(cell))continue;cells.add(cell);points.push([x-x0+.5,y-y0+.5]);}
 source.sampleCache.set(key,points);if(source.sampleCache.size>30)source.sampleCache.delete(source.sampleCache.keys().next().value);return points;
}
function renderProjection(canvas,ctx,box,zoom,sharedDisplay){
 if(!loaded?.problem)return;
 const source=loaded.problem.source,w=box[2]-box[0],h=box[3]-box[1],display=Math.max(.01,(canvas.clientWidth||canvas.width)/canvas.width),factor=zoom*display;
 const intensity=Number(el('alarmOpacity').value)/100;
 ctx.save();ctx.globalAlpha=intensity*.28;
 if(el('alarms').checked)ctx.drawImage(source.red,box[0],box[1],w,h,0,0,w,h);
 if(el('otherStates').checked)ctx.drawImage(source.others,box[0],box[1],w,h,0,0,w,h);
 const kinds=[...(el('alarms').checked?['C']:[]),...(el('otherStates').checked?['U','T']:[])];
 const step=Math.max(2,Math.ceil(10/(zoom*(sharedDisplay??display)))),radius=1.45/factor;
 ctx.globalAlpha=Math.max(.35,intensity);ctx.lineWidth=1.2/factor;
 for(const kind of kinds){ctx.strokeStyle='white';ctx.fillStyle=problemColors[kind];ctx.beginPath();for(const [x,y] of projectionSamples(source,kind,box,step)){ctx.moveTo(x+radius,y);ctx.arc(x,y,radius,0,Math.PI*2);}ctx.stroke();ctx.fill();}
 ctx.restore();
}
function visibleProblemLabels(model){
 if(!loaded?.problem||!el('problemLabels').checked)return [];
 const state=triggerState();if((state==='C'&&!el('alarms').checked)||(state!=='C'&&!el('otherStates').checked))return [];
 const meta=loaded.problem.meta,n=meta.counts[state],rows=meta.models[model][state].filter(row=>maskIsVisible(model,row.id)&&row.count>=Math.max(4,n*.01)).slice(0,3),focus=loaded.models[model].inspectionFocus;
 if(focus&&maskIsVisible(model,focus.id)&&!rows.some(row=>row.id===focus.id)){const row=meta.models[model][state].find(row=>row.id===focus.id);if(row)rows.push(row);}
 return rows;
}
function hasProjectedLabel(model,id){return visibleProblemLabels(model).some(row=>row.id===id);}
function drawProblemOverlay(canvas,ctx,model){
 if(!loaded?.problem||loaded.j.key!==activeKey)return;
 const [w,h]=loaded.j.wh,display=Math.max(.01,(canvas.clientWidth||w)/w),factor=scale*display;
 const focus=loaded.models[model].inspectionFocus;
 if(focus&&maskIsVisible(model,focus.id)&&focus.layer){ctx.save();ctx.globalAlpha=1;ctx.shadowColor='#00bed8';ctx.shadowBlur=3/factor;ctx.drawImage(focus.layer,0,0);ctx.restore();}
 renderProjection(canvas,ctx,loaded.j.box,scale,(canvases[0].clientWidth||w)/w);
 loaded.problem.tagRects[model]=[];
 const state=triggerState();ctx.save();ctx.font=(11/factor)+'px Segoe UI';ctx.textAlign='left';ctx.textBaseline='middle';
 const left=(centerX-.5/scale)*w,right=(centerX+.5/scale)*w,top=(centerY-.5/scale)*h,bottom=(centerY+.5/scale)*h;
 for(const row of visibleProblemLabels(model)){
  if(row.x<left||row.x>right||row.y<top||row.y>bottom)continue;
  const label=row.id===0?'未分割':'ID '+row.id,tw=ctx.measureText(label).width,rw=tw+6/factor,rh=16/factor,pad=2/factor;
  const offsets=[[6,-17],[6,5],[-rw*factor-6,-17],[-rw*factor-6,5],[6,-39],[6,27],[-rw*factor-6,-39],[-rw*factor-6,27]];
  let rectangle=null;
  for(const [dx,dy] of offsets){
   const r={id:row.id,x:Math.max(left+3/factor,Math.min(right-rw-3/factor,row.x+dx/factor)),y:Math.max(top+3/factor,Math.min(bottom-rh-3/factor,row.y+dy/factor)),w:rw,h:rh};
   if(loaded.problem.tagRects[model].some(a=>r.x<a.x+a.w+pad&&r.x+r.w+pad>a.x&&r.y<a.y+a.h+pad&&r.y+r.h+pad>a.y))continue;
   rectangle=r;break;
  }
  if(!rectangle)continue;
  const {x,y}=rectangle,endX=Math.max(x,Math.min(x+rw,row.x)),endY=Math.max(y,Math.min(y+rh,row.y));
  ctx.strokeStyle='white';ctx.lineWidth=1/factor;ctx.beginPath();ctx.moveTo(row.x,row.y);ctx.lineTo(endX,endY);ctx.stroke();
  ctx.fillStyle=state==='C'?'#a80b2b':state==='U'?'#475569':'#955f07';ctx.fillRect(x,y,rw,rh);ctx.fillStyle='white';ctx.fillText(label,x+3/factor,y+8/factor);
  loaded.problem.tagRects[model].push(rectangle);
 }
 ctx.restore();
}
function projectionLabelHit(model,x,y){
 const rects=loaded?.problem?.tagRects[model]||[];
 for(let i=rects.length-1;i>=0;i--){const r=rects[i];if(x>=r.x&&x<r.x+r.w&&y>=r.y&&y<r.y+r.h)return r.id;}
 return null;
}
function drawFullProblemOverlay(canvas,ctx){if(!loaded?.problem)return;renderProjection(canvas,ctx,[0,0,loaded.problem.source.w,loaded.problem.source.h],1);}
function zoomToProblemBox(box){
 if(!box)return;const [w,h]=loaded.j.wh,bw=Math.max(1,box[2]-box[0]),bh=Math.max(1,box[3]-box[1]);
 scale=Math.max(1,Math.min(10,w/(bw*1.2),h/(bh*1.2)));centerX=(box[0]+box[2])/(2*w);centerY=(box[1]+box[3])/(2*h);clampCenter();drawAll();
}
function focusProblemMask(model,id){
 if(!loaded?.problem||loaded.j.key!==activeKey)return;
 const row=loaded.problem.meta.models[model][triggerState()].find(r=>r.id===id);if(!row)return;
 inspectMask(model,id,true,row.projection_box);
}
function inspectMask(model,id,zoom=false,fallback=null){
 if(!loaded?.problem||loaded.j.key!==activeKey)return;
 if(id>0&&!maskIsVisible(model,id))setMaskVisible(model,id,true);
 let box=fallback,layer=null;
 if(id>0){
  const [w,h]=loaded.j.wh,data=loaded.models[model].idData;if(!data)return;
  const labels=new Uint16Array(w*h),rgba=new Uint8ClampedArray(data.length);let x0=w,y0=h,x1=0,y1=0;
  for(let i=0;i<labels.length;i++)labels[i]=data[i*4]+256*data[i*4+1];
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
   const i=y*w+x;if(labels[i]!==id)continue;x0=Math.min(x0,x);y0=Math.min(y0,y);x1=Math.max(x1,x+1);y1=Math.max(y1,y+1);
   const edge=x===0||y===0||x===w-1||y===h-1||labels[i-1]!==id||labels[i+1]!==id||labels[i-w]!==id||labels[i+w]!==id;
   rgba.set([0,201,223,edge?255:24],i*4);
  }
  if(x1>x0&&y1>y0){box=[x0,y0,x1,y1];layer=canvasFromPixels(w,h,rgba);}
 }
 loaded.problem.focus={model,id,box,layer};loaded.models[model].inspectionFocus={id,box,layer};refreshMaskVisibility(model);
 el('problemFocusStatus').textContent=(zoom?'已定位 ':'查看 ')+['原始 CropFormer','局部 CropFormer','SAM2.1'][model]+' · '+(id?'ID '+id:'未落入保留 mask 的区域');
 if(zoom)zoomToProblemBox(box);else drawAll();
}
function clearProblemFocus(){if(loaded?.problem){loaded.problem.focus=null;loaded.models.forEach((m,i)=>{m.inspectionFocus=null;refreshMaskVisibility(i);});}el('problemFocusStatus').textContent='';}
function currentProjectedState(x,y){
 if(!loaded?.problem)return '';const s=loaded.problem.source,fx=x+loaded.j.box[0],fy=y+loaded.j.box[1];return ['无异常投影','U 未分配投影','T 状态投影','C 冲突投影'][s.codes[fy*s.w+fx]||0];
}
['otherStates','alarmOpacity','problemLabels'].forEach(id=>el(id).oninput=drawAll);
el('focusProblem').onclick=()=>{if(!loaded?.problem)return;clearProblemFocus();zoomToProblemBox(loaded.problem.meta.state_boxes[triggerState()]);};
el('clearProblemFocus').onclick=()=>{clearProblemFocus();scale=1;centerX=centerY=.5;drawAll();};
el('referenceDetails').ontoggle=drawAll;
window.addEventListener('resize',()=>requestAnimationFrame(drawAll));

// Independent viewing controls; repair target IDs and native labels are untouched.
const maskDisplayPreferences=new Map();
let maskHoverIDs=[null,null,null];
function maskIDs(model){return Object.keys(loaded.models[model].data.colors).map(Number).sort((a,b)=>a-b);}
function maskIsVisible(model,id){return id===0||((intersectsROI(model,id)||loaded?.models[model]?.showUnrelated)&&!loaded?.models[model]?.hiddenMasks?.has(id));}
function prepareMaskDisplay(){
 maskHoverIDs=[null,null,null];
 for(let model=0;model<3;model++){
  const m=loaded.models[model],key=activeKey+'|'+model;
  m.hiddenMasks=new Set(maskDisplayPreferences.get(key)||[]);m.inspectionFocus=null;m.filteredImages=null;
  buildMaskLegend(model);refreshMaskVisibility(model);
 }
 refreshProblemSummary();
}
function rememberMaskDisplay(model){maskDisplayPreferences.set(activeKey+'|'+model,[...loaded.models[model].hiddenMasks]);}
function buildMaskLegend(model){
 const root=el('maskLegend'+model);root.replaceChildren();
 const m=loaded.models[model],related=new Map(loaded.problem.meta.models[model][triggerState()].map(row=>[row.id,row.count]));
 const area=new Map(m.data.positions.map(row=>[row.id,row.area]));
 const ids=maskIDs(model).sort((a,b)=>(related.get(b)||0)-(related.get(a)||0)||(area.get(b)||0)-(area.get(a)||0)||a-b);
 for(const id of ids){
  const row=document.createElement('div');row.className='maskRow';row.dataset.maskID=String(id);
  const label=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=maskIsVisible(model,id);check.dataset.visibilityID=String(id);
  check.setAttribute('aria-label','显示'+['原始 CropFormer','局部 CropFormer','SAM2.1'][model]+' mask ID '+id);
  check.onchange=()=>setMaskVisible(model,id,check.checked);
  const swatch=document.createElement('span');swatch.className='maskSwatch';swatch.style.background='rgb('+m.data.colors[id].join(',')+')';swatch.setAttribute('aria-hidden','true');
  const text=document.createElement('span');text.textContent='ID '+id;label.append(check,swatch,text);
  const highlight=document.createElement('button');highlight.textContent='高亮';highlight.setAttribute('aria-label','高亮'+['原始 CropFormer','局部 CropFormer','SAM2.1'][model]+' mask ID '+id);highlight.onclick=()=>inspectMask(model,id,false);
  const solo=document.createElement('button');solo.textContent='只看';solo.setAttribute('aria-label','只看'+['原始 CropFormer','局部 CropFormer','SAM2.1'][model]+' mask ID '+id);solo.onclick=()=>soloMask(model,id);
  if(related.has(id))row.title=problemNames[triggerState()]+'投影 '+related.get(id)+' 像素';
  row.append(label,highlight,solo);root.appendChild(row);
 }
}
function refreshMaskVisibility(model){
 if(!loaded?.models[model]?.hiddenMasks)return;
 const m=loaded.models[model],ids=maskIDs(model),shown=ids.filter(id=>maskIsVisible(model,id)).length;
 el('maskCount'+model).textContent='显示 '+shown+' / '+ids.length+' 个 mask';
 const active=m.inspectionFocus?.id;
 el('maskActive'+model).textContent=active?'当前查看：ID '+active+(maskIsVisible(model,active)?'':'（已隐藏）'):'';
 refreshMaskHover(model);
 el('maskLegend'+model).querySelectorAll('.maskRow').forEach(row=>{
  const id=Number(row.dataset.maskID),visible=maskIsVisible(model,id);row.querySelector('input').checked=visible;row.classList.toggle('hiddenMask',!visible);row.classList.toggle('inspectedMask',id===active&&visible);
 });decorateIntersectionLegend(model);
}
function invalidateVisibleImages(model){loaded.models[model].filteredImages=null;}
function maskImages(model){
 const m=loaded.models[model];if(!m.hiddenMasks?.size&&!m.data.relevance.no_intersection_ids.length)return {color:m.color,boundary:m.boundary};
 if(m.filteredImages)return m.filteredImages;
 const [w,h]=loaded.j.wh,colors=loadIDs(m.color),edges=loadIDs(m.boundary),ids=m.idData;
 const color=new Uint8ClampedArray(colors),boundary=new Uint8ClampedArray(edges);
 for(let i=0;i<w*h;i++){const p=i*4,id=ids[p]+256*ids[p+1];if(!maskIsVisible(model,id)){color[p+3]=0;boundary[p+3]=0;}}
 return m.filteredImages={color:canvasFromPixels(w,h,color),boundary:canvasFromPixels(w,h,boundary)};
}
function setMaskVisible(model,id,visible){
 if(!loaded?.models[model]?.hiddenMasks||id<=0)return;
 const m=loaded.models[model];if(visible)m.hiddenMasks.delete(id);else m.hiddenMasks.add(id);
 invalidateVisibleImages(model);rememberMaskDisplay(model);refreshMaskVisibility(model);refreshProblemSummary();drawAll();
}
function soloMask(model,id){
 if(!loaded?.models[model]?.hiddenMasks)return;
 loaded.models[model].hiddenMasks=new Set(maskIDs(model).filter(value=>value!==id));
 invalidateVisibleImages(model);rememberMaskDisplay(model);refreshMaskVisibility(model);refreshProblemSummary();inspectMask(model,id,false);
}
function showAllMasks(model){
 if(!loaded?.models[model]?.hiddenMasks)return;
 const m=loaded.models[model];m.hiddenMasks.clear();m.inspectionFocus=null;
 if(loaded.problem.focus?.model===model){loaded.problem.focus=null;el('problemFocusStatus').textContent='';}
 invalidateVisibleImages(model);rememberMaskDisplay(model);refreshMaskVisibility(model);refreshProblemSummary();drawAll();
}
function hideAllMasks(model){
 if(!loaded?.models[model]?.hiddenMasks)return;
 loaded.models[model].hiddenMasks=new Set(maskIDs(model));
 invalidateVisibleImages(model);rememberMaskDisplay(model);refreshMaskVisibility(model);refreshProblemSummary();drawAll();
}
function refreshMaskHover(model){const id=maskHoverIDs[model];el('maskHover'+model).textContent=id===null?'':id?'鼠标位置：ID '+id+(maskIsVisible(model,id)?'':'（已隐藏）'):'鼠标位置：未落入保留 mask';}
function showMaskIDsAtPoint(labels){if(!loaded)return;maskHoverIDs=labels;labels.forEach((id,model)=>refreshMaskHover(model));}
for(let model=0;model<3;model++){
 el('showAllMasks'+model).onclick=()=>showAllMasks(model);el('hideAllMasks'+model).onclick=()=>hideAllMasks(model);
 let press=null;const c=canvases[model];
 c.addEventListener('pointerdown',e=>press={x:e.clientX,y:e.clientY});
 c.addEventListener('pointerup',e=>{
  if(!press||Math.hypot(e.clientX-press.x,e.clientY-press.y)>=5||!loaded||loaded.j.key!==activeKey){press=null;return;}press=null;
  const p=nativePoint(e,c),[w,h]=loaded.j.wh,x=Math.floor(p.x*w),y=Math.floor(p.y*h);if(x<0||y<0||x>=w||y>=h)return;
  const removedID=rejectedPixel(model,x,y);if(removedID){e.cvprRejectedMaskID=removedID;rejectedNotice(model,removedID);return;}
  const data=loaded.models[model].idData;if(!data)return;const offset=(y*w+x)*4,id=projectionLabelHit(model,p.x*w,p.y*h)??(data[offset]+256*data[offset+1]);
  e.cvprViewedMaskID=id;
  if(id<=0){el('maskHover'+model).textContent='鼠标位置：未落入保留 mask';return;}
  if(e.shiftKey)setMaskVisible(model,id,!maskIsVisible(model,id));else inspectMask(model,id,false);
 });
 c.addEventListener('pointercancel',()=>press=null);
 c.addEventListener('mouseleave',()=>{maskHoverIDs=[null,null,null];for(let i=0;i<3;i++)refreshMaskHover(i);});
}

// Runs inside the existing viewer closure. Candidate labels remain unchanged.

const previousReferences={};
function refreshPreviousReference(current){
 const host=el('previousReference'),old=previousReferences[current.uid];host.replaceChildren();
 if(!old)return;
 const label=document.createElement('span');
 label.textContent='旧选择参考：视角 '+old.view_number+' / 帧 '+old.frame+' / '+modelNames[old.model_index]+' / ID '+old.mask_ids.join('、')+'。本轮需重新确认目标。';host.appendChild(label);
 const b=document.createElement('button');b.type='button';b.textContent='载入旧选择作草稿';
 b.onclick=async()=>{
  const i=current.views.findIndex(v=>v.number===old.view_number&&v.frame===old.frame);
  if(i<0||!current.views[i].variants.includes(old.crop_key)){selectionNotice('旧选择不在当前候选中。');return;}
  activeView=i;el('view').value=String(i);const v=current.views[i];
  el('crop').replaceChildren(...v.variants.map(k=>option(k,cropLabel(D.crops[k]))));el('crop').value=old.crop_key;
  await chooseCrop(old.crop_key);
  if(!loaded||loaded.j.key!==old.crop_key)return;
  chosenModel=old.model_index;chosenIDs=new Set(old.mask_ids);refreshSelection();el('numbers').checked=true;drawAll();
  selectionNotice('已载入旧草稿。请检查目标，删掉不相关 mask，再保存确认。');
 };host.appendChild(b);
}

const pilotUIDs=new Set(["room2-918c0a39ec3a355e", "room2-4fb8768bf9c4ac8e", "room2-cc78ce4b4393bb15", "room2-ae199e2c6951e5a3", "room2-3779323d1b73b6d8", "room2-d89e81d88fa55663", "room2-846561f1dce70f69", "room2-b24c00af3bf63474", "room2-9a0c4db1de376f2b", "room2-08248c1eef1b6a0e", "room2-0930fd24a110f5be", "room2-32a207d6993dcb14", "room2-ab47e0ea0626b9de", "room2-c9343a4583d29b1d", "room2-be6d2db8ec4281d7", "room2-58d12c89e9465050", "room2-c55b885fc040cd71", "room2-c30fd86dc9258ed1", "room2-6bc112111e899667", "room2-f0d4f5e7f367750b", "room2-b3c73bd1250086a8", "room2-bddeed0280eaf0a7", "room2-79b63467911537e5", "room2-08e75dd30dbc5c73", "room2-b4c4488c84960852", "room2-ddfccdee5d22298c", "room2-62a610165ed0f638", "room2-16d06b992339dac7", "room2-36c9b9f7868327ac", "room2-c401d0456b360359", "room2-285810af33f3380e", "room2-2ddf8aed098beb94"]);
const modelChoices=['original','cropformer_local','sam2_1_local'];
const modelNames=['原始 CropFormer','局部 CropFormer','SAM2.1'];
const qaMode=new URLSearchParams(location.search).get('qa')==='1';
const storageKey='cvpr-room2-e1-intersection-roi-20261015-v1'+(qaMode?'-qa':'');
let savedSelections={};
try{const cached=JSON.parse(localStorage.getItem(storageKey)||'{}');if(cached&&typeof cached==='object'&&!Array.isArray(cached))savedSelections=cached;}catch(e){el('selectionMessage').textContent='浏览器记录不可用，请及时导出选择。';}
let selectionContext='',chosenModel=null,chosenIDs=new Set();

function selectionNotice(message){el('selectionMessage').textContent=message;}
function selectionContextChanged(){
 const context=D.cases[activeCase].uid+'|'+activeKey;
 if(context!==selectionContext){
  selectionContext=context;chosenModel=null;chosenIDs.clear();el('targetNote').value='';
  const old=savedSelections[D.cases[activeCase].uid];
  if(old&&old.crop_key===activeKey){chosenModel=old.model_index;chosenIDs=new Set(old.mask_ids||[]);el('targetNote').value=old.target_notes||'';}
  selectionNotice('');
 }
 refreshSelection();
}
function exportPayload(){return {
 schema:'cvpr_roi_human_selection_v1',source_selection:'room2-e1-intersection-roi-20261015',intersection_filter:D.intersection_filter,overmerge_filter:D.overmerge_filter,
 exported_at:new Date().toISOString(),is_qa:qaMode,
 pilot_case_uids:[...pilotUIDs],total_fixed_cases:D.cases.length,source_total_fixed_cases:32,ranked_entries:60,scene_scope:'room2',asset_base_path:'',source_data_directory:"/data/chenkejun/CVPR/results/e1_roi_top20_20261015/review",baseline_method:'P1-A1-strict-vote / E1 prerepair',baseline_repaired:false,source_candidate_sha256:"4aa32c240bbcf5a4284d3021a905d357cfcb415f5a386f1c8ea7043d2c238e94",source_surface_sha256:"3f611dfe701a68259c71b94962ae6bae3c13fbac9c24f1e3eca5b414b8d2d4f3",annotation_complete:D.cases.every(c=>!!savedSelections[c.uid]),case_order:D.cases.map(c=>({uid:c.uid,rankings:c.rankings})),
 selections:D.cases.flatMap(c=>savedSelections[c.uid]?[savedSelections[c.uid]]:[])
};}
function refreshSelection(){
 const current=D.cases[activeCase],record=savedSelections[current.uid];refreshPreviousReference(current);
 const pilotDone=[...pilotUIDs].filter(uid=>savedSelections[uid]).length;
 el('selectionProgress').textContent=(qaMode?'功能测试 · ':'')+'room2 已确认 '+pilotDone+' / '+D.cases.length;
 el('savedSelection').textContent=record?'已保存：视角 '+record.view_number+' / 帧 '+record.frame+' / '+(record.choice==='unreliable'?'都不可靠 / 不通过':modelNames[record.model_index]+' / 目标 ID '+record.mask_ids.join('、')+(record.border_touching_mask_ids?.length?' / 边界 mask 已允许':'')):'本案例尚未保存';
 document.querySelectorAll('[data-model-choice]').forEach(b=>{const isActive=b.dataset.modelChoice===String(chosenModel);b.classList.toggle('chosen',isActive);b.setAttribute('aria-pressed',String(isActive));});
 document.querySelectorAll('.panel').forEach((p,i)=>p.classList.toggle('selectedPanel',i===chosenModel));
 el('selectedTargets').textContent=chosenModel===null?'先选一套方案，再在该栏点击质量合格的完整目标。':chosenModel===-1?'三套都不可靠 / 不通过：记录失败，保留地图。':'已选目标 ID：'+([...chosenIDs].sort((a,b)=>a-b).join('、')||'尚未选择');
 const list=el('targetChoices');list.replaceChildren();
 if(chosenModel!==null&&chosenModel>=0&&loaded&&loaded.j.key===activeKey){
  const involvement=new Map((loaded.problem?.meta.models[chosenModel][triggerState()]||[]).map(row=>[row.id,row.count]));
  maskIDs(chosenModel).filter(id=>intersectsROI(chosenModel,id)).map(id=>({id})).forEach(p=>{
   const b=document.createElement('button');b.className='targetChoice'+(chosenIDs.has(p.id)?' chosen':'');
   b.textContent='ID '+p.id;b.setAttribute('aria-label','选择目标 ID '+p.id);b.setAttribute('aria-pressed',String(chosenIDs.has(p.id)));
   b.onclick=()=>toggleTarget(p.id);list.appendChild(b);
  });
 }
 el('selectionJson').value=JSON.stringify(exportPayload(),null,2);
}
function toggleTarget(id){if(id<=0)return;if(!intersectsROI(chosenModel,id)){selectionNotice('ID '+id+' 与扩展异常表面无交集，仅作对照，未加入候选。');return;}if(chosenIDs.has(id))chosenIDs.delete(id);else chosenIDs.add(id);selectionNotice('');refreshSelection();el('numbers').checked=true;drawAll();}
function chooseModel(index){if(chosenModel!==index)chosenIDs.clear();chosenModel=index;selectionNotice('');refreshSelection();el('numbers').checked=index>=0;drawAll();}
function selectFromCanvas(event,canvas,index){
 if(event.shiftKey||event.cvprRejectedMaskID)return;
 if(chosenModel!==index||!loaded||loaded.j.key!==activeKey)return;
 const p=nativePoint(event,canvas),[w,h]=loaded.j.wh,x=Math.floor(p.x*w),y=Math.floor(p.y*h);
 if(x<0||y<0||x>=w||y>=h)return;
 const data=loaded.models[index].idData;if(!data){selectionNotice('像素编号无法读取，请使用下面的 ID 按钮。');return;}
 const offset=(y*w+x)*4,id=event.cvprViewedMaskID??projectionLabelHit(index,p.x*w,p.y*h)??(data[offset]+256*data[offset+1]);
 if(id>0)toggleTarget(id);else selectionNotice('这里没有目标 mask，请点击物体内部。');
}
function saveSelection(moveNext){
 if(chosenModel===null){selectionNotice('请先选择一套方案，或选择“都不可靠 / 不通过”。');return;}
 if(chosenModel>=0&&(!loaded||loaded.j.key!==activeKey)){selectionNotice('图像尚未加载完成。');return;}
 if(chosenModel>=0&&chosenIDs.size===0){selectionNotice('请点击质量合格的完整目标；可以选多个。');return;}
 const c=D.cases[activeCase],v=c.views[activeView],j=D.crops[activeKey];
 const ids=[...chosenIDs].sort((a,b)=>a-b),borderTouchingIDs=new Set();if(chosenModel>=0&&ids.some(id=>!intersectsROI(chosenModel,id))){selectionNotice('包含无交集 Mask，请重新选择相交候选。');return;}
 if(chosenModel>=0){
  const pixels=loaded.models[chosenModel].idData,[w,h]=j.wh,found=new Set(),cut=new Set();
  if(!pixels){selectionNotice('像素编号尚不可读，请刷新后重试。');return;}
  const desired=new Set(ids),fullW=loaded.full.naturalWidth,fullH=loaded.full.naturalHeight;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
   const p=(y*w+x)*4,id=pixels[p]+256*pixels[p+1];if(!desired.has(id))continue;found.add(id);
   if((x===0&&j.box[0]>0)||(x===w-1&&j.box[2]<fullW)||(y===0&&j.box[1]>0)||(y===h-1&&j.box[3]<fullH))cut.add(id);
  }
  if(ids.some(id=>!found.has(id))){selectionNotice('目标编号不属于当前图像，请重新选择。');return;}
  cut.forEach(id=>borderTouchingIDs.add(id));
 }
 const record={case_uid:c.uid,scene:c.scene,ROI:c.ROI,aliases:c.aliases,rankings:c.rankings,counts:c.counts,quality_status:chosenModel<0?'FAIL':'PASS',source_entry:v.source_entry,trigger_group:v.trigger_group,
  view_number:v.number,frame:v.frame,source_rgb:v.source_RGB,crop_key:activeKey,
  crop_xyxy:j.box,native_crop_wh:j.wh,crop_strategy:j.strategy,
  choice:chosenModel<0?'unreliable':modelChoices[chosenModel],model_index:chosenModel,
  mask_ids:ids,label_asset:chosenModel<0?null:'assets/'+activeKey+'_'+['original','local','sam'][chosenModel]+'_ids.png',review_label_asset:chosenModel<0?null:j.models[chosenModel].ids_file,overmerge_filter:D.overmerge_filter,
  scope:borderTouchingIDs.size?'selected_visible_masks':'selected_whole_visible_objects',artificial_crop_border_check:borderTouchingIDs.size?'TOUCHES_CROP_BORDER_ALLOWED':'PASS',border_touching_mask_ids:[...borderTouchingIDs].sort((a,b)=>a-b),crop_border_selection_allowed:true,
  intersection_filter:D.intersection_filter,intersection_support:chosenModel<0?[]:ids.map(id=>({id,...j.models[chosenModel].relevance.labels[id]})),selected_at:new Date().toISOString(),is_qa:qaMode,target_notes:el('targetNote').value.trim()};
 savedSelections[c.uid]=record;
 let persisted=true;try{localStorage.setItem(storageKey,JSON.stringify(savedSelections));}catch(e){persisted=false;}
 refreshSelection();buildCases();
 selectionNotice(persisted?('已保存到本浏览器。'+(borderTouchingIDs.size?'ID '+[...borderTouchingIDs].join('、')+' 触及裁剪边界，已允许保存。':'')):'已记录，但浏览器存储失败，请立即导出选择。');
 if(moveNext&&persisted){const position=filtered.indexOf(activeCase),next=filtered.slice(position+1).find(i=>!savedSelections[D.cases[i].uid])??filtered.find(i=>!savedSelections[D.cases[i].uid]);if(next!==undefined)chooseCase(next);else selectionNotice('32 个区域全部完成。请导出 room2 选择，放到 CVPR 文件夹，告诉 Codex“room2 已标好”。');}
}
function updateSelectionFilter(){
 filtered=D.cases.flatMap((c,i)=>(el('scene').value==='all'||c.rankings.some(r=>r.group===el('scene').value))&&(!el('pilotOnly').checked||pilotUIDs.has(c.uid))?[i]:[]);
 if(el('scene').value!=='all')filtered.sort((a,b)=>D.cases[a].rankings.find(r=>r.group===el('scene').value).rank-D.cases[b].rankings.find(r=>r.group===el('scene').value).rank);chooseCase(filtered[0]);
}
document.querySelectorAll('[data-model-choice]').forEach(b=>b.onclick=()=>chooseModel(Number(b.dataset.modelChoice)));
el('saveSelection').onclick=()=>saveSelection(false);el('saveAndNext').onclick=()=>saveSelection(true);
el('pilotOnly').onchange=updateSelectionFilter;
el('exportSelections').onclick=()=>{
 const file=new Blob([JSON.stringify(exportPayload(),null,2)],{type:'application/json'}),url=URL.createObjectURL(file),a=document.createElement('a');
 a.href=url;a.download=(qaMode?'QA_':'')+'room2_E1异常扩展相交候选_质量判定_20261015.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 selectionNotice('已导出 room2_E1异常扩展相交候选_质量判定_20261015.json。请放到本地 CVPR 文件夹，告诉 Codex“room2 已标好”。');
};
canvases.forEach((c,i)=>{
 let press=null;
 c.addEventListener('pointerdown',e=>press={x:e.clientX,y:e.clientY});
 c.addEventListener('pointerup',e=>{if(press&&Math.hypot(e.clientX-press.x,e.clientY-press.y)<5)selectFromCanvas(e,c,i);press=null;});
 c.addEventListener('pointercancel',()=>press=null);
});
el('scene').onchange=updateSelectionFilter;
filtered=D.cases.flatMap((c,i)=>pilotUIDs.has(c.uid)?[i]:[]);
refreshSelection();

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

window.threewayReview={data:D,getState:()=>({case:activeCase,view:activeView,key:activeKey,loaded:!!loaded,scale}),selectCase:chooseCase,selectView:chooseView,selectCrop:chooseCrop};
const requested=new URLSearchParams(location.search),requestedCase=D.cases.findIndex(c=>c.uid===requested.get('roi'));
chooseCase(requestedCase>=0?requestedCase:filtered[0]);
const requestedCrop=requested.get('crop');if(requestedCrop){const ci=D.cases[activeCase],vi=ci.views.findIndex(v=>v.variants.includes(requestedCrop));if(vi>=0){el('view').value=String(vi);chooseView(vi,requestedCrop);}}
})();
