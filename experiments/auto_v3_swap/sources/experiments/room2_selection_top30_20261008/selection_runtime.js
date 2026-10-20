(()=>{
'use strict';
const D=JSON.parse(document.getElementById('reviewData').textContent),el=id=>document.getElementById(id);
let activeCase=0,activeView=0,activeKey='',filtered=D.cases.map((_,i)=>i),loadToken=0,loaded=null;
let scale=1,centerX=.5,centerY=.5,drag=null;
const imageCache=new Map(),canvases=[el('canvas0'),el('canvas1'),el('canvas2')];
function getImage(src){if(!imageCache.has(src)){imageCache.set(src,new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error('图像无法加载：'+src));img.src=src;}));while(imageCache.size>90)imageCache.delete(imageCache.keys().next().value);}return imageCache.get(src);}
function loadIDs(img){const c=document.createElement('canvas');c.width=img.naturalWidth;c.height=img.naturalHeight;c.getContext('2d').drawImage(img,0,0);try{return c.getContext('2d').getImageData(0,0,c.width,c.height).data;}catch(e){if(e.name==='SecurityError')return null;throw e;}}
function buildCases(){el('cases').replaceChildren();filtered.forEach(i=>{const c=D.cases[i],b=document.createElement('button');b.className='case'+(i===activeCase?' active':'');b.dataset.caseIndex=i;b.innerHTML='<img alt=""><span><b></b><small></small></span>';b.querySelector('img').src=c.views[0].full_rgb;b.querySelector('b').textContent=String(c.order).padStart(2,'0')+' · '+c.ROI;b.querySelector('small').textContent=(savedSelections[c.uid]?'已选 · ':'')+'C '+c.counts.CONFLICT+' · U '+c.counts.U+' · T '+c.counts.T;b.onclick=()=>chooseCase(i);el('cases').appendChild(b);});}
function option(value,label){const o=document.createElement('option');o.value=value;o.textContent=label;return o;}
function chooseCase(i){activeCase=i;activeView=0;const c=D.cases[i];el('caseTitle').textContent=c.scene+' / '+c.ROI;el('caseInfo').textContent='原始冲突排名 '+c.order+' / 30 · C '+c.counts.CONFLICT+' 点 · 同区域编号 '+c.aliases.join(' / ');el('view').replaceChildren(...c.views.map((v,j)=>option(j,'视角 '+v.number+' · 帧 '+v.frame)));el('view').value='0';buildCases();updateNav();const saved=savedSelections[c.uid],resume=saved?c.views.findIndex(v=>v.frame===saved.frame&&v.number===saved.view_number):0;chooseView(Math.max(0,resume),saved?.crop_key);el('view').value=String(activeView);}
function updateNav(){const p=filtered.indexOf(activeCase);el('counter').textContent=(p+1)+' / '+filtered.length;el('previous').disabled=p<=0;el('next').disabled=p>=filtered.length-1;}
function cropLabel(j){if(j.strategy==='balanced')return '完整目标 + 15% 上下文（默认） · '+j.zoom.toFixed(2)+'×';if(j.strategy==='context')return '宽上下文 · '+j.zoom.toFixed(2)+'×';return '局部窗口 '+activeTileIndex(j.key)+' · '+j.zoom.toFixed(2)+'× · 目标点 '+Math.round(j.coverage*100)+'%';}
function activeTileIndex(key){const m=key.match(/_tile(\d+)$/);return m?m[1]:'';}
function chooseView(j,preferred){activeView=j;const v=D.cases[activeCase].views[j];el('crop').replaceChildren(...v.variants.map(k=>option(k,cropLabel(D.crops[k]))));const key=v.variants.includes(preferred)?preferred:v.recommended;el('crop').value=key;chooseCrop(key);}
async function chooseCrop(key){const token=++loadToken;activeKey=key;const j=D.crops[key],v=D.cases[activeCase].views[activeView];el('models').classList.add('loading');el('status').textContent='加载对比…';el('error').textContent='';scale=1;centerX=centerY=.5;
 const coverage=j.complete?'覆盖该视角全部目标状态点':'局部窗口覆盖 '+Math.round(j.coverage*100)+'% 目标状态点，物体可能延伸到窗口之外';
 const gain=j.zoom<1.05?'该区域接近整帧，完整目标输入没有明显放大；可切换局部窗口。':'相对原整帧像素尺度 '+j.zoom.toFixed(2)+' 倍。';
 selectionContextChanged();clearProblemFocus();el('projectionSummary').textContent='读取异常投影…';el('focusProblem').disabled=true;
 el('cropInfo').textContent='原始输入 '+j.wh.join(' × ')+' px → CropFormer 网络整图分支 '+j.network_input_wh.join(' × ')+' px；'+gain+' '+coverage+'。';
 el('meta0').textContent=j.models[0].count+' 个可见 mask · 读取原整帧缓存后裁剪';el('meta1').textContent=j.models[1].count+' 个可见 mask · 相同权重 / 0.5 阈值';el('meta2').textContent=j.models[2].count+' 个可见 mask · Hiera Large / 官方 AMG 默认';
 const balanced=D.crops[v.recommended];el('staticLink').href=balanced.comparison;el('staticLink').textContent='保存用四栏对比图（完整目标）';
 try{const [rgb,full,alarm,models]=await Promise.all([getImage(j.RGB),getImage(v.full_rgb),getImage(v.alarm),Promise.all(j.models.map(async m=>{const [color,boundary,ids]=await Promise.all([getImage(m.color),getImage(m.boundary),getImage(m.ids)]);return {color,boundary,idData:loadIDs(ids),data:m};}))]);if(token!==loadToken)return;loaded={j,rgb,full,alarm,models};prepareProblemAnnotations();prepareMaskDisplay();refreshSelection();[el('rgbRef'),...canvases].forEach(c=>{c.width=j.wh[0];c.height=j.wh[1];});el('largeRGB').src=v.full_rgb;drawAll();el('models').classList.remove('loading');el('status').textContent='';}catch(err){if(token===loadToken){el('error').textContent=err.message;el('status').textContent='加载失败';el('models').classList.remove('loading');}}
}
function transformed(ctx,w,h,callback){ctx.save();ctx.translate(w/2-centerX*w*scale,h/2-centerY*h*scale);ctx.scale(scale,scale);callback();ctx.restore();}
function drawCanvas(c,model){if(!loaded)return;const ctx=c.getContext('2d'),[w,h]=loaded.j.wh;ctx.clearRect(0,0,w,h);ctx.fillStyle='#e6ebf1';ctx.fillRect(0,0,w,h);transformed(ctx,w,h,()=>{ctx.imageSmoothingEnabled=true;if(model===null||!el('pure').checked)ctx.drawImage(loaded.rgb,0,0);if(model!==null){const m=loaded.models[model];ctx.globalAlpha=el('pure').checked?1:Number(el('opacity').value)/100;ctx.drawImage(maskImages(model).color,0,0);ctx.globalAlpha=1;if(el('boundary').checked)ctx.drawImage(maskImages(model).boundary,0,0);if(el('numbers').checked){ctx.font=(Math.max(11,w/45)/scale)+'px Segoe UI';ctx.textAlign='center';ctx.textBaseline='middle';m.data.positions.forEach(p=>{if(p.area<60||!maskIsVisible(model,p.id)||hasProjectedLabel(model,p.id))return;const label=String(p.id),tw=ctx.measureText(label).width;ctx.fillStyle='rgba(18,26,38,.78)';ctx.fillRect(p.x-tw/2-3/scale,p.y-8/scale,tw+6/scale,16/scale);ctx.fillStyle='white';ctx.fillText(label,p.x,p.y);});}}if(model!==null)drawProblemOverlay(c,ctx,model);});}
function drawFull(){if(!loaded)return;const c=el('fullRef');c.width=1200;c.height=680;const ctx=c.getContext('2d');ctx.drawImage(loaded.full,0,0,1200,680);drawFullProblemOverlay(c,ctx);const b=loaded.j.box;ctx.strokeStyle='#20d796';ctx.lineWidth=6;ctx.strokeRect(b[0]+3,b[1]+3,b[2]-b[0]-6,b[3]-b[1]-6);}
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
el('scene').append(...Object.entries(D.scene_counts).map(([s,n])=>option(s,s+' · '+n)));
el('scene').onchange=()=>{filtered=D.cases.flatMap((c,i)=>el('scene').value==='all'||c.scene===el('scene').value?[i]:[]);chooseCase(filtered[0]);};
el('view').onchange=()=>chooseView(Number(el('view').value));el('crop').onchange=()=>chooseCrop(el('crop').value);
el('previous').onclick=()=>{const p=filtered.indexOf(activeCase);if(p>0)chooseCase(filtered[p-1]);};el('next').onclick=()=>{const p=filtered.indexOf(activeCase);if(p<filtered.length-1)chooseCase(filtered[p+1]);};
['opacity','pure','boundary','numbers','alarms'].forEach(id=>el(id).oninput=drawAll);el('reset').onclick=()=>{clearProblemFocus();scale=1;centerX=centerY=.5;drawAll();};
el('fullRef').onclick=()=>el('imageDialog').showModal();el('closeDialog').onclick=()=>el('imageDialog').close();
document.addEventListener('keydown',e=>{if(['SELECT','INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key==='ArrowRight')el('next').click();if(e.key==='ArrowLeft')el('previous').click();});
// Annotation only: fixed state projections and model IDs remain immutable.
const PROJECTIONS=JSON.parse(el('projectionData').textContent);
const projectionImageCache=new WeakMap();
const problemNames={C:'冲突',U:'未分配',T:'T 状态'};
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
 const c=D.cases[activeCase];if(c.uid!==lastProblemCase){el('otherStates').checked=c.group!=='CONFLICT';lastProblemCase=c.uid;}
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
  summary.textContent=n?problemNames[name]+'投影落入 '+nonzero.length+' 个 mask'+(unassigned?' · 未落入 mask '+(unassignedPercent<1?'＜1':Math.round(unassignedPercent))+'%':''): '当前范围没有'+problemNames[name]+'投影';
  root.appendChild(summary);
  const addEntry=(row,parent)=>{
   const percent=row.count/Math.max(1,n)*100,b=document.createElement('button');b.className='problemID';b.textContent=(row.id===0?'未落入 mask':'ID '+row.id)+' · '+(percent<1?'＜1':Math.round(percent))+'%';
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
 el('problemFocusStatus').textContent=(zoom?'已定位 ':'查看 ')+['原始 CropFormer','局部 CropFormer','SAM2.1'][model]+' · '+(id?'ID '+id:'未落入 mask 的区域');
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
function maskIsVisible(model,id){return id===0||!loaded?.models[model]?.hiddenMasks?.has(id);}
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
 });
}
function invalidateVisibleImages(model){loaded.models[model].filteredImages=null;}
function maskImages(model){
 const m=loaded.models[model];if(!m.hiddenMasks?.size)return {color:m.color,boundary:m.boundary};
 if(m.filteredImages)return m.filteredImages;
 const [w,h]=loaded.j.wh,colors=loadIDs(m.color),edges=loadIDs(m.boundary),ids=m.idData;
 const color=new Uint8ClampedArray(colors),boundary=new Uint8ClampedArray(edges);
 for(let i=0;i<w*h;i++){const p=i*4,id=ids[p]+256*ids[p+1];if(m.hiddenMasks.has(id)){color[p+3]=0;boundary[p+3]=0;}}
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
function refreshMaskHover(model){const id=maskHoverIDs[model];el('maskHover'+model).textContent=id===null?'':id?'鼠标位置：ID '+id+(maskIsVisible(model,id)?'':'（已隐藏）'):'鼠标位置：未落入 mask';}
function showMaskIDsAtPoint(labels){if(!loaded)return;maskHoverIDs=labels;labels.forEach((id,model)=>refreshMaskHover(model));}
for(let model=0;model<3;model++){
 el('showAllMasks'+model).onclick=()=>showAllMasks(model);el('hideAllMasks'+model).onclick=()=>hideAllMasks(model);
 let press=null;const c=canvases[model];
 c.addEventListener('pointerdown',e=>press={x:e.clientX,y:e.clientY});
 c.addEventListener('pointerup',e=>{
  if(!press||Math.hypot(e.clientX-press.x,e.clientY-press.y)>=5||!loaded||loaded.j.key!==activeKey){press=null;return;}press=null;
  const p=nativePoint(e,c),[w,h]=loaded.j.wh,x=Math.floor(p.x*w),y=Math.floor(p.y*h);if(x<0||y<0||x>=w||y>=h)return;
  const data=loaded.models[model].idData;if(!data)return;const offset=(y*w+x)*4,id=projectionLabelHit(model,p.x*w,p.y*h)??(data[offset]+256*data[offset+1]);
  e.cvprViewedMaskID=id;
  if(id<=0){el('maskHover'+model).textContent='鼠标位置：未落入 mask';return;}
  if(e.shiftKey)setMaskVisible(model,id,!maskIsVisible(model,id));else inspectMask(model,id,false);
 });
 c.addEventListener('pointercancel',()=>press=null);
 c.addEventListener('mouseleave',()=>{maskHoverIDs=[null,null,null];for(let i=0;i<3;i++)refreshMaskHover(i);});
}

// Runs inside the existing viewer closure. Candidate labels remain unchanged.

const previousReferences={"room2-14047feb0df3aaa5": {"case_uid": "room2-14047feb0df3aaa5", "scene": "room2", "ROI": "ROI-C0002", "aliases": ["ROI-C0002", "ROI-U0001", "ROI-T0003"], "view_number": 1, "frame": 1065, "source_rgb": "room2/assets/frame001065.jpg", "crop_key": "room2_ROI-C0002_v1_f001065_balanced", "crop_xyxy": [0, 0, 1200, 680], "native_crop_wh": [1200, 680], "crop_strategy": "balanced", "choice": "original", "model_index": 0, "mask_ids": [1, 4, 5, 8, 11], "label_asset": "assets/room2_ROI-C0002_v1_f001065_balanced_original_ids.png", "scope": "selected_whole_visible_objects", "artificial_crop_border_check": "PASS", "selected_at": "2026-10-07T09:19:07.885Z", "is_qa": false, "target_notes": ""}, "room2-7704f54997e6d29b": {"case_uid": "room2-7704f54997e6d29b", "scene": "room2", "ROI": "ROI-C0003", "aliases": ["ROI-C0003", "ROI-U0006", "ROI-T0002"], "view_number": 1, "frame": 1400, "source_rgb": "room2/assets/frame001400.jpg", "crop_key": "room2_ROI-C0003_v1_f001400_balanced", "crop_xyxy": [144, 0, 1200, 680], "native_crop_wh": [1056, 680], "crop_strategy": "balanced", "choice": "original", "model_index": 0, "mask_ids": [2, 6, 10], "label_asset": "assets/room2_ROI-C0003_v1_f001400_balanced_original_ids.png", "scope": "selected_whole_visible_objects", "artificial_crop_border_check": "PASS", "selected_at": "2026-10-07T09:19:39.575Z", "is_qa": false, "target_notes": ""}, "room2-0ba2ef6c21b4c92d": {"case_uid": "room2-0ba2ef6c21b4c92d", "scene": "room2", "ROI": "ROI-C0009", "aliases": ["ROI-C0009"], "view_number": 1, "frame": 1085, "source_rgb": "room2/assets/frame001085.jpg", "crop_key": "room2_ROI-C0009_v1_f001085_balanced", "crop_xyxy": [800, 454, 1200, 680], "native_crop_wh": [400, 226], "crop_strategy": "balanced", "choice": "cropformer_local", "model_index": 1, "mask_ids": [4], "label_asset": "assets/room2_ROI-C0009_v1_f001085_balanced_local_ids.png", "scope": "selected_visible_masks", "artificial_crop_border_check": "TOUCHES_CROP_BORDER_ALLOWED", "border_touching_mask_ids": [4], "crop_border_selection_allowed": true, "selected_at": "2026-10-07T09:25:31.388Z", "is_qa": false, "target_notes": ""}, "room2-eadde8f5bac2ff37": {"case_uid": "room2-eadde8f5bac2ff37", "scene": "room2", "ROI": "ROI-C0008", "aliases": ["ROI-C0008"], "view_number": 1, "frame": 1590, "source_rgb": "room2/assets/frame001590.jpg", "crop_key": "room2_ROI-C0008_v1_f001590_balanced", "crop_xyxy": [0, 0, 1200, 680], "native_crop_wh": [1200, 680], "crop_strategy": "balanced", "choice": "sam2_1_local", "model_index": 2, "mask_ids": [8, 10, 12, 15], "label_asset": "assets/room2_ROI-C0008_v1_f001590_balanced_sam_ids.png", "scope": "selected_whole_visible_objects", "artificial_crop_border_check": "PASS", "border_touching_mask_ids": [], "crop_border_selection_allowed": true, "selected_at": "2026-10-07T09:26:14.794Z", "is_qa": false, "target_notes": ""}, "room2-6f08166618cd41f8": {"case_uid": "room2-6f08166618cd41f8", "scene": "room2", "ROI": "ROI-C0013", "aliases": ["ROI-C0013", "ROI-U0009", "ROI-T0010"], "view_number": 1, "frame": 795, "source_rgb": "room2/assets/frame000795.jpg", "crop_key": "room2_ROI-C0013_v1_f000795_balanced", "crop_xyxy": [183, 0, 583, 615], "native_crop_wh": [400, 615], "crop_strategy": "balanced", "choice": "unreliable", "model_index": -1, "mask_ids": [], "label_asset": null, "scope": "selected_whole_visible_objects", "artificial_crop_border_check": "PASS", "border_touching_mask_ids": [], "crop_border_selection_allowed": true, "selected_at": "2026-10-07T09:26:45.295Z", "is_qa": false, "target_notes": ""}};
function refreshPreviousReference(current){
 const host=el('previousReference'),old=previousReferences[current.uid];host.replaceChildren();
 if(!old)return;
 const label=document.createElement('span');
 label.textContent='旧选择参考：视角 '+old.view_number+' / 帧 '+old.frame+' / '+(old.choice==='unreliable'?'都不可靠':modelNames[old.model_index]+' / ID '+old.mask_ids.join('、'))+'。本轮需重新确认目标。';host.appendChild(label);
 const b=document.createElement('button');b.type='button';b.textContent='载入旧选择作草稿';
 b.onclick=async()=>{
  const i=current.views.findIndex(v=>v.number===old.view_number&&v.frame===old.frame);
  if(i<0||!current.views[i].variants.includes(old.crop_key)){selectionNotice('旧选择不在当前候选中。');return;}
  activeView=i;el('view').value=String(i);const v=current.views[i];
  el('crop').replaceChildren(...v.variants.map(k=>option(k,cropLabel(D.crops[k]))));el('crop').value=old.crop_key;
  await chooseCrop(old.crop_key);
  if(!loaded||loaded.j.key!==old.crop_key)return;
  chosenModel=old.model_index;chosenIDs=new Set(old.mask_ids);el('targetNote').value=old.target_notes||'';refreshSelection();el('numbers').checked=true;drawAll();
  selectionNotice('已载入旧草稿。请检查目标，删掉不相关 mask，再保存确认。');
 };host.appendChild(b);
}

const pilotUIDs=new Set(["room2-ec916c94e6c24185", "room2-14047feb0df3aaa5", "room2-7704f54997e6d29b", "room2-ea0a9c646a7b1129", "room2-74781753162bd8a3", "room2-d4df44dfbac3d805", "room2-b18070389a477f61", "room2-eadde8f5bac2ff37", "room2-0ba2ef6c21b4c92d", "room2-9fbb545ed05f1f98", "room2-8d83120d5b27ce5e", "room2-75ede2d925653a39", "room2-6f08166618cd41f8", "room2-00d8ce08c611b2a8", "room2-579feefc65a0d844", "room2-c9af20694ea516a9", "room2-b4a44441452f25db", "room2-9cfe3a2676f3d857", "room2-f2fdcadc88f2f5c8", "room2-484f8436078a799f", "room2-643fcda2169d6501", "room2-5b29bfae991c4fc0", "room2-5a3119a1c3664255", "room2-3fc8ed82d4bb8223", "room2-f1797fac02e52a4c", "room2-3c538142a11821ea", "room2-599d3ca182893959", "room2-2cc0a409cceececc", "room2-0b324734bca2ccef", "room2-25422e72833b7e1f"]);
const modelChoices=['original','cropformer_local','sam2_1_local'];
const modelNames=['原始 CropFormer','局部 CropFormer','SAM2.1'];
const qaMode=new URLSearchParams(location.search).get('qa')==='1';
const storageKey='cvpr-room2-top30-selection-20261008-v1'+(qaMode?'-qa':'');
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
 schema:'cvpr_roi_human_selection_v1',source_selection:'room2-top30-20261008',
 exported_at:new Date().toISOString(),is_qa:qaMode,
 pilot_case_uids:[...pilotUIDs],total_fixed_cases:D.cases.length,source_total_fixed_cases:30,scene_scope:'room2',asset_base_path:'',source_data_directory:'room2_selection_top30_20261008/review',
 ranking_group:'CONFLICT',ranking_basis:'original native conflict point count descending',
 source_candidate_sha256:"bb27650bc23bfbcb0f7b3d63dcdb5edc18c593ba9c8f15fb8e76a52f2820de25",source_surface_sha256:"42cdf7b1c621785235a03e87489b94248e46dfdf5a15beea19230ca4a73c155c",
 case_order:D.cases.map(c=>({rank:c.order,case_uid:c.uid,ROI:c.ROI,conflict_points:c.counts.CONFLICT})),annotation_complete:D.cases.every(c=>!!savedSelections[c.uid]),
 selections:D.cases.flatMap(c=>savedSelections[c.uid]?[savedSelections[c.uid]]:[])
};}
function refreshSelection(){
 const current=D.cases[activeCase],record=savedSelections[current.uid];refreshPreviousReference(current);
 const pilotDone=[...pilotUIDs].filter(uid=>savedSelections[uid]).length;
 el('selectionProgress').textContent=(qaMode?'功能测试 · ':'')+'room2 已确认 '+pilotDone+' / '+D.cases.length;
 el('savedSelection').textContent=record?'已保存：视角 '+record.view_number+' / 帧 '+record.frame+' / '+(record.choice==='unreliable'?'都不可靠':modelNames[record.model_index]+' / 目标 ID '+record.mask_ids.join('、')+(record.border_touching_mask_ids?.length?' / 边界 mask 已允许':'')):'本案例尚未保存';
 document.querySelectorAll('[data-model-choice]').forEach(b=>{const isActive=b.dataset.modelChoice===String(chosenModel);b.classList.toggle('chosen',isActive);b.setAttribute('aria-pressed',String(isActive));});
 document.querySelectorAll('.panel').forEach((p,i)=>p.classList.toggle('selectedPanel',i===chosenModel));
 el('selectedTargets').textContent=chosenModel===null?'先选一套方案，再在该栏点击要修复的完整目标。':chosenModel===-1?'三套都不可靠：记录失败，保留地图。':'已选目标 ID：'+([...chosenIDs].sort((a,b)=>a-b).join('、')||'尚未选择');
 const list=el('targetChoices');list.replaceChildren();
 if(chosenModel!==null&&chosenModel>=0&&loaded&&loaded.j.key===activeKey){
  const involvement=new Map((loaded.problem?.meta.models[chosenModel][triggerState()]||[]).map(row=>[row.id,row.count]));
  loaded.models[chosenModel].data.positions.slice().sort((a,b)=>(involvement.get(b.id)||0)-(involvement.get(a.id)||0)||b.area-a.area).forEach(p=>{
   const b=document.createElement('button');b.className='targetChoice'+(chosenIDs.has(p.id)?' chosen':'');
   b.textContent='ID '+p.id;b.setAttribute('aria-label','选择目标 ID '+p.id);b.setAttribute('aria-pressed',String(chosenIDs.has(p.id)));
   b.onclick=()=>toggleTarget(p.id);list.appendChild(b);
  });
 }
 el('selectionJson').value=JSON.stringify(exportPayload(),null,2);
}
function toggleTarget(id){if(id<=0)return;if(chosenIDs.has(id))chosenIDs.delete(id);else chosenIDs.add(id);selectionNotice('');refreshSelection();el('numbers').checked=true;drawAll();}
function chooseModel(index){if(chosenModel!==index)chosenIDs.clear();chosenModel=index;selectionNotice('');refreshSelection();el('numbers').checked=index>=0;drawAll();}
function selectFromCanvas(event,canvas,index){
 if(event.shiftKey)return;
 if(chosenModel!==index||!loaded||loaded.j.key!==activeKey)return;
 const p=nativePoint(event,canvas),[w,h]=loaded.j.wh,x=Math.floor(p.x*w),y=Math.floor(p.y*h);
 if(x<0||y<0||x>=w||y>=h)return;
 const data=loaded.models[index].idData;if(!data){selectionNotice('像素编号无法读取，请使用下面的 ID 按钮。');return;}
 const offset=(y*w+x)*4,id=event.cvprViewedMaskID??projectionLabelHit(index,p.x*w,p.y*h)??(data[offset]+256*data[offset+1]);
 if(id>0)toggleTarget(id);else selectionNotice('这里没有目标 mask，请点击物体内部。');
}
function saveSelection(moveNext){
 if(chosenModel===null){selectionNotice('请先选择一套方案，或选择“都不可靠”。');return;}
 if(chosenModel>=0&&(!loaded||loaded.j.key!==activeKey)){selectionNotice('图像尚未加载完成。');return;}
 if(chosenModel>=0&&chosenIDs.size===0){selectionNotice('请点击需要修复的完整目标；可以选多个。');return;}
 const c=D.cases[activeCase],v=c.views[activeView],j=D.crops[activeKey];
 const ids=[...chosenIDs].sort((a,b)=>a-b),borderTouchingIDs=new Set();
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
 const record={case_uid:c.uid,scene:c.scene,ROI:c.ROI,aliases:c.aliases,conflict_rank:c.order,conflict_points:c.counts.CONFLICT,
  view_number:v.number,frame:v.frame,source_rgb:v.source_RGB,crop_key:activeKey,
  crop_xyxy:j.box,native_crop_wh:j.wh,crop_strategy:j.strategy,
  choice:chosenModel<0?'unreliable':modelChoices[chosenModel],model_index:chosenModel,
  mask_ids:ids,label_asset:chosenModel<0?null:'assets/'+activeKey+'_'+['original','local','sam'][chosenModel]+'_ids.png',
  scope:borderTouchingIDs.size?'selected_visible_masks':'selected_whole_visible_objects',artificial_crop_border_check:borderTouchingIDs.size?'TOUCHES_CROP_BORDER_ALLOWED':'PASS',border_touching_mask_ids:[...borderTouchingIDs].sort((a,b)=>a-b),crop_border_selection_allowed:true,
  selected_at:new Date().toISOString(),is_qa:qaMode,target_notes:el('targetNote').value.trim()};
 savedSelections[c.uid]=record;
 let persisted=true;try{localStorage.setItem(storageKey,JSON.stringify(savedSelections));}catch(e){persisted=false;}
 refreshSelection();buildCases();
 selectionNotice(persisted?('已保存到本浏览器。'+(borderTouchingIDs.size?'ID '+[...borderTouchingIDs].join('、')+' 触及裁剪边界，已允许保存。':'')):'已记录，但浏览器存储失败，请立即导出选择。');
 if(moveNext&&persisted){const position=filtered.indexOf(activeCase),next=filtered.slice(position+1).find(i=>!savedSelections[D.cases[i].uid])??filtered.find(i=>!savedSelections[D.cases[i].uid]);if(next!==undefined)chooseCase(next);else selectionNotice('30 个区域全部完成。请导出 room2 选择，放到 CVPR 文件夹，告诉 Codex“room2 前30已标好”。');}
}
function updateSelectionFilter(){
 filtered=D.cases.flatMap((c,i)=>(el('scene').value==='all'||c.scene===el('scene').value)&&(!el('pilotOnly').checked||pilotUIDs.has(c.uid))?[i]:[]);
 chooseCase(filtered.includes(activeCase)?activeCase:filtered[0]);
}
document.querySelectorAll('[data-model-choice]').forEach(b=>b.onclick=()=>chooseModel(Number(b.dataset.modelChoice)));
el('saveSelection').onclick=()=>saveSelection(false);el('saveAndNext').onclick=()=>saveSelection(true);
el('pilotOnly').onchange=updateSelectionFilter;
el('exportSelections').onclick=()=>{
 const file=new Blob([JSON.stringify(exportPayload(),null,2)],{type:'application/json'}),url=URL.createObjectURL(file),a=document.createElement('a');
 a.href=url;a.download=(qaMode?'QA_':'')+'room2_前30_修复候选选择_20261008.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 selectionNotice('已导出 room2_前30_修复候选选择_20261008.json。请放到本地 CVPR 文件夹，告诉 Codex“room2 前30已标好”。');
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

window.threewayReview={data:D,getState:()=>({case:activeCase,view:activeView,key:activeKey,loaded:!!loaded,scale}),selectCase:chooseCase,selectView:chooseView,selectCrop:chooseCrop};
chooseCase(filtered[0]);
})();
