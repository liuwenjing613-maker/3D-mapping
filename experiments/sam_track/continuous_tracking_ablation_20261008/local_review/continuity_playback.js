const el = id => document.getElementById(id);
const canvas = el('canvas'), ctx = canvas.getContext('2d', {alpha:false});
const rawCanvas=el('rawCanvas'),rawCtx=rawCanvas.getContext('2d',{alpha:false});
const overlay = document.createElement('canvas'); overlay.width=1200; overlay.height=680;
const overlayCtx = overlay.getContext('2d');
const query=new URLSearchParams(location.search);
const SCENE=['room0','room2'].includes(query.get('scene'))?query.get('scene'):'room2';
const STORE='continuous_tracking_preview_'+SCENE;
const reasonNames={insufficient_visible_anchor:'可见目标锚点不足',anchor_coverage:'目标覆盖不足',spatial_extent:'追踪表面越界',foreign_old_family_overlap:'覆盖了其他旧家族',no_identified_old_family_observation:'未找到冻结旧家族'};
let data, current=0, currentImage, selected=new Set(), frameCache=new Map(), chunkCache=new Map();
let playing=false, playGeneration=0, drawGeneration=0, eligible=[], lockedHover=false, busyTimer;
const checks=new Map(), tags=new Map(), groups=new Map();

function showError(error){pause();el('error').hidden=false;el('error').textContent='回放读取失败：'+String(error.message||error);el('busy').hidden=true;}
function formattedFrame(fid){return 'f'+String(fid).padStart(4,'0');}
function safeFrame(value){return Math.max(0,Math.min(1999,Math.round(Number(value)||0)));}
function range(){let a=safeFrame(el('loopStart').value),b=safeFrame(el('loopEnd').value);if(a>b)[a,b]=[b,a];return [a,b];}
function currentWindow(oid){const name=el('policy').value;if(name==='unrestricted')return null;return data.continuity_policies[name].tracks.find(t=>t.track===oid);}
function allowed(oid,fid){const w=currentWindow(oid);return !w||(fid>=w.allowed_frame_range[0]&&fid<=w.allowed_frame_range[1]);}
function selectedBits(){let bits=0;for(const oid of selected)if(allowed(oid,current))bits|=1<<(oid-1);return bits>>>0;}
function activeArea(frame){return [...selected].reduce((sum,oid)=>sum+(allowed(oid,frame.frame)?frame.areas[oid-1]:0),0);}
function updateEligible(){
  const [a,b]=range();eligible=data.frames.filter(f=>f.frame>=a&&f.frame<=b&&(!el('skipEmpty').checked||el('mode').value==='rgb'||activeArea(f)>0)).map(f=>f.frame);
  el('onlyFrameList').textContent=`片段 ${formattedFrame(a)}–${formattedFrame(b)} · 符合当前筛选的帧 ${eligible.length}/${b-a+1} · 本次保存全部2000帧，没有时间抽样`;
  if(!eligible.length&&playing)pause();
  savePreferences();
}
function savePreferences(){if(!data)return;try{localStorage.setItem(STORE,JSON.stringify({annotation:data.annotation_sha256,selected:[...selected],frame:current,fps:el('fps').value,mode:el('mode').value,policy:el('policy').value,loop:el('loop').checked,skipEmpty:el('skipEmpty').checked,range:range(),opacity:el('opacity').value,outlines:el('outlines').checked}));}catch{}}
function restorePreferences(){try{const p=JSON.parse(localStorage.getItem(STORE));if(!p||p.annotation!==data.annotation_sha256)return;
  selected=new Set((p.selected||[]).filter(i=>i>=1&&i<=data.tracks.length));current=safeFrame(p.frame);
  for(const id of ['fps','mode','opacity','policy'])if(p[id]!=null)el(id).value=p[id];
  for(const id of ['loop','skipEmpty','outlines'])if(p[id]!=null)el(id).checked=!!p[id];
  if(p.range){el('loopStart').value=safeFrame(p.range[0]);el('loopEnd').value=safeFrame(p.range[1]);}
}catch{}}
function bytesToHex(bytes){return [...bytes].map(v=>v.toString(16).padStart(2,'0')).join('');}
async function getChunk(first){
  if(chunkCache.has(first)){const p=chunkCache.get(first);chunkCache.delete(first);chunkCache.set(first,p);return p;}
  const chunk=data.chunks.find(c=>c.first_frame===first);
  const task=(async()=>{const response=await fetch(chunk.path);if(!response.ok)throw Error(`帧包 ${first}：HTTP ${response.status}`);
    const bytes=await response.arrayBuffer();if(bytes.byteLength!==chunk.bytes)throw Error('帧包长度不符');
    const actual=bytesToHex(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)));if(actual!==chunk.sha256)throw Error('帧包校验未通过');return bytes;})();
  chunkCache.set(first,task);while(chunkCache.size>4)chunkCache.delete(chunkCache.keys().next().value);
  try{return await task;}catch(e){chunkCache.delete(first);throw e;}
}
async function decodeMask(bytes){
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  const pairsBuffer=await new Response(stream).arrayBuffer();if(pairsBuffer.byteLength%8)throw Error('mask编码长度错误');
  const pairs=new Uint32Array(pairsBuffer),pixels=new Uint32Array(data.width*data.height);let offset=0;
  for(let i=0;i<pairs.length;i+=2){const count=pairs[i],value=pairs[i+1];if(!count||value>>>data.tracks.length||offset+count>pixels.length)throw Error('mask记录越界');pixels.fill(value,offset,offset+count);offset+=count;}
  if(offset!==pixels.length)throw Error('mask没有覆盖完整原始画幅');return pixels;
}
async function getFrame(fid){
  if(frameCache.has(fid))return frameCache.get(fid);
  const task=(async()=>{const f=data.frames[fid],buf=await getChunk(f.chunk);const j=f.offset+f.rgb_bytes,k=j+f.mask_bytes;
    const [image,raw,used]=await Promise.all([createImageBitmap(new Blob([buf.slice(f.offset,j)],{type:'image/jpeg'})),decodeMask(buf.slice(j,k)),decodeMask(buf.slice(k,k+f.used_bytes))]);
    if(image.width!==data.width||image.height!==data.height){image.close();throw Error('RGB与mask原始尺寸不一致');}
    return {image,raw,used,meta:f};})();
  frameCache.set(fid,task);
  try{return await task;}catch(e){frameCache.delete(fid);throw e;}
}
function trimFrames(){for(const [fid,promise] of frameCache){if(frameCache.size<=8)break;if(fid===current)continue;frameCache.delete(fid);promise.then(f=>{if(f!==currentImage)f.image.close();}).catch(()=>{});}}
function prefetch(){const at=eligible.findIndex(fid=>fid>current);if(at<0)return;for(const fid of eligible.slice(at,at+3))getFrame(fid).catch(()=>{});trimFrames();}

function renderComparison(destination,pixels,image,bits){
  const mode=el('mode').value;
  if(mode==='mask'){destination.fillStyle='#111923';destination.fillRect(0,0,data.width,data.height);}
  else destination.drawImage(image,0,0);
  if(mode!=='rgb'){
    const targetColors=new Map();
    const rgba=new Uint8ClampedArray(pixels.length*4),alpha=mode==='mask'?255:Math.round(Number(el('opacity').value)*255),outlines=el('outlines').checked,w=data.width;
    for(let i=0;i<pixels.length;i++){
      const value=(pixels[i]&bits)>>>0;if(!value)continue;
      let colors=targetColors.get(value);if(!colors){colors=[];for(let oid=1;oid<=data.tracks.length;oid++)if(value&(1<<(oid-1)))colors.push(data.tracks[oid-1].color);targetColors.set(value,colors);}
      const x=i%w,y=(i/w)|0,color=colors.length===1?colors[0]:colors[((x+y)>>3)%colors.length];
      const edge=outlines&&(x===0||x===w-1||y===0||y===data.height-1||(pixels[i-1]&bits)!==value||(pixels[i+1]&bits)!==value||(pixels[i-w]&bits)!==value||(pixels[i+w]&bits)!==value);
      const j=i*4;rgba[j]=color[0];rgba[j+1]=color[1];rgba[j+2]=color[2];rgba[j+3]=edge?245:alpha;
    }
    overlayCtx.putImageData(new ImageData(rgba,data.width,data.height),0,0);destination.drawImage(overlay,0,0);
  }
}
function rawSelectionBits(){let bits=0;for(const oid of selected)bits|=1<<(oid-1);return bits>>>0;}
function paint(){
  if(!currentImage)return;const {image,meta}=currentImage,mode=el('mode').value;
  el('rawPanel').hidden=!el('compare').checked;el('comparison').classList.toggle('single',!el('compare').checked);
  renderComparison(ctx,currentImage.raw,image,selectedBits());
  if(el('compare').checked)renderComparison(rawCtx,currentImage.raw,image,rawSelectionBits());
  const present=data.tracks.filter(t=>selected.has(t.track)&&allowed(t.track,current)&&meta.areas[t.track-1]>0);
  el('frameTitle').textContent=`${formattedFrame(current)} / f1999`;
  el('visibleCount').textContent=`当前勾选 ${selected.size}/${data.tracks.length} 条 · 本帧非空 ${present.length} 条`;
  el('rawLabel').textContent='原始SAM轨迹 · 本帧'+data.tracks.filter(t=>selected.has(t.track)&&meta.areas[t.track-1]>0).length+'个非空mask';
  el('filteredLabel').textContent=el('policy').selectedOptions[0].textContent+' · 本帧'+present.length+'个非空mask';
  el('scrub').value=current;el('jumpFrame').value=current;
  const stopped=data.tracks.filter(t=>selected.has(t.track)&&!allowed(t.track,current));
  el('frameState').textContent=el('policy').value==='unrestricted'?'原始轨迹 · 无空帧截断':`截断预览 · 本帧已排除 ${stopped.length} 条`;
  el('frameState').className='badge';
  el('frameNote').textContent='空帧指原始分辨率保存mask的像素数为0。停止从各自种子向外计数，不按建图每5帧计数；不同ROI或共用身份的轨迹各自独立停止。';
  const report=data.continuity_policies[el('policy').value];
  el('policySummary').textContent=report?`${SCENE} · ${data.tracks.length}条轨迹 · 当前规则最多容许连续${report.max_empty_gap}空帧 · 到达停止条件后该方向不再恢复。`:`${SCENE} · ${data.tracks.length}条原轨迹 · 可查看消失以后重新出现的实际结果。`;
  for(const t of data.tracks){const tag=tags.get(t.track),area=meta.areas[t.track-1];
    if(!allowed(t.track,current)){const w=currentWindow(t.track);tag.textContent='断轨截除'+(area?' · 原有mask':'');tag.className='tag rejected';tag.title=`保留 f${w.allowed_frame_range[0]}–f${w.allowed_frame_range[1]}；原始本帧${area}像素。此方向已停止，不重新关联。`;}
    else if(!area){tag.textContent='本帧无mask';tag.className='tag empty';tag.title='原始SAM结果为空';}
    else if(!meta.mapping_frame){tag.textContent=`有mask · ${(area/1000).toFixed(1)}千像素`;tag.className='tag';tag.title='本帧原地图未建图';}
    else if(meta.accepted.includes(t.track)){tag.textContent='保留 · 原方案曾通过门槛';tag.className='tag used';tag.title=`原始 ${area} 像素，扣除冲突后前景 ${meta.used_areas[t.track-1]} 像素`;
    }else{const reasons=meta.rejections[String(t.track)]||[];tag.textContent='保留 · 原方案未通过门槛';tag.className='tag rejected';tag.title=reasons.map(r=>reasonNames[r]||(r.startsWith('other_seed_')?'覆盖其他选定目标':r)).join('；')||'未通过可靠性检查';}
  }
  el('hover').textContent='';lockedHover=false;savePreferences();
}
async function showFrame(fid){
  const request=++drawGeneration;fid=safeFrame(fid);clearTimeout(busyTimer);busyTimer=setTimeout(()=>{if(request===drawGeneration){el('busy').hidden=false;el('busy').textContent=`读取 ${formattedFrame(fid)}…`; }},150);
  try{const frame=await getFrame(fid);if(request!==drawGeneration)return false;current=fid;currentImage=frame;paint();prefetch();return true;}
  finally{if(request===drawGeneration){clearTimeout(busyTimer);el('busy').hidden=true;}}
}
function pause(){playing=false;playGeneration++;drawGeneration++;clearTimeout(busyTimer);el('busy').hidden=true;el('play').textContent='播放';}
async function play(){
  if(playing){pause();return;}if(!eligible.length){el('frameNote').textContent='当前筛选没有可播放帧，请勾选目标或调整片段。';return;}
  playing=true;el('play').textContent='暂停';const token=++playGeneration;
  try{
    while(playing&&token===playGeneration){const started=performance.now();let next=eligible.find(fid=>fid>current);
      if(next===undefined){if(!el('loop').checked){pause();break;}next=eligible[0];}
      if(token!==playGeneration)break;await showFrame(next);
      await new Promise(resolve=>setTimeout(resolve,Math.max(0,1000/Number(el('fps').value)-(performance.now()-started))));
    }
  }catch(e){showError(e);}
}
async function step(direction){pause();let fid;
  if(direction>0)fid=eligible.find(i=>i>current);else fid=eligible.findLast(i=>i<current);
  if(fid===undefined&&el('loop').checked)fid=direction>0?eligible[0]:eligible.at(-1);
  if(fid!==undefined)await showFrame(fid);
}
function syncCheckboxes(){for(const [oid,check] of checks)check.checked=selected.has(oid);for(const {check,ids} of groups.values()){check.checked=ids.every(i=>selected.has(i));check.indeterminate=!check.checked&&ids.some(i=>selected.has(i));}}
function filterChanged(){syncCheckboxes();updateEligible();paint();prefetch();}
function choose(ids){selected=new Set(ids);filterChanged();}
function makeLegend(){
  const byROI=new Map();for(const track of data.tracks){if(!byROI.has(track.ROI))byROI.set(track.ROI,[]);byROI.get(track.ROI).push(track);}
  for(const [ROI,tracks] of byROI){
    const section=document.createElement('div');section.className='group';const heading=document.createElement('div');heading.className='group-heading';
    const check=document.createElement('input');check.type='checkbox';check.setAttribute('aria-label',ROI+'整组');
    const title=document.createElement('span');title.textContent=ROI;const seed=document.createElement('button');seed.textContent='种子帧';seed.title='跳到原始人工种子帧';seed.onclick=()=>{pause();showFrame(tracks[0].seed_frame).catch(showError);};
    const ids=tracks.map(t=>t.track);check.onchange=()=>{for(const oid of ids){if(check.checked)selected.add(oid);else selected.delete(oid);}filterChanged();};groups.set(ROI,{check,ids});heading.append(check,title,seed);section.append(heading);
    for(const t of tracks){
      const row=document.createElement('div');row.className='target';const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.setAttribute('aria-label',`${t.ROI} mask${t.mask} ID${t.persistent_id}`);
      checkbox.onchange=()=>{if(checkbox.checked)selected.add(t.track);else selected.delete(t.track);filterChanged();};checks.set(t.track,checkbox);
      const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=`rgb(${t.color.join(',')})`;
      const text=document.createElement('div');text.className='name';text.textContent=`mask${t.mask} → ID${t.persistent_id}`;text.title=`轨迹${t.track} · 种子 ${formattedFrame(t.seed_frame)}${t.canonical!==t.track?' · 与轨迹'+t.canonical+'共用身份':''}`;
      const tag=document.createElement('span');tag.className='tag';tags.set(t.track,tag);text.append(tag);
      const solo=document.createElement('button');solo.textContent='单看';solo.setAttribute('aria-label',`单看 ${t.ROI} mask${t.mask}`);solo.onclick=()=>choose([t.track]);row.append(checkbox,swatch,text,solo);section.append(row);
    }
    el('legend').append(section);
  }
  syncCheckboxes();
}
function pixelInfo(event){
  if(!currentImage||lockedHover)return;const box=canvas.getBoundingClientRect(),x=Math.min(data.width-1,Math.max(0,Math.floor((event.clientX-box.left)/box.width*data.width))),y=Math.min(data.height-1,Math.max(0,Math.floor((event.clientY-box.top)/box.height*data.height)));
  const bits=(el('mode').value==='used'?currentImage.used:currentImage.raw)[y*data.width+x]&selectedBits();const matches=data.tracks.filter(t=>bits&(1<<(t.track-1)));
  el('hover').textContent=matches.length?matches.map(t=>`${t.ROI} mask${t.mask} → ID${t.persistent_id}`).join('；'):'该像素没有勾选目标的mask';
}
function wire(){
  el('scene').value=SCENE;el('scene').onchange=()=>{pause();location.href='playback.html?scene='+el('scene').value+'&policy='+el('policy').value;};el('policy').onchange=()=>{pause();updateEligible();paint();prefetch();};
  el('play').onclick=play;el('previous').onclick=()=>step(-1).catch(showError);el('next').onclick=()=>step(1).catch(showError);
  el('jump').onclick=()=>{pause();showFrame(el('jumpFrame').value).catch(showError);};el('jumpFrame').onkeydown=e=>{if(e.key==='Enter')el('jump').click();};
  el('scrub').oninput=()=>{pause();showFrame(el('scrub').value).catch(showError);};
  el('all').onclick=()=>choose(data.tracks.map(t=>t.track));el('none').onclick=()=>choose([]);el('blinds').onclick=()=>choose(data.tracks.filter(t=>(SCENE==='room2'?['ROI-C0001','ROI-C0008']:['ROI-C0011']).includes(t.ROI)).map(t=>t.track));
  el('floor').onclick=()=>choose(data.tracks.filter(t=>(SCENE==='room2'?[7,8,13,14,15]:[]).includes(t.track)).map(t=>t.track));
  el('compare').onchange=paint;el('enlarge').onclick=()=>{const pair=el('comparison');pair.classList.toggle('large');el('enlarge').textContent=pair.classList.contains('large')?'并排适应窗口':'上下放大画面';};
  for(const id of ['mode','skipEmpty'])el(id).onchange=()=>{updateEligible();paint();prefetch();};
  for(const id of ['loopStart','loopEnd']){const change=()=>{updateEligible();prefetch();};el(id).oninput=change;el(id).onchange=change;}
  for(const id of ['opacity','outlines'])el(id).oninput=()=>paint();for(const id of ['fps','loop'])el(id).onchange=savePreferences;
  el('setStart').onclick=()=>{el('loopStart').value=current;updateEligible();};el('setEnd').onclick=()=>{el('loopEnd').value=current;updateEligible();};el('resetRange').onclick=()=>{el('loopStart').value=0;el('loopEnd').value=1999;updateEligible();};
  el('saveFrame').onclick=()=>canvas.toBlob(blob=>{if(!blob)return;const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`${SCENE}_${formattedFrame(current)}_${el('policy').value}.png`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
  canvas.onmousemove=pixelInfo;canvas.onclick=e=>{if(lockedHover){lockedHover=false;pixelInfo(e);}else{pixelInfo(e);lockedHover=true;}};canvas.onmouseleave=()=>{if(!lockedHover)el('hover').textContent='';};
  document.addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA/.test(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();play();}else if(e.key==='ArrowRight'){e.preventDefault();step(1).catch(showError);}else if(e.key==='ArrowLeft'){e.preventDefault();step(-1).catch(showError);}});
}
async function main(){
  if(typeof DecompressionStream==='undefined')throw Error('当前浏览器不支持mask解码，请使用当前Codex内置浏览器或新版Chrome/Edge');
  const response=await fetch(SCENE+'_preview_data.json');if(!response.ok)throw Error('回放数据尚未准备完成');data=await response.json();
  if(data.status!=='PASS'||data.frames_count!==2000||data.tracks.length!==(SCENE==='room0'?30:25)||!data.native_masks_lossless)throw Error('回放数据完整性未通过');
  try{const response=await fetch('experiment_status.json');if(response.ok){const state=await response.json();if(state.status==='RUNNING')el('experimentNote').textContent='正在重建投票与地图，完整v3待完成。当前画面是原轨迹与截断mask同帧对照；被截除结果不能自动判为错误。';else if(state.status==='PASS')el('experimentNote').textContent='完整换票、扩散和统一v3对照已完成，地图及指标见总览页。当前只比较同一帧的mask，左边原轨迹、右边当前策略。';}}catch{}
  selected=new Set(data.tracks.map(t=>t.track));restorePreferences();if(['gap0','gap1','unrestricted'].includes(query.get('policy')))el('policy').value=query.get('policy');if(query.has('frame'))current=safeFrame(query.get('frame'));if(query.has('track')&&data.tracks.some(t=>t.track===Number(query.get('track'))))selected=new Set([Number(query.get('track'))]);el('pageTitle').textContent=SCENE+' · 连续空帧截断对照';el('floor').hidden=SCENE==='room0';el('sceneNote').textContent='只使用本轮实际保存的原始轨迹；没有增加目标、重新分配像素或更改原结果。';makeLegend();wire();updateEligible();await showFrame(current);
  for(const id of ['play','previous','next'])el(id).disabled=false;
  el('verification').textContent='复用已经校验的2000张原RGB和原始分辨率mask。只按冻结的连续性区间筛除轨迹位；保留像素与原结果完全相同。此页不代表新的投票、地图或v3结果。';
}
main().catch(showError);
