import * as THREE from './vendor/three.module.js';
import {OrbitControls} from './vendor/OrbitControls.js';
import {parsePLY} from './ply.js';
const $=s=>document.querySelector(s),fmt=n=>Number(n).toLocaleString('zh-CN');
const views=[],dataCache=new Map();
let manifest,linking=false,scheduled=false,initialized=false;
const cfg={mode:1,scope:0,pointSize:1.75,clip:false,clipZ:.9,trajectory:false};
const fail=e=>{const box=$('#error');box.hidden=false;box.textContent='展示未完成：'+(e.message||e)+'。请保留页面重试；可从面板链接读取原始实例 PLY。';console.error(e);};
window.addEventListener('unhandledrejection',e=>fail(e.reason));
const vertexShader='attribute vec3 instanceColor;attribute float instanceID;attribute float stateID;uniform float pointSize;uniform float pixelRatio;uniform int mode;uniform int scope;uniform float chosenID;uniform bool clipEnabled;uniform float clipZ;varying vec3 vColor;varying float visiblePoint;void main(){bool assigned=instanceID>0.0;bool show=(scope==0)||(scope==1&&assigned)||(scope==2&&!assigned)||(scope==3&&!assigned&&stateID<0.5)||(scope==4&&!assigned&&stateID>0.5&&stateID<1.5)||(scope==5&&!assigned&&stateID>2.5);show=show&&(chosenID<0.0||abs(instanceID-chosenID)<0.5)&&(!clipEnabled||position.z<=clipZ);visiblePoint=show?1.0:0.0;vec3 stateColor=assigned?vec3(.64,.69,.70):(stateID<0.5?vec3(0.0):(stateID<1.5?vec3(1.0,1.0,0.0):vec3(1.0,0.0,0.0)));vColor=mode==1?instanceColor:(mode==2?stateColor:vec3(.36,.49,.53));gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=pointSize*pixelRatio;}';
const fragmentShader='precision highp float;varying vec3 vColor;varying float visiblePoint;void main(){if(visiblePoint<0.5||distance(gl_PointCoord,vec2(.5))>.5)discard;gl_FragColor=vec4(vColor,1.0);}';
function schedule(){if(scheduled)return;scheduled=true;requestAnimationFrame(()=>{scheduled=false;for(const v of views)v.renderer.render(v.scene,v.camera);});}
function cameraChange(source){
 if(!linking&&$('#sync').checked){linking=true;for(const v of views){if(v===source)continue;v.camera.position.copy(source.camera.position);v.camera.up.copy(source.camera.up);v.controls.target.copy(source.controls.target);v.controls.update();}linking=false;}
 schedule();
}
function setPose(v,pose){v.camera.up.fromArray(pose.up||[0,0,1]);v.camera.position.fromArray(pose.eye);v.controls.target.copy(v.camera.position).add(new THREE.Vector3().fromArray(pose.forward).multiplyScalar(3));v.controls.update();}
function turn(v,yaw,pitch=0){const off=v.camera.position.clone().sub(v.controls.target),r=off.length(),a=Math.atan2(off.y,off.x)+yaw,p=Math.max(-1.52,Math.min(1.52,Math.asin(off.z/r)+pitch));v.camera.up.set(0,0,1);v.camera.position.set(r*Math.cos(p)*Math.cos(a),r*Math.cos(p)*Math.sin(a),r*Math.sin(p)).add(v.controls.target);v.controls.update();}
function preset(name){
 if(!manifest)return;linking=true;const all=manifest.maps.find(x=>x.key==='300_'+document.querySelector('input[name=branch]:checked').value+'_before');
 for(const v of views){
  if(['first','200','300'].includes(name))setPose(v,all.cameras[name==='first'?0:name==='200'?199:299]);
  else if(name==='left'||name==='right')turn(v,(name==='left'?1:-1)*Math.PI/6);
  else{const box=new THREE.Box3(new THREE.Vector3().fromArray(all.bbox[0]),new THREE.Vector3().fromArray(all.bbox[1])),center=box.getCenter(new THREE.Vector3());v.controls.target.copy(center);v.camera.up.set(0,0,1);v.camera.position.copy(center).add(name==='top'?new THREE.Vector3(0,-.005,8.4):new THREE.Vector3(6.3,-6.6,4.6));v.controls.update();}
 }
 linking=false;schedule();
}
function matches(data,i,id){
 if(id>=0&&data.ids[i]!==id)return false;if(cfg.clip&&data.positions[i*3+2]>cfg.clipZ)return false;
 const a=data.ids[i]>0,s=data.states[i],f=cfg.scope;return f===0||f===1&&a||f===2&&!a||f===3&&!a&&s===0||f===4&&!a&&s===1||f===5&&!a&&s===3;
}
function updateSettings(v){
 const u=v.material?.uniforms;if(!u)return;u.mode.value=cfg.mode;u.scope.value=cfg.scope;u.pointSize.value=cfg.pointSize;u.clipEnabled.value=cfg.clip;u.clipZ.value=cfg.clipZ;u.chosenID.value=Number(v.select.value);if(v.path)v.path.visible=cfg.trajectory;
 let shown=0;for(let i=0;i<v.data.count;i++)if(matches(v.data,i,u.chosenID.value))shown++;
 v.panel.querySelector('.view-count').textContent='完整 '+fmt(v.data.count)+' · 显示 '+fmt(shown)+' 点';schedule();
}
function updateAll(){for(const v of views)updateSettings(v);}
async function fetchPLY(row,onProgress){
 const response=await fetch(row.ply,{cache:'no-store'});if(!response.ok)throw Error(row.title+' 的 PLY 读取失败');
 const bytes=new Uint8Array(row.bytes),reader=response.body.getReader();let offset=0;
 while(true){const {done,value}=await reader.read();if(done)break;if(offset+value.byteLength>row.bytes)throw Error('PLY 大于清单字节数');bytes.set(value,offset);offset+=value.byteLength;onProgress(100*offset/row.bytes);}
 if(offset!==row.bytes)throw Error('PLY 字节数与清单不一致');
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(x=>x.toString(16).padStart(2,'0')).join('');
 if(digest!==row.sha256)throw Error('PLY SHA256 核验失败');
 const data=parsePLY(bytes.buffer);if(data.count!==row.points)throw Error('PLY 点数与清单不一致');return data;
}
function setRow(v,row){
 v.row=row;v.panel.dataset.map=row.key;v.panel.dataset.loaded='false';v.panel.querySelector('h2').textContent=row.title;v.panel.querySelector('.tag').textContent=row.tag;
 v.panel.querySelector('.stats').innerHTML='<span><strong>'+fmt(row.points)+'</strong> 点</span><span><strong>'+row.instances+'</strong> 实例</span><span>覆盖率 <strong>'+(100*row.assigned/row.points).toFixed(2)+'%</strong></span><span>U '+fmt(row.states.U)+' / T '+fmt(row.states.T)+' / C '+fmt(row.states.C)+'</span>';
 const selected=v.select.value||'-1';v.select.innerHTML='<option value="-1">全部</option>'+Object.entries(row.instance_counts).map(([id,n])=>'<option value="'+id+'">ID '+id+' · '+fmt(n)+' 点</option>').join('');
 if(selected!=='-1'&&!row.instance_counts[selected])v.select.insertAdjacentHTML('beforeend','<option value="'+selected+'">ID '+selected+' · 此图不存在</option>');v.select.value=selected;
 v.panel.querySelector('.download-row').innerHTML=row.downloads.map(x=>'<a href="'+x.file+'" download>'+x.title+'</a>').join('')+'<span>raw 0–'+row.last_frame+' · 原始实例 ID</span>';
 if(v.path){v.scene.remove(v.path);v.path.geometry.dispose();v.path.material.dispose();}
 const pg=new THREE.BufferGeometry().setFromPoints(row.cameras.map(p=>new THREE.Vector3().fromArray(p.eye)));v.path=new THREE.Line(pg,new THREE.LineBasicMaterial({color:0x007e82}));v.path.visible=cfg.trajectory;v.scene.add(v.path);
 v.renderer.domElement.setAttribute('aria-label',row.title+' 实例 PLY，拖动旋转、滚轮缩放');
}
function makePanel(row){
 const panel=document.createElement('section');panel.className='map-panel';
 panel.innerHTML='<div class="panel-head"><h2></h2><span class="tag"></span></div><div class="stats"></div><div class="viewport"><div class="load-state"><div class="spinner"></div><span>读取实际实例 PLY…</span></div><span class="orientation">Z ↑ · 米 · 全量实例点云</span></div><div class="panel-controls"><label>实例 <select class="instance-select"><option value="-1">全部</option></select></label><span class="view-count">等待 PLY 载入</span></div><div class="download-row"></div>';
 $('#maps').append(panel);
 const viewport=panel.querySelector('.viewport'),renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0xe9eef0,1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.NoToneMapping;renderer.domElement.tabIndex=0;viewport.prepend(renderer.domElement);
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(60,1,.015,100);camera.up.set(0,0,1);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;controls.minDistance=.06;controls.maxDistance=60;controls.zoomToCursor=true;
 const v={row,panel,viewport,renderer,scene,camera,controls,select:panel.querySelector('.instance-select'),loadVersion:0};views.push(v);setRow(v,row);setPose(v,manifest.maps[0].cameras[0]);controls.addEventListener('change',()=>cameraChange(v));
 const resize=()=>{const w=viewport.clientWidth,h=viewport.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();schedule();};new ResizeObserver(resize).observe(viewport);resize();
 v.select.onchange=()=>updateSettings(v);
 renderer.domElement.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();turn(v,e.key==='ArrowLeft'?.12:e.key==='ArrowRight'?-.12:0,e.key==='ArrowUp'?.08:e.key==='ArrowDown'?-.08:0);});
 renderer.domElement.addEventListener('dblclick',e=>{if(!v.cloud||v.panel.dataset.loaded!=='true')return;const rect=renderer.domElement.getBoundingClientRect(),mouse=new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),ray=new THREE.Raycaster();ray.params.Points.threshold=.035;ray.setFromCamera(mouse,camera);const hit=ray.intersectObject(v.cloud).find(h=>matches(v.data,h.index,Number(v.select.value)));if(hit){controls.target.fromArray(v.data.positions,hit.index*3);controls.update();}});
 return v;
}
async function loadMap(v,row=v.row){
 const version=++v.loadVersion;if(v.cloud){v.scene.remove(v.cloud);v.cloud.geometry.dispose();v.material.dispose();v.cloud=null;v.material=null;}setRow(v,row);
 const loading=v.panel.querySelector('.load-state'),text=loading.querySelector('span');loading.hidden=false;loading.querySelector('.spinner').hidden=false;text.textContent='载入 '+row.title+'…';schedule();
 try{
  let task=dataCache.get(row.key);
  if(!task){task=fetchPLY(row,p=>{if(version===v.loadVersion)text.textContent='读取服务器实例 PLY · '+p.toFixed(0)+'%';});dataCache.set(row.key,task);task.catch(()=>dataCache.delete(row.key));}
  const data=await task;if(version!==v.loadVersion)return;v.data=data;text.textContent='点数与 SHA256 已核验 · 建立 3D…';
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(data.positions,3));geometry.setAttribute('instanceColor',new THREE.Uint8BufferAttribute(data.rgb,3,true));geometry.setAttribute('instanceID',new THREE.BufferAttribute(data.ids,1));geometry.setAttribute('stateID',new THREE.BufferAttribute(data.states,1));geometry.computeBoundingSphere();
  v.material=new THREE.ShaderMaterial({vertexShader,fragmentShader,uniforms:{pointSize:{value:cfg.pointSize},pixelRatio:{value:v.renderer.getPixelRatio()},mode:{value:cfg.mode},scope:{value:cfg.scope},chosenID:{value:-1},clipEnabled:{value:cfg.clip},clipZ:{value:cfg.clipZ}},depthTest:true,depthWrite:true,toneMapped:false});v.cloud=new THREE.Points(geometry,v.material);v.cloud.frustumCulled=false;v.scene.add(v.cloud);updateSettings(v);loading.hidden=true;v.panel.dataset.loaded='true';schedule();
 }catch(e){if(version!==v.loadVersion)return;text.textContent='载入失败：'+e.message;loading.querySelector('.spinner').hidden=true;fail(e);}
}
function chooseBranch(branch){if(!manifest||views.length!==4)return;document.title='room0 · 200/300 实例四图 · '+branch+' 分支';const keys=['200_'+branch+'_before','200_'+branch+'_after','300_'+branch+'_before','300_'+branch+'_after'];Promise.all(views.map((v,i)=>loadMap(v,manifest.maps.find(r=>r.key===keys[i]))));}
function sharedID(value){for(const v of views){if(value!=='-1'&&!Array.from(v.select.options).some(o=>o.value===value))v.select.insertAdjacentHTML('beforeend','<option value="'+value+'">ID '+value+' · 此图不存在</option>');v.select.value=value;updateSettings(v);}}
$('#mode').onchange=e=>{cfg.mode=Number(e.target.value);updateAll();};$('#scope').onchange=e=>{cfg.scope=Number(e.target.value);updateAll();};$('#shared-id').onchange=e=>sharedID(e.target.value);
$('#point-size').oninput=e=>{cfg.pointSize=Number(e.target.value);$('#size-value').value=cfg.pointSize;updateAll();};
$('#clip').onchange=e=>{cfg.clip=e.target.checked;$('#clip-z').disabled=!cfg.clip;updateAll();};$('#clip-z').oninput=e=>{cfg.clipZ=Number(e.target.value);$('#clip-value').value=cfg.clipZ.toFixed(2)+' m';updateAll();};
$('#trajectory').onchange=e=>{cfg.trajectory=e.target.checked;updateAll();};$('#sync').onchange=e=>{if(e.target.checked&&views[0])cameraChange(views[0]);};
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>preset(b.dataset.view));
document.querySelectorAll('input[name=branch]').forEach(r=>r.onchange=()=>chooseBranch(r.value));
$('#fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('#comparison').requestFullscreen();}catch(e){fail(e);}};
document.addEventListener('fullscreenchange',()=>{$('#fullscreen').textContent=document.fullscreenElement?'退出全屏':'四图全屏';schedule();});
$('#reset').onclick=()=>{cfg.mode=1;cfg.scope=0;cfg.pointSize=1.75;cfg.clip=false;cfg.clipZ=.9;cfg.trajectory=false;$('#mode').value='1';$('#scope').value='0';$('#point-size').value='1.75';$('#size-value').value='1.75';$('#clip').checked=false;$('#clip-z').disabled=true;$('#clip-z').value='.9';$('#clip-value').value='0.90 m';$('#trajectory').checked=false;$('#shared-id').value='-1';sharedID('-1');updateAll();preset('first');};
async function showTiming(){try{const r=await fetch('timing.json',{cache:'no-store'});if(!r.ok){$('#timing').textContent='修复仍在执行；完成后列出实际阶段耗时。';return;}const t=await r.json();$('#timing').innerHTML='<table><thead><tr><th>阶段</th><th>耗时（秒）</th><th>执行范围 / 计时口径</th></tr></thead><tbody>'+t.rows.map(x=>'<tr><td>'+x.stage+'</td><td class="num">'+(typeof x.seconds==='number'?x.seconds.toFixed(3):x.seconds)+'</td><td>'+x.note+'</td></tr>').join('')+'</tbody></table><p>'+t.note+'</p>';if(t.attempt_rows?.length)$('#timing').insertAdjacentHTML('beforeend','<p><b>失败与重试过程 · 单独记录</b> · <a href="attempts_timings.json" target="_blank">完整记录</a></p><table><thead><tr><th>尝试</th><th>A/B</th><th>耗时（秒）</th><th>结果</th></tr></thead><tbody>'+t.attempt_rows.map(a=>'<tr><td>'+a.title+' · '+a.attempt+'</td><td>'+a.group+'</td><td class="num">'+a.seconds.toFixed(3)+'</td><td>'+esc(a.note||a.status)+'</td></tr>').join('')+'</tbody></table>');}catch(e){$('#timing').textContent='计时记录尚未发布。';}}
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function showProgress(){
 try{const r=await fetch('progress.json',{cache:'no-store'});if(!r.ok)return;const p=await r.json();$('#progress').innerHTML=p.runs.map(run=>'<article class="progress-card"><h2>'+run.checkpoint+' 帧 · '+run.group+' 分支</h2>'+run.stages.map(s=>'<div class="stage-row"><span>'+esc(s.name)+'</span><b class="status-'+esc(s.status)+'">'+({'PASS':'已完成','RUNNING':'执行中','PENDING':'等待','FAILED':'失败，待处理'}[s.status]||esc(s.status))+'</b></div>').join('')+'<p class="progress-detail">'+esc(run.message||'等待前序检查点完成。')+'</p></article>').join('');}catch(e){$('#progress').textContent='进度记录暂未读取成功；当前没有发布完整四图。';}
}

async function boot(){
 await showProgress();
 try{
  const r=await fetch('manifest.json',{cache:'no-store'});if(!r.ok)throw Error('读取四图结果清单失败');const m=await r.json();
  if(m.status!=='PASS'){let msg=m.message||'等待两组实际修复与状态提交核验。';try{const q=await fetch('status.json',{cache:'no-store'});if(q.ok){const s=await q.json();if(s.message)msg=s.message;}}catch(e){}$('#waiting').textContent=msg+' 未完成步骤保持等待；八份实际实例 PLY 全部核验后自动载入。';await showTiming();setTimeout(boot,12000);return;}
  if(initialized)return;initialized=true;manifest=m;$('#waiting').hidden=true;
  const ids=[...new Set(m.maps.flatMap(r=>Object.keys(r.instance_counts).map(Number)))].sort((a,b)=>a-b);$('#shared-id').insertAdjacentHTML('beforeend',ids.map(id=>'<option value="'+id+'">ID '+id+'</option>').join(''));
  const p=m.palette_metrics||{};$('#palette-info').textContent='联合色盘 '+p.instance_count+' 个 ID · 相邻最小 ΔE76 '+Number(p.minimum_neighbor_deltaE76).toFixed(1);
  $('#verification').textContent='已核验：'+Object.entries(m.source_counts).map(([k,n])=>k.replace('_',' 帧 · ')+': '+fmt(n)+' 点').join('；')+'。同一分支、同一检查点修复前后坐标逐点相同，八份 PLY 均无降采样，原地图哈希未变。';
  const branch=document.querySelector('input[name=branch]:checked').value,keys=['200_'+branch+'_before','200_'+branch+'_after','300_'+branch+'_before','300_'+branch+'_after'];for(const key of keys)makePanel(m.maps.find(r=>r.key===key));await Promise.all(views.map(v=>loadMap(v)));await showTiming();
 }catch(e){fail(e);if(!initialized)setTimeout(boot,15000);}
}
boot();
