import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { hiddenInstanceIDs } from './mask_visibility.js?v=20261007';

const metadata=JSON.parse(document.getElementById('plyData').textContent);
const byUid=new Map(metadata.cases.map(c=>[c.case_uid,c]));
const el=id=>document.getElementById(id),panel=el('plyFloat'),stage=el('plyStage'),canvas=el('plyCanvas');
let renderer,scene,camera,controls,cloud,material,wire,geometry,model=null,uid=null,scope='local',condition=2;
let desired=null,loadToken=0,loadedPath=null,lastFocus=null,frameRequest=null,failed=false;
let paletteTexture=null,paletteUID=null;
const palettePromise=fetch('instance_palettes.json?v=20261007').then(r=>{if(!r.ok)throw Error('原 PLY 配色加载失败');return r.json();});
const states=['baseline','seed_only','seed_plus_short_track'];
const stateNames=['P1-A1 原版（holes_geodesic）','仅种子修复','短程追踪修复'];
const vertex=`
precision highp float;
attribute vec3 aLabels; attribute float aSeedOwner;
uniform float uCondition; uniform float uMode; uniform float uOnly; uniform float uShowUnassigned;
uniform sampler2D uPalette; uniform float uPaletteWidth;
uniform float uCount; uniform float uTargets[16]; uniform float uHiddenCount; uniform float uHiddenIDs[16]; uniform vec3 uMaskColors[16]; uniform float uPointSize;
varying vec3 vColor; varying float vVisible;
vec3 hsv(float h){vec3 p=abs(fract(vec3(h)+vec3(0.,2./3.,1./3.))*6.-3.);return .98*mix(vec3(1.),clamp(p-1.,0.,1.),.7);}
void main(){
 float id=uCondition<.5?aLabels.x:uCondition<1.5?aLabels.y:aLabels.z;
 bool selected=false; vec3 selection=vec3(1.);
 for(int i=0;i<16;i++){if(float(i)<uCount&&abs(id-uTargets[i])<.1){selected=true;selection=uMaskColors[i];}}
 vVisible=(uOnly>.5&&!selected)?0.:1.;for(int i=0;i<16;i++){if(float(i)<uHiddenCount&&abs(id-uHiddenIDs[i])<.1)vVisible=0.;}
 if(uShowUnassigned<.5&&id<=0.)vVisible=0.;
 if(uMode<.5)vColor=color;
 else if(uMode>1.5&&uMode<2.5){vColor=color*.35; if(abs(id-aLabels.x)>.1){vColor=(aLabels.x<=0.&&id>0.)?vec3(.14,.92,.64):(aLabels.x>0.&&id<=0.)?vec3(1.,.27,.33):vec3(1.,.68,.16);}}
 else{vColor=id>0.?texture2D(uPalette,vec2((id+.5)/uPaletteWidth,.5)).rgb:vec3(89./255.);if(uMode>2.5&&selected)vColor=selection;}
 vec4 p=modelViewMatrix*vec4(position,1.);gl_Position=projectionMatrix*p;gl_PointSize=uPointSize;
}`;
const fragment=`precision highp float; varying vec3 vColor; varying float vVisible;void main(){if(vVisible<.5)discard;if(length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(vColor,1.);}`;

function draw(){if(failed||!renderer||panel.hidden||panel.classList.contains('minimized'))return;renderer.render(scene,camera);}
function schedule(){if(frameRequest)return;frameRequest=requestAnimationFrame(()=>{frameRequest=null;draw();});}
function resize(){if(!renderer)return;const w=stage.clientWidth,h=stage.clientHeight;if(w<1||h<1)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();schedule();}
function initialize(){
 try{
 renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false,preserveDrawingBuffer:true});
 renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.5));renderer.setClearColor(0x172338);
 scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(58,1,.02,100);camera.up.set(0,0,1);
 controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.screenSpacePanning=true;controls.minDistance=.08;controls.maxDistance=40;controls.addEventListener('change',schedule);
 material=new THREE.ShaderMaterial({vertexShader:vertex,fragmentShader:fragment,vertexColors:true,uniforms:{
   uCondition:{value:2},uMode:{value:1},uOnly:{value:0},uShowUnassigned:{value:1},uCount:{value:0},uTargets:{value:new Float32Array(16)},uHiddenCount:{value:0},uHiddenIDs:{value:new Float32Array(16)},
   uPalette:{value:null},uPaletteWidth:{value:1},
   uMaskColors:{value:Array.from({length:16},()=>new THREE.Vector3())},uPointSize:{value:2}}});
 new ResizeObserver(resize).observe(stage);resize();
 }catch(error){failed=true;el('plyBusy').textContent='3D 初始化失败：'+error.message;el('plyStatus').textContent='完整和局部 PLY 可通过下方链接下载。';}
}

function parsePLY(buffer,expected){
 const bytes=new Uint8Array(buffer),header=new TextDecoder().decode(bytes.subarray(0,4096));
 const end=header.indexOf('end_header\n');if(end<0)throw Error('PLY 头不完整');
 const offset=end+'end_header\n'.length;const match=header.slice(0,end).match(/element vertex (\d+)/);
 if(!match||!header.includes('format binary_little_endian 1.0'))throw Error('PLY 格式不匹配');
 const n=+match[1],stride=metadata.record_stride_bytes;
 if(n!==expected||stride!==31||offset+n*stride!==buffer.byteLength)throw Error('PLY 点数或数据长度校验失败');
 const positions=new Float32Array(n*3),colors=new Float32Array(n*3),labels=new Float32Array(n*3),owners=new Float32Array(n);
 const summaries=Array.from({length:3},()=>({assigned:0,unassigned:0,changed:0}));
 const data=new DataView(buffer);
 for(let i=0,j=offset;i<n;i++,j+=stride){
   for(let k=0;k<3;k++){positions[i*3+k]=data.getFloat32(j+k*4,true);colors[i*3+k]=data.getUint8(j+12+k)/255;labels[i*3+k]=data.getInt32(j+15+k*4,true);}
   owners[i]=data.getInt32(j+27,true);
   for(let k=0;k<3;k++){const id=labels[i*3+k];summaries[k][id>0?'assigned':'unassigned']++;if(id!==labels[i*3])summaries[k].changed++;}
 }
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(positions,3));g.setAttribute('color',new THREE.BufferAttribute(colors,3));g.setAttribute('aLabels',new THREE.BufferAttribute(labels,3));g.setAttribute('aSeedOwner',new THREE.BufferAttribute(owners,1));g.computeBoundingBox();g.computeBoundingSphere();
 g.userData.labelSummaries=summaries;return g;
}

function selection(){
 const c=desired?.c;if(!c)return[];
 if(!el('plyFollow').checked)return[];
 const visible=new Set(desired.visible_track_ids||[]);
 return c.objects.filter(o=>visible.has(o.track_id)&&(desired.solo===null||o.track_id===desired.solo));
}
function updateUniforms(){
 if(!material||!desired)return;
 const selected=selection(),targets=material.uniforms.uTargets.value,colors=material.uniforms.uMaskColors.value;
 targets.fill(-9999);selected.slice(0,16).forEach((o,i)=>{targets[i]=o.audit.persistent_id;const rgb=desired.c.object_colors[o.track_id];colors[i].set(rgb[0]/255,rgb[1]/255,rgb[2]/255);});
 material.uniforms.uCount.value=Math.min(selected.length,16);material.uniforms.uCondition.value=condition;material.uniforms.uMode.value=+el('plyColor').value;
 material.uniforms.uShowUnassigned.value=el('plyUnassigned').checked?1:0;
 const hiddenIDs=hiddenInstanceIDs(desired,el('plyFollow').checked);material.uniforms.uHiddenIDs.value.fill(-9999);hiddenIDs.slice(0,16).forEach((id,i)=>material.uniforms.uHiddenIDs.value[i]=id);material.uniforms.uHiddenCount.value=Math.min(hiddenIDs.length,16);material.uniforms.uOnly.value=el('plyOnly').checked&&el('plyFollow').checked?1:0;material.uniforms.uPointSize.value=+el('plySize').value;
 updateWire();status();schedule();
}
function objectMetadata(){return model?.objects.find(o=>o.track_id===desired?.solo);}
function selectedBounds(){const o=objectMetadata();return o?.seed_surface_bounds||model?.local_bounds;}
function updateWire(){
 if(!scene||!model)return;
 if(wire){scene.remove(wire);wire.geometry.dispose();wire.material.dispose();wire=null;}
 const b=selectedBounds();if(!el('plySeedBox').checked||!b)return;
 const lo=new THREE.Vector3(...b[0]),hi=new THREE.Vector3(...b[1]),size=hi.clone().sub(lo),center=lo.clone().add(hi).multiplyScalar(.5);
 const box=new THREE.BoxGeometry(Math.max(size.x,.02),Math.max(size.y,.02),Math.max(size.z,.02));
 wire=new THREE.LineSegments(new THREE.EdgesGeometry(box),new THREE.LineBasicMaterial({color:0xffcf55,depthTest:false,transparent:true,opacity:.95}));
 box.dispose();wire.position.copy(center);wire.renderOrder=10;scene.add(wire);
}
function focus(full=false){
 if(!model||!camera)return;
 const b=full?model.native_bounds:selectedBounds();if(!b)return;
 const lo=new THREE.Vector3(...b[0]),hi=new THREE.Vector3(...b[1]),center=lo.clone().add(hi).multiplyScalar(.5),radius=Math.max(.25,hi.distanceTo(lo)*.5);
 let offset;
 if(model.seed_camera_to_world&&!full){const p=model.seed_camera_to_world;offset=new THREE.Vector3(p[0][3],p[1][3],p[2][3]).sub(center);if(offset.length()<.15)offset.set(1,-1,.6);offset.normalize().multiplyScalar(Math.max(.8,radius*2.4));}
 else{offset=new THREE.Vector3(1,-1,.7).normalize().multiplyScalar(radius*2.5);}
 camera.position.copy(center.clone().add(offset));controls.target.copy(center);camera.near=.02;camera.far=Math.max(100,radius*12);camera.updateProjectionMatrix();controls.update();schedule();
}
function status(){
 if(!model)return;
 const chosen=objectMetadata(),s=states[condition],points=geometry?.getAttribute('position')?.count;
 const idText=chosen?`mask ${chosen.native_mask_id} ↔ 3D ${chosen.persistent_id}；此 ID 全场景 ${chosen.full_scene_identity_points[s].toLocaleString()} 点`:'显示当前案例的实例点云';
 const counts=geometry?.userData.labelSummaries?.[condition],base=geometry?.userData.labelSummaries?.[0],delta=counts&&base?counts.unassigned-base.unassigned:0;
 el('plyVersion').textContent=`当前版本：${stateNames[condition]} · 原地图：20261002 / P1-A1 / holes_geodesic`;
 el('plyStatus').textContent=`${scope==='local'?'局部':'完整场景'} ${points?points.toLocaleString():'—'} 点 · ${idText}。${counts?`本范围已分配 ${counts.assigned.toLocaleString()} 点；未分配（灰色）${counts.unassigned.toLocaleString()} 点${condition?`，较原版 ${delta>=0?'+':''}${delta.toLocaleString()}`:''}。`:''}${model.status!=='READY'?'本例跳过，三个版本相同。':condition?`本范围改变 ${counts?.changed.toLocaleString()||'—'} 个标签。`:'显示原版标签。'}`;
 el('plyScopeText').textContent=scope==='local'?'原密度局部范围':'完整场景';
}
async function ensureModel(c){
 if(failed)return;
 const next=byUid.get(c.case_uid);if(!next){el('plyStatus').textContent='当前案例尚无 PLY';return;}
 model=next;uid=c.case_uid;
 el('plyTitle').textContent=`3D PLY · ${c.scene} ${c.ROI}`;
 el('plyDownload').href=model.full.path;el('plyDownloadLocal').href=model.local.path;
 const source=model[scope];if(loadedPath===source.path){updateUniforms();return;}
 const token=++loadToken;el('plyBusy').hidden=false;el('plyBusy').textContent=`正在加载${scope==='local'?'局部':'完整场景'} PLY · ${(source.bytes/1048576).toFixed(1)} MB…`;
 try{
   const palettes=await palettePromise;if(token!==loadToken)return;
   if(paletteUID!==c.case_uid){
     const original=palettes.scenes[c.scene]?.id_colors;if(!original)throw Error('原 PLY 配色缺少当前场景');
     const width=Math.max(1,...Object.keys(original).map(Number),...c.objects.map(o=>o.audit.persistent_id))+1,data=new Uint8Array(width*4);
     for(let id=0;id<width;id++){const rgb=original[id]||[220,130,200];data.set([...rgb,255],id*4);}
     for(const o of c.objects){const id=o.audit.persistent_id;if(!original[id])data.set([...c.object_colors[o.track_id],255],id*4);}
     if(paletteTexture)paletteTexture.dispose();paletteTexture=new THREE.DataTexture(data,width,1,THREE.RGBAFormat);paletteTexture.minFilter=paletteTexture.magFilter=THREE.NearestFilter;paletteTexture.generateMipmaps=false;paletteTexture.needsUpdate=true;material.uniforms.uPalette.value=paletteTexture;material.uniforms.uPaletteWidth.value=width;paletteUID=c.case_uid;
   }
   const response=await fetch(source.path);if(!response.ok)throw Error('PLY HTTP '+response.status);
   const buffer=await response.arrayBuffer();if(token!==loadToken)return;
   const incoming=parsePLY(buffer,source.points);if(token!==loadToken){incoming.dispose();return;}
   if(cloud)scene.remove(cloud);if(geometry)geometry.dispose();
   geometry=incoming;cloud=new THREE.Points(geometry,material);cloud.frustumCulled=false;scene.add(cloud);loadedPath=source.path;
   lastFocus=desired.solo;updateUniforms();focus(scope==='full');resize();el('plyBusy').hidden=true;
 }catch(error){if(token===loadToken){el('plyBusy').textContent=error.message;el('plyStatus').textContent='PLY 加载失败，可刷新或使用下载链接。';}}
}
async function sync(state){
 const changedCase=uid!==state.c.case_uid,changedMask=desired?.solo!==state.solo;
 desired=state;
 if(changedCase){scope='local';el('plyLocal').classList.add('active');el('plyFull').classList.remove('active');}
 await ensureModel(state.c);
 if(changedMask&&!changedCase&&lastFocus!==state.solo){lastFocus=state.solo;focus();}
}

el('plyLaunch').onclick=()=>{panel.hidden=false;el('plyLaunch').hidden=true;panel.classList.remove('minimized');el('plyMinimize').textContent='收起';resize();};
el('plyClose').onclick=()=>{panel.hidden=true;el('plyLaunch').hidden=false;};
el('plyMinimize').onclick=()=>{const min=panel.classList.toggle('minimized');el('plyMinimize').textContent=min?'展开':'收起';el('plyMinimize').setAttribute('aria-label',min?'展开 3D 窗口':'最小化 3D 窗口');resize();};
document.querySelectorAll('[data-ply-state]').forEach(b=>b.onclick=()=>{condition=+b.dataset.plyState;document.querySelectorAll('[data-ply-state]').forEach(x=>x.classList.toggle('active',x===b));updateUniforms();});
el('plyColor').onchange=updateUniforms;el('plyOnly').onchange=updateUniforms;el('plyUnassigned').onchange=updateUniforms;el('plySize').oninput=updateUniforms;el('plySeedBox').onchange=updateUniforms;
el('plyFollow').onchange=()=>{updateUniforms();focus();};el('plyFocus').onclick=()=>focus();el('plyReset').onclick=()=>focus(scope==='full');
function setScope(s){scope=s;el('plyLocal').classList.toggle('active',s==='local');el('plyFull').classList.toggle('active',s==='full');ensureModel(desired.c);}
el('plyLocal').onclick=()=>setScope('local');el('plyFull').onclick=()=>setScope('full');
let drag=null;const handle=el('plyDrag');
handle.addEventListener('pointerdown',e=>{if(e.target.closest('button'))return;const r=panel.getBoundingClientRect();drag={x:e.clientX,y:e.clientY,left:r.left,top:r.top};handle.setPointerCapture(e.pointerId);});
handle.addEventListener('pointermove',e=>{if(!drag)return;const x=Math.min(Math.max(4,drag.left+e.clientX-drag.x),Math.max(4,innerWidth-panel.offsetWidth-4));const y=Math.min(Math.max(4,drag.top+e.clientY-drag.y),Math.max(4,innerHeight-panel.offsetHeight-4));panel.style.left=x+'px';panel.style.top=y+'px';panel.style.right='auto';panel.style.bottom='auto';});
handle.addEventListener('pointerup',()=>{drag=null;});handle.addEventListener('pointercancel',()=>{drag=null;});
window.addEventListener('resize',resize);
initialize();window.pilotPlyViewer={sync};if(window.pilotPlyApi)sync(window.pilotPlyApi.state());
