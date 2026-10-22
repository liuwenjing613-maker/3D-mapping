import * as THREE from './vendor/three.module.js';
import {OrbitControls} from './vendor/OrbitControls.js';
import {parsePLY,instanceColors} from './ply.js?v=1';
const $=s=>document.querySelector(s),fmt=n=>Number(n).toLocaleString('zh-CN');
const views=[];let manifest,linking=false,scheduled=false;
const cfg={mode:1,scope:0,pointSize:2,clip:false,clipZ:.9,trajectory:false};
const fail=e=>{const box=$('#error');box.hidden=false;box.textContent='3D 加载未完成：'+(e.message||e)+'。可使用各面板的原始 PLY 链接。';console.error(e);};
window.addEventListener('unhandledrejection',e=>fail(e.reason));
const vertexShader=`
attribute vec3 rgbColor;
attribute vec3 instanceColor;
attribute float instanceID;
attribute float stateID;
uniform float pointSize;
uniform float pixelRatio;
uniform int mode;
uniform int scope;
uniform float chosenID;
uniform bool clipEnabled;
uniform float clipZ;
varying vec3 vColor;
varying float visiblePoint;
void main(){
 bool assigned=instanceID>0.0;
 bool show=(scope==0)||(scope==1&&assigned)||(scope==2&&!assigned)||(scope==3&&!assigned&&stateID<0.5)||(scope==4&&!assigned&&stateID>0.5&&stateID<1.5)||(scope==5&&!assigned&&stateID>2.5);
 show=show&&(chosenID<0.0||abs(instanceID-chosenID)<0.5)&&(!clipEnabled||position.z<=clipZ);
 visiblePoint=show?1.0:0.0;
 vec3 stateColor=assigned?vec3(.64,.69,.70):(stateID<0.5?vec3(0.0):(stateID<1.5?vec3(1.0,1.0,0.0):vec3(1.0,0.0,0.0)));
 vColor=mode==0?rgbColor:(mode==1?instanceColor:(mode==2?stateColor:vec3(.36,.49,.53)));
 gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);
 gl_PointSize=pointSize*pixelRatio;
}`;
const fragmentShader=`
precision highp float;
varying vec3 vColor;
varying float visiblePoint;
void main(){
 if(visiblePoint<0.5||distance(gl_PointCoord,vec2(.5))>.5)discard;
 gl_FragColor=vec4(vColor,1.0);
}`;
function schedule(){if(scheduled)return;scheduled=true;requestAnimationFrame(()=>{scheduled=false;for(const v of views)v.renderer.render(v.scene,v.camera);});}
function cameraChange(source){
 if(!linking&&$('#sync').checked){linking=true;for(const v of views){if(v===source)continue;v.camera.position.copy(source.camera.position);v.camera.up.copy(source.camera.up);v.controls.target.copy(source.controls.target);v.controls.update();}linking=false;}
 schedule();
}
function setPose(v,pose){v.camera.up.set(0,0,1);v.camera.position.fromArray(pose.eye);v.controls.target.copy(v.camera.position).add(new THREE.Vector3().fromArray(pose.forward).multiplyScalar(3));v.controls.update();}
function turn(v,yaw,pitch=0){const off=v.camera.position.clone().sub(v.controls.target),r=off.length(),a=Math.atan2(off.y,off.x)+yaw,p=Math.max(-1.52,Math.min(1.52,Math.asin(off.z/r)+pitch));v.camera.position.set(r*Math.cos(p)*Math.cos(a),r*Math.cos(p)*Math.sin(a),r*Math.sin(p)).add(v.controls.target);v.controls.update();}
function preset(name){
 linking=true;
 for(const v of views){
  if(name==='first')setPose(v,manifest.maps[0].cameras[0]);
  else if(name==='left'||name==='right')turn(v,(name==='left'?1:-1)*Math.PI/6);
  else{const box=new THREE.Box3(new THREE.Vector3().fromArray(manifest.maps[manifest.maps.length-1].bbox[0]),new THREE.Vector3().fromArray(manifest.maps[manifest.maps.length-1].bbox[1])),center=box.getCenter(new THREE.Vector3());
   v.controls.target.copy(center);v.camera.up.set(0,0,1);v.camera.position.copy(center).add(name==='top'?new THREE.Vector3(0,-.005,8.4):new THREE.Vector3(6.3,-6.6,4.6));v.controls.update();
  }
 }
 linking=false;schedule();
}
function matches(data,i,id){
 if(id>=0&&data.ids[i]!==id)return false;
 if(cfg.clip&&data.positions[i*3+2]>cfg.clipZ)return false;
 const assigned=data.ids[i]>0,s=data.states[i],f=cfg.scope;
 return f===0||f===1&&assigned||f===2&&!assigned||f===3&&!assigned&&s===0||f===4&&!assigned&&s===1||f===5&&!assigned&&s===3;
}
function updateSettings(v){
 const u=v.material?.uniforms;if(!u)return;
 u.mode.value=cfg.mode;u.scope.value=cfg.scope;u.pointSize.value=cfg.pointSize;u.clipEnabled.value=cfg.clip;u.clipZ.value=cfg.clipZ;u.chosenID.value=Number(v.select.value);v.path.visible=cfg.trajectory;
 let shown=0;for(let i=0;i<v.data.count;i++)if(matches(v.data,i,u.chosenID.value))shown++;
 v.panel.querySelector('.view-count').textContent='已载入 '+fmt(v.data.count)+' 点 · 当前筛选 '+fmt(shown)+' 点';
 schedule();
}
function updateAll(){for(const v of views)updateSettings(v);}
async function fetchPLY(row,onProgress){
 const response=await fetch(row.ply,{cache:'no-store'});if(!response.ok)throw Error(row.frames+' 帧 PLY 读取失败');
 const bytes=new Uint8Array(row.bytes),reader=response.body.getReader();let offset=0;
 while(true){const {done,value}=await reader.read();if(done)break;bytes.set(value,offset);offset+=value.byteLength;onProgress(Math.min(100,100*offset/row.bytes));}
 if(offset!==row.bytes)throw Error('PLY 字节数与清单不一致');
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(x=>x.toString(16).padStart(2,'0')).join('');
 if(digest!==row.sha256)throw Error('PLY 哈希核验失败');
 return parsePLY(bytes.buffer);
}
function makePanel(row){
 const panel=document.createElement('section');panel.className='map-panel';panel.dataset.frames=row.frames;
 panel.innerHTML=`<div class="panel-head"><div><h2>${row.frames} 帧</h2><div class="cutoff">原始编号 0, 5, …, ${row.last_frame}</div></div><span class="tag">修复前实际地图</span></div>
 <div class="stats"><div><strong>${fmt(row.points)}</strong><span>完整表面点</span></div><div><strong>${fmt(row.assigned)}</strong><span>已赋实例 ID · ${row.instances} 个实例</span></div><div><strong>${(100*row.assigned/row.points).toFixed(2)}%</strong><span>实例覆盖率</span></div></div>
 <div class="stage-states"><span>剩余 U <b>${fmt(row.states.U)}</b></span><span>T <b>${fmt(row.states.T)}</b></span><span>C <b>${fmt(row.states.C)}</b></span></div>
 <div class="viewport"><div class="load-state"><div class="spinner"></div><span>准备加载 ${(row.bytes/1e6).toFixed(1)} MB PLY…</span></div><span class="orientation">Z ↑ · 米 · 全量点云</span></div>
 <div class="panel-controls"><label>实例 ID <select class="instance-select" aria-label="${row.frames} 帧实例 ID"><option value="-1">全部</option>${Object.entries(row.instance_counts).map(([id,n])=>`<option value="${id}">ID ${id} · ${fmt(n)} 点</option>`).join('')}</select></label><button class="last-view">本进度末帧视角</button><span class="view-count">等待 PLY 载入</span></div>
 <div class="download-row">${row.downloads.map(x=>`<a href="${x.file}" download>${x.title}</a>`).join('')}</div>`;
 $('#maps').append(panel);
 const viewport=panel.querySelector('.viewport'),renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
 renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0xe9eef0,1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.NoToneMapping;
 renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label',row.frames+' 帧 3D 点云，可拖动旋转、滚轮缩放');viewport.prepend(renderer.domElement);
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(60,1,.015,100);camera.up.set(0,0,1);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;controls.minDistance=.06;controls.maxDistance=60;controls.zoomToCursor=true;
 const v={row,panel,viewport,renderer,scene,camera,controls,select:panel.querySelector('.instance-select')};views.push(v);
 setPose(v,manifest.maps[0].cameras[0]);controls.addEventListener('change',()=>cameraChange(v));
 const resize=()=>{const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();schedule();};new ResizeObserver(resize).observe(viewport);resize();
 v.select.onchange=()=>updateSettings(v);
 panel.querySelector('.last-view').onclick=()=>setPose(v,row.cameras.at(-1));
 renderer.domElement.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();turn(v,e.key==='ArrowLeft'?.12:e.key==='ArrowRight'?-.12:0,e.key==='ArrowUp'?.08:e.key==='ArrowDown'?-.08:0);});
 const pathGeometry=new THREE.BufferGeometry().setFromPoints(row.cameras.map(p=>new THREE.Vector3().fromArray(p.eye)));
 v.path=new THREE.Line(pathGeometry,new THREE.LineBasicMaterial({color:0x007e82}));v.path.visible=false;scene.add(v.path);
 renderer.domElement.addEventListener('dblclick',e=>{
  if(!v.cloud)return;const rect=renderer.domElement.getBoundingClientRect(),mouse=new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),ray=new THREE.Raycaster();
  ray.params.Points.threshold=.035;ray.setFromCamera(mouse,camera);
  const hit=ray.intersectObject(v.cloud).find(h=>matches(v.data,h.index,Number(v.select.value)));
  if(hit){controls.target.fromArray(v.data.positions,hit.index*3);controls.update();}
 });
 return v;
}
async function loadMap(v){
 const loading=v.panel.querySelector('.load-state'),text=loading.querySelector('span');
 try{
  v.data=await fetchPLY(v.row,p=>text.textContent='从服务器载入实际 PLY · '+p.toFixed(0)+'%');
  if(v.data.count!==v.row.points)throw Error('点数与前缀地图不一致');
  text.textContent='PLY 哈希已通过 · 建立 3D 显示…';
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(v.data.positions,3));
  geometry.setAttribute('rgbColor',new THREE.Uint8BufferAttribute(v.data.rgb,3,true));geometry.setAttribute('instanceColor',new THREE.Uint8BufferAttribute(instanceColors(v.data.ids,v.data.states),3,true));
  geometry.setAttribute('instanceID',new THREE.BufferAttribute(v.data.ids,1));geometry.setAttribute('stateID',new THREE.BufferAttribute(v.data.states,1));geometry.computeBoundingSphere();
  v.material=new THREE.ShaderMaterial({vertexShader,fragmentShader,uniforms:{pointSize:{value:cfg.pointSize},pixelRatio:{value:v.renderer.getPixelRatio()},mode:{value:cfg.mode},scope:{value:cfg.scope},chosenID:{value:-1},clipEnabled:{value:cfg.clip},clipZ:{value:cfg.clipZ}},depthTest:true,depthWrite:true,toneMapped:false});
  v.cloud=new THREE.Points(geometry,v.material);v.cloud.frustumCulled=false;v.scene.add(v.cloud);
  updateSettings(v);loading.hidden=true;v.panel.dataset.loaded='true';schedule();
 }catch(e){text.textContent='载入失败：'+e.message;loading.querySelector('.spinner').hidden=true;fail(e);}
}
$('#mode').onchange=e=>{cfg.mode=Number(e.target.value);updateAll();};
$('#scope').onchange=e=>{cfg.scope=Number(e.target.value);updateAll();};
$('#point-size').oninput=e=>{cfg.pointSize=Number(e.target.value);$('#size-value').value=cfg.pointSize;updateAll();};
$('#clip').onchange=e=>{cfg.clip=e.target.checked;$('#clip-z').disabled=!cfg.clip;updateAll();};
$('#clip-z').oninput=e=>{cfg.clipZ=Number(e.target.value);$('#clip-value').value=cfg.clipZ.toFixed(2)+' m';updateAll();};
$('#trajectory').onchange=e=>{cfg.trajectory=e.target.checked;updateAll();};
$('#sync').onchange=e=>{if(e.target.checked&&views[0])cameraChange(views[0]);};
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>preset(b.dataset.view));
$('#reset').onclick=()=>{cfg.mode=1;cfg.scope=0;cfg.pointSize=2;cfg.clip=false;cfg.clipZ=.9;cfg.trajectory=false;$('#mode').value='1';$('#scope').value='0';$('#point-size').value='2';$('#size-value').value='2';$('#clip').checked=false;$('#clip-z').disabled=true;$('#clip-z').value='.9';$('#clip-value').value='0.90 m';$('#trajectory').checked=false;for(const v of views)v.select.value='-1';updateAll();preset('first');};
try{const response=await fetch('manifest.json',{cache:'no-store'});if(!response.ok)throw Error('地图清单读取失败');manifest=await response.json();$('#maps').innerHTML='';const maps=manifest.maps.map(makePanel);await Promise.all(maps.map(loadMap));}catch(e){fail(e);}
