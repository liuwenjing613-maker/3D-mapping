import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const meta=JSON.parse(document.getElementById('modelData').textContent), el=id=>document.getElementById(id);
const titles=['原基线','旧：两对象同时合格','旧：无质量门槛','新：SAM2候选逐对象','新：原mask逐对象'];
const canvas=el('canvas'), panel=el('float'), stage=el('stage');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0x172338);
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(58,1,.02,100);
camera.up.set(0,0,1);
const controls=new OrbitControls(camera,canvas);controls.screenSpacePanning=true;
let geometry,cloud,condition=3,scope='local',loadToken=0;
const palettes=await fetch('instance_palettes.json').then(r=>r.json()), original=palettes.scenes.room2.id_colors;
const width=Math.max(355,...Object.keys(original).map(Number))+1, pixels=new Uint8Array(width*4);
for(let id=0;id<width;id++)pixels.set([...(original[id]||[220,130,200]),255],id*4);
pixels.set([40,186,245,255],52*4);pixels.set([64,234,120,255],355*4);
const palette=new THREE.DataTexture(pixels,width,1,THREE.RGBAFormat);
palette.minFilter=palette.magFilter=THREE.NearestFilter;palette.needsUpdate=true;
const material=new THREE.ShaderMaterial({vertexColors:true,uniforms:{uState:{value:condition},uMode:{value:1},
  uPalette:{value:palette},uWidth:{value:width},uOnly:{value:0},uBlue:{value:1},uGreen:{value:1},uUnknown:{value:1},uSize:{value:2}},
vertexShader:`precision highp float;attribute vec3 aLabels0;attribute vec2 aLabels1;
uniform float uState,uMode,uWidth,uOnly,uBlue,uGreen,uUnknown,uSize;uniform sampler2D uPalette;
varying vec3 vColor;varying float vVisible;
void main(){float id=uState<.5?aLabels0.x:uState<1.5?aLabels0.y:uState<2.5?aLabels0.z:uState<3.5?aLabels1.x:aLabels1.y;
bool blue=abs(id-52.)<.1,green=abs(id-355.)<.1;
vVisible=1.;if((uOnly>.5&&!blue&&!green)||(blue&&uBlue<.5)||(green&&uGreen<.5)||(id<=0.&&uUnknown<.5))vVisible=0.;
vColor=id>0.?texture2D(uPalette,vec2((id+.5)/uWidth,.5)).rgb:vec3(.35);
if(uMode<.5)vColor=color;
if(uMode>1.5){vColor=color*.3;if(abs(id-aLabels0.x)>.1)vColor=aLabels0.x<=0.?vec3(.14,.92,.64):id<=0.?vec3(1.,.27,.33):vec3(1.,.68,.16);}
gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);gl_PointSize=uSize;}`,
fragmentShader:`precision highp float;varying vec3 vColor;varying float vVisible;void main(){if(vVisible<.5||length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(vColor,1.);}`});
function draw(){if(!panel.hidden&&!panel.classList.contains('mini'))renderer.render(scene,camera);}
function resize(){const w=stage.clientWidth,h=stage.clientHeight;if(w&&h){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();draw();}}
new ResizeObserver(resize).observe(stage);controls.addEventListener('change',draw);
function focus(){const b=scope==='local'?meta.seed_bounds:meta.native_bounds;
const lo=new THREE.Vector3(...b[0]),hi=new THREE.Vector3(...b[1]),center=lo.clone().add(hi).multiplyScalar(.5),radius=Math.max(.25,hi.distanceTo(lo)*.5);
const pose=meta.seed_camera_to_world;
const direction=scope==='local'?new THREE.Vector3(pose[0][3],pose[1][3],pose[2][3]).sub(center):new THREE.Vector3(1,-1,.7);
if(direction.length()<.1)direction.set(1,-1,.6);
camera.position.copy(center.clone().add(direction.normalize().multiplyScalar(Math.max(.8,radius*2.4))));controls.target.copy(center);controls.update();draw();}
function status(){const s=geometry?.userData.stats?.[condition];if(s)el('modelStatus').textContent=`${titles[condition]} · ${scope==='local'?'局部':'完整场景'} ${geometry.attributes.position.count.toLocaleString()} 点 · 已分配 ${s.assigned.toLocaleString()} · 灰点 ${s.unknown.toLocaleString()} · 标签变化 ${s.changed.toLocaleString()}`;}
function update(){material.uniforms.uState.value=condition;material.uniforms.uMode.value=+el('colorMode').value;
material.uniforms.uOnly.value=el('only').checked?1:0;material.uniforms.uBlue.value=el('blue').checked?1:0;
material.uniforms.uGreen.value=el('green').checked?1:0;material.uniforms.uUnknown.value=el('unknown').checked?1:0;
material.uniforms.uSize.value=+el('pointSize').value;status();draw();}
async function load(){const token=++loadToken;el('busy').hidden=false;el('busy').textContent='正在加载真实 PLY…';
try{const response=await fetch(meta[scope].path);if(!response.ok)throw Error('PLY HTTP '+response.status);
const buffer=await response.arrayBuffer();if(token!==loadToken)return;
const bytes=new Uint8Array(buffer),header=new TextDecoder().decode(bytes.subarray(0,4096)),end=header.indexOf('end_header\n');
if(end<0)throw Error('PLY header');const offset=end+11,n=+header.match(/element vertex (\d+)/)[1],stride=meta.record_stride_bytes;
if(stride!==39||n!==meta[scope].points||offset+n*stride!==buffer.byteLength)throw Error('PLY 点数校验失败');
const data=new DataView(buffer),positions=new Float32Array(n*3),colors=new Float32Array(n*3),a=new Float32Array(n*3),b=new Float32Array(n*2);
const stats=Array.from({length:5},()=>({assigned:0,unknown:0,changed:0}));
for(let i=0,j=offset;i<n;i++,j+=stride){for(let k=0;k<3;k++){positions[i*3+k]=data.getFloat32(j+k*4,true);colors[i*3+k]=data.getUint8(j+12+k)/255;}
for(let k=0;k<5;k++){const id=data.getInt32(j+15+k*4,true);if(k<3)a[i*3+k]=id;else b[i*2+k-3]=id;stats[k][id>0?'assigned':'unknown']++;if(id!==a[i*3])stats[k].changed++;}}
const incoming=new THREE.BufferGeometry();incoming.setAttribute('position',new THREE.BufferAttribute(positions,3));incoming.setAttribute('color',new THREE.BufferAttribute(colors,3));
incoming.setAttribute('aLabels0',new THREE.BufferAttribute(a,3));incoming.setAttribute('aLabels1',new THREE.BufferAttribute(b,2));incoming.userData.stats=stats;
if(cloud)scene.remove(cloud);if(geometry)geometry.dispose();geometry=incoming;cloud=new THREE.Points(geometry,material);cloud.frustumCulled=false;scene.add(cloud);
el('busy').hidden=true;window.objectwiseModel={points:n,states:stats,scope,ply:meta[scope].path};update();focus();resize();
}catch(error){el('busy').textContent=error.message;}}
el('state').onchange=()=>{condition=+el('state').value;update();};
el('scope').onchange=()=>{scope=el('scope').value;load();};
['colorMode','only','blue','green','unknown','pointSize'].forEach(id=>el(id).addEventListener('input',update));
el('reset').onclick=focus;el('close').onclick=()=>{panel.hidden=true;el('open').hidden=false;};
el('open').onclick=()=>{panel.hidden=false;el('open').hidden=true;resize();};
el('minimize').onclick=()=>{panel.classList.toggle('mini');resize();};
let drag=null;el('drag').addEventListener('pointerdown',event=>{if(event.target.closest('button'))return;
const b=panel.getBoundingClientRect();drag={x:event.clientX,y:event.clientY,left:b.left,top:b.top};el('drag').setPointerCapture(event.pointerId);});
el('drag').addEventListener('pointermove',event=>{if(!drag)return;panel.style.right='auto';panel.style.bottom='auto';
panel.style.left=Math.max(0,Math.min(innerWidth-panel.offsetWidth,drag.left+event.clientX-drag.x))+'px';panel.style.top=Math.max(0,Math.min(innerHeight-45,drag.top+event.clientY-drag.y))+'px';});
el('drag').addEventListener('pointerup',()=>drag=null);
load();
