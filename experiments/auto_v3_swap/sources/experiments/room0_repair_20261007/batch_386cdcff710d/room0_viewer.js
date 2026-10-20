import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
const meta=JSON.parse(document.getElementById('modelData').textContent),el=id=>document.getElementById(id);
const titles=['原始地图','仅种子修复','完整双向追踪修复'];
const renderer=new THREE.WebGLRenderer({canvas:el('canvas'),antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0x172338);
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(58,1,.02,100);camera.up.set(0,0,1);
const controls=new OrbitControls(camera,el('canvas'));controls.screenSpacePanning=true;
let geometry,cloud,condition=2,scope='local',uid=el('caseSelect').value,token=0;
const colors=(await fetch('instance_colors.json').then(r=>r.json())).id_colors;
const width=Math.max(361,...Object.keys(colors).map(Number))+1,pixels=new Uint8Array(width*4),targetPixels=new Uint8Array(width*4);
for(let id=0;id<width;id++)pixels.set([...(colors[id]||[220,130,200]),255],id*4);
const palette=new THREE.DataTexture(pixels,width,1,THREE.RGBAFormat),targets=new THREE.DataTexture(targetPixels,width,1,THREE.RGBAFormat);
for(const texture of [palette,targets]){texture.minFilter=texture.magFilter=THREE.NearestFilter;texture.needsUpdate=true;}
const material=new THREE.ShaderMaterial({vertexColors:true,uniforms:{uState:{value:2},uMode:{value:1},uPalette:{value:palette},uTargets:{value:targets},
 uWidth:{value:width},uOnly:{value:0},uUnknown:{value:1},uSize:{value:2}},
 vertexShader:`precision highp float;attribute vec3 aLabels;uniform float uState,uMode,uWidth,uOnly,uUnknown,uSize;
 uniform sampler2D uPalette,uTargets;varying vec3 vColor;varying float vVisible;
 void main(){float id=uState<.5?aLabels.x:uState<1.5?aLabels.y:aLabels.z;
 vec2 uv=vec2((max(0.,id)+.5)/uWidth,.5);vec4 p=texture2D(uPalette,uv);float target=texture2D(uTargets,uv).r;
 vVisible=1.;if((id>0.&&p.a<.5)||(id>0.&&uOnly>.5&&target<.5)||(id<=0.&&uUnknown<.5))vVisible=0.;
 vColor=id>0.?p.rgb:vec3(.35);if(uMode<.5)vColor=color;
 if(uMode>1.5){vColor=color*.28;if(abs(id-aLabels.x)>.1)vColor=aLabels.x<=0.?vec3(.14,.92,.64):id<=0.?vec3(1.,.27,.33):vec3(1.,.68,.16);}
 gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);gl_PointSize=uSize;}`,
 fragmentShader:`precision highp float;varying vec3 vColor;varying float vVisible;
 void main(){if(vVisible<.5||length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(vColor,1.);}`});
function draw(){if(!el('float').hidden&&!el('float').classList.contains('mini'))renderer.render(scene,camera);}
function resize(){const w=el('stage').clientWidth,h=el('stage').clientHeight;if(w&&h){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();draw();}}
new ResizeObserver(resize).observe(el('stage'));controls.addEventListener('change',draw);
function focus(){const entry=meta.cases[uid],b=scope==='full'?meta.native_bounds:entry.bounds;
 const lo=new THREE.Vector3(...b[0]),hi=new THREE.Vector3(...b[1]),center=lo.clone().add(hi).multiplyScalar(.5),radius=Math.max(.25,lo.distanceTo(hi)*.5);
 const p=entry.camera_to_world;const direction=scope==='full'?new THREE.Vector3(1,-1,.8):new THREE.Vector3(p[0][3],p[1][3],p[2][3]).sub(center);
 if(direction.length()<.1)direction.set(1,-1,.6);camera.position.copy(center.clone().add(direction.normalize().multiplyScalar(Math.max(.8,radius*(scope==='full'?2.3:1.8)))));controls.target.copy(center);controls.update();draw();}
function update(){material.uniforms.uState.value=condition;material.uniforms.uMode.value=+el('colorMode').value;
 material.uniforms.uOnly.value=el('only').checked?1:0;material.uniforms.uUnknown.value=el('unknown').checked?1:0;material.uniforms.uSize.value=+el('pointSize').value;
 const s=geometry?.userData.stats?.[condition];if(s)el('modelStatus').textContent=`${titles[condition]} · ${scope==='full'?'完整room0':meta.cases[uid].ROI} · ${geometry.attributes.position.count.toLocaleString()}点 · 灰点${s.unknown.toLocaleString()} · 标签变化${s.changed.toLocaleString()}`;
 draw();}
function legend(){pixels.forEach((_,i)=>{if(i%4===3)pixels[i]=255;});targetPixels.fill(0);el('idLegend').replaceChildren();
 const ids=Object.keys(meta.identity_sources).map(Number).sort((a,b)=>a-b);el('only').disabled=!ids.length;if(!ids.length)el('only').checked=false;
 for(const id of ids){targetPixels.set([255,255,255,255],id*4);const label=document.createElement('label'),checkbox=document.createElement('input'),dot=document.createElement('i');
 checkbox.type='checkbox';checkbox.checked=true;checkbox.dataset.pid=id;dot.style.background=`rgb(${colors[id]||[220,130,200]})`;label.title=meta.identity_sources[id].join('；');
 checkbox.onchange=()=>{pixels[id*4+3]=checkbox.checked?255:0;palette.needsUpdate=true;draw();};label.append(checkbox,dot,document.createTextNode(' ID'+id));el('idLegend').append(label);}
 palette.needsUpdate=true;targets.needsUpdate=true;
 document.querySelectorAll('[data-case]').forEach(section=>section.hidden=section.dataset.case!==uid);
 el('caseStatus').textContent=meta.cases[uid].ROI+' · 种子帧 '+meta.cases[uid].frame;}
async function load(){const localToken=++token,entry=scope==='full'?meta.full:meta.cases[uid];el('busy').hidden=false;el('busy').textContent='加载真实PLY并校验…';
 try{const response=await fetch(entry.path);if(!response.ok)throw Error('PLY HTTP '+response.status);const buffer=await response.arrayBuffer();
 const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),b=>b.toString(16).padStart(2,'0')).join('');
 if(hash!==entry.sha256)throw Error('PLY来源校验失败');if(localToken!==token)return;
 const header=new TextDecoder().decode(new Uint8Array(buffer,0,Math.min(4096,buffer.byteLength))),end=header.indexOf('end_header\n');
 if(end<0)throw Error('PLY header');const offset=end+11,n=+header.match(/element vertex (\d+)/)[1],stride=meta.record_stride_bytes;
 if(stride!==31||n!==entry.points||offset+n*stride!==buffer.byteLength)throw Error('PLY点数或格式错误');
 const view=new DataView(buffer),positions=new Float32Array(n*3),rgb=new Float32Array(n*3),labels=new Float32Array(n*3);
 const stats=Array.from({length:3},()=>({assigned:0,unknown:0,changed:0}));
 for(let i=0,j=offset;i<n;i++,j+=stride){for(let k=0;k<3;k++){positions[i*3+k]=view.getFloat32(j+k*4,true);rgb[i*3+k]=view.getUint8(j+12+k)/255;
  const id=view.getInt32(j+15+k*4,true);labels[i*3+k]=id;stats[k][id>0?'assigned':'unknown']++;if(id!==labels[i*3])stats[k].changed++;}}
 const next=new THREE.BufferGeometry();next.setAttribute('position',new THREE.BufferAttribute(positions,3));next.setAttribute('color',new THREE.BufferAttribute(rgb,3));next.setAttribute('aLabels',new THREE.BufferAttribute(labels,3));next.userData.stats=stats;
 if(cloud)scene.remove(cloud);if(geometry)geometry.dispose();geometry=next;cloud=new THREE.Points(geometry,material);cloud.frustumCulled=false;scene.add(cloud);
 el('busy').hidden=true;window.room0Model={points:n,states:stats,scope,case_uid:uid,sha256:hash,hashVerified:true,ply:entry.path};update();focus();resize();
 }catch(error){el('busy').textContent=error.message;}}
el('state').onchange=()=>{condition=+el('state').value;update();};el('scope').onchange=()=>{scope=el('scope').value;load();};
el('caseSelect').onchange=()=>{uid=el('caseSelect').value;legend();if(scope==='local')load();else{focus();update();}};
for(const id of ['colorMode','only','unknown','pointSize'])el(id).addEventListener('input',update);
el('allIds').onclick=()=>{el('idLegend').querySelectorAll('input').forEach(input=>{input.checked=true;pixels[Number(input.dataset.pid)*4+3]=255;});palette.needsUpdate=true;draw();};
el('reset').onclick=focus;el('close').onclick=()=>{el('float').hidden=true;el('open').hidden=false;};
el('open').onclick=()=>{el('float').hidden=false;el('open').hidden=true;resize();};el('minimize').onclick=()=>{el('float').classList.toggle('mini');resize();};
let drag=null;el('drag').addEventListener('pointerdown',event=>{if(event.target.closest('button'))return;const b=el('float').getBoundingClientRect();drag={x:event.clientX,y:event.clientY,left:b.left,top:b.top};el('drag').setPointerCapture(event.pointerId);});
el('drag').addEventListener('pointermove',event=>{if(!drag)return;const p=el('float');p.style.right='auto';p.style.bottom='auto';p.style.left=Math.max(0,Math.min(innerWidth-p.offsetWidth,drag.left+event.clientX-drag.x))+'px';p.style.top=Math.max(0,Math.min(innerHeight-45,drag.top+event.clientY-drag.y))+'px';});
el('drag').addEventListener('pointerup',()=>drag=null);legend();load();
