import * as THREE from '../room2_top30_repair_20261008/vendor/three.module.js';
import {OrbitControls} from '../room2_top30_repair_20261008/vendor/OrbitControls.js';
const el=id=>document.getElementById(id),params=new URLSearchParams(location.search);
if(['room0','room2'].includes(params.get('scene')))el('scene').value=params.get('scene');
const camera=new THREE.PerspectiveCamera(58,1,.02,120);camera.up.set(0,0,1);
const sharedTarget=new THREE.Vector3();
let metadata,geometry,palette,targets,pixels,targetPixels,token=0;
const panels=['left','middle','right'].map((id,index)=>{
 const canvas=el(id),renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});
 renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor(0x172338);
 const scene=new THREE.Scene(),controls=new OrbitControls(camera,canvas);
 controls.target=sharedTarget;controls.screenSpacePanning=true;controls.addEventListener('change',draw);
 return {id,index,renderer,scene,controls,cloud:null,material:null};
});
function draw(){for(const p of panels){const w=p.renderer.domElement.clientWidth,h=p.renderer.domElement.clientHeight;if(!w||!h)continue;
 p.renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();p.renderer.render(p.scene,camera);}}
new ResizeObserver(draw).observe(document.querySelector('.panels'));
const shader={
 vertexShader:`precision highp float;attribute vec4 aLabels;uniform float uState,uMode,uWidth,uUnknown,uOthers,uChangesOnly,uSize,uLocal;
 uniform vec3 uMin,uMax;uniform sampler2D uPalette,uTargets;varying vec3 vColor;varying float vVisible;
 void main(){float id=uState<.5?aLabels.x:uState<1.5?aLabels.y:uState<2.5?aLabels.z:aLabels.w;float old=aLabels.y;
 vec2 uv=vec2((max(0.,id)+.5)/uWidth,.5);vec4 p=texture2D(uPalette,uv);float target=texture2D(uTargets,uv).r;
 vVisible=1.;if((id>0.&&p.a<.5)||(id>0.&&uOthers<.5&&target<.5)||(id<=0.&&uUnknown<.5))vVisible=0.;
 if(uChangesOnly>.5&&abs(id-old)<.1)vVisible=0.;
 if(uLocal>.5&&(position.x<uMin.x||position.x>uMax.x||position.y<uMin.y||position.y>uMax.y||position.z<uMin.z||position.z>uMax.z))vVisible=0.;
 vColor=id>0.?p.rgb:vec3(.412);if(uMode>.5){vColor=vec3(.21);if(abs(id-old)>.1)vColor=old<=0.?vec3(.14,.92,.64):id<=0.?vec3(1.,.27,.33):vec3(1.,.68,.16);}
 gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);gl_PointSize=uSize;}`,
 fragmentShader:`precision highp float;varying vec3 vColor;varying float vVisible;void main(){if(vVisible<.5||length(gl_PointCoord-vec2(.5))>.5)discard;gl_FragColor=vec4(vColor,1.);}`
};
function currentCase(){return metadata?.cases?.[el('caseSelect').value];}
function focus(){if(!metadata)return;const entry=currentCase(),b=entry?.bounds||metadata.native_bounds;
 const lo=new THREE.Vector3(...b[0]),hi=new THREE.Vector3(...b[1]),center=lo.clone().add(hi).multiplyScalar(.5),radius=Math.max(.25,lo.distanceTo(hi)*.5);
 const p=entry?.camera_to_world,direction=p?new THREE.Vector3(p[0][3],p[1][3],p[2][3]).sub(center):new THREE.Vector3(1,-1,.8);
 if(direction.length()<.1)direction.set(1,-1,.6);camera.position.copy(center.clone().add(direction.normalize().multiplyScalar(Math.max(.8,radius*(entry?1.8:2.3)))));
 sharedTarget.copy(center);panels.forEach(p=>p.controls.update());draw();}
function update(){if(!metadata||!geometry)return;const entry=currentCase(),b=entry?.bounds||metadata.native_bounds,states=[+el('reference').value,2,3];
 for(const p of panels){const u=p.material.uniforms;u.uState.value=states[p.index];u.uMode.value=+el('colorMode').value;
  u.uUnknown.value=el('unknown').checked?1:0;u.uOthers.value=el('others').checked?1:0;u.uChangesOnly.value=el('changesOnly').checked?1:0;
  u.uSize.value=+el('pointSize').value;u.uLocal.value=entry?1:0;u.uMin.value.set(...b[0]);u.uMax.value.set(...b[1]);
  const name=metadata.state_names[states[p.index]],s=metadata.stats[name];
  el(p.id+'Stats').textContent=`完整地图：灰点 ${s.unknown.toLocaleString()} · 与原完整追踪相比变化 ${s.changed_vs_unrestricted.toLocaleString()} 点`;
 }
 el('referenceTitle').textContent=+el('reference').value===1?'原完整追踪修复':'原始P1-A1地图';
 el('deltaLegend').hidden=+el('colorMode').value!==1;
 el('maskLink').href='playback.html?scene='+el('scene').value+'&policy=gap1';draw();}
function legend(){if(!metadata)return;el('idLegend').replaceChildren();targetPixels.fill(0);
 for(let i=3;i<pixels.length;i+=4)pixels[i]=255;
 const entry=currentCase(),ids=entry?.target_ids||metadata.selected_persistent_ids;
 for(const id of ids){targetPixels.set([255,255,255,255],id*4);const label=document.createElement('label'),check=document.createElement('input'),dot=document.createElement('i');
  check.type='checkbox';check.checked=true;check.dataset.pid=id;check.setAttribute('aria-label','3D ID'+id);
  dot.style.background=`rgb(${metadata.id_colors[id]})`;label.title=(metadata.identity_sources[id]||[]).join('；');
  check.onchange=()=>{pixels[id*4+3]=check.checked?255:0;palette.needsUpdate=true;draw();};label.append(check,dot,document.createTextNode('ID'+id));el('idLegend').append(label);
 }
 palette.needsUpdate=true;targets.needsUpdate=true;
}
async function load(){const mine=++token,sceneName=el('scene').value;el('busy').hidden=false;el('busy').textContent='读取完整地图并校验来源…';
 try{const response=await fetch('models/'+sceneName+'_comparison_pack.json');if(!response.ok)throw Error('地图元数据尚未就绪');const next=await response.json();
  if(next.status!=='PASS'||next.stride_bytes!==28||next.downsampled||next.GT_used_for_rendering)throw Error('地图导出校验条件不满足');
  const raw=await fetch('models/'+sceneName+'_comparison_all_points.bin');if(!raw.ok)throw Error('地图文件 HTTP '+raw.status);const buffer=await raw.arrayBuffer();
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),b=>b.toString(16).padStart(2,'0')).join('');
  if(hash!==next.sha256||buffer.byteLength!==next.points*28)throw Error('完整地图SHA或点数校验失败');if(mine!==token)return;
  const view=new DataView(buffer),positions=new Float32Array(next.points*3),labels=new Float32Array(next.points*4),actualStats=Object.fromEntries(next.state_names.map(name=>[name,{unknown:0,changed:0}]));
  for(let i=0,j=0;i<next.points;i++,j+=28){for(let k=0;k<3;k++)positions[i*3+k]=view.getFloat32(j+k*4,true);
   const old=view.getInt32(j+16,true);for(let k=0;k<4;k++){const id=view.getInt32(j+12+k*4,true);labels[i*4+k]=id;const s=actualStats[next.state_names[k]];if(id<=0)s.unknown++;if(id!==old)s.changed++;}}
  for(const name of next.state_names){if(actualStats[name].unknown!==next.stats[name].unknown||actualStats[name].changed!==next.stats[name].changed_vs_unrestricted)throw Error('地图标签统计校验失败');}
  if(geometry)geometry.dispose();geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('aLabels',new THREE.BufferAttribute(labels,4));metadata=next;
  const width=Math.max(...Object.keys(next.id_colors).map(Number))+1;pixels=new Uint8Array(width*4);targetPixels=new Uint8Array(width*4);
  for(const [id,rgb] of Object.entries(next.id_colors))pixels.set([...rgb,255],Number(id)*4);
  if(palette)palette.dispose();if(targets)targets.dispose();palette=new THREE.DataTexture(pixels,width,1,THREE.RGBAFormat);targets=new THREE.DataTexture(targetPixels,width,1,THREE.RGBAFormat);
  for(const t of [palette,targets]){t.minFilter=t.magFilter=THREE.NearestFilter;t.needsUpdate=true;}
  for(const p of panels){if(p.cloud)p.scene.remove(p.cloud);if(p.material)p.material.dispose();p.material=new THREE.ShaderMaterial({...shader,uniforms:{
   uState:{value:1},uMode:{value:0},uWidth:{value:width},uPalette:{value:palette},uTargets:{value:targets},uUnknown:{value:1},uOthers:{value:1},uChangesOnly:{value:0},uSize:{value:2},uLocal:{value:0},uMin:{value:new THREE.Vector3()},uMax:{value:new THREE.Vector3()}}});
   p.cloud=new THREE.Points(geometry,p.material);p.cloud.frustumCulled=false;p.scene.add(p.cloud);}
  const select=el('caseSelect');select.replaceChildren(new Option('完整场景','full'));
  Object.entries(next.cases).sort((a,b)=>a[1].ROI.localeCompare(b[1].ROI)).forEach(([uid,row])=>select.add(new Option(row.ROI+' · f'+row.frame,uid)));
  const requested=params.get('roi');if(requested){const row=Object.entries(next.cases).find(([uid,row])=>row.ROI===requested);if(row)select.value=row[0];}
  el('downloads').replaceChildren();for(const name of ['gap0','gap1']){const a=document.createElement('a');a.href='models/'+sceneName+'_'+name+'_full_instance.ply';a.download=sceneName+'_'+name+'_full_instance.ply';a.textContent='下载完整彩色PLY：'+(name==='gap0'?'严格连续':'允许空1帧');el('downloads').append(a);}
  el('proof').textContent=`校验通过 · ${next.points.toLocaleString()} 个真实表面点 · 未抽样 · 四份标签共用原始几何 · ${next.gap0_gap1_array_equality.instance_id?'两档最终实例标签逐点相同':'两档最终实例标签存在差异'}。`;
  el('busy').hidden=true;legend();update();focus();
 }catch(error){el('busy').textContent=error.message;}}
el('scene').onchange=load;el('caseSelect').onchange=()=>{legend();update();focus();};
for(const id of ['reference','colorMode','unknown','others','changesOnly','pointSize'])el(id).addEventListener('input',update);
el('reset').onclick=focus;el('allIds').onclick=()=>{el('idLegend').querySelectorAll('input').forEach(input=>{input.checked=true;pixels[Number(input.dataset.pid)*4+3]=255;});palette.needsUpdate=true;draw();};
load();
