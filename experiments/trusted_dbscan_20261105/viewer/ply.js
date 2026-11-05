
const THREE=await import('three');
const {OrbitControls}=await import('three/addons/controls/OrbitControls.js');
const $=x=>document.getElementById(x),root='./trusted_dbscan/ply_view/', panes=[];
let manifest,current,sceneData,gen=0,spinning=false,loaded=false,files=[],activeView;
const topPlane=new THREE.Plane(new THREE.Vector3(0,0,-1),1e6);
const camera=new THREE.PerspectiveCamera(45,1,.001,500);camera.up.set(0,0,1);
for(const name of ['left','right']){
 const el=$(name),renderer=new THREE.WebGLRenderer({antialias:true});renderer.localClippingEnabled=true;renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#0b1420');el.append(renderer.domElement);
 const scene=new THREE.Scene(),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;
 const cloud=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({size:3,sizeAttenuation:false,vertexColors:true}));
 const gt=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({size:2,sizeAttenuation:false,color:0xffffff,transparent:true,opacity:.65,depthTest:false}));
 cloud.material.clippingPlanes=[topPlane];gt.material.clippingPlanes=[topPlane];scene.add(cloud,gt);const p={el,renderer,scene,controls,cloud,gt,box:null,data:null};panes.push(p);
 controls.addEventListener('change',()=>{for(const q of panes)if(q!==p)q.controls.target.copy(controls.target)});
 controls.addEventListener('start',()=>{spinning=false;$('spin').textContent='▶ 自动旋转'});
}
let last=performance.now();
function frame(now){
 requestAnimationFrame(frame);const dt=Math.min((now-last)/1000,.04);last=now;
 if(spinning&&loaded){const c=panes[0].controls.target;camera.position.sub(c).applyAxisAngle(new THREE.Vector3(0,0,1),dt*.22).add(c);camera.lookAt(c)}
 for(const p of panes){const w=p.el.clientWidth,h=p.el.clientHeight;if(p.renderer.domElement.width!==Math.round(w*p.renderer.getPixelRatio())||p.renderer.domElement.height!==Math.round(h*p.renderer.getPixelRatio()))p.renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();p.renderer.render(p.scene,camera)}
}requestAnimationFrame(frame);
async function readPly(path){
 const r=await fetch(path);if(!r.ok)throw Error('点云读取失败 '+r.status+' '+path);
 const b=await r.arrayBuffer(),u=new Uint8Array(b),head=new TextDecoder().decode(u.subarray(0,1500)),end=head.indexOf('end_header\n')+11;
 if(end<11||!head.includes('binary_little_endian'))throw Error('PLY 格式不匹配');
 const n=Number(head.match(/element vertex (\d+)/)[1]);if(b.byteLength!==end+n*22)throw Error('PLY 点数与文件长度不符');
 const d=new DataView(b),pos=new Float32Array(n*3),color=new Uint8Array(n*3),rgb=new Uint8Array(n*3),ids=new Int32Array(n);
 for(let i=0;i<n;i++){let off=end+i*22;for(let k=0;k<3;k++){pos[i*3+k]=d.getFloat32(off+k*4,true);color[i*3+k]=u[off+12+k];rgb[i*3+k]=u[off+15+k]}ids[i]=d.getInt32(off+18,true)}
 return {pos,color,rgb,ids,n};
}
const linear=Float32Array.from({length:256},(_,i)=>{const c=i/255;return c<=.04045?c/12.92:Math.pow((c+.055)/1.055,2.4)});
function recolor(){
 for(let arm=0;arm<2;arm++){const p=panes[arm],d=p.data;if(!d)continue;const out=new Float32Array(d.n*3),src=$('color').value==='rgb'?d.rgb:d.color,sel=new Set(current?.picked?.[arm]||[]),fade=$('focus').checked&&!!current;
 for(let i=0;i<d.n;i++){const bg=fade&&!sel.has(d.ids[i]);for(let k=0;k<3;k++)out[i*3+k]=bg?linear[d.rgb[i*3+k]]*.14:linear[src[i*3+k]]}
 p.cloud.geometry.setAttribute('color',new THREE.BufferAttribute(out,3));p.cloud.material.size=Number($('size').value);p.gt.visible=!!current&&$('gt').checked;if(p.box)p.box.visible=$('box').checked;
 }}
function reset(top=false){
 if(!activeView)return;const c=new THREE.Vector3(...activeView.center),e=Math.max(...activeView.extent,.08),dir=top?new THREE.Vector3(.001,-.001,1):new THREE.Vector3(...(current?.direction||[1,-1,.8])).normalize();
 const aspect=panes[0].el.clientWidth/panes[0].el.clientHeight,dist=e*1.35/Math.min(1,aspect);
 camera.up.set(0,top?1:0,top?0:1);camera.position.copy(c).addScaledVector(dir,dist);camera.near=Math.max(.0005,e/10000);camera.far=Math.max(100,e*100);
 for(const p of panes){p.controls.target.copy(c);p.controls.update()}camera.lookAt(c);
}
function story(){
 if(!current)return `${sceneData.scene} 完整场景。两边同一固定几何，颜色表示各自真实原生实例。可切换到局部案例查看退化细节。`;
 const s=current.scene,g=current.gt,base=`${s} · GT ${g} ${current.name}：IoU ${current.iou[0].toFixed(4)} → ${current.iou[1].toFixed(4)}。`;
 const stories={'room0_92':'旧版历史核心补全获益更大，书本覆盖更完整；本次后处理仍改善了书本，但未达到旧版。','room0_30':'可信策略结果仍缺少一大块百叶窗覆盖，后处理只带来小幅补灰。','room0_85':'书本仍混在其他实例中；切换“关联 ID 的完整范围”检查混并。','room0_58':'部分缺失仍在；补灰扩大预测范围，主实例 IoU 略降。','room0_4':'书本的混并没有被后处理拆开。','office0_9':'右侧沙发仍分成 ID79 与 ID572；两种颜色是真实分裂，未强行统一。','office0_22':'植物小幅改善但仍在 IoU 0.5 以下。','office1_18':'瓶子仍混在 ID9，缺失的独立身份没有重建。','room2_38':'改善案例，用于对照观察可信策略保留的实例区域。'};
 return base+(stories[`${s}_${g}`]||'');
}
async function choose(){
 const ticket=++gen;loaded=false;spinning=false;$('spin').textContent='▶ 自动旋转';$('error').textContent='';$('loading').textContent='正在从服务器加载两侧 PLY…';
 for(const p of panes){p.cloud.visible=false;p.gt.visible=false;if(p.box)p.box.visible=false}
 for(let a=0;a<2;a++){$('metric'+a).textContent='—';$('legend'+a).textContent='';$('ply'+a).removeAttribute('href')}$('story').textContent='正在切换案例…';$('counts').textContent='';
 try{
 const key=$('case').value;current=manifest.cases.find(c=>c.key===key)||null;sceneData=manifest.scenes.find(s=>s.scene===(current?current.scene:key.replace('scene_','')));
 $('scope').disabled=!current;for(const id of ['focus','box','gt'])$(id).disabled=!current;
 const mode=$('scope').value,pre=root+sceneData.scene+'/';files=['old','trusted'].map(a=>pre+(current?`${current.gt}_${mode}_${a}`:a)+'.ply');
 const data=await Promise.all(files.map(readPly)),gtdata=current?await readPly(pre+current.gt+'_gt.ply'):null;if(ticket!==gen)return;
 for(let arm=0;arm<2;arm++){
  const p=panes[arm];p.data=data[arm];p.cloud.geometry.dispose();p.cloud.geometry=new THREE.BufferGeometry();p.cloud.geometry.setAttribute('position',new THREE.BufferAttribute(data[arm].pos,3));p.cloud.visible=true;
  p.gt.geometry.dispose();p.gt.geometry=new THREE.BufferGeometry();if(gtdata)p.gt.geometry.setAttribute('position',new THREE.BufferAttribute(gtdata.pos,3));
  if(p.box){p.scene.remove(p.box);p.box.geometry.dispose();p.box.material.dispose();p.box=null}
  if(current){p.box=new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(...current.bbox[0]),new THREE.Vector3(...current.bbox[1])),0xffffff);p.box.material.clippingPlanes=[topPlane];p.box.material.transparent=true;p.box.material.opacity=.45;p.scene.add(p.box)}
  $('metric'+arm).textContent=current?'完整 R2 · IoU '+current.iou[arm].toFixed(4):sceneData.scene+' · 同一最终 400 帧几何';
  $('legend'+arm).innerHTML=current?current.legend[arm].map(x=>'<span><i class="dot" style="background:rgb('+x.color.join(',')+')"></i>ID '+x.id+' · 命中 '+x.intersection+'</span>').join(''):'颜色为原生实例；灰色为未分配';
  $('ply'+arm).href=files[arm];$('ply'+arm).setAttribute('download','');
 }
 activeView=current?(mode==='local'?current.modes.local.view:current.modes.involved.view):sceneData.view;$('clipTop').value='0';applyClip();recolor();reset();loaded=true;
 $('story').textContent=story();$('story').className='note '+(current&&current.delta<0?'bad':'good');
 const count=current?current.modes[mode]:{sampled:sceneData.sampled_points,total:sceneData.total_points};
 $('counts').textContent='每侧显示 '+count.sampled.toLocaleString()+' / '+count.total.toLocaleString()+' 点 · 两侧相同抽样';
 $('loading').textContent='✓ 两侧 PLY 已加载 · 视角同步';document.title=(current?current.scene+' GT'+current.gt:sceneData.scene)+' · 实例 PLY 对比';
 }catch(e){if(ticket===gen){$('error').textContent=String(e);$('loading').textContent='加载未完成'}}
}
function applyClip(){if(!activeView)return;const percent=Number($('clipTop').value),lo=activeView.center[2]-activeView.extent[2]/2,hi=activeView.center[2]+activeView.extent[2]/2;topPlane.constant=percent===0?hi+.01:hi-(hi-lo)*percent/100;$('clipLabel').textContent=percent===0?'保留全部':'裁去上部 '+percent+'% · 高度 ≤ '+topPlane.constant.toFixed(2)+' m';}
$('clipTop').oninput=applyClip;$('clipReset').onclick=()=>{$('clipTop').value='0';applyClip()};
$('case').onchange=choose;$('scope').onchange=choose;for(const i of ['color','focus','box','gt','size'])$(i).oninput=recolor;
$('reset').onclick=()=>reset();$('top').onclick=()=>reset(true);$('spin').onclick=()=>{spinning=!spinning;$('spin').textContent=spinning?'❚❚ 暂停旋转':'▶ 自动旋转'};
$('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else $('container').requestFullscreen()};
try{
 const r=await fetch(root+'manifest.json');manifest=await r.json();if(manifest.status!=='PASS')throw Error('PLY 尚在生成，请稍后刷新');
 $('case').innerHTML='<optgroup label="实例对比">'+manifest.cases.map(c=>'<option value="'+c.key+'">'+c.scene+' · GT'+c.gt+' '+c.name+(c.delta<0?' ↓':' ↑')+'</option>').join('')+'</optgroup><optgroup label="完整场景">'+manifest.scenes.map(s=>'<option value="scene_'+s.scene+'">'+s.scene+' · 完整场景</option>').join('')+'</optgroup>';
 $('case').value='room0_92';await choose();
}catch(e){$('error').textContent=String(e)}
