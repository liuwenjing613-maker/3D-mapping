from pathlib import Path
import json,gzip,hashlib,colorsys,re
import numpy as np
O=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101');T=O/'ablations/dbscan_history_seeded_20261102';OLD=O/'ablations/dbscan_identity_completion_guarded_20261102';V=O/'instance_comparison_viewer';W=T/'viewer';W.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
summary=read(T/'full8_comparison.json');assert summary['status']=='PASS' and summary['GT']==350
dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4')])
manifest=dict(status='PASS',scenes={},methods=['main','online_B'],labels={'main':'main','online_B':'v2-B'},arms=['guarded','history_seeded'],arm_labels={'guarded':'上一版 DBSCAN','history_seeded':'调整版：历史核心辅助'},pooled=summary['pooled'],GT=350,settings=summary['settings'])
for s in summary['scenes']:
 manifest['scenes'][s]={}
 for v in manifest['methods']:
  manifest['scenes'][s][v]={};base=T/'base'/v/s;families=read(base/'complete.json')['families'];original=load(O/'published'/v/s/'final_B/instance_surface.npz');xyz=original['xyz_m'];old=original['instance_id'];n=len(old)
  ply0=V/s/(v+'.ply')
  with ply0.open('rb') as f:head=f.read(4096);offset=head.index(b'end_header\n')+11
  raw=np.memmap(ply0,dtype=dtype,mode='r',offset=offset,shape=(n,));np.testing.assert_array_equal(raw['instance_id'],old);oldrgb=np.column_stack([raw[k] for k in ['red','green','blue']]);u,ix=np.unique(old,return_index=True);palette={int(i):oldrgb[j].copy() for i,j in zip(u,ix) if i>0}
  for arm,p in [('guarded',base/'merge_fill'),('history_seeded',T/v/s)]:
   r=read(p/'complete.json');after=load(p/'instance_surface.npz');np.testing.assert_array_equal(xyz,after['xyz_m']);np.testing.assert_array_equal(original['rgb'],after['rgb']);new=after['instance_id'];un,inv=np.unique(new,return_inverse=True)
   for iid in un[un>0]:
    if int(iid) not in palette:palette[int(iid)]=np.rint(np.array(colorsys.hsv_to_rgb((int(iid)*.61803398875)%1,.8,.95))*255).astype(np.uint8)
   rgb=np.array([palette[int(i)] if i>0 else [95,104,116] for i in un],np.uint8)[inv];folder=W/s/v/arm;folder.mkdir(parents=True,exist_ok=True)
   if not (folder/'original.ply').exists():(folder/'original.ply').symlink_to(ply0)
   ply=folder/'temporary.ply';arr=np.empty(n,dtype=dtype)
   for i,k in enumerate(['x','y','z']):arr[k]=xyz[:,i]
   for i,k in enumerate(['red','green','blue']):arr[k]=rgb[:,i]
   arr['instance_id']=new
   header=('ply\nformat binary_little_endian 1.0\ncomment temporary DBSCAN history-core completion; unchanged native vote >=2 >=67%\nelement vertex '+str(n)+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\nend_header\n').encode()
   if not ply.exists():
    with ply.open('wb') as f:f.write(header);arr.tofile(f)
   check=np.memmap(ply,dtype=dtype,mode='r',offset=len(header),shape=(n,));np.testing.assert_array_equal(check['instance_id'],new)
   for i,k in enumerate(['x','y','z']):np.testing.assert_array_equal(check[k],xyz[:,i])
   if not (folder/'surface.bin.gz').exists():
    with gzip.open(folder/'surface.bin.gz','wb',compresslevel=1) as f:
     for a in [np.array([n],'<u4'),xyz.astype('<f4'),oldrgb,rgb,old.astype('<i4'),new.astype('<i4')]:f.write(a.tobytes())
   manifest['scenes'][s][v][arm]=dict(count=n,bbox=[xyz.min(0).tolist(),xyz.max(0).tolist()],binary=f'{s}/{v}/{arm}/surface.bin',ply_original=f'{s}/{v}/{arm}/original.ply',ply_temporary=f'{s}/{v}/{arm}/temporary.ply',ply_sha256=sha(ply),old_instances=int((np.unique(old)>0).sum()),new_instances=int((un>0).sum()),old_gray=r['original_gray'],new_gray=r['temporary_gray'],old_metrics=r['original_metrics'],new_metrics=r['temporary_metrics'],families=families,changes={k:r[k] for k in ['gray_to_assigned','assigned_to_gray','assigned_changed_ID']},geometry_GT_protocol_equal=True,same_ID_same_color=True)
   print('PLY_PASS',s,v,arm,flush=True)
(W/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
html=(OLD/'viewer/index.html').read_text();html=html.replace('DBSCAN 临时对照 · 实例合并与灰点补全','DBSCAN 八场景验证 · 历史核心辅助补全').replace('历史支持逐实例聚类 → 有证据的 ID 合并与去重重投票 → 末尾 B 一次 → 局部灰点补全。','有证据的 ID 合并 → 原规则重投票 → 末尾 B 一次 → DBSCAN 补灰＋历史修复核心恢复。').replace('<details open>','<details>')
html=re.sub(r'<select id="arm".*?</select>','<select id="arm" aria-label="处理方式"><option value="history_seeded">调整版：历史核心辅助</option><option value="guarded">上一版 DBSCAN</option></select>',html)
html=html.replace('<button id="reset">','<label>局部放大<select id="roi" aria-label="局部放大"><option value="">全景</option><option value="sidebook">room0 边桌书本</option><option value="coffeebook">room0 茶几书本</option></select></label><button id="reset">')
html=html.replace('<a href="./first_trial.json">第一版完整指标（含退化）</a><a href="./conservative_trial.json">保守模式指标</a>','<a href="../room0-books/">书本根因与逐点审计</a><a href="./final_audit.json">八场景一致性审计</a>')
html=html.replace('覆盖更多模式仍有边界错分；保守模式补点较少。','部分书面仍分配为桌子，新增方案不强制合并书桌；残余错误按八场景评价报告。').replace('viewer.js?v=3','viewer.js?v=1')
(W/'index.html').write_text(html)
js=(OLD/'viewer/viewer.js').read_text();js=js.replace("同2场景 pooled","同8场景 pooled").replace("arm==='original'?'merge_fill':arm","arm==='original'?'history_seeded':arm").replace("'office1 + room0 · '","'完整8场景 · '").replace("局部 DBSCAN：5 cm / min_samples 10 · 临时开发对照","DBSCAN + 已提交历史修复核心 · 临时开发对照")
start=js.index('function displayColors(');end=js.index('async function load()',start)
js=js[:start]+'''function displayColors(focus=false){if(!current)return;const family=$('family').value?new Set($('family').value.split(',').map(Number)):null,roi=$('scene').value==='room0'?$('roi').value:'',rb=roi==='sidebook'?[[4.95,2.1,-.97],[5.50,2.85,-.72]]:roi==='coffeebook'?[[3.25,.75,-1.08],[3.8,1.28,-.92]]:null,highlight=$('highlight').checked,{n,positions,oldIDs,newIDs,colors}=current,selectedIndex=[],box=new THREE.Box3();
for(let i=0;i<n;i++){const p=[positions[3*i],positions[3*i+1],positions[3*i+2]];if((!family||family.has(oldIDs[i])||family.has(newIDs[i]))&&(!rb||p.every((x,j)=>x>=rb[0][j]&&x<=rb[1][j]))){selectedIndex.push(i);if(family||rb)box.expandByPoint(new THREE.Vector3(...p))}}
for(let side=0;side<2;side++){const rgb=new Uint8Array(colors[side]);if(highlight)for(let i=0;i<n;i++){if(oldIDs[i]===newIDs[i]){rgb[3*i]*=.12;rgb[3*i+1]*=.12;rgb[3*i+2]*=.12}else rgb.set(oldIDs[i]<0?[79,238,216]:newIDs[i]<0?[255,100,110]:[255,172,87],3*i)}meshes[side].geometry.setAttribute('color',new THREE.BufferAttribute(rgb,3,true));meshes[side].geometry.setIndex(family||rb?new THREE.BufferAttribute(new Uint32Array(selectedIndex),1):null)}
if(focus){bounds=(family||rb)&&selectedIndex.length?box:wholeBounds;fit()}render()}
''' +js[end:]
js=js.replace("history.replaceState(null,'','?scene='+sc+'&method='+method+'&arm='+arm)","history.replaceState(null,'','?scene='+sc+'&method='+method+'&arm='+arm+'&roi='+$('roi').value)")
js=js.replace('setClip();resize();displayColors();fit();','setClip();resize();displayColors(true);')
js=js.replace("const q=new URLSearchParams(location.search);","const q=new URLSearchParams(location.search);if(['sidebook','coffeebook'].includes(q.get('roi')))$('roi').value=q.get('roi');")
js=js.replace("$('family').value='';displayColors(true)","$('family').value='';$('roi').value='';displayColors(true)")
js+="\n$('roi').addEventListener('change',()=>{history.replaceState(null,'','?scene='+$('scene').value+'&method='+$('method').value+'&arm='+$('arm').value+'&roi='+$('roi').value);displayColors(true)});\n"
(W/'viewer.js').write_text(js)
for name,target in [('vendor',V/'vendor'),('comparison.json',T/'full8_comparison.json'),('final_audit.json',T/'final_audit.json')]:
 if not (W/name).exists():(W/name).symlink_to(target,target_is_directory=name=='vendor')
if not (V/'dbscan-history').exists():(V/'dbscan-history').symlink_to(W,target_is_directory=True)
(T/'viewer_complete.json').write_text(json.dumps(dict(status='PASS',pairs=32,full_resolution=True,coordinates_and_ids_verified=True,same_ID_same_color=True)))
print('HISTORY_VIEWER_PASS',flush=True)
