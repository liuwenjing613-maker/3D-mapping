from common100 import *
import numpy as np,collections
from PIL import Image
from scipy.ndimage import binary_fill_holes
def holes(mask):
 filled=binary_fill_holes(mask);n=int(mask.sum());full=int(filled.sum())
 yy,xx=np.where(mask)
 return dict(foreground_pixels=n,hole_pixels=full-n,filled_pixels=full,hole_fraction=(full-n)/full if full else None,
  bbox_fill=n/((int(xx.max()-xx.min())+1)*(int(yy.max()-yy.min())+1)) if n else None)
def main():
 tick=time.perf_counter();summary=read(R/'gate_complete.json');seeds=read(R/'seed_id_map.json')['objects']
 cases=read(R/'seed_diagnostics.json')['cases']
 vis=R/'visuals';vis.mkdir(exist_ok=True)
 for case in cases:
  uid=case['case_uid'];folder=vis/uid;folder.mkdir(exist_ok=True)
  for fid in range(496):
   src=R/'cases'/uid/'tracking'/f'f{fid:06d}.png';labels=np.asarray(Image.open(src),np.uint16)
   assert int(labels.max())<256
   Image.fromarray(labels.astype(np.uint8)).save(folder/f'f{fid:06d}.png')
 gatefolder=vis/'gate';gatefolder.mkdir(exist_ok=True)
 for fid in FIDS:
  labels=np.asarray(Image.open(R/'gate/accepted_masks'/f'f{fid:06d}.png'),np.uint16)
  assert int(labels.max())<256;Image.fromarray(labels.astype(np.uint8)).save(gatefolder/f'f{fid:06d}.png')
 details={};seedholes={}
 for item in seeds:
  tid=item['track_id'];p=item['source'];mask=np.asarray(Image.open(S/p['mask']))>0
  seedholes[str(tid)]=holes(mask)
  row=summary['tracks'][str(tid)]
  relevant=row['persistent_id']==91 or row['old_family_id']==91 or row['seed_support_ID91_points']/max(1,row['seed_projected_points'])>=.5
  if not relevant:continue
  uid=row['case_uid'];history=[]
  for fid in range(496):
   labels=np.asarray(Image.open(R/'cases'/uid/'tracking'/f'f{fid:06d}.png'),np.uint16)
   history.append(dict(frame=fid,**holes(labels==tid)))
  fid=row['source_frame'];np.testing.assert_array_equal(np.asarray(Image.open(R/'cases'/uid/'tracking'/f'f{fid:06d}.png'))==tid,mask)
  nonempty=[r for r in history if r['foreground_pixels']]
  details[str(tid)]=dict(track_id=tid,group_id=row['group_id'],mask_id=row['mask_id'],source_frame=fid,seed_exact=True,seed=seedholes[str(tid)],frames=history,
   nonempty_frames=len(nonempty),frames_with_closed_holes=sum(r['hole_pixels']>0 for r in nonempty),
   min_hole_fraction=min(r['hole_fraction'] for r in nonempty) if nonempty else None,
   max_hole_fraction=max(r['hole_fraction'] for r in nonempty) if nonempty else None,
   nearby_frames=[history[i] for i in sorted(set(max(0,min(495,fid+d)) for d in [-10,-5,-1,0,1,5,10]))],
   note='Closed binary holes only. Changing view, occlusion and open contours can alter this ratio; no mask filling was used for inference or gate.')
 dump(R/'ID91_holes.json',dict(status='PASS',group=GROUP,tracks=details,diagnostic_only=True,seed_masks_modified=False))
 manifest=dict(status='PASS',group=GROUP,summary=summary,seed_holes=seedholes,ID91_hole_tracks=list(details),raw_frames=496,
  input_range='1–100 建图帧' if GROUP=='A' else '51–100 建图帧',tracking_range='原始帧 0–495 双向，496 帧；门控覆盖全部100建图帧',
  raw_rgb_root='../inputs/scene/results/',source_seed_range=[0,495] if GROUP=='A' else [250,495],repair100_applied=False)
 dump(R/'web_manifest.json',manifest)
 html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>100帧追踪对照</title><style>body{font:15px/1.6 system-ui,sans-serif;margin:0;background:#f1f4fa;color:#17243a}main{max-width:1300px;margin:auto;padding:24px}h1{font-size:25px;margin:0}a{color:#2258ab}.bar,.card{background:white;padding:16px;margin:15px 0;border-radius:10px}.status{background:#e0f3e9;padding:12px;border-left:5px solid #21835e}.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap}select,button,input{font:inherit}select,button{padding:6px 10px}input[type=range]{min-width:280px;flex:1}.panels{display:grid;grid-template-columns:1fr 1fr;gap:16px}canvas{width:100%;height:auto;background:#ddd}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px 10px;border-bottom:1px solid #dde3ed;text-align:left;font-variant-numeric:tabular-nums}tr.active{background:#e5edff}.low td{background:#fff2d9}.notice{background:#fff2d9;padding:12px;border-left:5px solid #b27811}.badge{font-size:12px;color:#82530c}td button{border:0;background:transparent;color:#215aae;cursor:pointer}.muted{color:#647189;font-size:13px}.scroll{overflow:auto}.metric{font-variant-numeric:tabular-nums}#holeChart{width:100%;height:90px;border-bottom:1px solid #bbb}#error{color:#a21d1d}@media(max-width:760px){main{padding:14px}.panels{grid-template-columns:1fr}input[type=range]{min-width:180px}}</style>
<main><p><a href="../../compare100/">返回 A/B 对照</a> · <a href="gate_complete.json">门控总结</a> · <a href="ID91_holes.json">ID91孔洞诊断</a></p>
<h1 id="title">100帧追踪对照</h1><p id="description"></p><div class="status">仅生成追踪预测与原门控评估。第一次50帧修复已用于建图，第二次100帧修复未执行；未改票、未提交地图状态。</div>
<div class="card"><div class="controls"><label>目标 <select id="target"></select></label><button id="seed">回到原始种子帧</button><button id="prev">−5帧</button><button id="next">+5帧</button><label><input type="checkbox" id="crop" checked>局部放大</label><select id="mode"><option value="tracking">原追踪预测</option><option value="gate">通过门控后</option></select></div><div class="controls"><label>原始帧 <input id="number" type="number" min="0" max="495" value="0" style="width:75px"></label><input type="range" id="frame" min="0" max="495" value="0" step="1"></div>
<p id="metrics" class="metric"></p><div class="panels"><div><canvas id="overlay"></canvas><div class="muted">真实 RGB + 当前目标蓝色覆盖</div></div><div><canvas id="mask"></canvas><div class="muted" id="maskCaption">当前目标原二值预测；白色前景，黑色背景与孔洞</div></div></div><p id="gateNote"></p><p id="conflicts" class="muted"></p><p id="error"></p></div>
<div class="card"><h2>ID91 窗户重复候选：内部孔洞</h2><p id="holeText">选择ID91相关种子查看孔洞记录。</p><p id="windowGateNote"></p><svg id="holeChart" viewBox="0 0 1000 90" preserveAspectRatio="none"></svg><p class="muted">孔洞比例=封闭内部背景像素÷填孔后面积。填孔只用于统计，没有改动种子或预测。视角、遮挡和轮廓开口会影响该值，不能把跨视角比例下降单独当作质量提升。橙线为原始种子帧。</p></div>
<div class="notice"><strong>通过帧数不是准确率。</strong>不同目标的可见范围不同，不能用 A/B 总体通过比例直接判断优劣。初始Gate与冲突后Gate的差值反映跨案例重叠弃权的影响，不能直接归因于追踪失败。<span id="lowGate"></span>下列关联只是假设执行修复时的建议，当前100帧地图没有提交这些ID变更。</div><div class="scroll"><table><thead><tr><th>种子</th><th>来源原始帧</th><th>拟关联ID（未提交）</th><th>非空追踪/496</th><th>初始Gate/100</th><th>冲突后Gate/100</th><th>拒绝原因</th></tr></thead><tbody id="rows"></tbody></table></div><p id="timing" class="muted"></p></main>
<script>
const $=id=>document.getElementById(id);let data,decisions,holeData,association,track,revision=0;const assocByTrack={};
const pad=n=>String(n).padStart(6,'0');const img=src=>new Promise((resolve,reject)=>{let i=new Image;i.onload=()=>resolve(i);i.onerror=()=>reject(Error('读取失败 '+src));i.src=src});
function select(tid){$('target').value=tid;track=data.summary.tracks[tid];$('frame').value=track.source_frame;$('number').value=track.source_frame;draw();}
function num(v){return (v*100).toFixed(1)+'%'}
async function draw(){const rev=++revision;try{
 const tid=+$('target').value;track=data.summary.tracks[tid];let fid=+$('frame').value;fid=Math.max(0,Math.min(495,fid));$('number').value=fid;$('error').textContent='';
 const gate=$('mode').value==='gate',mapping=fid%5===0;const gateRow=decisions.frames.find(x=>x.frame===fid);const g=gateRow?.objects[tid];const g0=gateRow?.initial_objects[tid];const accepted=gateRow?.accepted_track_ids.includes(tid);
 $('maskCaption').textContent=gate&&mapping?'通过门控后的当前目标二值结果；已含跨案例重叠弃权':'当前目标原二值预测；白色前景，黑色背景与孔洞';$('gateNote').textContent=mapping?('初始Gate：'+(g0.accepted?'通过':'拒绝')+'，锚点覆盖 '+num(g0.tracked_coverage)+'；跨案例重叠弃权后：'+(accepted?'通过':'拒绝')+'，锚点覆盖 '+num(g.tracked_coverage)+'。可见锚点 '+g.visible_seed_samples+'/'+g.seed_anchor_count+'，bbox内支持 '+num(g.new_surface_bbox_fraction)+'。'+(g.reasons.length?'最终拒绝原因：'+g.reasons.join(', '):'')):'该原始帧参与稠密视频追踪与记忆更新，但不属于stride=5建图帧，因此不执行门控。';
 if(gate&&!mapping){$('gateNote').textContent+=' 此处显示原追踪预测。';}
 const partners=(track.overlap_partners||[]).slice(0,3);$('conflicts').textContent=track.initial_accepted_mapping_frames===undefined?'':'本目标初始Gate通过 '+track.initial_accepted_mapping_frames+'/100 帧，冲突后 '+track.accepted_mapping_frames+'/100 帧，因交叠掉出 '+track.overlap_abstention_dropped_frames+' 帧。'+(partners.length?'主要共同覆盖对象（掉出帧上的二维重叠累计，非3D点）：'+partners.map(p=>p.other_group_id+' '+p.overlap_pixels_sum_on_target_dropped_frames.toLocaleString()+'像素').join('；')+'。':'');const windowGate=data.summary.window91_observation_gate?.[tid];$('windowGateNote').textContent=windowGate?'本案例旧身份集合 ['+(windowGate.case_old_family_ids||[]).join(', ')+']。原门控只要求案例中任一旧身份有观测。'+(windowGate.missing_old_family_rejection_frames.length?'原始帧 '+windowGate.missing_old_family_rejection_frames.join('、')+' 因案例旧观测缺失被直接Gate拒绝'+(windowGate.exact_frame_match?'，与前端整观测错关联的5帧完全对应。':'。'):'没有帧因案例旧观测缺失而拒绝；这不等于ID91在每帧都存在旧观测。')+'不同联合分组可能改变这个前提，不能把通过数提高当成质量提升。完整partial＋E1能否弥补仍未验证；100帧实际修复尚未执行。':'';
 const maskPath=(gate&&mapping?'visuals/gate/':'visuals/'+track.case_uid+'/')+'f'+pad(fid)+'.png';
 const [raw,labels]=await Promise.all([img(data.raw_rgb_root+'frame'+pad(fid)+'.jpg'),img(maskPath)]);if(rev!==revision)return;
 const scratch=document.createElement('canvas');scratch.width=1200;scratch.height=680;const ctx=scratch.getContext('2d',{willReadFrequently:true});ctx.drawImage(labels,0,0);const d=ctx.getImageData(0,0,1200,680);let minX=1200,minY=680,maxX=-1,maxY=-1,n=0;
 const maskImage=new ImageData(1200,680),overlay=new ImageData(1200,680);
 for(let y=0;y<680;y++)for(let x=0;x<1200;x++){let i=(y*1200+x)*4,on=d.data[i]===tid;maskImage.data[i]=maskImage.data[i+1]=maskImage.data[i+2]=on?255:0;maskImage.data[i+3]=255;if(on){n++;minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);overlay.data[i]=15;overlay.data[i+1]=90;overlay.data[i+2]=255;overlay.data[i+3]=160;}}
 let box=[0,0,1200,680];if($('crop').checked&&n){let margin=35;box=[Math.max(0,minX-margin),Math.max(0,minY-margin),Math.min(1200,maxX+margin+1),Math.min(680,maxY+margin+1)];}
 const [x,y,x1,y1]=box,w=x1-x,h=y1-y;for(let id of ['overlay','mask']){$(id).width=w;$(id).height=h}
 ctx.putImageData(overlay,0,0);let o=$('overlay').getContext('2d');o.drawImage(raw,x,y,w,h,0,0,w,h);o.drawImage(scratch,x,y,w,h,0,0,w,h);
 ctx.putImageData(maskImage,0,0);$('mask').getContext('2d').drawImage(scratch,x,y,w,h,0,0,w,h);
 $('metrics').textContent=track.group_id+' / '+track.mask_id+' · 原始帧 '+fid+' · 来源帧 '+track.source_frame+' · 当前预测前景 '+n.toLocaleString()+' 像素'+(fid===track.source_frame?(gate&&mapping?' · 来源帧，门控输出含冲突弃权':' · 原种子逐像素保持'):'');
 const holes=holeData.tracks[tid];if(holes){const now=holes.frames[fid];$('holeText').textContent='ID91相关种子 '+track.group_id+'：原种子内部孔洞 '+holes.seed.hole_pixels.toLocaleString()+' 像素（'+num(holes.seed.hole_fraction||0)+'）；当前帧 '+now.hole_pixels.toLocaleString()+' 像素（'+(now.hole_fraction===null?'无前景':num(now.hole_fraction))+'）。来源帧预测与原种子逐像素一致。';
 const points=holes.frames.map(r=>(r.frame/495*1000).toFixed(1)+','+(85-(r.hole_fraction||0)*80).toFixed(1)).join(' '),seed=holes.source_frame/495*1000,cur=fid/495*1000;
 $('holeChart').innerHTML='<polyline fill="none" stroke="#225cdd" stroke-width="2" points="'+points+'"/><line x1="'+seed+'" x2="'+seed+'" y1="0" y2="90" stroke="#e49b18" stroke-width="2"/><line x1="'+cur+'" x2="'+cur+'" y1="0" y2="90" stroke="#6a7180" stroke-dasharray="4 4"/>';
 }else{$('holeText').textContent='当前目标不是主要ID91重复候选。原种子孔洞 '+data.seed_holes[tid].hole_pixels+' 像素。';$('holeChart').innerHTML='';}
 }catch(e){$('error').textContent=e.message}}
(async()=>{[data,decisions,holeData,association]=await Promise.all(['web_manifest.json','gate/frame_decisions.json','ID91_holes.json','global_association.json'].map(x=>fetch(x).then(r=>r.json())));
 document.title=data.group+'组 · 100帧追踪对照';for(const row of association.objects)for(const tid of row.member_track_ids)assocByTrack[tid]=row;const low=Object.values(data.summary.tracks).filter(row=>row.accepted_mapping_frames<=10);$('lowGate').textContent=low.length?' 本组需逐帧检查：'+low.map(row=>row.group_id+' 仅 '+row.accepted_mapping_frames+'/100 帧通过').join('，')+'；此处提醒不改变筛选或门控。':'';$('title').textContent=data.group+'组 · 100帧原种子追踪与门控';$('description').textContent='候选来源：'+data.input_range+'。'+data.tracking_range+'。共'+data.summary.objects+'个种子、'+data.summary.cases+'个联合追踪案例。';
 for(const [tid,row] of Object.entries(data.summary.tracks)){let o=document.createElement('option');o.value=tid;const idKind=assocByTrack[tid]?.association==='distinct_new_identity'?'新ID建议':'沿用已有ID';o.textContent=row.group_id+' / '+row.mask_id+' · 拟ID '+row.persistent_id+'（'+idKind+'，未提交）';$('target').appendChild(o);let tr=document.createElement('tr');if(row.accepted_mapping_frames<=10)tr.className='low';tr.innerHTML='<td><button>'+row.group_id+' / '+row.mask_id+'</button></td><td>'+row.source_frame+'</td><td>'+row.persistent_id+' <span class="badge">'+idKind+' · 未提交</span></td><td>'+row.raw_nonempty_frames+'</td><td>'+decisions.frames.filter(f=>f.initial_objects[tid].accepted).length+'</td><td>'+row.accepted_mapping_frames+'</td><td>'+Object.entries(row.rejection_reasons).map(([k,n])=>k+': '+n).join('; ')+'</td>';tr.querySelector('button').onclick=()=>select(tid);$('rows').appendChild(tr);}
 $('timing').textContent='完整校验/关联/Gate入口 '+data.summary.seconds.toFixed(2)+'秒，其中门控本体 '+data.summary.gate_seconds.toFixed(2)+'秒（内含子项，不相加）。导出和网页展示另计；追踪队列耗时见A/B对照总表。';
 $('target').onchange=()=>select($('target').value);$('frame').oninput=draw;$('number').onchange=()=>{$('frame').value=$('number').value;draw()};$('seed').onclick=()=>select($('target').value);$('prev').onclick=()=>{$('frame').value=Math.max(0,+$('frame').value-5);draw()};$('next').onclick=()=>{$('frame').value=Math.min(495,+$('frame').value+5);draw()};$('crop').onchange=draw;$('mode').onchange=draw;
 const query=new URLSearchParams(location.search),requested=query.get('target');select(requested&&data.summary.tracks[requested]?requested:(data.ID91_hole_tracks[0]||Object.keys(data.summary.tracks)[0]));if(query.has('frame')){$('frame').value=Math.max(0,Math.min(495,Number(query.get('frame'))||0));draw();}})().catch(e=>$('error').textContent=e.message);
</script></html>'''
 (R/'index.html').write_text(html,encoding='utf-8')
 web=TOP/'review/tracking100'
 if not web.exists():web.symlink_to(TRACKROOT,target_is_directory=True)
 else:assert web.resolve()==TRACKROOT.resolve()
 dump(R/'render_complete.json',dict(status='PASS',group=GROUP,seconds=time.perf_counter()-tick,ID91_tracks=list(details),previews=len(cases)*496+100,
  page_url=f'http://127.0.0.1:8798/online-repair/tracking100/{GROUP}/',inference_masks_modified=False))
 print(json.dumps(dict(status='PASS',group=GROUP,seconds=time.perf_counter()-tick,ID91_tracks=list(details))),flush=True)
if __name__=='__main__':main()
