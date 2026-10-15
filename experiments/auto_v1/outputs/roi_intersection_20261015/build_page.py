from pathlib import Path
import json,re,os,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006';OLD=ROOT/'room2_E1_retained_ROI_top20_20261015';OUT=ROOT/'room2_E1_intersection_ROI_20261015'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reb(p):
 q=(OLD/p).resolve();assert q.is_file();return os.path.relpath(q,OUT).replace('\\','/')
d=read(OLD/'viewer_data.json');p=read(OLD/'projection_annotations.json');r=read(OUT/'intersection.json');s=read(OLD/'selection_setup.json');before={n:sha(OLD/n) for n in ['index.html','viewer_data.json','selection_setup.json']}
assets=set()
for c in d['cases']:
 c['expansion']=r['cases'][c['uid']]
 for v in c['views']:
  for field in ['full_rgb','alarm']:v[field]=reb(v[field]);assets.add(v[field])
  f=r['frames'][c['uid']+':'+str(v['frame'])];v['expansion']=f['expansion'];assets.add(v['expansion'])
for key,crop in d['crops'].items():
 ir=r['crops'][key];crop['intersection_frame']=r['frames'][ir['frame_key']]
 for field in ['RGB','comparison']:
  if field in crop:crop[field]=reb(crop[field]);assets.add(crop[field])
 for mi,m in enumerate(crop['models']):
  for field in ['color','boundary','ids_file']:m[field]=reb(m[field]);assets.add(m[field])
  m['ids']=m['ids_file'];m['relevance']=ir['models'][mi]
  assert set(map(int,m['colors']))==set(m['relevance']['eligible_ids']+m['relevance']['no_intersection_ids'])
  if 'removed_source' in m:m['removed_source']['ids_file']=reb(m['removed_source']['ids_file']);assets.add(m['removed_source']['ids_file'])
for v in p['views'].values():v['alarm_asset']=reb(v['alarm_asset'])
d['name']='room2-e1-intersection-roi-20261015';d['intersection_filter']=dict(rule=r['rule'],stats=r['stats'],source_sha256=sha(OUT/'intersection.json'))
save(OUT/'viewer_data.json',d);save(OUT/'projection_annotations.json',p)
html=(OLD/'index.html').read_text(encoding='utf-8')
for tag,value in [('reviewData',d),('projectionData',p)]:
 encoded=json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
 html,n=re.subn(r'(<script id="'+tag+r'" type="application/json">).*?(</script>)',lambda m:m[1]+encoded+m[2],html,flags=re.S);assert n==1
def change(a,b):
 global html
 assert a in html,a[:100];html=html.replace(a,b)
change('room2 · 筛选后 ROI 质量检查','room2 · 异常点扩展与相交候选检查')
change('三类各前 20 · 60 个入榜名额 / 32 个 ROI · 拟剔除 360 个候选','三类各前 20 · 32 个 ROI · 三维扩展 2 cm · 默认显示相交候选')
html=re.sub(r'<div class="room2Workflow">.*?</div>','<div class="room2Workflow">先保留通过过合并筛选、且与当前 ROI 的扩展异常表面有交集的 Mask。无交集 Mask 显示时保留原分割颜色和统一透明度，以“无交集”标记区分；红色拟剔除项仍由原开关控制。相交只表示与本次 ROI 相关，请继续人工检查质量。</div>',html,count=1)
bar='<div class="intersectionToolbar"><label><input type="checkbox" id="showUnrelated">显示无交集 Mask（对照）</label><label><input type="checkbox" id="showExpansion" checked>显示新增扩展点（青色）</label><span id="intersectionStatus"></span></div><div class="intersectionSummary" id="expansionSummary"></div>'
change('<div class="rejectedToolbar">',bar+'<div class="rejectedToolbar">')
for i in range(3):
 anchor='<div class="rejectedPanel" id="rejectedPanel'+str(i)+'">'
 panel='<div class="intersectionPanel"><label><input type="checkbox" id="showUnrelated'+str(i)+'" aria-label="显示'+['原始 CropFormer','局部 CropFormer','SAM2.1'][i]+'无交集 Mask">显示本栏无交集 Mask</label><small id="intersectionCount'+str(i)+'"></small></div>'
 change(anchor,panel+anchor)
change('</style>', '.intersectionToolbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:9px 12px;margin-top:10px;border:1px solid #87d6de;background:#effbfd;border-radius:8px;font-size:13px}.intersectionToolbar label{font-weight:600;display:flex;align-items:center;gap:4px}.intersectionToolbar span,.intersectionSummary{font-size:12px;color:#435c68}.intersectionSummary{padding:7px 0;line-height:1.6}.intersectionPanel{padding:7px 10px;background:#f0fafb;border-top:1px solid #cee7eb;font-size:12px}.intersectionPanel small{display:block;margin-top:4px;color:#496671}.intersectionBadge{color:#007b8b;font-size:10px}.intersectionBadge.outside{color:#64748b}'+ '</style>')
change('const [rgb,full,alarm,models]=await Promise.all([getImage(j.RGB),getImage(v.full_rgb),getImage(v.alarm),','const [rgb,full,alarm,expansion,models]=await Promise.all([getImage(j.RGB),getImage(v.full_rgb),getImage(v.alarm),getImage(v.expansion),')
change('loaded={j,rgb,full,alarm,models};prepareProblemAnnotations();prepareMaskDisplay();','loaded={j,rgb,full,alarm,expansion,models};prepareProblemAnnotations();prepareIntersection();prepareMaskDisplay();')
change("return id===0||!loaded?.models[model]?.hiddenMasks?.has(id);", "return id===0||((intersectsROI(model,id)||loaded?.models[model]?.showUnrelated)&&!loaded?.models[model]?.hiddenMasks?.has(id));")
change('if(!m.hiddenMasks?.size)return {color:m.color,boundary:m.boundary};','if(!m.hiddenMasks?.size&&!m.data.relevance.no_intersection_ids.length)return {color:m.color,boundary:m.boundary};')
change('if(m.hiddenMasks.has(id)){color[p+3]=0;boundary[p+3]=0;}', 'if(!maskIsVisible(model,id)){color[p+3]=0;boundary[p+3]=0;}')
change("const label=String(p.id),tw=", "const label=intersectsROI(model,p.id)?String(p.id):'无交集 '+p.id,tw=")
change("row.classList.toggle('inspectedMask',id===active&&visible);\n });", "row.classList.toggle('inspectedMask',id===active&&visible);\n });decorateIntersectionLegend(model);")
change('drawRejectedFill(ctx,model);drawProblemOverlay', 'drawRejectedFill(ctx,model);drawExpansion(ctx);drawProblemOverlay')
change('drawFullProblemOverlay(c,ctx);const b=', 'drawExpansion(ctx,true);drawFullProblemOverlay(c,ctx);const b=')
change("el('otherStates').checked=meta.group!=='CONFLICT'", "el('otherStates').checked=true")
change("loaded.models[chosenModel].data.positions.slice().sort((a,b)=>(involvement.get(b.id)||0)-(involvement.get(a.id)||0)||b.area-a.area).forEach(p=>{", "maskIDs(chosenModel).filter(id=>intersectsROI(chosenModel,id)).map(id=>({id})).forEach(p=>{")
change('function toggleTarget(id){if(id<=0)return;', "function toggleTarget(id){if(id<=0)return;if(!intersectsROI(chosenModel,id)){selectionNotice('ID '+id+' 与扩展异常表面无交集，仅作对照，未加入候选。');return;}")
change('const ids=[...chosenIDs].sort((a,b)=>a-b),borderTouchingIDs=new Set();', "const ids=[...chosenIDs].sort((a,b)=>a-b),borderTouchingIDs=new Set();if(chosenModel>=0&&ids.some(id=>!intersectsROI(chosenModel,id))){selectionNotice('包含无交集 Mask，请重新选择相交候选。');return;}")
change('source_selection:\'room2-e1-retained-roi-top20-20261015\',overmerge_filter:D.overmerge_filter,', "source_selection:'room2-e1-intersection-roi-20261015',intersection_filter:D.intersection_filter,overmerge_filter:D.overmerge_filter,")
change('cvpr-room2-e1-retained-roi-top20-20261015-v1','cvpr-room2-e1-intersection-roi-20261015-v1')
change('room2_E1各前20筛选后ROI_质量判定_20261015','room2_E1异常扩展相交候选_质量判定_20261015')
# Clean the inherited duplicate review_label_asset field while preserving original label IDs.
dup="review_label_asset:chosenModel<0?null:'assets/'+activeKey+'_'+['original','local','sam'][chosenModel]+'_ids.png',"
html=html.replace(dup,'')
change('selected_at:new Date().toISOString(),is_qa:qaMode,', "intersection_filter:D.intersection_filter,intersection_support:chosenModel<0?[]:ids.map(id=>({id,...j.models[chosenModel].relevance.labels[id]})),selected_at:new Date().toISOString(),is_qa:qaMode,")
change('保存四栏对比图（仅保留 Mask）','原全量对照图（未应用交集筛选）')
change('window.threewayReview=',(H/'intersection_ui.js').read_text(encoding='utf-8')+'\nwindow.threewayReview=')
(OUT/'index.html').write_text(html,encoding='utf-8');runtime=re.findall(r'<script>\s*(.*?)</script>',html,re.S);assert len(runtime)==1;(H/'runtime.js').write_text(runtime[0],encoding='utf-8')
counts=[]
for key,crop in d['crops'].items():
 counts.append(dict(key=key,intersects=[len(m['relevance']['eligible_ids']) for m in crop['models']],no_intersection=[len(m['relevance']['no_intersection_ids']) for m in crop['models']],overmerge=[len(m['removed_ids']) for m in crop['models']]))
save(OUT/'mask_counts.json',dict(models=['original_cropformer','local_cropformer','sam2.1'],counting_unit='mask occurrences per crop',stats=r['stats'],crops=counts))
projection_audit=read(OLD/'projection_annotation_audit.json');projection_audit['intersection_audit_file']='audit.json';save(OUT/'projection_annotation_audit.json',projection_audit)
s.update(status='READY_FOR_INTERSECTION_REVIEW',name=d['name'],intersection_filter=d['intersection_filter'],isolated_storage_key='cvpr-room2-e1-intersection-roi-20261015-v1',page_sha256=sha(OUT/'index.html'),export_filename='room2_E1异常扩展相交候选_质量判定_20261015.json');save(OUT/'selection_setup.json',s)
(OUT/'plan.txt').write_text('当前32个ROI的全部原异常点（C/U/T）作为A；在冻结TSDF表面按欧氏距离扩展2厘米（体素1厘米的两倍），不改变状态、身份和几何。沿用原可见性检查，Mask有效可见顶点与A+有至少1个交点即相关。先应用旧过合并规则，相交数量不作质量排序。原图Mask按完整原帧范围检查；局部Mask按其实际裁剪预测范围检查。交集在裁剪外的原图候选会提示，可展开RGB参考查看全帧。青色为新增表面点投影，不是二维膨胀；图上重合的点只画一次，数字按不同TSDF顶点计数。无交集项显示时保留原分割颜色和统一透明度，以“无交集”标记区分，不能选为本次种子；拟过合并项仍用红色对照。尚未运行IoU/包含关系提名、修复或评价。',encoding='utf-8')
for path in assets:assert (OUT/path).resolve().is_file(),path
assert before=={n:sha(OLD/n) for n in before}
report=dict(status='PASS',assets_verified=len(assets),ROIs=len(d['cases']),crops=len(d['crops']),source_masks_unchanged=True,old_page_unchanged=True,stats=r['stats']);save(H/'page_verification.json',report);print(json.dumps(report))
