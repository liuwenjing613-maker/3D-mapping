from pathlib import Path
import json,re,hashlib
H=Path(__file__).resolve().parent; W=H.parents[1]
WEB=W/'results/固定案例_三模型对比_20261006'
T=W/'outputs/room0_selection_20261007/room0_selection_template.html'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,t):p.write_text(t,encoding='utf-8',newline='\n')
for scene in ['room0','room2']:
 d=WEB/(scene+'_strict_vote_annotation_20261010')
 if not (d/'viewer_data.json').exists():continue
 data=read(d/'viewer_data.json');proj=read(d/'projection_annotations.json');setup=read(d/'selection_setup.json')
 assert len(data['cases'])==30 and sum(len(c['views']) for c in data['cases'])==90
 assert [c['ROI'] for c in data['cases']]==[f'ROI-C{i:04d}' for i in range(1,31)]
 counts=[c['counts']['CONFLICT'] for c in data['cases']];assert counts==sorted(counts,reverse=True)
 assert set(data['crops'])==set(proj['crops'])
 t=T.read_text(encoding='utf-8').replace('room0',scene)
 t=t.replace(scene+' · 人工修复选择',scene+' · strict-vote 重新标注')
 t=t.replace('14 个固定问题区域 · 42 个视角','P1-A1-strict-vote · 冲突前 30 个区域 · 90 个视角')
 t=t.replace('14 个区域','30 个区域')
 t=t.replace('按区域查看，换视角确认物体完整。已保存的区域可随时修改。','严格一票未修复基线，按冲突点数降序。新一轮独立标注，旧选择不载入。')
 a=t.index('const previousReferences=');b=t.index('\nfunction refreshPreviousReference',a)
 t=t[:a]+'const previousReferences={};'+t[b:]
 t=re.sub(r'const pilotUIDs=new Set\(\[.*?\]\);','const pilotUIDs=new Set('+json.dumps([c['uid'] for c in data['cases']])+');',t,count=1)
 storage='cvpr-'+scene+'-strict-vote-selection-20261010-v1'
 t=t.replace('cvpr-'+scene+'-selection-f49eb927-20261007-v1',storage)
 t=t.replace("source_selection:'selection-20261006-f49eb927'","source_selection:'"+scene+"-strict-vote-top30-20261010'")
 t=t.replace("source_total_fixed_cases:37,scene_scope:'"+scene+"',asset_base_path:'../',source_data_directory:'roi_threeway_fixed_20261006/selection-20261006-f49eb927/review',",
  "source_total_fixed_cases:30,scene_scope:'"+scene+"',asset_base_path:'',source_data_directory:'"+setup['source_data_directory']+"/review',\n"+
  "baseline_method:'P1-A1-strict-vote',baseline_repaired:false,vote_policy:'one vote per frame and point; multi-ID abstention',evaluation_protocol:'object_observed_repair / revision 2',\n"+
  "ranking_group:'CONFLICT',ranking_basis:'strict native C points intersect final unassigned; whole connected components; count descending',source_candidate_sha256:"+json.dumps(setup['source_candidate_sha256'])+",source_surface_sha256:"+json.dumps(setup['source_surface_sha256'])+",\n"+
  "case_order:D.cases.map(c=>({rank:c.order,case_uid:c.uid,ROI:c.ROI,conflict_points:c.counts.CONFLICT})),annotation_complete:D.cases.every(c=>!!savedSelections[c.uid]),")
 t=t.replace("const record={case_uid:c.uid,scene:c.scene,ROI:c.ROI,aliases:c.aliases,", "const record={case_uid:c.uid,scene:c.scene,ROI:c.ROI,aliases:c.aliases,conflict_rank:c.order,conflict_points:c.counts.CONFLICT,")
 t=t.replace("'"+scene+" 区域 '+c.order+' · 同区域编号 '","'strict-vote 冲突排名 '+c.order+' / 30 · C '+c.counts.CONFLICT+' 点 · 同区域编号 '")
 t=t.replace("b.querySelector('b').textContent=c.scene+' / '+c.ROI;","b.querySelector('b').textContent=String(c.order).padStart(2,'0')+' · '+c.ROI;")
 t=t.replace(scene+'_修复候选选择_20261007.json',scene+'_strict_vote_人工标注_20261010.json')
 t=t.replace('联合换票 → 3D 与统一 v3 对比','严格一票＋歧义弃权重算 → 3D 与 revision 2 对比')
 t=t.replace('当前：等待你标注，尚未开始追踪。','当前：strict-vote 未修复基线的新标注。')
 t=t.replace('例如：餐桌左侧两把椅子；与上一区域是同一把椅子','建议：mask 5 与 ROI-C0002 的 mask 9 是同一把椅子')
 t=t.replace('href="../mask_counts.json"','href="mask_counts.json"').replace('href="../projection_annotation_audit.json"','href="projection_annotation_audit.json"')
 t=t.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
 t=t.replace('__PROJECTIONS__',json.dumps(proj,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
 assert 'f49eb927' not in t and '统一 v3' not in t and '旧5例' not in t
 runtime=re.findall(r'<script>\s*(.*?)</script>',t,re.S);assert len(runtime)==1
 save(d/'index.html',t);save(H/scene/'selection_runtime.js',runtime[0])
 save(d/'plan.txt',scene+' / P1-A1-strict-vote 新一轮人工标注\n'
  +'使用未人工修复的 strict-vote 最终地图；同帧同点多个不同 ID 弃权，同 ID 合并为一票。\n'
  +'ROI 与旧版规则一致：最终未归属点中的原生 U/T/C，图上一环连接形成完整连通区域；按 C 点数降序，连通分量编号打破平局，取前30。\n'
  +'每个 ROI 三个历史视角。原始 CropFormer、局部 CropFormer、SAM2.1 使用相同 RGB 裁剪比较。\n'
  +'选择一个最佳视角/裁剪/模型，点选完整目标，可多选；无法可靠选择则标记都不可靠。\n'
  +'跨 ROI 为同一物体时，请在目标说明中注明 ROI 和 Mask 编号。保存全部30例后导出 JSON 到本地 CVPR。\n'
  +'本轮独立存储，不载入旧人工选择；QA 测试记录隔离。当前未追踪、未执行新修复。\n'
  +'后续评估统一使用 main 的 object_observed_repair / revision 2；ROI 和模型候选生成不使用 GT。\n')
 setup.update(baseline_method='P1-A1-strict-vote',baseline_repaired=False,previous_reference_count=0,human_records_created=0,page_sha256=sha(d/'index.html'),isolated_storage_key=storage)
 save(d/'selection_setup.json',json.dumps(setup,ensure_ascii=False,indent=2))
 save(H/scene/'build_receipt.json',json.dumps(setup,ensure_ascii=False,indent=2))
 print(scene,setup['status'],setup['candidate_crop_count'])
