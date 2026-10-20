"""Apply room0's annotation UI to an isolated, ranked room2 top-30 review."""
from pathlib import Path
import hashlib,json,re

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
SOURCE=WORK/'results/固定案例_三模型对比_20261006'
DEST=SOURCE/'room2_selection_top30_20261008'
TEMPLATE=WORK/'outputs/room0_selection_20261007/room0_selection_template.html'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,s):
 with p.open('w',encoding='utf-8',newline='\n') as f:f.write(s)
def replace_once(s,a,b):
 assert s.count(a)==1,(a,s.count(a))
 return s.replace(a,b,1)

def main():
 protected=[SOURCE/'viewer_data.json',SOURCE/'select.html',SOURCE/'room0_selection_20261007/index.html',
  SOURCE/'room2_selection_20261007/index.html',WORK/'room2_修复候选选择_20261007.json',WORK/'room0_修复候选选择_20261007.json',TEMPLATE]
 hashes={str(p):sha(p) for p in protected if p.exists()}
 d=read(DEST/'viewer_data.json');p=read(DEST/'projection_annotations.json');setup=read(DEST/'selection_setup.json')
 assert len(d['cases'])==30 and sum(len(c['views']) for c in d['cases'])==90
 assert [c['ROI'] for c in d['cases']]==[f'ROI-C{i:04d}' for i in range(1,31)]
 assert set(d['crops'])==set(p['crops']);uids=[c['uid'] for c in d['cases']]
 old=read(WORK/'room2_修复候选选择_20261007.json')
 assert old['scene_scope']=='room2' and not old['is_qa']
 refs={r['case_uid']:r for r in old['selections'] if r['case_uid'] in uids and not r.get('is_qa')}
 assert len(refs)==5
 text=TEMPLATE.read_text(encoding='utf-8').replace('room0','room2')
 text=replace_once(text,'<title>room2 · 人工修复选择</title>','<title>room2 前30 · 人工修复选择</title>')
 text=replace_once(text,'<h1>room2 · 人工修复选择</h1>','<h1>room2 前30 · 人工修复选择</h1>')
 text=replace_once(text,'>导出 room2 选择</button>','>导出 room2 前30 选择</button>')
 text=replace_once(text,'14 个固定问题区域 · 42 个视角','冲突排序前 30 个区域 · 90 个视角')
 text=text.replace('全部 14 个区域','排序前 30 个区域').replace('14 个区域全部完成。','30 个区域全部完成。')
 text=replace_once(text,'按区域查看，换视角确认物体完整。已保存的区域可随时修改。',
  '按原始冲突点数从多到少排序。每区确认一次；旧5例可载入草稿，再检查保存。')
 text=replace_once(text,"'room2 区域 '+c.order+' · 同区域编号 '","'原始冲突排名 '+c.order+' / 30 · C '+c.counts.CONFLICT+' 点 · 同区域编号 '")
 text=replace_once(text,"b.querySelector('b').textContent=c.scene+' / '+c.ROI;","b.querySelector('b').textContent=String(c.order).padStart(2,'0')+' · '+c.ROI;")
 a=text.index('const previousReferences=');b=text.index('\nfunction refreshPreviousReference',a)
 text=text[:a]+'const previousReferences='+json.dumps(refs,ensure_ascii=False)+';'+text[b:]
 text=replace_once(text,"modelNames[old.model_index]+' / ID '+old.mask_ids.join('、')","(old.choice==='unreliable'?'都不可靠':modelNames[old.model_index]+' / ID '+old.mask_ids.join('、'))")
 text=replace_once(text,"chosenModel=old.model_index;chosenIDs=new Set(old.mask_ids);refreshSelection();", "chosenModel=old.model_index;chosenIDs=new Set(old.mask_ids);el('targetNote').value=old.target_notes||'';refreshSelection();")
 text,n=re.subn(r'const pilotUIDs=new Set\(\[.*?\]\);','const pilotUIDs=new Set('+json.dumps(uids)+');',text,count=1);assert n==1
 text=text.replace('cvpr-room2-selection-f49eb927-20261007-v1','cvpr-room2-top30-selection-20261008-v1')
 text=text.replace("source_selection:'selection-20261006-f49eb927'","source_selection:'room2-top30-20261008'")
 text=replace_once(text,"source_total_fixed_cases:37,scene_scope:'room2',asset_base_path:'../',source_data_directory:'roi_threeway_fixed_20261006/selection-20261006-f49eb927/review',",
  "source_total_fixed_cases:30,scene_scope:'room2',asset_base_path:'',source_data_directory:'room2_selection_top30_20261008/review',"+
  "\n ranking_group:'CONFLICT',ranking_basis:'original native conflict point count descending',"+
  "\n source_candidate_sha256:"+json.dumps(setup['source_candidate_sha256'])+",source_surface_sha256:"+json.dumps(setup['source_surface_sha256'])+","+
  "\n case_order:D.cases.map(c=>({rank:c.order,case_uid:c.uid,ROI:c.ROI,conflict_points:c.counts.CONFLICT})),annotation_complete:D.cases.every(c=>!!savedSelections[c.uid]),")
 text=replace_once(text,"const record={case_uid:c.uid,scene:c.scene,ROI:c.ROI,aliases:c.aliases,", "const record={case_uid:c.uid,scene:c.scene,ROI:c.ROI,aliases:c.aliases,conflict_rank:c.order,conflict_points:c.counts.CONFLICT,")
 text=text.replace('room2_修复候选选择_20261007.json','room2_前30_修复候选选择_20261008.json')
 text=text.replace('“room2 已标好”','“room2 前30已标好”')
 text=replace_once(text,'href="../mask_counts.json"','href="mask_counts.json"')
 text=replace_once(text,'href="../projection_annotation_audit.json"','href="projection_annotation_audit.json"')
 page=text.replace('__DATA__',json.dumps(d,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
 page=page.replace('__PROJECTIONS__',json.dumps(p,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
 assert '__DATA__' not in page and '__PROJECTIONS__' not in page and 'room0' not in page
 runtime=re.findall(r'<script>\s*(.*?)</script>',page,re.S);assert len(runtime)==1
 save(HERE/'selection_template.html',text);save(HERE/'selection_runtime.js',runtime[0]);save(DEST/'index.html',page)
 save(DEST/'previous_room2_reference.json',json.dumps(refs,ensure_ascii=False,indent=2)+'\n')
 plan=('room2 原始冲突排名前30：ROI-C0001～ROI-C0030，90个固定视角。\n'
  '每区切换视角、裁剪，比较原始CropFormer、局部CropFormer、SAM2.1。\n'
  '选择一套模型后，点选冲突处需要修复的完整目标；多个物体分别选多个mask。\n'
  '每个mask都可显隐、只看和高亮；红点标出冲突位置。\n'
  '三套都差时选“都不可靠”。触及裁剪边界可保存，后续会核对完整度。\n'
  '旧5例仅作参考，点击载入草稿后仍需确认保存，不自动计入进度。\n'
  '每区保存并下一个，30/30后导出room2_前30_修复候选选择_20261008.json。\n'
  '导出文件放到本地CVPR文件夹，告诉Codex“room2 前30已标好”。\n'
  '现在仅准备候选；未追踪、未修改地图。收到记录后核对重复物体，双向SAM2.1追踪，联合撤回/补票，再展示3D与统一v3评估。\n')
 save(DEST/'plan.txt',plan)
 for path,h in hashes.items():assert sha(Path(path))==h,path
 setup.update(previous_reference_count=len(refs),human_records_created=0,page_sha256=sha(DEST/'index.html'),protected_files_unchanged=True)
 save(DEST/'selection_setup.json',json.dumps(setup,ensure_ascii=False,indent=2)+'\n')
 save(HERE/'build_receipt.json',json.dumps(dict(setup,protected_hashes=hashes),ensure_ascii=False,indent=2)+'\n')
 final={str(p.relative_to(DEST)).replace('\\','/'):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='final_files_sha256.json'}
 save(DEST/'final_files_sha256.json',json.dumps(final,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:setup[k] for k in ['status','case_count','view_count','candidate_crop_count','previous_reference_count','human_records_created','protected_files_unchanged']},ensure_ascii=False))

if __name__=='__main__':main()
