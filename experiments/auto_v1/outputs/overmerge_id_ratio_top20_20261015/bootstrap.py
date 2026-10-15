from pathlib import Path
H=Path(__file__).resolve().parent;OLD=H.parent/'overmerge_id_ratio_20261014'
names=['score.py','transfer.py','build_retained.py','verify_retained.py','add_rejected_toggle.py','rejected_comparison.js']
for name in names:
 t=(OLD/name).read_text(encoding='utf-8')
 for a,b in [('e1_roi_top10_20261014','e1_roi_top20_20261015'),('overmerge_id_ratio_20261014','overmerge_id_ratio_top20_20261015'),('room2_E1_prerepair_CUT_top10_20261014','room2_E1_prerepair_CUT_top20_20261015'),('room2_E1_retained_ROI_20261014','room2_E1_retained_ROI_top20_20261015'),('room2_E1_overmerge_audit_20261014','room2_E1_overmerge_audit_top20_20261015'),('cut-top10-20261014','cut-top20-20261015'),('CUT-top10-20261014','CUT-top20-20261015'),('retained-roi-20261014','retained-roi-top20-20261015'),('room2_E1筛选后ROI_质量判定_20261014','room2_E1各前20筛选后ROI_质量判定_20261015'),('room2_E1修复前三状态前十_质量判定_20261014','room2_E1修复前三状态前二十_质量判定_20261015')]:t=t.replace(a,b)
 if name=='score.py':
  t=t.replace('assert len(jobs)==412', "assert len(jobs)==len(viewer['crops'])")
  t=t.replace('assert occurrences==18303', "assert occurrences==sum(m['count'] for crop in viewer['crops'].values() for m in crop['models'])")
  t=t.replace('ROIs=19,rank_entries=30,view_entries=90','ROIs=32,rank_entries=60,view_entries=180')
 if name=='build_retained.py':
  t=t.replace("audit_page='../'+AUDIT.name+'/index.html'", "audit_page='index.html'")
  t=t.replace('冲突 / 未分配 / 证据不足各前 10 · 30 个名额去重为 19 个 ROI · 三路分割','冲突 / 未分配 / 证据不足各前 20 · 60 个名额去重为 32 个 ROI · 三路分割')
  t=t.replace("'沿用原 ROI 检查格式 · 已排除 194 个拟过合并候选 · 19 个 ROI / 三路分割'", "'三类各前 20 · 60 个入榜名额 / 32 个 ROI · 拟剔除 '+str(a['stats']['reject'])+' 个候选'")
  t=t.replace('<a href="../\'+AUDIT.name+\'/index.html">查看拟剔除项</a>。','')
  t=t.replace('unique_rejected=194',"unique_rejected=a['stats']['reject']").replace('ROIs=19,view_entries=90,crops=412',"ROIs=32,view_entries=180,crops=len(d['crops'])")
  t=t.replace('==18303',"==a['stats']['occurrences']")
  # Keep exports linked to original source masks, even when a template uses dynamic labels.
  needle=" (OUT/'index.html').write_text(html,encoding='utf-8')"
  previous="label_asset:chosenModel<0?null:j.models[chosenModel].ids_file,overmerge_filter:D.overmerge_filter"
  replacement="label_asset:chosenModel<0?null:'assets/'+activeKey+'_'+['original','local','sam'][chosenModel]+'_ids.png',review_label_asset:chosenModel<0?null:j.models[chosenModel].ids_file,overmerge_filter:D.overmerge_filter"
  patch=' html=html.replace('+repr(previous)+','+repr(replacement)+')\n'
  t=t.replace(needle,patch+needle)
 if name=='verify_retained.py':
  t=t.replace("len(new['cases'])==19 and len(new['crops'])==412", "len(new['cases'])==32 and len(new['crops'])==len(old['crops'])").replace('==90','==180').replace('ROIs=19,views=90,crops=412',"ROIs=32,views=180,crops=len(new['crops'])")
 (H/name).write_text(t,encoding='utf-8')
print('CREATED',len(names))
