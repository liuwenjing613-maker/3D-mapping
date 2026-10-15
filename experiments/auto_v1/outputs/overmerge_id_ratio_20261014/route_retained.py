from pathlib import Path
import re
H=Path(__file__).resolve().parent;ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006'
dest='../room2_E1_retained_ROI_20261014/index.html'
for p in [H/'index.html',ROOT/'room2_E1_overmerge_audit_20261014/index.html']:
 text=p.read_text(encoding='utf-8')
 text=text.replace('先看红色项是否误删，再看保留项是否漏筛。所有判断均可撤销、保存和导出。','拟剔除项在这里核验；保留项回到原来的 ROI 三模型页面检查。')
 text=text.replace('href="../room2_E1_prerepair_CUT_top10_20261014/index.html" target="_blank">原三模型候选页 ↗','href="'+dest+'">保留项 ROI 检查 ↗')
 text=text.replace('查漏筛 · 保留 <b id="retainCount">','保留项 · 按 ROI 检查 <b id="retainCount">')
 text=text.replace('<button data-filter="all">全部</button>','')
 p.write_text(text,encoding='utf-8')
for p in [H/'app.js',ROOT/'room2_E1_overmerge_audit_20261014/app.js']:
 text=p.read_text(encoding='utf-8')
 old="document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{decisionFilter=b.dataset.filter;updateTabs();filterCandidates(false);});"
 new="document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{if(b.dataset.filter==='retained'){location.href='"+dest+"'+(QA?'?qa=1':'');return;}decisionFilter=b.dataset.filter;updateTabs();filterCandidates(false);});"
 assert old in text or new in text;text=text.replace(old,new)
 text=text.replace("function openFromOverview(cid){const c=D.candidates[cid];decisionFilter=", "function openFromOverview(cid){const c=D.candidates[cid];if(isRetained(c)){const j=D.jobs[c.occ[0][0]],query=new URLSearchParams({roi:j.uid,crop:j.key});if(QA)query.set('qa','1');location.href='"+dest+"?'+query.toString();return;}decisionFilter=")
 text=text.replace('保留 · 继续查漏筛','保留 · 按 ROI 检查')
 p.write_text(text,encoding='utf-8')
print('RETAINED_LINK_ROUTED')
page=ROOT/'room2_E1_retained_ROI_20261014/index.html'
html=page.read_text(encoding='utf-8')
html=html.replace("label_asset:chosenModel<0?null:j.models[chosenModel].ids_file,overmerge_filter:D.overmerge_filter", "label_asset:chosenModel<0?null:'assets/'+activeKey+'_'+['original','local','sam'][chosenModel]+'_ids.png',review_label_asset:chosenModel<0?null:j.models[chosenModel].ids_file,overmerge_filter:D.overmerge_filter")
page.write_text(html,encoding='utf-8')
runtime=re.findall(r'<script>\s*(.*?)</script>',html,re.S);assert len(runtime)==1
(H/'retained_runtime.js').write_text(runtime[0],encoding='utf-8')
