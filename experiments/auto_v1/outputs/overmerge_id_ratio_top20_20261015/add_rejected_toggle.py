from pathlib import Path
import json,re,hashlib
H=Path(__file__).resolve().parent;ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006'
OLD=ROOT/'room2_E1_prerepair_CUT_top20_20261015';OUT=ROOT/'room2_E1_retained_ROI_top20_20261015'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
d=read(OUT/'viewer_data.json');old=read(OLD/'viewer_data.json');count=0
for key,crop in d['crops'].items():
 for mi,m in enumerate(crop['models']):
  if not m['removed_ids']:continue
  orig=old['crops'][key]['models'][mi]
  m['removed_source']=dict(ids_file='../'+OLD.name+'/'+orig['ids_file'],colors={str(i):orig['colors'][str(i)] for i in m['removed_ids']})
  assert (OUT/m['removed_source']['ids_file']).resolve().is_file();count+=1
save(OUT/'viewer_data.json',d)
page=OUT/'index.html';html=page.read_text(encoding='utf-8')
encoded=json.dumps(d,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
html,n=re.subn(r'(<script id="reviewData" type="application/json">).*?(</script>)',lambda m:m[1]+encoded+m[2],html,flags=re.S);assert n==1
if 'id="showRejected"' not in html:
 control='<div class="rejectedToolbar"><label><input type="checkbox" id="showRejected">显示拟剔除 Mask（对照检查）</label><span id="rejectedStatus" role="status"></span></div>'
 html=html.replace('<div class="problemToolbar"',control+'\n<div class="problemToolbar"',1)
 for i in range(3):
  anchor='<div class="meta" id="meta'+str(i)+'"></div>'
  control='<div class="rejectedPanel" id="rejectedPanel'+str(i)+'"><label><input type="checkbox" id="showRejected'+str(i)+'" aria-label="显示'+['原始 CropFormer','局部 CropFormer','SAM2.1'][i]+'拟剔除 Mask">显示本栏拟剔除</label><small id="rejectedCount'+str(i)+'"></small></div>'
  assert anchor in html;html=html.replace(anchor,anchor+control,1)
 html=html.replace('</style>','.rejectedToolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:9px 12px;margin:10px 0;background:#fff6f7;border:1px solid #e7b2bb;border-radius:8px;font-size:13px}.rejectedToolbar label{font-weight:600;color:#a71f3b;display:flex;align-items:center;gap:5px}.rejectedToolbar span{font-size:12px;color:#7e5360}.rejectedPanel{padding:7px 10px;border-top:1px solid #eee;font-size:12px;background:#fafbfc}.rejectedPanel label{display:flex;align-items:center;gap:4px}.rejectedPanel small{display:block;color:#90616c;margin:3px 0}.rejectedPanel.showingRejected{background:#fff0f3;border-color:#e5a6b4}\n</style>',1)
 html=html.replace('已按本轮规则排除拟过合并 Mask。','拟剔除 Mask 默认隐藏，可勾选“显示拟剔除 Mask”对照检查。')
 html=html.replace('prepareProblemAnnotations();prepareMaskDisplay();refreshSelection();','prepareProblemAnnotations();prepareMaskDisplay();prepareRejectedComparison();refreshSelection();')
 html=html.replace("if(model!==null)drawProblemOverlay(c,ctx,model);", "if(model!==null){drawRejectedFill(ctx,model);drawProblemOverlay(c,ctx,model);drawRejectedLabels(c,ctx,model);}")
 needle="const data=loaded.models[model].idData;if(!data)return;const offset=(y*w+x)*4,id=projectionLabelHit"
 replacement="const removedID=rejectedPixel(model,x,y);if(removedID){e.cvprRejectedMaskID=removedID;rejectedNotice(model,removedID);return;}\n  const data=loaded.models[model].idData;if(!data)return;const offset=(y*w+x)*4,id=projectionLabelHit"
 assert needle in html;html=html.replace(needle,replacement,1)
 html=html.replace('if(event.shiftKey)return;','if(event.shiftKey||event.cvprRejectedMaskID)return;',1)
 html=html.replace("el('staticLink').textContent='保存四栏对比图（当前默认范围）'", "el('staticLink').textContent='保存四栏对比图（仅保留 Mask）'")
 html=html.replace('>保存四栏对比图（当前默认范围）<','>保存四栏对比图（仅保留 Mask）<')
 code=(H/'rejected_comparison.js').read_text(encoding='utf-8')
 anchor='window.threewayReview=';assert anchor in html;html=html.replace(anchor,code+'\n'+anchor,1)
else:
 # Refresh helper functions without duplicating UI or changing saved-selection keys.
 start=html.index('// Display-only comparison:');end=html.index('window.threewayReview=',start)
 html=html[:start]+(H/'rejected_comparison.js').read_text(encoding='utf-8')+'\n'+html[end:]
page.write_text(html,encoding='utf-8')
runtime=re.findall(r'<script>\s*(.*?)</script>',html,re.S);assert len(runtime)==1
(H/'retained_runtime.js').write_text(runtime[0],encoding='utf-8')
setup=read(OUT/'selection_setup.json');setup['rejected_display_toggle']=dict(default='hidden',global_toggle=True,per_model_toggle=True,display_only=True);setup['page_sha256']=hashlib.sha256(page.read_bytes()).hexdigest();save(OUT/'selection_setup.json',setup)
print(json.dumps(dict(status='READY',model_crops_with_rejected=count,storage_key_unchanged=True,filtered_label_assets_unchanged=True)))
