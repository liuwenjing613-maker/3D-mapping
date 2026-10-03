from pathlib import Path
import json,hashlib,zipfile
import numpy as np
from PIL import Image
P=Path(__file__).parent;A=P/'inference_arrays';D=Path('D:/Users/刘雯静/Downloads/CVPR/results/P1-A1_room0_局部CropFormer重分割_20261003')
m=json.loads((A/'input_manifest.json').read_text());audit=json.loads((A/'inference_audit.json').read_text());rec={r['key']:r for r in map(json.loads,(A/'inference_records.jsonl').read_text().splitlines())}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
count=0
for j in m['jobs']:
 if j['kind']!='ROI_crop':continue
 key=j['key'];rgb=A/'inputs'/f'{key}_rgb.png';old=A/'inputs'/f'{key}_baseline.png';new=A/'inference'/f'{key}.png'
 assert sha(rgb)==j['RGB_file_sha256'] and sha(old)==j['baseline_file_sha256'] and sha(new)==rec[key]['mask_file_sha256']
 folder=D/j['ROI']/f'视角{j["view"]}';label='紧裁剪' if j['strategy']=='tight' else '证据来源裁剪' if j['strategy']=='witness' else '宽裁剪'
 assert np.array_equal(np.asarray(Image.open(new)),np.asarray(Image.open(folder/f'{label}_局部mask_ID.png')))
 assert np.array_equal(np.asarray(Image.open(rgb)),np.asarray(Image.open(folder/f'{label}_输入RGB.png')))
 assert (folder/f'{label}_RGB与mask对照.png').exists();count+=1
assert count==159 and len({r['rank'] for r in m['cases']})==32 and not list(D.rglob('*.ply'))
controls=[r['frozen_cache_comparison'] for r in rec.values() if r['kind']=='full_frame_control'];assert len(controls)==79 and all(r['identical_label_image'] for r in controls)
probes=json.loads((P/'witness_pixel_diagnostics.json').read_text());assert len(probes)==24
(D/'证据像素重新分割诊断.json').write_text(json.dumps(probes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
audit.update({'native_saved_RGB_and_label_masks_exactly_match_inference':True,'selected_ROIs':32,'original_selected_ROI_views':96,'wide_ROI_crop_inputs':39,'competition_source_RGB_inputs':24,'all_full_frame_controls_pixel_identical_to_frozen_cache':True,'browser_QA':'PASS: 32 ROI choices; 159 image sets decode; crop/view/pixel selection; mobile no overflow','diagnostic_scope':'pure CropFormer RGB crop re-inference; visual frontend comparison, not a new 3D v3 evaluation','unassigned_witness_pixels_after_reinference':sum(p['after_L']==0 for p in probes),'no_new_PLY':True})
(D/'结果核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
conclusion='本次真实重新运行CropFormer：32个ROI、96个已选历史RGB视角，另含39张宽裁剪和24张此前展示的竞争来源RGB，共159次局部分割。79张对应整帧对照与原缓存逐像素完全相同。\n'
conclusion+='所有局部推理保持原权重与0.5阈值，不使用GT提示、深度细化或手工mask提示。三维地图没有执行修复。\n'
conclusion+='视觉判断：局部裁剪有个别边界收益，未表现为稳定通用修复。例如ROI-0063的105帧证据像素从墙面F4变为局部L3（花瓶形状）；ROI-0035的1765帧和ROI-0038的1670帧代表像素均变为L0未分割。\n'
conclusion+='24个证据来源像素中有7个在重新分割后变为L0。该统计只描述24个固定证据样本，不是GT准确率或v3建图指标。\n'
conclusion+='查看每个ROI目录中的三个原选视角和两个竞争来源视角（若有）；切换宽／紧裁剪，重点查看目标完整性、背景混入、漏分和额外碎片，不能只看mask数量。\n'
(D/'查看结论.txt').write_text(conclusion,encoding='utf-8')
zpath=D.parent/'room0_局部CropFormer_32ROI全部对照.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for p in sorted(D.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(D)))
uploads=[[str((P/f).resolve()),'/home/chenkejun/CVPR/experiments/p1a1_roi_cropformer_zoom_20261003/'+f] for f in ['prepare_inputs.py','prepare_witness_inputs.py','infer_crops.py','render_zoom_comparison.py','check_witness_pixels.py','finish_zoom_review.py','zoom_preview_template.html','assemble_zoom_preview.py','test_zoom_preview.cjs','start_native_inference.sh','prepare_driver_links.sh']]
uploads.extend([[str(zpath),'/data/chenkejun/CVPR/revisable_instance_map/p1a1_roi_cropformer_zoom_20261003/room0_32ROI_all_comparisons.zip'],[str(D/'结果核验.json'),'/data/chenkejun/CVPR/revisable_instance_map/p1a1_roi_cropformer_zoom_20261003/visual_review_audit.json'],[str(D/'查看结论.txt'),'/home/chenkejun/CVPR/experiments/p1a1_roi_cropformer_zoom_20261003/review_conclusions_CN.txt'],[str(D/'证据像素重新分割诊断.json'),'/data/chenkejun/CVPR/revisable_instance_map/p1a1_roi_cropformer_zoom_20261003/witness_pixel_diagnostics.json']])
(P/'final_upload.json').write_text(json.dumps(uploads,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'status':'PASS','crop_images_verified':count,'full_frame_controls_identical':len(controls),'witness_unassigned':7,'zip_bytes':zpath.stat().st_size},ensure_ascii=False))
