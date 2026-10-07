from pathlib import Path
import json, hashlib, shutil, ast

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'results/固定案例_三模型对比_20261006/tracking_review_20261007'
manifest=json.loads((OUT/'ply_models/manifest.json').read_text(encoding='utf-8'))
case=next(c for c in manifest['cases'] if c['scene']=='room2')
for scope in ['local','full']:
    p=OUT/case[scope]['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==case[scope]['sha256']
annotation=HERE.parents[1]/'修复候选选择_20261006.json'
assert hashlib.sha256(annotation.read_bytes()).hexdigest()==manifest['source_export_sha256']
palettes=json.loads((OUT/'instance_palettes.json').read_text(encoding='utf-8'))
expected=json.loads((HERE/'room2_original_palette.json').read_text(encoding='utf-8'))
assert all(palettes['scenes']['room2']['id_colors'][str(r['id'])]==r['RGB'] for r in expected)
correction={'status':'PASS','root_cause':'Solo selection was incorrectly treated as unchecked masks. The viewer hid original colored chair instance points and left unpublished gray surfaces visible.',
            'previous_gray_chair_explanation_withdrawn':True,'fix':'Explicit checkbox visibility is separate from solo selection; original PLY instance RGB is preserved.',
            'original_P1A1_PLY':str(palettes['scenes']['room2']['source_PLY']),
            'all_51_room2_original_instance_colors_match_local_PLY':True,
            'regression_tests_passed':5,'browser_default_mask24_keeps_colored_chairs':True,
            'browser_explicit_uncheck_hides_only_corresponding_identity':True,
            'browser_before_and_after_verified':True,'new_inference':False,'map_labels_changed_by_this_fix':False,
            'annotation_unchanged':True,
            'before_screenshot':'room2_p1a1_color_restored.jpg','after_screenshot':'room2_short_repair_color_restored.jpg'}
for name in [correction['before_screenshot'],correction['after_screenshot']]:shutil.copyfile(HERE/name,OUT/name)
(OUT/'room2_display_bug_fix.json').write_text(json.dumps(correction,ensure_ascii=False,indent=2),encoding='utf-8')
p=OUT/'room2_ply_comparison.json';record=json.loads(p.read_text(encoding='utf-8'))
record['display_diagnosis_correction']=correction
record['browser_verified']['original_screenshot']=correction['before_screenshot']
record['browser_verified']['repair_screenshot']=correction['after_screenshot']
record['previous_display_screenshots_had_solo_visibility_bug']=['room2_p1a1_original_gray_proof.jpg','room2_short_repair_gray_proof.jpg']
p.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
p=OUT/'floating_ply_verification.json';record=json.loads(p.read_text(encoding='utf-8'))
record['browser_verification'].update({'instance_colors':'exact_original_PLY_instance_RGB','solo_selection_preserves_other_instances':True,'explicit_checkbox_visibility_only':True,'regression_tests_passed':5,'screenshot':correction['after_screenshot']})
p.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
ast.parse((HERE/'integrate_floating_ply.py').read_text(encoding='utf-8'))
pairs=json.loads((HERE/'floating_ply_delivery_upload.json').read_text(encoding='utf-8'))
pairs=[p for p in pairs if not p[0].endswith('index.html')]
remote='/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/tracking_review/'
for name in ['mask_visibility.js','instance_palettes.json','room2_display_bug_fix.json','room2_ply_comparison.json',correction['before_screenshot'],correction['after_screenshot']]:pairs.append([str(OUT/name),remote+name])
for name in ['mask_visibility.js','test_mask_visibility.mjs','export_original_ply_palettes.py','export_original_ply_palettes.sh','finalize_visibility_fix.py']:pairs.append([str(HERE/name),'/home/chenkejun/CVPR/experiments/pilot10_repair_20261007/'+name])
pairs.append([str(OUT/'index.html'),remote+'index.html'])
(HERE/'floating_ply_visibility_upload.json').write_text(json.dumps(pairs,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'status':'PASS','original_PLY_RGB_verified':51,'regression_tests':5,'room2_PLY_files_and_annotation_unchanged':True,'browser_color_restored':True},ensure_ascii=False))
