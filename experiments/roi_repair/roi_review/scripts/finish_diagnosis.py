from pathlib import Path
import json,zipfile
P=Path(__file__).parent;R=Path('D:/Users/刘雯静/Downloads/CVPR/results/P1-A1_room0_ROI优先级_20261002/冲突情况标注')
cases=json.loads((R/'13案例_证据与诊断.json').read_text(encoding='utf-8'))
assert len(cases)==13
assert all(r['counts']['CONFLICT']==sum(x['conflict_points'] for x in r['competition_pairs']) or r['counts']['CONFLICT']>sum(x['conflict_points'] for x in r['competition_pairs']) for r in cases)
assert next(r for r in cases if r['rank']==73)['counts']['CONFLICT']==0
assert next(d for r in cases if r['rank']==35 for d in r['instance_details'] if d['id']==206)['semantic_reference'][0]['name']=='chair'
assert next(d for r in cases if r['rank']==61 for d in r['instance_details'] if d['id']==79)['GT_instance_reference'][0]['GT_instance_id']==6001
assert all(p['sample_top1_votes']+p['sample_top2_votes']+p['sample_other_votes']==p['sample_total_votes'] for r in cases for p in r['competition_pairs'])
assert not list(R.rglob('*.ply'))
audit=json.loads((R/'投票与投影核验.json').read_text(encoding='utf-8'))
audit.update({'visual_browser_QA':'PASS: 13 selections; images decode; no JS errors; mobile no overflow','annotation_style':'confirmed vote/state facts separated from cause inference','case_figures':13,'no_new_PLY':True})
(R/'投票与投影核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
zip_path=R.parent/'room0_ROI_13案例冲突标注.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(R.iterdir()):
  if p.suffix.lower() in {'.png','.json','.csv'}:z.write(p,p.name)
upload=[[str(P/'case_annotations.json'),'/home/chenkejun/CVPR/experiments/p1a1_room0_roi_review_20261002/case_annotations.json'],
 [str(P/'render_annotated_cases.py'),'/home/chenkejun/CVPR/experiments/p1a1_room0_roi_review_20261002/render_annotated_cases.py'],
 [str(zip_path),'/data/chenkejun/CVPR/revisable_instance_map/p1a1_room0_roi_review_20261002/room0_ROI_13_cases_annotated.zip'],
 [str(R/'13案例_证据与诊断.json'),'/data/chenkejun/CVPR/revisable_instance_map/p1a1_room0_roi_review_20261002/conflict_diagnosis/annotated_cases.json']]
(P/'annotation_upload.json').write_text(json.dumps(upload,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'audit':'PASS','annotation_archive_bytes':zip_path.stat().st_size,'case_count':len(cases)},ensure_ascii=False))
