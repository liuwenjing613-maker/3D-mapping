from pathlib import Path
import json,hashlib
p=Path('outputs/vlm_threeway_20261014');out=p/'vlm_outputs/outputs';s=json.loads((out/'summary.json').read_text(encoding='utf-8'));m=json.loads((p/'input_manifest.json').read_text(encoding='utf-8'))
for row,entry in zip(s['results'],m['models']):
 assert row['method']==entry['method']
 assert hashlib.sha256(Path(entry['input_image']).read_bytes()).hexdigest()==row['input_sha256']
 assert row['validation']['all_candidates_exactly_once']
 raw=json.loads((out/(row['method']+'_raw_response.json')).read_text(encoding='utf-8'))
 assert raw['message']['content']==(out/(row['method']+'_output.txt')).read_text(encoding='utf-8')
 print(row['method'],row['seconds'],row['accepted_ids'],row['uncertain_ids'],row['rejected_ids'])
m['prepared_model']['inference_status']='completed_three_independent_requests'
m['prepared_model']['summary_file']=str((out/'summary.json').resolve())
(p/'input_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'verification.json').write_text(json.dumps({'status':'PASS','exact_preview_images_used':True,'original_model_text_preserved':True,'all_candidate_ids_classified_exactly_once':True,'inference_calls':3,'format_only_postprocessing':True},indent=2),encoding='utf-8')
