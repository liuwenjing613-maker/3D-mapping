from pathlib import Path
import hashlib,json
WORK=Path(__file__).resolve().parent
WEB=WORK.parents[1]/'results/固定案例_三模型对比_20261006/continuous_tracking_ablation_20261008'
complete=json.loads((WORK/'server_results/complete.json').read_text(encoding='utf-8'))
facts=json.loads((WORK/'final_publish_receipt.json').read_text(encoding='utf-8'))['facts']
record={'status':'PASS','experiment':'seed-centered independent forward/reverse continuity gap0 vs gap1',
 'only_algorithmic_intervention':'Suppress saved raw track masks after reaching the per-direction empty-frame limit',
 'empty_definition':'area=0 of original max-logit-partitioned native 1200x680 mask; count every original frame',
 'SAM_GPU_rerun':False,'all_four_vote_repair_and_geodesic_diffusion_runs_completed':True,
 'GT_used_for_filter_or_repair':False,'original_outputs_preserved':True,'facts':facts,
 'numeric_gates_weights_diffusion_and_v3_code_protocol_flags_unchanged':True,
 'reporting_only_changes':['Per-single-identity fixed-surface diagnostics, as in the original evaluator, preserving multiple selected parts of one GT'],
 'display_only_changes':['Room2 restored identity 113 lacked a table entry; use the original renderer palette [249,74,245]; preserve every explicit original color',
     'Same-frame paired native mask canvases, optional black-background masks',
     'Four-state full-point 3D pack, synchronized cameras and per-identity visibility controls'],
 'not_implemented_in_current_algorithm':['GT-assisted filtering','Reacquisition/new seeds','Changing the original observation retirement behavior',
     'Weakening replacement gates','New appearance/3D identity re-identification'],
 'streaming_tracker_generated_but_not_GPU_tested':'track_with_continuity.py; separate experimental runner, not installed as original tracker',
 'server_code_directory':'/home/chenkejun/CVPR/experiments/continuous_tracking_ablation_20261008',
 'server_result_directory':'/data/chenkejun/CVPR/revisable_instance_map/continuous_tracking_ablation_20261008',
 'local_PLY_directory':str(WORK.parents[1]/'可视化ply/连续性截断_20261008'),
 'review_URL':'http://127.0.0.1:8785/continuous_tracking_ablation_20261008/index.html',
 'evaluation_protocol_status':'Same current v3 debug protocol as originals, still unfrozen; not formal benchmark',
 'checks':['10 continuity tests','4000 original native frame mask validations','original frozen inputs SHA checks',
    'complete frame replay equals delta vote ledger','exact withdrawal rollback','outside-scope final changes zero',
    'colored full PLY geometry/IDs/colors match aligned binary packs and frozen server SHA'],
 'source_file_SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in WORK.glob('*.py')}}
(WORK/'execution_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Execution record saved')
