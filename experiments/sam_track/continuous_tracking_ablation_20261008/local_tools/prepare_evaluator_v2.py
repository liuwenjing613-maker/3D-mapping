"""Keep frozen repair code intact; fix supplementary reports for multiple seed parts of one GT."""
from pathlib import Path
import hashlib
import json
import difflib

WORK=Path(__file__).resolve().parent
source=(WORK/'run_ablation.py').read_text(encoding='utf-8')
old="'fixed_identity_diagnostics':engine.f.target_diagnostics(mapping,arrays['instance_id'],gt.instance_id,valid,identity_gt),"
assert source.count(old)==1
new="'fixed_identity_diagnostics':{'targets':[engine.f.target_diagnostics(mapping,arrays['instance_id'],gt.instance_id,valid,{pid:gid})['targets'][0] for pid,gid in sorted(identity_gt.items())], 'multiple_selected_parts_of_same_GT_preserved':True, 'same_single_identity_diagnostics_as_original_evaluate_batch':True},"
modified=source.replace(old,new)
modified=modified.replace("parser.add_argument('--stage',choices=['repair','evaluate'],required=True)","parser.add_argument('--stage',choices=['evaluate'],required=True)")
assert modified!=source
modified=modified.replace("'evaluator_sha256':engine.r.sha(e.__file__),'flags':e.FLAGS,'protocol_sha256':engine.r.sha(e.PROTOCOL),", "'evaluator_sha256':engine.r.sha(e.__file__),'flags':e.FLAGS,'protocol_sha256':engine.r.sha(e.PROTOCOL),\n        'supplementary_reporting_wrapper':str(Path(__file__).resolve()),'supplementary_reporting_wrapper_sha256':engine.r.sha(__file__),")
target=WORK/'evaluate_ablation_v2.py'
target.write_text(modified,encoding='utf-8')
(WORK/'evaluator_reporting_diff.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile='frozen_run_ablation.py',tofile='evaluate_ablation_v2.py')),encoding='utf-8')
(WORK/'evaluator_reporting_revision.json').write_text(json.dumps({'status':'REPORTING_FIX_ONLY',
 'frozen_repair_wrapper_modified':False,'donor_sha256':hashlib.sha256(source.encode()).hexdigest(),
 'reporting_wrapper_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
 'change':'Call existing supplementary fixed-surface diagnostics once per persistent identity, as the original evaluate_batch does; preserve multiple selected parts of the same GT',
 'human_masks_votes_diffusion_or_metric_evaluator_modified':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Prepared separate evaluation wrapper: fixed per-identity supplementary statistics; frozen repair and v3 unchanged')
