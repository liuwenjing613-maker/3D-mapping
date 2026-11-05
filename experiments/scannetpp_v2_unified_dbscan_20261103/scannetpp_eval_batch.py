"""Dataset-scope adapter around unchanged current-main cmd_eval_batch.

The installed CLI exposes ScanNet++ scope validation for adapt-surface/pairs only.
Validate every canonical input using that same scope function, then call its
unchanged pooled implementation. No historical/debug protocol or GT override.
"""
import sys,json,csv
from pathlib import Path
from dbscan_common import C,MAIN,read,sha,dump,protocol
sys.path.insert(0,str(MAIN))
from unified_eval.cli import build_parser,cmd_eval_batch
from unified_eval.current_protocol import require_current_protocol,reject_current_debug_overrides
from unified_eval.scannetpp_scope import require_scannetpp_scope
lock=protocol();plan=read(C/'plan.json')
args=build_parser().parse_args(['eval-batch',*sys.argv[1:]])
assert not args.historical_protocol
require_current_protocol(args.config,historical_protocol=False);reject_current_debug_overrides(args)
entries=read(args.scenes)['scenes'];expected={s['scene']:s['gt'] for s in plan['scenes']};seen=set();sources=[]
for row in entries:
    gt=Path(row['gt']);require_scannetpp_scope(gt,Path(plan['scope_lock']))
    matching=[s for s,x in expected.items() if str(gt.resolve())==str(Path(x['gt_path']).resolve()) and sha(gt)==x['gt_sha256']];assert len(matching)==1
    scene=matching[0];assert scene not in seen;seen.add(scene)
    prediction=Path(row['prediction']);adapter=read(prediction.parent/'adapter_manifest.json')
    assert adapter['gt_file_sha256']==expected[scene]['gt_sha256'] and adapter['profile_config_sha256']==lock['config_sha256']
    sources.append(dict(scene=scene,gt_sha256=sha(gt),prediction_sha256=sha(prediction),diagnostic_sha256=sha(row['diagnostic_prediction']),adapter_manifest_sha256=sha(prediction.parent/'adapter_manifest.json')))
assert seen==set(expected)
cmd_eval_batch(args)
# Verify its per-scene computations reproduce the already validated adapt-surface outputs.
with (args.out/'per_scene.csv').open() as f:rows=list(csv.DictReader(f))
metrics={read(Path(row['prediction']).parent/'metrics.json')['per_scene'][0]['scene_id']:read(Path(row['prediction']).parent/'metrics.json')['per_scene'][0] for row in entries}
for row in rows:
    m=metrics[row['scene_id']]
    for key in ['CA_AP_uniform','CA_AP50_uniform','CA_AP25_uniform']:assert abs(float(row[key])-m[key])<1e-12,key
    for key in ['TP','FP','FN']:assert int(row[key])==m['CA_PQ'][key],key
summary=read(args.out/'summary.json');assert summary['CA_PRF1_0_5']['TP']+summary['CA_PRF1_0_5']['FN']==70
dump(args.out/'scannetpp_scope_adapter_audit.json',dict(status='PASS',scope_lock_sha256=sha(plan['scope_lock']),GT=70,inputs=sources,per_scene_reproduces_validated_adapt_surface=True,unchanged_main_batch_function=True,main_cli_sha256=sha(MAIN/'unified_eval/cli.py'),profile_config_sha256=lock['config_sha256'],frozen=False))
