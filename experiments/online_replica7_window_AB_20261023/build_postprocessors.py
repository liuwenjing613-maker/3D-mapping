"""Adapt existing independent audit, main R2 and lossless PLY export to this batch."""
from pathlib import Path
import ast
C=Path(__file__).resolve().parent
S=C.parent/'online_full_scene_room0_AB_20261023'
REMOTE='/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023'
OUT='/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023'
def put(name,s):ast.parse(s);(C/name).write_text(s,encoding='utf-8')
def change(s,a,b):assert a in s,a;return s.replace(a,b)
s=(S/'audit_completed.py').read_text(encoding='utf-8').replace('import hashlib, json, time','import hashlib, json, time, sys')
s=change(s,"OUT = Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')",f"BATCH=Path({OUT!r})\nSCENE,GROUP=sys.argv[1:3]\nassert SCENE in ['room1','room2','office0','office1','office2','office3','office4'] and GROUP in ['A','B']\nOUT=BATCH/SCENE")
a=s.index('    previous_path =');b=s.index('    previous = read(previous_path)',a)
s=s[:a]+"    old={50:0,100:50,200:100,300:200,400:300}[n]\n    previous_path=OUT/str(old)/group/'repair/commit_complete.json' if old else OUT/'initial'/group/'initial_complete.json'\n"+s[b:]
s=s.replace('frame{n-99}', 'frame{old+1}').replace("and first['repaired_state_used']", "and first['repaired_state_used']==(old>0)").replace("(n - 100) * 5", "old * 5")
s=s.replace("    commit = read(commit_path)","    commit = read(commit_path)\n    policy=read(root/'candidates/window_source_policy.json');assert policy['status']=='PASS'")
a=s.index('def main():')
s=s[:a]+'''def main():
    rows=[check(GROUP,n) for n in [50,100,200,300,400]]
    assert all(r['status']=='PASS' for r in rows)
    source=HERE.parent/'online_full_scene_room0_AB_20261023'
    for n in [50,100,200,300,400]:
        for name in ['track_case.py','gate_v1.py','objectwise_ops.py','strict_vote_ops.py','joint_batch_ops.py','tracking_policy.json','gate_policy.json']:
            assert sha(HERE/SCENE/'tracking_code'/str(n)/GROUP/name)==sha(source/'tracking_code/400'/GROUP/name)
    assert sha(HERE/SCENE/'track_pure_gate.py')==sha(source/'track_pure_gate.py')
    result=dict(status='PASS',scene=SCENE,branch=GROUP,mapping_frames=400,generated_unix=time.time(),checkpoints=rows,
        source_files_verified=len(HASH_CACHE),initialization='zero frames; no inherited room0 history',
        scope='A/B candidate-search strategies; checkpoint surface reconstruction; window-only SAM special source at all checkpoints',
        limitation='Earlier room0 real original/optimized closed-loop equivalence was not exact. This run does not resolve it.')
    dest=OUT/'verification'/f'{GROUP}_causal_audit.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\\n')
    print(json.dumps(dict(status='PASS',scene=SCENE,branch=GROUP,verified_files=len(HASH_CACHE),path=str(dest))))
if __name__=='__main__':main()
'''
put('audit_branch.py',s)
s=(S/'evaluate_final.py').read_text(encoding='utf-8')
s=change(s,"OUT = Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')\nGROUP = sys.argv[1]",f"BATCH=Path({OUT!r})\nSCENE,GROUP=sys.argv[1:3]\nGT_COUNTS=dict(room1=45,room2=52,office0=31,office1=23,office2=41,office3=58,office4=28)\nassert SCENE in GT_COUNTS\nOUT=BATCH/SCENE")
s=s.replace("['canonical_GT']['room0']","['canonical_GT'][SCENE]").replace("'scene': 'room0'","'scene': SCENE").replace('online_checkpoint_room0_{GROUP}','online_window_{SCENE}_{GROUP}')
s=s.replace("OUT / 'adapter_audit.json'","BATCH / 'adapter_audit.json'").replace("'scene_evaluable_GT': 72","'scene_evaluable_GT': GT_COUNTS[SCENE]")
s=s.replace("    mapping = read(","    assert read(OUT/'verification'/f'{GROUP}_causal_audit.json')['status']=='PASS'\n    mapping = read(")
put('evaluate_branch.py',s)
s=(S/'export_final.py').read_text(encoding='utf-8')
s=change(s,"OUT = Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')",f"OUT=Path({OUT!r})")
s=change(s,"group = sys.argv[1]","scene,group,count,stage=sys.argv[1:5]\nassert scene in ['room1','room2','office0','office1','office2','office3','office4']\nn=int(count);assert n in [50,100,200,300,400] and stage in ['before','after']")
s=s.replace("root = OUT / '400' / group","root=OUT/scene/str(n)/group").replace("commit['processed_frames'] == 400","commit['processed_frames'] == n")
s=change(s,"source = Path(commit['final_map'])\nassert hashlib.sha256(source.read_bytes()).hexdigest() == commit['final_map_sha256']", "source=Path(commit['final_map']) if stage=='after' else root/'mapping/baseline/final/instance_surface.npz'\nsource_hash=hashlib.sha256(source.read_bytes()).hexdigest()\nif stage=='after':assert source_hash==commit['final_map_sha256']")
s=s.replace("dest = OUT / 'deliverables'","dest=OUT/'deliverables'/scene").replace('dest.mkdir(exist_ok=True)','dest.mkdir(parents=True,exist_ok=True)').replace("f'room0_{group}_400_repaired.ply'","f'{scene}_{group}_{n}_{stage}.ply'").replace("f'room0_{group}_400_export.json'","f'{scene}_{group}_{n}_{stage}_export.json'")
s=s.replace("'branch': group,", "'scene':scene,'branch':group,'checkpoint':n,'stage':stage,").replace("'source_sha256': commit['final_map_sha256']","'source_sha256':source_hash")
put('export_checkpoint.py',s)
print('POSTPROCESSORS_READY')
