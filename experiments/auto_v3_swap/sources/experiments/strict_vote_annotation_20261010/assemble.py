from pathlib import Path
import json
H=Path(__file__).resolve().parent; W=H.parents[1]
C='/home/chenkejun/CVPR/experiments/strict_vote_annotation_20261010'
R='/data/chenkejun/CVPR/results/strict_vote_annotation_20261010'
old=W/'outputs/room2_selection_top30_20261008'
roi=W/'worktrees/ROI-repair/experiments/roi_repair/roi_selection_cut30'
def save(p,t):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,encoding='utf-8',newline='\n')
t=(roi/'common.py').read_text(encoding='utf-8')
t=t.replace("'/data/chenkejun/CVPR/revisable_instance_map/roi_selection_cut30_20261006'",repr(R+'/catalog'))
t=t.replace("surface=BASE/f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz'", "surface=Path(f'/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/{scene}/abstain/final/instance_surface.npz')")
t=t.replace("evidence=BASE/f'native/{scene}/P1-A1/surface_p0/surface_evidence.npz'", "evidence=Path(f'/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/{scene}/abstain/surface_evidence.npz')")
save(H/'common.py',t)
t=(roi/'build_candidates.py').read_text(encoding='utf-8')
a=t.index(" if scene in ['room0','room2']:");b=t.index(" status('准备完整区域与固定索引')",a)
t=t[:a]+t[b:]
t=t.replace("scene in ['room0','room2']",'False').replace('P1-A1 frozen whole-connected','P1-A1-strict-vote unrepaired whole-connected').replace('original final P1-A1','strict-vote final P1-A1')
# Read-only source verification before any ROI generation; fixed geometry and frozen prediction hash.
t=t.replace(" surface,evidence,graphpath,config=paths(scene)"," surface,evidence,graphpath,config=paths(scene)\n freeze=read('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009/predictions_freeze.json')['scenes'][scene]['predictions']['strict_baseline_final']\n assert str(surface)==freeze['path'] and sha(surface)==freeze['sha256']")
save(H/'build_candidates.py',t)
uploads=[[str(H/n),C+'/'+n] for n in ['common.py','build_candidates.py']]
for scene in ['room0','room2']:
 d=H/scene; remote=C+'/'+scene; root=R+'/'+scene
 for n in ['prepare_inputs.py','infer_cropformer.py','infer_sam.py','render_helpers.py','render_annotation.py']:
  t=(old/n).read_text(encoding='utf-8')
  t=t.replace('/home/chenkejun/CVPR/experiments/room2_selection_top30_20261008',remote)
  t=t.replace('/data/chenkejun/CVPR/revisable_instance_map/room2_selection_top30_20261008',root)
  t=t.replace('/data/chenkejun/CVPR/revisable_instance_map/roi_selection_cut30_20261006/room2',R+'/catalog/'+scene)
  t=t.replace('room2-top30-20261008',scene+'-strict-vote-top30-20261010')
  t=t.replace('room2',scene)
  # Force fresh views/crops from strict candidates. Cached model predictions require identical RGB only.
  if n=='prepare_inputs.py':
   t=t.replace("CATALOG/'selection.json',",'')
   t=t.replace("old_cases={c['uid']:c for c in old['cases'] if c['scene']=='"+scene+"'}",'old_cases={}')
   t=t.replace('previous_five_cases_candidates_preserved=True','previous_candidates_used_as_human_choices=False')
   t=t.replace('original CONFLICT absolute','strict-vote CONFLICT absolute')
  if n=='render_annotation.py':
   t=t.replace("len(manifest['cache_reuse'])","len(manifest['cache_reuse'])")
   t=t.replace('previous_five_cases_candidates_preserved=True','previous_candidates_used_as_human_choices=False')
   t=t.replace('original absolute conflict','strict-vote absolute conflict')
   t=t.replace('unified v3; candidate selection only, no quality metrics claimed','object_observed_repair / revision 2; candidate selection only; GT not used')
   t=t.replace('cvpr-'+scene+'-top30-selection-20261008-v1','cvpr-'+scene+'-strict-vote-selection-20261010-v1')
   t=t.replace(scene+'_前30_修复候选选择_20261008.json',scene+'_strict_vote_人工标注_20261010.json')
  save(d/n,t);uploads.append([str(d/n),remote+'/'+n])
save(H/'upload.json',json.dumps(uploads,ensure_ascii=False))
save(H/'mkdir.sh','mkdir -p '+C+'/room0 '+C+'/room2 '+R+'/catalog '+R+'/room0 '+R+'/room2\n')
save(H/'launch_catalog.sh',f'''python3 - <<'PY'
import subprocess,os
from pathlib import Path
for s in ['room0','room2']:
 r=Path('{R}')/s
 with (r/'catalog.log').open('ab') as log:
  p=subprocess.Popen(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-u','{C}/build_candidates.py',s],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}})
 (r/'catalog.pid').write_text(str(p.pid));print(s,p.pid)
PY
''')
save(H/'progress.sh',f'''python3 - <<'PY'
from pathlib import Path
r=Path('{R}')
for s in ['room0','room2']:
 print(s)
 for name in ['catalog','prepare','cropformer','sam2_1','render']:
  p=r/s/(name+'.log')
  if p.exists():print(name,p.read_text()[-500:])
PY
''')
print('ASSEMBLED',len(uploads))
