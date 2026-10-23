from pathlib import Path
import json,hashlib,shutil,subprocess,time
R=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023')
C=R/'experiments/online_replica7_window_AB_20261023'
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
E=C/'evidence/launch';E.mkdir(parents=True,exist_ok=True)
for name in ['run_plan.json','adapter_audit.json','catalog_path_adapter_fix.json','frozen_docstring_restore.json','frozen_kernel_audit.json','launch.json','post_launch.json']:
 shutil.copyfile(O/name,E/name)
files=[]
for p in C.rglob('*'):
 if not p.is_file() or p.is_symlink() or 'evidence' in p.relative_to(C).parts or '__pycache__' in p.parts:continue
 if p.suffix not in ['.py','.json','.txt','.js','.css','.html']:continue
 assert p.stat().st_size<10*1024*1024,str(p)
 files.append(dict(path=str(p.relative_to(C)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
(E/'code_manifest.json').write_text(json.dumps(dict(created_unix=time.time(),files=files),indent=2)+'\n')
rel=str(C.relative_to(R))
subprocess.run(['git','add','--',rel],cwd=R,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only','--',rel],cwd=R,text=True).splitlines()
assert staged and all(p.startswith(rel+'/') for p in staged)
subprocess.run(['git','commit','--only','-m','Run seven Replica scenes online with window-only SAM routing', '--',rel],cwd=R,check=True,stdout=subprocess.DEVNULL)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
receipt=dict(status='PASS',commit=commit,repository=str(R),code_files=len(files),staged_files=len(staged),pushed=False)
(O/'launch_git_archive.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
