from pathlib import Path
import argparse,hashlib,json,subprocess
p=argparse.ArgumentParser();p.add_argument('--deployed',action='store_true');p.add_argument('--audit-results',action='store_true');a=p.parse_args()
b=Path(__file__).resolve().parent;m=json.loads((b/'source_manifest.json').read_text());cfg=json.loads((b/'frozen.json').read_text())
errors=[]
for e in m['files']:
 for f in ([b/e['file'],Path(e['deployed'])] if a.deployed else [b/e['file']]):
  if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=e['sha256']:errors.append(str(f))
if errors:raise SystemExit('Source mismatch: '+repr(errors))
print('PASS source files:',len(m['files']))
if a.audit_results:
 if not a.deployed:raise SystemExit('--audit-results requires --deployed')
 subprocess.run(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python',cfg['source_root']+'/audit_uniform_actual_commits.py','--require-complete'],check=True)
