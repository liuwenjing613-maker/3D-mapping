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

if a.audit_results and cfg.get('paper_final'):
 r=Path(cfg['result_root'])
 d=json.loads((r/'final_audit_and_comparison.json').read_text())
 assert d['status']=='PASS_COMPLETE' and d['new_actual_commits']==32 and d['reused_first50_commits']==8
 assert hashlib.sha256((r/'comparison/comparison.json').read_bytes()).hexdigest()==d['comparison_sha256']
 cmp=json.loads((r/'comparison/comparison.json').read_text())
 assert cfg['pooled']==cmp['results']['online_final_trusted']
 assert len(d['retained_label_audit'])==8 and all(x['assigned_labels_unchanged'] and x['new_labels_vote_supported'] for x in d['retained_label_audit'])
 print('PASS Replica paper-final actual execution, retained-label audit and exact metrics')
