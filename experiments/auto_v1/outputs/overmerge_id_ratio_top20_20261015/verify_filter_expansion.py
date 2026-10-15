from pathlib import Path
import json
H=Path(__file__).resolve().parent;ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
old=read(ROOT/'room2_E1_overmerge_audit_20261014/data.json');new=read(H/'data.json')
assert new['rule']==old['rule']
previous={c['id']:c for c in old['candidates']};expanded={c['id']:c for c in new['candidates']}
assert previous.keys()<=expanded.keys()
for cid,c in previous.items():
 for f in ['N','ids','states','n2','r2','decision','frame','model','pixels','box']:assert c[f]==expanded[cid][f],(cid,f)
for c in expanded.values():
 N=sum(c['states']);assert N==c['N'];n2=c['ids'][1][1] if len(c['ids'])>1 else 0
 assert c['n2']==n2
 expected='reject' if n2>=10 and 5*n2>=N else ('no_support' if N==0 else 'keep')
 assert c['decision']==expected
report=dict(status='PASS',rules_unchanged=True,previous_candidates_exactly_preserved=len(previous),total_candidates_checked=len(expanded),new_candidates=len(expanded)-len(previous),stats=new['stats'])
(H/'filter_expansion_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report))
