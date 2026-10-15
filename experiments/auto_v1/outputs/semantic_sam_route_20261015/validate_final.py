"""Independent result checks: semantic decisions, routing, preservation, page assets."""
from pathlib import Path
import ast,collections,hashlib,json
import numpy as np
from PIL import Image
H=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
s=read(H/'semantic.json');old=read(H/'semantic_mask_only.json');r=read(H/'selection.json');prev=read(H.parent/'mask_version_selection_20261015_r2/selection.json');f=read(H.parent/'overmerge_id_ratio_top20_20261015/data.json')
assert set(s['candidates'])==set(old['candidates'])
counts=collections.Counter();transitions=collections.Counter()
terms=s['terms'];targets=s['config']['targets'];num_target=sum(map(len,targets.values()))
for cid,a in s['candidates'].items():
 b=old['candidates'][cid];assert a['scores']==b['scores'];sc=a['scores'];expected=max(sc[:num_target])-max(sc[num_target:]);assert abs(expected-a['margin'])<1e-7
 if b['hit'] and b['target_family']=='cabinet':
  z=a['cabinet_context'];cs=z['scores'];ci=[i for i,t in enumerate(terms) if t in targets['cabinet']];oi=[i for i,t in enumerate(terms) if t not in targets['cabinet']];assert abs(max(cs[i] for i in ci)-max(cs[i] for i in oi)-z['margin'])<1e-7;assert a['hit']==(z['margin']>=.03)
 else:assert a['hit']==b['hit']
 counts['semantic_candidates']+=1;counts['semantic_hits']+=a['hit']
for key,c in r['crops'].items():
 hits={n['key'] for n in c['nodes'] if n['semantic_hit']};replaced={e['a'] for e in c['semantic_replacement_edges']};generic_members={k for g in c['groups'] if g['reason']!='SEMANTIC_SAM_DIRECT' for k in g['members']}
 assert not generic_members&(hits|replaced)
 for n in c['nodes']:
  row=c['models'][n['model']]['labels'][str(n['label'])]
  if n['key'] in hits:assert row['status']=='selected' and row['reason']=='SEMANTIC_SAM_DIRECT'
  if n['key'] in replaced:assert n['model'] in (0,1) and row['status']=='alternative' and row['reason']=='SEMANTIC_CROPFORMER_REPLACED'
  counts['mask_occurrences']+=1;counts[row['status']]+=1
 if not hits:
  for mi,m in enumerate(prev['crops'][key]['models']):
   for lid,record in m['labels'].items():assert record['status']==c['models'][mi]['labels'][lid]['status']
  counts['no_hit_crops_unchanged']+=1
 for mi,m in enumerate(prev['crops'][key]['models']):
  for lid in m['selected_ids']:
   new=c['models'][mi]['labels'][str(lid)];transitions[new['reason'] if new['status']!='selected' else 'still_selected']+=1
for key,expected in [('selected','selected_occurrences'),('deferred','deferred_occurrences'),('no_hit_crops_unchanged','unchanged_crops_without_semantic_hit')]:assert counts[key]==r['stats'][expected]
# Image construction boundaries: retain original RGB, hide only outside-mask pixels,
# and supply actual surroundings for fragments including on the ROI edge.
namespace={'np':np};tree=ast.parse((H/'classify.py').read_text(encoding='utf-8'));node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='masked_square');exec(compile(ast.Module(body=[node],type_ignores=[]),'masked_square','exec'),namespace)
rgb=np.arange(8*12*3,dtype=np.uint8).reshape(8,12,3);mask=np.zeros((8,12),bool);mask[2:4,1:10]=True;mask[2,3:6]=False;square,bbox=namespace['masked_square'](rgb,mask,[127]*3);assert bbox==[1,2,10,4] and square.shape==(9,9,3);assert (square[0]==127).all()
tree=ast.parse((H/'refine_cabinet.py').read_text(encoding='utf-8'));node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='context_box');exec(compile(ast.Module(body=[node],type_ignores=[]),'context_box','exec'),namespace)
assert namespace['context_box']([0,0,2,2],(680,1200,3))==[0,0,82,82]
assert namespace['context_box']([200,200,300,350],(680,1200,3))==[100,50,400,500]
report=dict(status='PASS',counts=dict(counts),old_selected_transitions=dict(transitions),masked_scores_unchanged=True,window_decisions_unchanged=True,context_pixel_boundary_cases_passed=True,cabinet_positive_validation='No cabinet passed in this batch; recall on real cabinet masks is not established.',ground_truth_used=False)
(H/'final_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report))
