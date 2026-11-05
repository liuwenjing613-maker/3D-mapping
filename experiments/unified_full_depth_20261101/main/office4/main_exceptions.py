"""Bind historical main exceptions to exact binary masks, never to newly assigned G numbers."""
from common_continue import *
import copy
def main():
 selection_path=CANDIDATE_ROOT/'v2/selection.json';manifest_path=CHOSEN
 selection=read(selection_path);chosen=read(manifest_path);original=read(S/'seed_manifest.json')
 specifications=read(CONT/'historical_exceptions.json');audit=[]
 keep=set(selection['retained_group_ids']);records={r['group_id']:r for r in selection['records']}
 restored=[]
 for old in specifications['pr_retain']:
  matches=[g for g in original['groups'] if g['primary']['mask_sha256']==old['mask_sha256'] and g['primary']['frame']==old['frame']]
  assert len(matches)<=1
  if not matches:audit.append(dict(**old,applied=False,reason='original binary primary not selected on new geometry'));continue
  g=matches[0];gid=g['id'];record=records[gid]
  if gid in keep:audit.append(dict(**old,current_group_id=gid,applied=False,reason='already retained'));continue
  if record['terminal_stage']!='pr_filter':
   audit.append(dict(**old,current_group_id=gid,applied=False,reason='excluded at an earlier stage; original exception only covered p/r'));continue
  keep.add(gid);restored.append(gid)
  record.update(decision='retain',terminal_stage='historical_exact_mask_override')
  record['stage_decisions'].append(dict(stage='historical_exact_mask_override',decision='retain',evidence=old))
  audit.append(dict(**old,current_group_id=gid,applied=True,reason='same original frame and binary mask, final p/r exclusion'))
 selection['retained_group_ids']=sorted(keep);selection['manual_restored_group_ids']=restored
 selection['final_pr_excluded_group_ids']=sorted(set(selection['final_pr_excluded_group_ids'])-set(restored))
 selection['stages'][-1].update(retained=len(keep),retained_group_ids=sorted(keep),removed=len(selection['final_pr_excluded_group_ids']),removed_group_ids=selection['final_pr_excluded_group_ids'],manual_restored_group_ids=restored)
 chosen['groups']=[dict(copy.deepcopy(g),original_track_id=i) for i,g in enumerate(original['groups'],1) if g['id'] in keep]
 chosen['manual_overrides']=audit
 dump(selection_path,selection);chosen['selection_sha256']=digest(selection_path);dump(manifest_path,chosen)
 done=read(CANDIDATE_ROOT/'v2/complete.json');done.update(retained=len(keep),manual_overrides=audit,selection_sha256=digest(selection_path),manifest_sha256=digest(manifest_path));dump(CANDIDATE_ROOT/'v2/complete.json',done)
 source=read(CANDIDATE_ROOT/'grouping_r2/semantic_exclusions.json').get('manual_source_corrections',{})
 dump(CANDIDATE_ROOT/'main_exceptions.json',dict(status='PASS',PR_exceptions=audit,source_exceptions=source,
  identity_binding='exact original binary-mask/frame identity; generated G/M numbers are never used as identity',GT_used=False))
if __name__=='__main__':main()
