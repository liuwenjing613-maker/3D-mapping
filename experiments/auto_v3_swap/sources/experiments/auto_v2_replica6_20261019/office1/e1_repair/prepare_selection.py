"""Translate the completed automatic selection, with no room2 exceptions."""
from pathlib import Path
import hashlib,json
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office1')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 complete=read(R/'v2/complete.json');assert complete['status']=='PASS'
 source=read(R/'v1/seed_manifest.json');chosen=read(R/'v2/seed_manifest.json');decision=read(R/'v2/selection.json');kept=set(decision['retained_group_ids'])
 assert not decision['manual_restored_group_ids'] and kept=={g['id'] for g in chosen['groups']}
 assert sha(R/'v2/selection.json')==complete['selection_sha256'] and sha(R/'v2/seed_manifest.json')==complete['manifest_sha256']
 inc=[];exc=[]
 for tid,g in enumerate(source['groups'],1):
  p=g['primary'];row=dict(group_id=g['id'],original_track_id=tid,mask_id=p['id'],frame=p['frame'],mask_sha256=p['mask_sha256'],mask_relative_path=p['mask'])
  (inc if g['id'] in kept else exc).append(row)
 selection=dict(schema='automatic-office1-original-E1-v1',status='confirmed',version='auto-v2-office1-semantic-pr-20261019',qa_only=False,scene='office1',user_request='User-approved latest automatic scheme: exempt window/cabinet SegFormer top classes from stage 5 only; rerun original E1 and main R2',included=inc,excluded=exc,manifest_sha256=sha(R/'v1/seed_manifest.json'),selected_source_manifest=str(R/'v2/seed_manifest.json'),selected_source_manifest_sha256=complete['manifest_sha256'],selected_source_decisions=str(R/'v2/selection.json'),selected_source_decisions_sha256=complete['selection_sha256'],pr_filter_applied=True,semantic_pr_exempt_group_ids=decision['semantic_pr_exempt_group_ids'],user_retained_exceptions=[],baseline_sha256=read(R/'v2/measurements.json')['baseline_sha256'])
 assert inc
 p=C/'confirmed_selection.json';assert not p.exists();p.write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
 (R/'e1_repair').mkdir(exist_ok=True);(R/'e1_repair/confirmed_selection.json').write_bytes(p.read_bytes())
 print('SELECTION',len(inc),len(exc),flush=True)
if __name__=='__main__':main()
