from pathlib import Path
import hashlib,json,zipfile
C=Path('/home/chenkejun/CVPR/experiments/frame_point_replacement_20261021');R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(R/'final_complete.json')['status']=='PASS'
assert read(R/'all_runs_complete.json')['status']=='PASS'
assert read(R/'control_scores_complete.json')['status']=='PASS'
out=R/'code_and_receipts.zip';assert not out.exists()
with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(C.glob('*.py')):z.write(p,'code/'+p.name)
 z.write(C/'config.json','code/config.json')
 for p in sorted(R.glob('*.json')):z.write(p,'results/'+p.name)
 for p in sorted((R/'diagnostics').glob('*.json')):z.write(p,'results/diagnostics/'+p.name)
 for group in ['automatic','manual']:
  for scene in ['room0','room1','room2','office0','office1','office2','office3','office4']:
   d=R/group/scene
   for name in ['predictions_freeze.json','policy.json','mapping_complete.json','evaluation_complete.json','evaluation_failure.json','evaluation_recovery.json','evaluation_r2/protocol_freeze.json','direct_controls_r2/prediction_lock.json']:
    p=d/name
    if p.exists():z.write(p,'results/'+str(p.relative_to(R)))
receipt=read(R/'delivery_receipt.json');receipt['files'][out.name]=sha(out)
receipt['files']['final_complete.json']=sha(R/'final_complete.json')
(R/'delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='PASS',archive=str(out),bytes=out.stat().st_size,sha256=sha(out))))
