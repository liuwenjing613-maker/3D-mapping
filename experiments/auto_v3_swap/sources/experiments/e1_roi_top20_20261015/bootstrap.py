"""Extend the inspected top-10 workflow without changing selection/model rules."""
from pathlib import Path
import re
H=Path(__file__).resolve().parent;OLD=H.parent/'e1_roi_top10_20261014'
files=['prepare_inputs.py','audit_inputs.py','normalize_manifest.py','infer_cropformer.py','infer_sam.py','render_helpers.py','render_annotation.py','finalize_projections.py','pipeline.py','transfer.py','download_resume.py','unpack_verify.py','build_page.py','verify_delivery.py']
for name in files:
 t=(OLD/name).read_text(encoding='utf-8').replace('top10','top20').replace('20261014','20261015')
 # These numeric tokens in these scripts are batch dimensions, except RGB blue=30.
 if name!='render_helpers.py':t=re.sub(r'\b19\b','32',t);t=re.sub(r'\b90\b','180',t);t=t.replace('共90个','共180个')
 for a,b in [('ranked_entries=30','ranked_entries=60'),('ranked_entries:30','ranked_entries:60'),("'ranked_entries':30","'ranked_entries':60"),('==30','==60'),('30 entries','60 entries'),('30 个','60 个'),('前 10','前 20'),('前10','前20'),('前十','前二十'),('<=10','<=20'),('[:10]','[:20]'),('==10','==20'),('g:10','g:20'),('range(1,11)','range(1,21)'),('crops=412',"crops=len(v['crops'])")]:t=t.replace(a,b)
 if name=='prepare_inputs.py':
  t=t.replace("OLD=Path('/data/chenkejun/CVPR/results/strict_vote_annotation_20261010/room2')","OLD=Path('/data/chenkejun/CVPR/results/e1_roi_top10_20261014')")
  t=t.replace('Freeze the original room2 conflict top 30','Extend E1 prerepair room2 C/U/T to top 20 each')
 if name=='pipeline.py':
  t=t.replace("assert run('render_annotation').wait()==0,'render failed'","assert run('render_annotation').wait()==0,'render failed'\n assert run('finalize_projections').wait()==0,'projection finalization failed'")
 if name=='transfer.py':
  t=t.replace("'normalize_manifest.py']","'normalize_manifest.py','finalize_projections.py']")
 if name=='verify_delivery.py':
  t=t.replace("len(v['crops'])==412", "len(v['crops'])==s['candidate_crop_count']").replace("'CONFLICT':10,'U':10,'T':10", "'CONFLICT':20,'U':20,'T':20")
 (H/name).write_text(t,encoding='utf-8')
print('CREATED',len(files))
