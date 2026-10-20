"""Validate cached/new predictions and emit metadata needed by selection.

No resampling, inference, mask editing or human selections are performed here.
The final review renders seed masks directly from these frozen source arrays.
"""
import copy,json
import numpy as np
from PIL import Image
from prepare_inputs import ROOT,SOURCE,NAME,read,write,sha

def main():
 import normalize_manifest
 m=read(ROOT/'input_manifest.json')
 for folder in ['cropformer','sam2_1']:
  assert read(ROOT/folder/'audit.json')['status']=='PASS'
 assert read(ROOT/'cropformer/audit.json')['control_min_pixel_agreement']>=.999
 viewer=dict(name=NAME,cases=copy.deepcopy(m['cases']),crops={},scene_counts={'office0':len(m['cases'])},total_views=m['preferred_views'])
 for case in viewer['cases']:
  for v in case['views']:v['full_rgb']=str(SOURCE/v['source_RGB'])
 for j in m['jobs']:
  if j['kind']!='crop':continue
  assert sha(j['input_file'])==j['RGB_sha256'] and sha(j['baseline_file'])==j['baseline_sha256']
  paths=[j['baseline_file'],ROOT/'cropformer'/(j['key']+'.png'),ROOT/'sam2_1'/(j['key']+'.png')]
  models=[]
  for index,p in enumerate(paths):
   a=np.asarray(Image.open(p));assert list(a.shape)==j['shape']
   if index:
    rec=read(p.with_suffix('.json'));assert sha(p)==rec['mask_sha256'] and rec['RGB_sha256']==j['RGB_sha256']
   models.append(dict(count=int(len(np.unique(a[a>0]))),ids_file=str(p)))
  viewer['crops'][j['key']]=dict(models=models,box=j['crop_xyxy'])
 assert len(viewer['cases'])==len(m['cases']) and sum(len(c['views']) for c in viewer['cases'])==m['preferred_views']
 for p,h in m['protected_source_hashes'].items():assert sha(p)==h
 write(ROOT/'review/viewer_data.json',viewer)
 write(ROOT/'pipeline_complete.json',dict(status='PASS',scene='office0',cases=len(m['cases']),views=m['preferred_views'],crops=len(viewer['crops']),all_prediction_hashes_verified=True))
 print('FRONTEND_COMPLETE',len(viewer['crops']),flush=True)
if __name__=='__main__':main()
