from native_runtime import C,O,container
import sys,subprocess,json,numpy as np
scene=sys.argv[1]
for mode in ['kernel_original','kernel_geometry','pilot','full']:
 root=O/'native'/scene/mode;root.parent.mkdir(parents=True,exist_ok=True)
 if (root/'complete.json').exists():continue
 with (O/(scene+'_geometry_'+mode+'.log')).open('a') as f:
  subprocess.run(container(['/usr/bin/python3','-B','/experiment/native_geometry.py',scene,mode]),stdout=f,stderr=subprocess.STDOUT,check=True)
 if mode=='kernel_geometry':
  for i in range(3):
   with np.load(root/f'state{i}.npz') as a,np.load(root.parent/'kernel_original'/f'state{i}.npz') as b:
    assert set(a.files)==set(b.files)
    for k in a.files:np.testing.assert_array_equal(a[k],b[k])
  (root.parent/'kernel_test.json').write_text(json.dumps(dict(status='PASS',batches=3,all_TSDF_fields_bit_exact=True)))
 if mode=='full':
  for n in [1,3]:
   with np.load(root/str(n)/'tsdf.npz') as a,np.load(root.parent/'pilot'/str(n)/'tsdf.npz') as b:
    for k in a.files:np.testing.assert_array_equal(a[k],b[k])
  (root.parent/'prefix_test.json').write_text(json.dumps(dict(status='PASS',pilot_full_prefixes_bit_exact=[1,3])))
