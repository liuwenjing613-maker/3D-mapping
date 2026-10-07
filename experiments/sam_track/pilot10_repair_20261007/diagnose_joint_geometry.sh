set -eu
/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python - <<'PY'
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
import run_repairs as r
import evaluate_pilot_v3 as e
out=r.ROOT/'joint_mask_replacement_20261007/room2_mask14_mask24'
assert json.loads((out/'status.json').read_text())['status']=='PASS'
gt=e.load_gt(r.INPUT/'room2/ground_truth/gt.npz')
with np.load(r.BASE/'final/room2/P1-A1/diffusion/holes_geodesic/instance_surface.npz') as z:xyz=z['xyz_m']
tree=cKDTree(xyz)
rows=[]
for gid in [4001,4002]:
    vertices=gt.xyz_ref[(gt.instance_id==gid)&gt.valid_vertex_mask&~gt.ignore_vertex_mask]
    distance,nearest=tree.query(vertices,workers=8)
    rows.append({'GT_id':gid,'valid_GT_vertices':len(vertices),
        'GT_vertices_within_1cm_of_any_TSDF_vertex':int(np.sum(distance<.01)),
        'recall_and_IoU_upper_bound_under_1cm_point_adapter':float(np.mean(distance<.01)),
        'distance_quantiles_m':np.quantile(distance,[.1,.25,.5,.75,.9,.95]).tolist(),
        'diagnostic_only_coverage_by_distance_m':{str(d):float(np.mean(distance<d)) for d in [.005,.01,.015,.02,.03]},
        'nearest_displacement_mean_m':(vertices-xyz[nearest]).mean(0).tolist()})
data={'status':'PASS','posthoc_only':True,'GT_used_to_change_repair':False,'v3_flags_or_thresholds_changed':False,
    'diagnostic_claim':'upper bound for current fixed TSDF vertex point representation and Euclidean distance adapter; does not establish true mesh surface missingness',
    'cases':rows}
r.dump(out/'geometry_diagnostic.json',data)
print(json.dumps(data,ensure_ascii=False))
PY
