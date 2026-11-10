"""Existing frozen holes_geodesic publication; strict votes and confirmed IDs stay fixed."""
from common import *
from dataclasses import asdict
sys.path.insert(0,str(SRC))
from revisable_instance_map.surface_evidence import reduce_surface_votes,CONFIRMED
from revisable_instance_map.offline_surface_assignment import AssignmentSettings,assign_surface
sys.path.insert(0,str(OLD_C))
from strict_votes import BASE
def fill(scene):
 dst=O/'scenes'/scene
 ownroot=OLD/'ours'/f'{scene}_400_main'
 original_path=ownroot/'maps/f001995.npz';checkpoint=read(ownroot/'checkpoint.json')
 vote_path=ownroot/checkpoint['votes']
 assert sha(vote_path)==checkpoint['votes_sha256']
 original=load(original_path);v=load(vote_path);a=load(dst/'normals.npz');color=load(dst/'rgb.npz')
 xyz=original['xyz_m'];labels=original['instance_id'];n=len(xyz)
 for other in [v['xyz'],a['xyz_m']]:np.testing.assert_array_equal(xyz,other)
 assert v['frames'].tolist()==list(range(0,2000,5))
 evidence=reduce_surface_votes(v['keys'],v['votes'],n,BASE,2,.67)
 np.testing.assert_array_equal(labels,np.where(evidence['state']==CONFIRMED,evidence['top1_instance_id'],-1))
 valid=a['normal_valid']&color['color_valid']
 assert np.all(valid[labels>0]),'A published competitor lacks verified attributes' 
 ix=np.flatnonzero(valid);mapping=np.full(n,-1,np.int32);mapping[ix]=np.arange(len(ix),dtype=np.int32)
 point=v['keys']//BASE;keep=valid[point]
 pair=dict(surface_point_index=mapping[point[keep]],instance_id=(v['keys'][keep]%BASE).astype(np.int32),
           frame_votes=v['votes'][keep])
 ev={k:arr[ix] for k,arr in evidence.items()}
 settings=AssignmentSettings()
 expected=read(Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')/f'final/{scene}/P1-A1/diffusion/materialization_report.json')['settings_sha256']
 settings_hash=hashlib.sha256(json.dumps(asdict(settings),sort_keys=True,separators=(',',':')).encode()).hexdigest()
 assert settings_hash==expected
 before={str(p):sha(p) for p in [original_path,vote_path,dst/'normals.npz',dst/'rgb.npz']}
 def progress(stage,values):print('FILL',scene,stage,values,flush=True)
 variants,detail,stats=assign_surface(xyz[ix],a['normals'][ix],color['rgb'][ix],ev,pair,labels[ix],settings,progress)
 final=labels.copy();final[ix]=variants['holes_geodesic']
 changed=final!=labels;filled=(labels<=0)&(final>0)
 assert np.array_equal(changed,filled)
 np.testing.assert_array_equal(final[labels>0],labels[labels>0])
 assert set(np.unique(final[final>0])).issubset(set(original['native_instance_ids']))
 np.testing.assert_array_equal(final[~valid],labels[~valid])
 # Keep diagnostic indices in the explicitly saved subset index space.
 save(dst/'completion_diagnostics.npz',valid_to_original_index=ix,**detail)
 save(dst/'instance_surface.npz',xyz_m=xyz,instance_id=final,native_instance_ids=original['native_instance_ids'])
 save(dst/'point_provenance.npz',original_instance_id=labels,final_instance_id=final,
  filled=filled,attributes_valid=valid,original_state=evidence['state'])
 for p,h in before.items():assert sha(p)==h
 report=dict(status='PASS',scene=scene,frames=400,stage='final_publication_only',
  definition='strict one-vote plus unchanged frozen holes_geodesic; no observation/association/merge/split repair',
  filled_points=int(filled.sum()),assigned_before=int((labels>0).sum()),assigned_after=int((final>0).sum()),
  unassigned_after=int((final<=0).sum()),surface_points=n,invalid_attributes=int((~valid).sum()),
  excluded_confirmed_points=int(((labels>0)&~valid).sum()),existing_IDs_changed=0,new_instance_IDs=0,
  source_sha256=before,prediction_sha256=sha(dst/'instance_surface.npz'),geometry_sha256=digest(xyz),
  settings=asdict(settings),settings_sha256=settings_hash,GT_used_for_prediction=False,
  labels_frozen_before_evaluation_unix=time.time(),feedback_to_association_or_votes=False,statistics=stats)
 dump(dst/'mapping_complete.json',report);print('FILL_COMPLETE',scene,report['filled_points'],flush=True)
if __name__=='__main__':fill(sys.argv[1])
