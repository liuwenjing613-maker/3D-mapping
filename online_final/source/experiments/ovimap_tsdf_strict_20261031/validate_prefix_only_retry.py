from pathlib import Path
import sys,os,subprocess,json
C=Path(__file__).resolve().parent;sys.path.insert(0,str(C))
from native_runner import container as original_container
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
R=Path('/data/chenkejun/ovimap_runtime_20260908')
SOURCE=Path('/home/chenkejun/beauty/conceptgraphs/data/Replica')
inputs=O/'prefix_only_inputs';scene=inputs/'Replica/room0'
dst=O/'prefix_only_native_retry';dst.mkdir()
def container(*args,**kwargs):
 cmd=original_container(*args,**kwargs)
 return cmd[:1]+['--ro-bind',str(SOURCE),str(SOURCE)]+cmd[1:]
inner=['/usr/bin/python3','-B','/experiment/native_instrumented.py','--dataset','replica','--task','Nyu40','--scene_num','room0','--data_folder',str(inputs/'Replica'),'--result_folder',str(dst),'--start','0','--end','10','--step','5','--data_association','2','--inst_association','4','--seg_graph_confidence','3','--temp_panoptics_folder',str(inputs/'masks'),'--use_temp_geometrics','--temp_geometrics_folder',str(inputs/'geometric'),'--intermediate_seg_folder',str(dst/'fused'),'--num_threads','1','--log','aligned','--skip_feature_extraction']
with (dst/'run.log').open('w') as f:subprocess.run(container(inner,extra={'CVPR_READOUT_MODE':'trace','CVPR_EXPECTED_COUNT':2,'OMP_NUM_THREADS':1,'OPENBLAS_NUM_THREADS':1}),stdout=f,stderr=subprocess.STDOUT,check=True)
import numpy as np
for fid in [0,5]:
 for kind in ['geometry','ovi']:
  with np.load(dst/kind/('f%06d.npz'%fid)) as a,np.load(O/'native/room0_2_trace'/kind/('f%06d.npz'%fid)) as b:
   for k in a.files:np.testing.assert_array_equal(a[k],b[k])
with np.load(dst/'tsdf_final.npz') as a,np.load(O/'native/room0_2_trace/tsdf_final.npz') as b:
 for k in a.files:np.testing.assert_array_equal(a[k],b[k])
from baseline import get_frame,sample,extract_frame_observations,OnlineVoxelAssociator,reduce_surface_votes,CONFIRMED,load
from strict_votes_bounded import IncrementalVotes,BASE
config=json.loads((Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs/room0.json')).read_text())
config['source']['scene_root']=str(scene);config['source']['mask_root']=str(inputs/'masks')
assoc=OnlineVoxelAssociator(allow_multiple_observations_per_instance_per_frame=True,improved_association=False,valid_first_sampling=False,association_mode='probabilistic',relative_weight_temperature=.1,null_candidate_score=.35)
history=[];cache=IncrementalVotes()
rows=[json.loads(x) for x in (dst/'prefix_audit.jsonl').read_text().splitlines()]
for row in rows:
 fid=row['frame_id'];frame,sha,_=get_frame('room0',fid,row,config)
 obs=extract_frame_observations(frame,'Replica','room0',sha);dec=assoc.process_frame(frame,obs)
 history.append(sample(frame,{d['mask_local_id']:d['instance_id'] for d in dec}))
 x=load(dst/'geometry'/('f%06d.npz'%fid))['xyz_m'];k,v=cache.refresh(x,history)
 ev=reduce_surface_votes(k,v,len(x),BASE,2,.67);label=np.where(ev['state']==CONFIRMED,ev['top1_instance_id'],-1).astype(np.int32)
 reference=load(O/'ours/room0_400_main/maps'/('f%06d.npz'%fid))
 np.testing.assert_array_equal(label,reference['instance_id'])
report={'status':'PASS','configured_source_contains_only_frames':[0,5],'trajectory_last_row':5,
 'future_RGB_D_masks_and_geometric_segments_absent_from_configured_source':True,
 'native_tsdf_geometry_ids_bit_exact':True,'ours_prefix_labels_bit_exact':True}
(O/'prefix_input_isolation_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
