from pathlib import Path
import sys,subprocess,os,json
from native_runtime import container,C,O
scene=sys.argv[1];count=int(sys.argv[2]) if len(sys.argv)>2 else 200
dst=O/'ovimap'/scene/str(count);dst.mkdir(parents=True,exist_ok=True)
if (dst/'complete.json').exists():sys.exit(0)
if (dst/'run.log').exists():raise RuntimeError('Inspect prior attempt before restarting')
inner=['/usr/bin/python3','-B','/experiment/ovi_instrumented.py','--dataset','replica','--task','Nyu40','--scene_num','input','--data_folder',str(O/'inputs'/scene),'--result_folder',str(dst),'--start','0','--end',str(count*5),'--step','5','--data_association','2','--inst_association','4','--seg_graph_confidence','3','--temp_panoptics_folder',str(O/'inputs'/scene/'masks'),'--save_temp_geometrics','--temp_geometrics_folder',str(dst/'geometrics'),'--intermediate_seg_folder',str(dst/'fused'),'--num_threads','1','--log','ScanNet_aligned','--skip_feature_extraction']
with (dst/'run.log').open('w') as f:subprocess.run(container(inner,extra=dict(CVPR_EXPECTED_COUNT=count,CVPR_READOUT_MODE='terminal')),stdout=f,stderr=subprocess.STDOUT,check=True)
assert json.loads((dst/'readout_complete.json').read_text())['status']=='PASS'
(dst/'complete.json').write_text(json.dumps(dict(status='PASS',scene=scene,frames=count,instance_mapping_only=True,semantic_feature_extraction=False,geometry='native segmented TSDF')))
