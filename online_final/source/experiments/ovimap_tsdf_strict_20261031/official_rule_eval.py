"""OVI-MAP instance evaluation rules on immutable integer-labeled native surfaces.
This is an evaluation-rule adapter, not the official VLM/color postprocessing pipeline.
"""
from pathlib import Path
import ast,contextlib,hashlib,importlib.util,io,json,os,sys,time
import numpy as np
from plyfile import PlyData,PlyElement
import open3d as o3d
import open3d.core as o3c

C=Path(__file__).resolve().parent
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
S=Path('/home/chenkejun/beauty/ovimap_aligned_eval_20260908/official_ovimap')
GTROOT=Path('/data/chenkejun/CVPR/results/OVI-MAP/paper200')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False));tmp.replace(p)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

# Use the exact upstream function body, avoiding unrelated VLM/visualization imports.
source=(S/'scripts/utils/mesh_postprocess_utils.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='project_to_triangle_mesh')
projection_source=ast.get_source_segment(source,node)
scope={'np':np,'o3c':o3c}
exec(compile(projection_source,str(S/'scripts/utils/mesh_postprocess_utils.py'),'exec'),scope)
official_project=scope['project_to_triangle_mesh']
spec=importlib.util.spec_from_file_location('upstream_ovi_instance_evaluator',S/'scripts/eval_inst_seg.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)

def capture_return(fn,keys,*args,**kwargs):
 captured={};previous=sys.getprofile()
 def observer(frame,event,arg):
  if event=='return' and frame.f_code is fn.__code__:
   captured.update({k:frame.f_locals[k] for k in keys})
 sys.setprofile(observer)
 try:result=fn(*args,**kwargs)
 finally:sys.setprofile(previous)
 assert set(captured)==set(keys)
 return result,captured

def project_pair(native_xyz,ref_xyz,own,ovi):
 assert native_xyz.dtype==np.float32 and ref_xyz.dtype==np.float32
 for labels in [own,ovi]:
  assert labels.shape==(len(native_xyz),)
  assert np.all((labels>=0)&(labels<=65535))
 mesh=o3d.t.geometry.TriangleMesh(o3c.Device('CPU:0'))
 mesh.vertex.positions=o3c.Tensor(native_xyz)
 mesh.vertex.colors=o3c.Tensor(np.zeros((len(native_xyz),3),np.float32))
 mesh.vertex.labels=o3c.Tensor(own.astype(np.uint16))
 result,state=capture_return(official_project,['indices','matched','distances_sqr'],mesh,o3c.Tensor(ref_xyz),
   label2=ovi.astype(np.uint16),dist_thresh=.05,device=o3c.Device('CPU:0'))
 own_out,ovi_out=result[1],result[3]
 index=state['indices'];matched=state['matched']
 np.testing.assert_array_equal(own_out,np.where(matched,own[index],0))
 np.testing.assert_array_equal(ovi_out,np.where(matched,ovi[index],0))
 return own_out,ovi_out,{'indices':index,'matched':matched,
  'distances_squared_m':state['distances_sqr'].numpy().reshape(-1)}

def write_labels(p,xyz,labels):
 vertices=np.empty(len(labels),dtype=[('x','f4'),('y','f4'),('z','f4'),('label','u2')])
 for i,key in enumerate(['x','y','z']):vertices[key]=xyz[:,i]
 vertices['label']=labels
 # The upstream instance scorer reads vertex labels only; preserve GT vertex order.
 PlyData([PlyElement.describe(vertices,'vertex')],text=False).write(p)

def score(gt,pred):
 fields=['gt_inst_iou_scores','weighted_iou_scores','prec_25','prec_50','prec_75','rec_25','rec_50','rec_75',
  'tp_25','fp_25','tp_50','fp_50','tp_75','fp_75']
 stdout=io.StringIO()
 with contextlib.redirect_stdout(stdout):
  _,values=capture_return(official.assign_pred_inst_to_gt_inst,fields,str(gt),str(pred),str(Path(pred).parent))
 metrics={'mIoU':float(np.mean(values['gt_inst_iou_scores'])),'wIoU':float(np.sum(values['weighted_iou_scores']))}
 for t in [25,50,75]:
  metrics['P@'+str(t)]=float(values['prec_'+str(t)])
  metrics['R@'+str(t)]=float(values['rec_'+str(t)])
  metrics['TP@'+str(t)]=int(values['tp_'+str(t)])
  metrics['FP@'+str(t)]=int(values['fp_'+str(t)])
 metrics['GT_count']=len(values['gt_inst_iou_scores'])
 order=['mIoU','wIoU','P@75','R@75','P@50','R@50','P@25','R@25']
 assert stdout.getvalue().strip()=='\t'.join(format(metrics[k],'.4f') for k in order)
 return metrics,stdout.getvalue()

def evaluate(scene):
 dst=O/'official_rule_evaluation'/scene;dst.mkdir(parents=True,exist_ok=True)
 if (dst/'complete.json').exists():return read(dst/'complete.json')
 native=O/'native'/f'{scene}_400_trace';ours=O/'ours'/f'{scene}_400_main'
 assert read(native/'readout_complete.json')['processed_frames']==400
 assert read(ours/'complete.json')['processed_frames']==400
 a=[json.loads(x) for x in (native/'prefix_audit.jsonl').read_text().splitlines()]
 b=[json.loads(x) for x in (ours/'audit.jsonl').read_text().splitlines()]
 assert len(a)==len(b)==400
 for i,(x,y) in enumerate(zip(a,b)):
  assert x['frame_id']==y['frame_id']==i*5
  assert x['geometry_sha256']==y['geometry_sha256']
  assert x['readonly_verified'] and y['strict_vote_verified']
 for count in [1,2,5,10,20,50,100,200,300,400]:assert b[count-1]['full_replay_verified']
 paths={'ours':ours/'maps/f001995.npz','ovimap':native/'ovi/f001995.npz'}
 locks={k:sha(p) for k,p in paths.items()}
 dump(dst/'prediction_lock.json',{'sha256':locks,'frozen_unix':time.time(),'GT_not_loaded_by_this_adapter_yet':True})
 own=load(paths['ours']);ovi=load(paths['ovimap'])
 np.testing.assert_array_equal(own['xyz_m'],ovi['xyz_m'])
 xyz=own['xyz_m'];assert digest(xyz)==a[-1]['geometry_sha256']
 gt=GTROOT/scene/'gt_instance_mesh.ply'
 vertices=PlyData.read(gt)['vertex'].data
 ref=np.column_stack([vertices[k] for k in ['x','y','z']]).astype(np.float32)
 gt_ids,gt_counts=np.unique(vertices['label'],return_counts=True)
 out_own,out_ovi,correspondence=project_pair(xyz,ref,np.maximum(own['instance_id'],0),np.maximum(ovi['instance_id'],0))
 cache=dst/'shared_correspondence.npz'
 np.savez_compressed(cache,**correspondence,native_xyz_sha256=np.array(digest(xyz)),gt_xyz_sha256=np.array(digest(ref)),max_distance_m=np.array(.05))
 result={}
 for name,labels in [('ours',out_own),('ovimap',out_ovi)]:
  pred=dst/(name+'_on_gt.ply');write_labels(pred,ref,labels)
  metrics,stdout=score(gt,pred);result[name]=metrics
  (dst/(name+'_upstream_stdout.txt')).write_text(stdout)
  dump(dst/(name+'_metrics.json'),metrics)
  assert sha(paths[name])==locks[name]
 report={'status':'PASS','scene':scene,'evaluation':'OVI-MAP evaluation-rule adapter',
  'original_git_commit':'f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424','frames':400,
  'GT_file':str(gt),'GT_file_sha256':sha(gt),'GT_vertices':len(ref),
  'GT_scope':'official full GT mesh; no R2 observed-object qualification',
  'GT_instances_min10':int(np.sum((gt_ids!=0)&(gt_counts>=10))),
  'max_distance_m':.05,'one_shared_projection_for_both_methods':True,
  'projection_source_sha256':sha(S/'scripts/utils/mesh_postprocess_utils.py'),
  'scorer_source_sha256':sha(S/'scripts/eval_inst_seg.py'),
  'prediction_sha256':locks,'correspondence_sha256':sha(cache),'native_xyz_sha256':digest(xyz),
  'geometrically_mapped_GT_vertices':int(correspondence['matched'].sum()),
  'VLM_view_filter_used':False,'triangle_color_reassignment_used':False,
  'labels':'direct integer native IDs; unassigned becomes zero',
  'metric_note':'Upstream per-prediction best-IoU P/R are not AP and do not enforce one-to-one instance matching; recall may exceed one. Preserve upstream arithmetic.',
  'comparison':result}
 for m in result.values():assert m['GT_count']==report['GT_instances_min10']
 dump(dst/'complete.json',report);print('OFFICIAL_RULE_DONE',scene,json.dumps(result),flush=True)
 return report

def aggregate():
 reports={s:read(O/'official_rule_evaluation'/s/'complete.json') for s in SCENES}
 assert all(r['status']=='PASS' for r in reports.values())
 metrics=['mIoU','wIoU','P@75','R@75','P@50','R@50','P@25','R@25']
 summary={m:{k:float(np.mean([reports[s]['comparison'][m][k] for s in SCENES])) for k in metrics} for m in ['ours','ovimap']}
 report={'status':'PASS','scene_count':8,'frames_per_scene':400,'aggregation':'unweighted mean of eight scene scores',
  'GT_count_total':sum(r['GT_instances_min10'] for r in reports.values()),'comparison':summary,
  'evaluation':'OVI-MAP evaluation-rule adapter, not full official postprocessing or paper-200-frame reproduction'}
 dump(O/'official_rule_comparison.json',report);return report
if __name__=='__main__':
 print(json.dumps(aggregate() if sys.argv[1]=='aggregate' else evaluate(sys.argv[1])),flush=True)
