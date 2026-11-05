"""Read-only algorithm benchmark; outputs isolated from existing cache/maps."""
import os,time,json,resource,sys,subprocess
from pathlib import Path
import numpy as np
from instance_repair.two_audit_cache import ObservationCache,write_json
from instance_repair.two_audit_sources import MappingSource,sha256_file
from instance_repair.two_audit_projection import ProjectionSettings,project_frame
from instance_repair.two_audit_matching import TrackSettings,TrackBuilder,Observation
ROOT=Path('/data/chenkejun/CVPR/results/two-audit/replica_room0_office2_v1_20261104')
OUT=ROOT/'performance_20261104'
def read(p):return json.loads(Path(p).read_text())
def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
def summary(v):
 a=np.asarray(v,float)
 return dict(total_s=float(a.sum()),mean_ms=float(a.mean()*1000),p50_ms=float(np.percentile(a,50)*1000),p95_ms=float(np.percentile(a,95)*1000),max_ms=float(a.max()*1000))
def run(scene,stage):
 t0=time.perf_counter();cpu0=time.process_time();base=rss();rows=[]
 output=OUT/(scene+'_'+stage);output.mkdir(parents=True,exist_ok=False)
 cache=ObservationCache(ROOT/'observations'/scene);m=cache.manifest
 settings=ProjectionSettings(**{k:m['projection'][k] for k in ProjectionSettings.__dataclass_fields__})
 if stage=='projection':
  source=MappingSource(m['mapping_root'],m['mode'],m['checkpoint'])
  cache.require_geometry(source.xyz)
  init=time.perf_counter()-t0
  for fid in source.frame_ids:
   t=time.perf_counter();frame=source.load_frame(fid,settings.max_depth_m);load=time.perf_counter()-t
   t=time.perf_counter();pack,stats=project_frame(frame,source.xyz,settings);project=time.perf_counter()-t
   pack.update(frame_id=np.array(fid,np.int64),geometry_id=np.array(source.geometry['geometry_id']),source_mask_sha256=np.array(frame.sources['mask']['sha256']))
   t=time.perf_counter();np.savez_compressed(output/('f%06d.npz'%fid),**pack);write=time.perf_counter()-t
   t=time.perf_counter();expected=cache.read_frame(fid,geometry_id=cache.geometry_id)
   assert set(pack)==set(expected)
   for k in pack:assert np.array_equal(pack[k],expected[k]),(fid,k)
   check=time.perf_counter()-t
   rows.append(dict(frame=fid,load_s=load,project_s=project,write_s=write,check_s=check,peak_rss_mib=rss()))
   if len(rows)%50==0:print(scene,stage,len(rows),'elapsed',round(time.perf_counter()-t0,1),flush=True)
  extra=dict(surface_points=len(source.xyz),cache_bytes=sum(p.stat().st_size for p in output.glob('*.npz')),all_frames_exactly_equal=True)
  fields=['load','project','write','check']
 else:
  old=read(ROOT/'stable_tracks'/scene/'manifest.json')
  setting=TrackSettings(**{k:old['settings'][k] for k in TrackSettings.__dataclass_fields__})
  builder=TrackBuilder(cache.geometry_id,m['geometry']['shape'][0],setting)
  init=time.perf_counter()-t0
  for fid in cache.frames:
   t=time.perf_counter();pack=cache.read_frame(fid,geometry_id=cache.geometry_id);load=time.perf_counter()-t
   t=time.perf_counter();obs=[]
   for i,r in enumerate(cache.by_frame[fid]):
    lo,hi=map(int,pack['mask_offsets'][i:i+2]);obs.append(Observation(r['observation_key'],r['mask_local_id'],pack['mask_surface_indices'][lo:hi]))
   prepare=time.perf_counter()-t
   before=builder.stats['comparisons'];t=time.perf_counter()
   builder.process_frame(fid,pack['visible_surface_indices'],obs,np.asarray(cache.frames[fid]['camera_to_world'],np.float64),geometry_id=cache.geometry_id)
   match=time.perf_counter()-t
   rows.append(dict(frame=fid,load_s=load,prepare_s=prepare,match_s=match,tracks=len(builder.tracks),comparisons=builder.stats['comparisons']-before,peak_rss_mib=rss()))
   if len(rows)%50==0:print(scene,stage,len(rows),'elapsed',round(time.perf_counter()-t0,1),flush=True)
  actual=builder.records();expected=read(ROOT/'stable_tracks'/scene/'tracks.json')
  for r in expected:r.pop('non_border_member_count',None)
  assert actual==expected,'Track result differs'
  extra=dict(surface_points=m['geometry']['shape'][0],stats=dict(builder.stats),tracks=len(actual),stable=sum(t['status']=='STABLE' for t in actual),tracks_exactly_equal=True)
  fields=['load','prepare','match']
 result=dict(status='PASS',scene=scene,stage=stage,frames=len(rows),initialization_s=init,wall_s=time.perf_counter()-t0,cpu_s=time.process_time()-cpu0,peak_rss_mib=rss(),baseline_rss_mib=base,metrics={k:summary([r[k+'_s'] for r in rows]) for k in fields},blocks=[dict(frame_start=rows[i]['frame'],frame_end=rows[min(i+99,len(rows)-1)]['frame'],metrics={k:summary([r[k+'_s'] for r in rows[i:i+100]]) for k in fields}) for i in range(0,len(rows),100)],rows=rows,**extra)
 write_json(output/'result.json',result);print('RESULT',scene,stage,result['wall_s'],result['peak_rss_mib'],flush=True)
if __name__=='__main__':
 if len(sys.argv)==3:run(*sys.argv[1:])
 else:
  OUT.mkdir(exist_ok=False)
  write_json(OUT/'environment.json',dict(cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],affinity=list(os.sched_getaffinity(0)),loadavg=Path('/proc/loadavg').read_text(),threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},numpy=np.__version__,python=sys.version,method='one full replay per scene/stage; OS page cache not flushed; CPU only; matching omits diagnostic evidence gzip; validation timed separately',code_hashes={str(p):sha256_file(p) for p in Path('instance_repair').glob('two_audit_*.py')}))
  for scene in ['room0','office2']:
   for stage in ['projection','matching']:
    subprocess.run([sys.executable,'-u',__file__,scene,stage],check=True)
  write_json(OUT/'complete.json',dict(status='PASS',results=[str(p) for p in OUT.glob('*/result.json')]))
