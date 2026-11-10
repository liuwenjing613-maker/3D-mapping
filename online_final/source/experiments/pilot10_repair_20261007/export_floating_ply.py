"""Export exact frozen geometry and all three label maps as binary PLY."""
from pathlib import Path
import hashlib, json, sys, tarfile
import numpy as np

ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
BASE=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
INPUT=Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main')
SNAPSHOT=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot')
OUT=ROOT/'tracking_review/ply_models'
sys.path.insert(0,str(SNAPSHOT/'revisable_instance_map/src'))
from revisable_instance_map.frame_io import ReplicaFrameSource

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def bbox(x):return [x.min(0).tolist(),x.max(0).tolist()] if len(x) else None

DTYPE=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),
                ('baseline_id','<i4'),('seed_id','<i4'),('instance_id','<i4'),('seed_track_id','<i4')])

def write_ply(path,xyz,rgb,before,seed,after,owners,indices):
    n=len(indices)
    records=np.empty(n,DTYPE)
    for j,k in enumerate(['x','y','z']):records[k]=xyz[indices,j]
    colors=np.rint(np.clip(rgb[indices],0,1)*255).astype(np.uint8)
    for j,k in enumerate(['red','green','blue']):records[k]=colors[:,j]
    for key,value in [('baseline_id',before),('seed_id',seed),('instance_id',after),('seed_track_id',owners)]:records[key]=value[indices]
    header='\n'.join(['ply','format binary_little_endian 1.0',
       'comment Frozen CVPR pilot map; coordinates preserved; no GT in export',
       'comment instance_id is seed_plus_short_track; baseline_id and seed_id are comparison states',
       f'element vertex {n}','property float x','property float y','property float z',
       'property uchar red','property uchar green','property uchar blue',
       'property int baseline_id','property int seed_id','property int instance_id','property int seed_track_id','end_header','']).encode('ascii')
    with path.open('wb') as f:f.write(header);records.tofile(f)
    restored=np.memmap(path,dtype=DTYPE,mode='r',offset=len(header),shape=(n,))
    np.testing.assert_array_equal(np.column_stack([restored[k] for k in ['x','y','z']]),xyz[indices])
    for key,value in [('baseline_id',before),('seed_id',seed),('instance_id',after),('seed_track_id',owners)]:np.testing.assert_array_equal(restored[key],value[indices])
    del restored
    return {'path':'ply_models/'+path.name,'sha256':sha(path),'points':n,'bytes':path.stat().st_size}

def main():
    OUT.mkdir(exist_ok=True)
    seeds=json.loads((ROOT/'human_seeds_20ad6139511b/seed_manifest.json').read_text())
    freeze=json.loads((ROOT/'v3/evaluation_freeze.json').read_text()) if (ROOT/'v3/evaluation_freeze.json').is_file() else json.loads((ROOT/'evaluation_freeze.json').read_text())
    cases=[]
    for c in seeds['seeds']:
        uid,scene=c['case_uid'],c['scene']
        row={k:c[k] for k in ['case_uid','scene','ROI','status']}
        path=BASE/f'final/{scene}/P1-A1/diffusion/holes_geodesic/instance_surface.npz'
        assert sha(path)==freeze['predictions']['baseline:'+scene]['sha256']
        source_hashes={str(path):sha(path)}
        base=load(path); xyz,rgb,before=base['xyz_m'],base['rgb'],base['instance_id']
        owners=np.zeros(len(xyz),np.int32)
        if c['status']=='READY':
            paths=[ROOT/'repair'/uid/k/'final/instance_surface.npz' for k in ['seed_only','seed_plus_short_track']]
            maps=[]
            for p,k in zip(paths,['seed_only','seed_plus_short_track']):
                assert sha(p)==freeze['predictions'][uid+':'+k]['sha256']
                source_hashes[str(p)]=sha(p)
                m=load(p)
                np.testing.assert_array_equal(m['xyz_m'],xyz);np.testing.assert_array_equal(m['rgb'],rgb)
                maps.append(m['instance_id'])
            seed,after=maps
            support=load(ROOT/'repair'/uid/'seed_projected_support.npz')
            pts,tracks=support['surface_point_index'],support['track_id']
            for tid in sorted(set(tracks.tolist())):
                p=np.unique(pts[tracks==tid]);owners[p]=np.where(owners[p]==0,tid,-1)
            objects=[]
            association=json.loads((ROOT/'repair'/uid/'frozen_association.json').read_text())
            for o in association['objects']:
                p=np.unique(pts[tracks==o['track_id']]); target=o['persistent_id']
                objects.append({**o,'seed_surface_bounds':bbox(xyz[p]),'seed_surface_points':len(p),
                                'full_scene_identity_points':{k:int(np.sum(a==target)) for k,a in [('baseline',before),('seed_only',seed),('seed_plus_short_track',after)]}})
            seed_points=np.unique(pts)
            target_bounds=np.array(bbox(xyz[seed_points]),dtype=float)
            # A local view contains every seed surface and every changed point,
            # with a fixed context margin. There is no preview downsampling.
            lo,hi=target_bounds[0]-.55,target_bounds[1]+.55
            include=np.all((xyz>=lo)&(xyz<=hi),axis=1)|(before!=after)|(before!=seed)
            indices=np.flatnonzero(include)
            source=ReplicaFrameSource(INPUT/scene/'configs/raw.json')
            frame=source.load_frame(c['source_choice']['frame'])
            row['seed_camera_to_world']=frame.camera_to_world.tolist()
            row['seed_frame']=c['source_choice']['frame']
            row['objects']=objects
        else:
            seed,after=before,before
            indices=np.arange(len(xyz))
            row['objects']=[]
        row['native_bounds']=bbox(xyz)
        row['local_bounds']=bbox(xyz[indices])
        row['label_changes']={'seed_only':int(np.sum(seed!=before)),'seed_plus_short_track':int(np.sum(after!=before))}
        row['source_maps']=source_hashes
        row['local']=write_ply(OUT/(uid+'_local.ply'),xyz,rgb,before,seed,after,owners,indices)
        row['full']=write_ply(OUT/(uid+'_full.ply'),xyz,rgb,before,seed,after,owners,np.arange(len(xyz)))
        assert all(sha(p)==h for p,h in source_hashes.items())
        cases.append(row)
        print(json.dumps({'case':uid,'local_points':len(indices),'full_points':len(xyz),'changed':row['label_changes']},ensure_ascii=False),flush=True)
    data={'status':'PASS','cases':cases,'native_geometry_and_labels_preserved_exactly':True,'preview_downsampling':False,
          'GT_used':False,'point_cloud_not_triangle_mesh':True,'units':'metres','world_up':[0,0,1],
          'record_stride_bytes':DTYPE.itemsize,'properties':['x','y','z','red','green','blue','baseline_id','seed_id','instance_id','seed_track_id'],
          'PLY_default_instance_id':'seed_plus_short_track','seed_track_id':'human seed surface owner; -1 ambiguous; 0 not seed-covered',
          'source_export_sha256':seeds['source_export_sha256'],'script_sha256':sha(__file__)}
    (OUT/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    archive=ROOT/'floating_ply_bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(OUT,arcname='ply_models')
    print(json.dumps({'status':'PASS','archive_bytes':archive.stat().st_size,'sha256':sha(archive),'cases':len(cases)}),flush=True)

if __name__=='__main__':main()
