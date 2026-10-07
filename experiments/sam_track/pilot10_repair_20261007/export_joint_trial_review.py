"""Equal-geometry projections and exact PLY states for the joint chair trial."""
from pathlib import Path
import json,tarfile
import numpy as np
from PIL import Image
import run_repairs as r
import audit_and_render as render

WORK=r.ROOT/'joint_mask_replacement_20261007/room2_mask14_mask24'
OUT=WORK/'review'
UID='room2-14047feb0df3aaa5'
DTYPE=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),
    ('baseline_id','<i4'),('pixel_control_id','<i4'),('instance_id','<i4'),('seed_track_id','<i4')])

def bbox(x):return [x.min(0).tolist(),x.max(0).tolist()]
def ply(path,xyz,rgb,labels,owners,indices):
    data=np.empty(len(indices),DTYPE)
    for k,name in enumerate(['x','y','z']):data[name]=xyz[indices,k]
    color=np.rint(np.clip(rgb[indices],0,1)*255).astype(np.uint8)
    for k,name in enumerate(['red','green','blue']):data[name]=color[:,k]
    for name,values in [('baseline_id',labels['baseline']),('pixel_control_id',labels['pixel_control']),('instance_id',labels['whole_mask']),('seed_track_id',owners)]:data[name]=values[indices]
    header='\n'.join(['ply','format binary_little_endian 1.0','comment Frozen TSDF; instance_id is whole-old-mask joint replacement',
        f'element vertex {len(indices)}','property float x','property float y','property float z',
        'property uchar red','property uchar green','property uchar blue','property int baseline_id',
        'property int pixel_control_id','property int instance_id','property int seed_track_id','end_header','']).encode('ascii')
    with path.open('wb') as f:f.write(header);data.tofile(f)
    restored=np.memmap(path,mode='r',dtype=DTYPE,offset=len(header),shape=(len(indices),))
    np.testing.assert_array_equal(np.column_stack([restored[k] for k in ['x','y','z']]),xyz[indices])
    for name,values in [('baseline_id',labels['baseline']),('pixel_control_id',labels['pixel_control']),('instance_id',labels['whole_mask']),('seed_track_id',owners)]:np.testing.assert_array_equal(restored[name],values[indices])
    return {'path':'ply_models/'+path.name,'sha256':r.sha(path),'bytes':path.stat().st_size,'points':len(indices)}

def main():
    assert json.loads((WORK/'status.json').read_text())['status']=='PASS'
    OUT.mkdir(exist_ok=True)
    for sub in ['assets','ply_models']:(OUT/sub).mkdir(exist_ok=True)
    summary=json.loads((WORK/'v3_summary.json').read_text())
    repair=json.loads((WORK/'repair_summary.json').read_text())
    frozen=json.loads((WORK/'evaluation_freeze.json').read_text())
    base_path=r.BASE/'final/room2/P1-A1/diffusion/holes_geodesic/instance_surface.npz'
    base=r.load_npz(base_path);xyz,rgb=base['xyz_m'],base['rgb']
    labels={'baseline':base['instance_id']};sources={str(base_path):r.sha(base_path)}
    for arm in ['pixel_control','whole_mask']:
        path=WORK/arm/'final/instance_surface.npz'
        assert r.sha(path)==frozen['predictions'][arm]['sha256']
        d=r.load_npz(path)
        np.testing.assert_array_equal(d['xyz_m'],xyz);np.testing.assert_array_equal(d['rgb'],rgb)
        labels[arm]=d['instance_id'];sources[str(path)]=r.sha(path)
    source=r.ReplicaFrameSource(r.INPUT/'room2/configs/raw.json')
    support=r.load_npz(r.ROOT/'repair'/UID/'seed_projected_support.npz')
    objects=[];owners=np.zeros(len(xyz),np.int32);seed_points=[]
    for tid,mid,pid in [(4,14,52),(7,24,355)]:
        p=np.unique(support['surface_point_index'][support['track_id']==tid]);owners[p]=tid;seed_points.append(p)
        objects.append({'track_id':tid,'native_mask_id':mid,'persistent_id':pid,'seed_surface_bounds':bbox(xyz[p]),
            'seed_surface_points':len(p),'full_scene_identity_points':{k:int(np.sum(a==pid)) for k,a in labels.items()}})
    seed_points=np.unique(np.concatenate(seed_points));bounds=np.array(bbox(xyz[seed_points]))
    changed=(labels['baseline']!=labels['pixel_control'])|(labels['baseline']!=labels['whole_mask'])
    indices=np.flatnonzero(np.all((xyz>=bounds[0]-.55)&(xyz<=bounds[1]+.55),axis=1)|changed)
    seedframe=source.load_frame(1065)
    model={'case_uid':UID,'scene':'room2','ROI':'mask14 + mask24','status':'READY','seed_frame':1065,
        'seed_camera_to_world':seedframe.camera_to_world.tolist(),'objects':objects,'native_bounds':bbox(xyz),
        'local_bounds':bbox(xyz[indices]),'source_maps':sources,
        'label_changes':{k:int(np.sum(labels[k]!=labels['baseline'])) for k in ['pixel_control','whole_mask']}}
    model['local']=ply(OUT/'ply_models/joint_local.ply',xyz,rgb,labels,owners,indices)
    model['full']=ply(OUT/'ply_models/joint_full.ply',xyz,rgb,labels,owners,np.arange(len(xyz)))
    r.dump(OUT/'ply_models/manifest.json',{'status':'PASS','cases':[model],'record_stride_bytes':31,
        'native_geometry_and_labels_preserved_exactly':True,'preview_downsampling':False,'GT_used_in_rendering':False})
    case=next(c for c in json.loads((Path(__file__).parent/'review_manifest.json').read_text())['cases'] if c['case_uid']==UID)
    decisions=json.loads((WORK/'frame_decisions.json').read_text())
    by_frame={x['frame']:x for x in decisions['frames']}
    table=render.palette(max(int(x.max()) for x in labels.values()));table[52]=[40,186,245];table[355]=[64,234,120]
    views=[]
    for view in case['views']:
        fid=view['frame'];frame=source.load_frame(fid);box=tuple(view['box'])
        pixel,point=render.visible_pixels(xyz,frame)
        assets={}
        for condition,values in labels.items():
            filename=f'assets/f{fid:06d}_{condition}.jpg'
            Image.fromarray(render.label_overlay(frame.rgb,pixel,point,values,table)).crop(box).save(OUT/filename,quality=94)
            assets[condition]=filename
        raw=f'assets/f{fid:06d}_rgb.jpg';Image.fromarray(frame.rgb).crop(box).save(OUT/raw,quality=94);assets['RGB']=raw
        raw_track=np.array(Image.open(WORK/'tracking'/f'f{fid:06d}.png'))
        target_rgb=frame.rgb.copy()
        for tid,pid in [(4,52),(7,355)]:
            mask=raw_track==tid
            target_rgb[mask]=np.rint(target_rgb[mask]*.45+table[pid]*.55).astype(np.uint8)
        track=f'assets/f{fid:06d}_tracking.jpg';Image.fromarray(target_rgb).crop(box).save(OUT/track,quality=94);assets['tracking']=track
        views.append({**view,'assets':assets,'decision':by_frame[fid],
            'visible_geometry_pixel_count':len(pixel),'same_visibility_and_geometry_for_all_states':True})
    # Fixed requested views include accepted and rejected frames without selection by score.
    report={'status':'PASS','summary':summary,'views':views,'model':model,
        'frame_decisions':decisions,'source_hashes':sources,'rendering_uses_GT':False,
        'geometry_depth_and_camera_equal_across_states':True,'no_source_map_modified':True}
    r.dump(OUT/'review_data.json',report)
    for original,digest in sources.items():assert r.sha(original)==digest
    archive=WORK/'review_bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(OUT,arcname='joint_mask_replacement_20261007')
    r.dump(WORK/'review_bundle.json',{'status':'PASS','path':str(archive),'sha256':r.sha(archive),'bytes':archive.stat().st_size})
    print(json.dumps({'status':'PASS','local_points':len(indices),'full_points':len(xyz),'archive_bytes':archive.stat().st_size,'archive_sha256':r.sha(archive)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
