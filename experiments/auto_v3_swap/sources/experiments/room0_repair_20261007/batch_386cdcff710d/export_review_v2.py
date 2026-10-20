"""Export real, equal-geometry PLY states and original-view room0 comparisons."""
from pathlib import Path
import json,sys,tarfile
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d')
LEGACY=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
sys.path.insert(0,str(LEGACY));sys.path.insert(0,str(LEGACY/'strict_local_repair_20261007'))
import run_repairs as r
import audit_and_render as render
import fixed_surface_repair as f
NAMES=['baseline','seed_only','full_track']
DTYPE=np.dtype([(n,'<f4') for n in ['x','y','z']]+[(n,'u1') for n in ['red','green','blue']]+[(n+'_id','<i4') for n in NAMES]+[('seed_track_id','<i4')])

def bounds(xyz):return [xyz.min(0).tolist(),xyz.max(0).tolist()]

def write_ply(path,xyz,rgb,labels,owners,indices):
    data=np.empty(len(indices),DTYPE)
    for k,name in enumerate(['x','y','z']):data[name]=xyz[indices,k]
    colors=np.rint(np.clip(rgb[indices],0,1)*255).astype(np.uint8)
    for k,name in enumerate(['red','green','blue']):data[name]=colors[:,k]
    for name in NAMES:data[name+'_id']=labels[name][indices]
    data['seed_track_id']=owners[indices]
    lines=['ply','format binary_little_endian 1.0','comment Exact frozen TSDF; baseline/seed-only/full-track labels',
           'element vertex %d'%len(indices),'property float x','property float y','property float z',
           'property uchar red','property uchar green','property uchar blue']
    lines.extend('property int '+name+'_id' for name in NAMES);lines.extend(['property int seed_track_id','end_header',''])
    header='\n'.join(lines).encode('ascii')
    with path.open('wb') as stream:stream.write(header);data.tofile(stream)
    restored=np.memmap(path,mode='r',dtype=DTYPE,offset=len(header),shape=(len(indices),))
    np.testing.assert_array_equal(np.c_[restored['x'],restored['y'],restored['z']],xyz[indices])
    np.testing.assert_array_equal(np.c_[restored['red'],restored['green'],restored['blue']],colors)
    for name in NAMES:np.testing.assert_array_equal(restored[name+'_id'],labels[name][indices])
    return {'path':'ply_models/'+path.name,'sha256':r.sha(path),'points':len(indices),'bytes':path.stat().st_size}

def main():
    complete=json.loads((ROOT/'repair_complete.json').read_text());assert complete['status']=='PASS'
    repairroot=Path(complete['outputs_root']);evaluation=json.loads((ROOT/'evaluation_summary.json').read_text());assert evaluation['status']=='PASS'
    out=ROOT/'review'
    for d in ['assets','ply_models','reports']:(out/d).mkdir(parents=True,exist_ok=True)
    paths={'baseline':r.BASE/'final/room0/P1-A1/diffusion/holes_geodesic/instance_surface.npz',
           **{name:repairroot/name/'final/instance_surface.npz' for name in NAMES[1:]}}
    maps={name:f.load_arrays(p) for name,p in paths.items()};baseline=maps['baseline'];xyz,rgb=baseline['xyz_m'],baseline['rgb']
    for value in maps.values():f.require_same_geometry(baseline,value)
    labels={name:value['instance_id'] for name,value in maps.items()}
    source=r.ReplicaFrameSource(r.INPUT/'room0/configs/raw.json')
    human=json.loads((ROOT/'seed_manifest.json').read_text())['seeds'];selected={row['case_uid']:row for row in human}
    cases=json.loads((ROOT/'selection_views.json').read_text())['cases']
    association=json.loads((ROOT/'global_association.json').read_text());objects={row['canonical_track_id']:row for row in association['objects']}
    canonical={oid:row['canonical_track_id'] for row in association['objects'] for oid in row['member_track_ids']}
    colors=json.loads((ROOT/'instance_palettes.json').read_text())['scenes']['room0']['id_colors']
    new_pids=sorted({obj['persistent_id'] for obj in objects.values() if str(obj['persistent_id']) not in colors})
    import colorsys
    swatches=[[48,223,148],[244,106,156],[89,185,250],[244,214,71],[169,112,239],[249,157,83]]
    newcolors={pid:swatches[i] if i<len(swatches) else [round(v*255) for v in colorsys.hsv_to_rgb((i*.61803398875+.19)%1,.76,.97)] for i,pid in enumerate(new_pids)}
    colors.update({str(pid):value for pid,value in newcolors.items()})
    palette=render.palette(max(max(int(value.max()) for value in labels.values()),max(map(int,colors)),max(obj['persistent_id'] for obj in objects.values())))
    assert all(obj['persistent_id']<len(palette) for obj in objects.values())
    for pid,color in colors.items():
        if int(pid)<len(palette):palette[int(pid)]=color
    f.atomic_json(out/'instance_colors.json',{'scene':'room0','id_colors':colors,'original_ID_colors_preserved':True,'new_ID_colors':newcolors})
    owners=np.zeros(len(xyz),np.int32);supports={}
    for row in human:
        if row['status']!='READY':continue
        z=f.load_arrays(ROOT/'cases'/row['case_uid']/'seed_projected_support.npz')
        for obj in row['objects']:
            oid=obj['track_id'];p=np.unique(z['surface_point_index'][z['track_id']==oid]);supports[oid]=p
            owners[p]=np.where(owners[p]==0,canonical[oid],np.minimum(owners[p],canonical[oid]))
    model={'scene':'room0','state_names':NAMES,'record_stride_bytes':DTYPE.itemsize,'native_bounds':bounds(xyz),
           'full':write_ply(out/'ply_models/room0_full.ply',xyz,rgb,labels,owners,np.arange(len(xyz))),
           'cases':{},'source_hashes':{str(p):r.sha(p) for p in paths.values()},'preview_downsampling':False}
    reports={name:json.loads((repairroot/name/'complete.json').read_text()) for name in NAMES[1:]}
    timeline=json.loads((repairroot/'full_track/frame_decisions.json').read_text())
    pair=f.load_arrays(repairroot/'full_track/native/surface_instance_frame_votes.npz')
    final=labels['full_track'];seed_audits=[]
    for row in human:
        if row['status']!='READY':continue
        raw_tracking=json.loads((ROOT/'cases'/row['case_uid']/'tracking/complete.json').read_text())
        assert raw_tracking['status']=='PASS' and len(raw_tracking['frames'])==2000
        for obj in row['objects']:
            oid=obj['track_id'];pid=objects[canonical[oid]]['persistent_id'];p=supports[oid]
            nonempty=[frame['frame'] for frame in raw_tracking['frames'] if frame['areas'][str(oid)]>0]
            sel=np.isin(pair['surface_point_index'],p);ids,n=np.unique(pair['instance_id'][sel],return_counts=True)
            vote_counts={str(int(i)):int(pair['frame_votes'][sel][pair['instance_id'][sel]==i].sum()) for i in ids}
            reasons={}
            for decision in timeline['frames']:
                check=decision.get('objects',{}).get(str(oid),{})
                for reason in check.get('reasons',[]):reasons[reason]=reasons.get(reason,0)+1
            seed_audits.append({'track_id':oid,'case_uid':row['case_uid'],'native_mask_id':obj['native_mask_id'],
                  'canonical_track_id':canonical[oid],'persistent_id':pid,'old_family_id':obj['old_family_id'],
                  'seed_points':len(p),'desired_identity_seed_surface_fraction_before':float(np.mean(labels['baseline'][p]==pid)),
                  'desired_identity_seed_surface_fraction_after':float(np.mean(final[p]==pid)),
                  'seed_surface_unknown_before':int(np.sum(labels['baseline'][p]<=0)),'seed_surface_unknown_after':int(np.sum(final[p]<=0)),
                  'seed_surface_final_identity_vote_counts':vote_counts,'rejection_reason_counts':reasons,
                  'raw_tracking_nonempty_frames':len(nonempty),'raw_tracking_empty_frames':2000-len(nonempty),
                  'raw_tracking_first_nonempty_frame':min(nonempty) if nonempty else None,
                  'raw_tracking_last_nonempty_frame':max(nonempty) if nonempty else None,
                  'raw_nonempty_is_not_reliability_acceptance':True,
                  'accepted_mapping_frames':reports['full_track']['counts']['accepted_frames_by_track'][str(oid)]})
    views=[];track_views=[]
    for case in cases:
        uid=case['uid'];choice=selected[uid]['source_choice'];fid=choice['frame'];frame=source.load_frame(fid)
        if selected[uid]['status']=='READY':p=np.unique(np.concatenate([supports[o['track_id']] for o in selected[uid]['objects']]))
        else:
            crop=np.zeros(frame.mask_local.shape,np.uint16);x0,y0,x1,y1=choice['crop_xyxy'];crop[y0:y1,x0:x1]=1
            from dataclasses import replace
            p,_,_=r.project_frame_regions(replace(frame,mask_local=crop),__import__('scipy.spatial',fromlist=['cKDTree']).cKDTree(xyz),pixel_stride=2,max_distance_m=.015,workers=8)
            p=np.unique(p)
        b=np.asarray(bounds(xyz[p]));indices=np.flatnonzero(np.all((xyz>=b[0]-.45)&(xyz<=b[1]+.45),axis=1))
        target_ids=sorted({obj['old_family_id'] for obj in selected[uid].get('objects',[])} | {objects[canonical[obj['track_id']]]['persistent_id'] for obj in selected[uid].get('objects',[])})
        model['cases'][uid]={'ROI':case['ROI'],'frame':fid,'bounds':b.tolist(),'camera_to_world':frame.camera_to_world.tolist(),
                 'target_ids':target_ids,**write_ply(out/'ply_models'/(case['ROI']+'.ply'),xyz,rgb,labels,owners,indices)}
        for view in case['views']:
            vf=view['frame'];fr=source.load_frame(vf);box=tuple(choice['crop_xyxy'] if vf==fid else view['source_crop'])
            pixel,surface=render.visible_pixels(xyz,fr);assets={}
            for name in NAMES:
                namefile=uid+'_f%06d_'%vf+name+'.jpg';Image.fromarray(render.label_overlay(fr.rgb,pixel,surface,labels[name],palette)).crop(box).save(out/'assets'/namefile,quality=93)
                assets[name]='assets/'+namefile
            rgbfile=uid+'_f%06d_rgb.jpg'%vf;Image.fromarray(fr.rgb).crop(box).save(out/'assets'/rgbfile,quality=93);assets['rgb']='assets/'+rgbfile
            deltafile=uid+'_f%06d_delta.jpg'%vf;Image.fromarray(render.delta_overlay(fr.rgb,pixel,surface,labels['baseline'],labels['full_track'])).crop(box).save(out/'assets'/deltafile,quality=93);assets['delta']='assets/'+deltafile
            views.append({'case_uid':uid,'ROI':case['ROI'],'frame':vf,'box':box,'assets':assets,'same_depth_geometry_selection':True})
        if selected[uid]['status']=='READY':
            for tf in sorted(set([0,250,500,750,1000,1250,1500,1750,1995,fid])):
                tracked=np.array(Image.open(ROOT/'cases'/uid/'tracking'/('f%06d.png'%tf)),np.uint16)
                fr=source.load_frame(tf);paint=fr.rgb.copy()
                for oid in np.unique(tracked):
                    if not oid:continue
                    mask=tracked==oid;pid=objects[canonical[int(oid)]]['persistent_id']
                    paint[mask]=(paint[mask].astype(float)*.4+palette[pid]*.6).astype(np.uint8)
                    edge=mask & (~np.roll(mask,1,0)|~np.roll(mask,-1,0)|~np.roll(mask,1,1)|~np.roll(mask,-1,1));paint[edge]=palette[pid]
                filename=uid+'_track_f%06d.jpg'%tf;Image.fromarray(paint).resize((900,510)).save(out/'assets'/filename,quality=92)
                decision=next((d for d in timeline['frames'] if d['frame']==tf),{})
                track_views.append({'case_uid':uid,'frame':tf,'asset':'assets/'+filename,'used_track_ids':decision.get('accepted_track_ids',[]),
                                    'only_this_case_used_track_ids':[oid for oid in decision.get('accepted_track_ids',[]) if oid in [o['track_id'] for o in selected[uid]['objects']]]})
    data={'status':'PASS','annotation_sha256':complete['annotation_sha256'],'human_selections':human,'global_association':association,
          'repair_reports':reports,'evaluation':evaluation,'model':model,'views':views,'track_views':track_views,
          'seed_audits':seed_audits,'timeline':timeline,'tracking_validation':json.loads((ROOT/'tracking_validation.json').read_text()),
          'rendering_uses_GT':False,'all_PLY_xyz_rgb_label_roundtrips_exact':True,
          'runtime_failure_records':[str(p.relative_to(ROOT)) for p in ROOT.rglob('*failure.json')],
          'unreliable_ROI_has_no_independent_intervention_but_can_show_shared_changes':True}
    f.atomic_json(out/'comparison_data.json',data)
    for filename in ['repair_complete.json','evaluation_summary.json','evaluation_freeze.json','input_freeze.json','global_association.json',
                     'tracking_validation.json','seed_manifest.json','cross_seed_diagnostics.json','alias_review.json','tracking_queue_complete.json']:
        f.atomic_json(out/'reports'/filename,json.loads((ROOT/filename).read_text()))
    for name in NAMES[1:]:
        for filename in ['complete.json','source_freeze.json']:
            f.atomic_json(out/'reports'/(name+'_'+filename),json.loads((repairroot/name/filename).read_text()))
    archive=ROOT/'review_bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:bundle.add(out,arcname='room0_repair_batch_20261007')
    record={'status':'PASS','path':str(archive),'sha256':r.sha(archive),'bytes':archive.stat().st_size,'full_points':len(xyz),'PLY_roundtrip_verified':True}
    f.atomic_json(ROOT/'review_bundle.json',record);print(json.dumps(record),flush=True)

if __name__=='__main__':main()
