"""Freeze annotation pixels and GT-free seed diagnostics before tracking."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,os,sys
import numpy as np
from PIL import Image,ImageDraw

LEGACY=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
sys.path.insert(0,str(LEGACY)); sys.path.insert(0,str(LEGACY/'strict_local_repair_20261007'))
import run_repairs as r
import fixed_surface_repair as f
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')

def main():
    inp=ROOT/'inputs'
    ingest=json.loads((inp/'local_ingest.json').read_text())
    assert r.sha(inp/'human_selection.json')==ingest['annotation_sha256']
    scene=r.Scene('room2'); scene.baseline_control()
    assert scene.source_hashes[str(r.BASE/'final/room2/P1-A1/diffusion/holes_geodesic/instance_surface.npz')]==ingest['source_surface_sha256']
    page=Path('/data/chenkejun/CVPR/revisable_instance_map/room2_selection_top30_20261008/review')
    for row in ingest['seeds']:
        if row['status']=='READY':
            assert r.sha(page/row['source_label_relative_path'])==row['source_label_sha256']
    scene.verify_unchanged()
    manifest=[]; diagnostics=[]; supports={}; contact=[]
    gpus=['GPU-d31841e8-d3ae-e7f2-f581-8dc866e5c129','GPU-b003c3ac-453c-dd7b-90b0-31ad9adb3163','GPU-9a11b2b0-1816-1e33-9dd1-395cf0aaf9dd']
    for row in ingest['seeds']:
        if row['status']!='READY':
            manifest.append(row); continue
        choice=row['source_choice']; uid=row['case_uid']; fid=choice['frame']
        original=scene.source.load_frame(fid)
        cropped=np.array(Image.open(inp/row['seed_file']),np.uint16)
        assert r.sha(inp/row['seed_file'])==row['seed_sha256']
        selected=cropped.copy()
        if choice['choice']=='original':
            selected[:]=0
            x0,y0,x1,y1=choice['crop_xyxy']
            for obj in row['objects']:
                oid,mid=obj['track_id'],obj['native_mask_id']
                np.testing.assert_array_equal(cropped[y0:y1,x0:x1]==oid,original.mask_local[y0:y1,x0:x1]==mid)
                selected[original.mask_local==mid]=oid
                np.testing.assert_array_equal(selected==oid,original.mask_local==mid)
        out=ROOT/'cases'/uid; out.mkdir(parents=True,exist_ok=True)
        seedfile=out/'prepared_seed.png'; Image.fromarray(selected).save(seedfile)
        pts,locals_,stats=r.project_frame_regions(replace(original,mask_local=selected),scene.tree,pixel_stride=2,max_distance_m=.015,workers=8)
        f.atomic_npz(out/'seed_projected_support.npz',surface_point_index=pts,track_id=locals_)
        objects=[]; families=set()
        for obj in row['objects']:
            oid=obj['track_id']; mask=selected==oid; p=np.unique(pts[locals_==oid]); assert len(p)
            mids,counts=np.unique(original.mask_local[mask],return_counts=True)
            hist=sorted([{'old_local_id':int(mid),'old_persistent_id':int(scene.lookup[(fid,int(mid))]) if mid else -1,'pixels':int(n),'fraction':float(n/mask.sum())} for mid,n in zip(mids,counts)],key=lambda x:-x['pixels'])
            positive=[x for x in hist if x['old_persistent_id']>0]
            assert positive
            # Freeze the dominant actual old observation as the candidate family;
            # the full histogram stays available for inspection, with no GT input.
            family=positive[0]['old_persistent_id']; families.add(family)
            oldids,oldn=np.unique(scene.final[p],return_counts=True)
            bounds=np.stack([scene.xyz[p].min(0),scene.xyz[p].max(0)])
            record={**obj,'seed_pixels':int(mask.sum()),'projected_seed_points':len(p),'old_family_id':family,
                    'old_pixel_overlap':hist,'baseline_seed_support_labels':dict(zip(oldids.astype(str).tolist(),oldn.tolist())),
                    'seed_xyz_bounds_m':bounds.tolist()}
            objects.append(record); supports[oid]=p
            rgb=original.rgb.copy(); color=np.array([(oid*53)%180+60,(oid*97)%180+60,(oid*137)%180+60],np.uint8)
            rgb[mask]=(rgb[mask].astype(float)*.35+color*.65).astype(np.uint8)
            im=Image.fromarray(rgb).resize((600,340)); draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,600,27),fill='black'); draw.text((8,8),f'{choice["ROI"]} frame {fid} mask {obj["native_mask_id"]} -> track {oid} old PID {family}',fill='white')
            contact.append(im)
        casecfg={'schema_version':1,'scene':'room2','case_uid':uid,'experiment_mode':'offline_retrospective',
                 'seed_mode':'original_reassociate' if choice['choice']=='original' else 'new_mask_reassociate',
                 'seed_frame':fid,'seed_manifest':str(ROOT/'seed_manifest.json'),
                 'objects':[{'track_id':o['track_id'],**({'original_mask_id':o['native_mask_id']} if choice['choice']=='original' else {})} for o in objects],
                 'old_family_ids':sorted(families),'policy_file':str(LEGACY/'objectwise_repair_20261007/policy.json'),
                 'output_dir':str(out),'tracking':{'checkpoint':'/data/chenkejun/CVPR/models/sam2.1_hiera_large.pt',
                     'checkpoint_sha256':'2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318',
                     'repo':'/home/chenkejun/CVPR/repos/sam2_1','python':'/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python',
                     'cuda_visible_devices':gpus[len(diagnostics)%len(gpus)],'directions':['forward','reverse']}}
        f.atomic_json(HERE/(uid+'.json'),casecfg)
        association={'status':'PREPARED_FOR_TRACKING_ONLY','scene':'room2','case_uid':uid,'seed_frame':fid,
                     'seed_file':str(seedfile),'seed_sha256':r.sha(seedfile),'objects':objects,
                     'GT_used':False,'persistent_identity_assignment_pending_global_review':True}
        f.atomic_json(out/'frozen_association.json',association)
        diagnostics.append({'case_uid':uid,'source_choice':choice,'objects':objects,'old_family_ids':sorted(families),
                            'projection':stats,'seed_sha256':r.sha(seedfile),'config_path':str(HERE/(uid+'.json'))})
        manifest.append({**row,'objects':objects,'seed_file':str(seedfile.relative_to(ROOT)),'seed_sha256':r.sha(seedfile)})
    pairs=[]
    for a,pa in supports.items():
        for b,pb in supports.items():
            if a>=b: continue
            inter=len(np.intersect1d(pa,pb))
            if inter:
                pairs.append({'track_a':a,'track_b':b,'intersection':inter,'dice':2*inter/(len(pa)+len(pb)),
                              'coverage_a':inter/len(pa),'coverage_b':inter/len(pb)})
    canvas=Image.new('RGB',(1800,340*((len(contact)+2)//3)),(245,245,245))
    for i,im in enumerate(contact): canvas.paste(im,(600*(i%3),340*(i//3)))
    canvas.save(ROOT/'seed_contact.jpg',quality=92)
    for start in range(0,len(contact),6):
        sheet=Image.new('RGB',(1800,680),(245,245,245))
        for k,im in enumerate(contact[start:start+6]):sheet.paste(im,(600*(k%3),340*(k//3)))
        sheet.save(ROOT/('seed_contact_%02d.jpg'%(start//6+1)),quality=95)
    f.atomic_json(ROOT/'seed_manifest.json',{'status':'PASS','scene':'room2','annotation_sha256':ingest['annotation_sha256'],
                    'repair_cases':len(diagnostics),'selected_masks':len(supports),'total_ranked_cases':30,'unannotated_cases':ingest['unannotated_cases'],'seeds':manifest,'GT_used':False})
    f.atomic_json(ROOT/'seed_diagnostics.json',{'status':'PASS','GT_used':False,'cases':diagnostics,'cross_case_surface_overlap':pairs})
    raw=json.loads((r.INPUT/'room2/configs/raw.json').read_text()); src=raw['source']; rgbroot=Path(src['scene_root'])
    rgbhashes={str(rgbroot/src['rgb_pattern'].format(frame=fid)):r.sha(rgbroot/src['rgb_pattern'].format(frame=fid)) for fid in range(2000)}
    f.atomic_json(ROOT/'input_freeze.json',{'status':'FROZEN_BEFORE_INFERENCE','original_baseline_source_hashes':scene.source_hashes,
                  'annotation_sha256':ingest['annotation_sha256'],'raw_rgb_hashes':rgbhashes,
                  'seed_and_config_hashes':{str(p):r.sha(p) for p in [ROOT/'seed_manifest.json',ROOT/'seed_diagnostics.json',*[Path(c['config_path']) for c in diagnostics],*[ROOT/c['seed_file'] for c in manifest if c['status']=='READY']]},
                  'prepare_code_sha256':r.sha(__file__),'tracking_code_path':str(LEGACY/'objectwise_repair_20261007/track_case.py'),
                  'tracking_code_sha256':r.sha(LEGACY/'objectwise_repair_20261007/track_case.py'), 'GT_used':False})
    scene.verify_unchanged()
    print(json.dumps({'status':'PASS','cases':len(diagnostics),'objects':len(supports),'cross_seed_overlaps':pairs},ensure_ascii=False))

if __name__=='__main__': main()
