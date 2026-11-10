"""SAM2.1 full sequence with only the two original human chair seeds."""
from pathlib import Path
import gc,hashlib,json,os,sys,time
import numpy as np
from PIL import Image
import torch

ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
UID='room2-14047feb0df3aaa5'
WORK=ROOT/'joint_mask_replacement_20261007/room2_mask14_mask24'
OUT=WORK/'tracking'
REPO=Path('/home/chenkejun/CVPR/repos/sam2_1')
CKPT=Path('/data/chenkejun/CVPR/models/sam2.1_hiera_large.pt')
sys.path.insert(0,str(REPO))
from sam2.build_sam import build_sam2_video_predictor

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        d=json.loads((OUT/'complete.json').read_text())
        assert d['code_sha256']==sha(__file__)
        print('Existing verified tracking is preserved',flush=True);return
    manifest=json.loads((ROOT/'human_seeds_20ad6139511b/seed_manifest.json').read_text())
    seed=next(s for s in manifest['seeds'] if s['case_uid']==UID)
    file=ROOT/'human_seeds_20ad6139511b'/seed['seed_file']
    assert sha(file)==seed['seed_sha256']
    original=np.array(Image.open(file),np.uint16)
    ids=[4,7]
    assert {o['track_id']:o['native_mask_id'] for o in seed['objects']}[4]==14
    assert {o['track_id']:o['native_mask_id'] for o in seed['objects']}[7]==24
    immutable=np.where(np.isin(original,ids),original,0).astype(np.uint16)
    derived=OUT/'derived_joint_seed.png';Image.fromarray(immutable).save(derived)
    assert sha(CKPT)=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
    cfg=json.loads(Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2/configs/raw.json').read_text())
    source=cfg['source'];rgbroot=Path(source['scene_root'])
    poses=np.loadtxt(rgbroot/source['trajectory']).reshape(-1,4,4)
    total=len(poses);fid=seed['source_choice']['frame'];records={}
    assert total==2000
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    start=time.monotonic()
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(CKPT),device='cuda',apply_postprocessing=False)
    status={'status':'RUNNING','case_uid':UID,'native_mask_ids':[14,24],'track_ids':ids,'seed_frame':fid,
        'objects':2,'total_frames':total,'completed_frames':0,'original_seed_sha256':sha(file),
        'derived_seed_sha256':sha(derived),'code_sha256':sha(__file__),'checkpoint_sha256':sha(CKPT),
        'GT_used':False,'other_six_seed_masks_used':False,'GPU_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'partition':'largest logit; smaller track ID breaks ties'}
    dump(OUT/'status.json',status)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for reverse,rawframes in [(False,list(range(fid,total))),(True,list(range(fid+1)))]:
            video=OUT/('rgb_reverse' if reverse else 'rgb_forward');video.mkdir(exist_ok=True)
            for i,raw in enumerate(rawframes):
                original_rgb=rgbroot/source['rgb_pattern'].format(frame=raw)
                link=video/f'{i:05d}.jpg'
                if link.is_symlink():assert link.resolve()==original_rgb.resolve()
                else:link.symlink_to(original_rgb.resolve())
            seed_idx=rawframes.index(fid)
            state=predictor.init_state(str(video),offload_video_to_cpu=True,offload_state_to_cpu=True)
            for oid in ids:predictor.add_new_mask(state,seed_idx,oid,immutable==oid)
            for index,obj_ids,logits in predictor.propagate_in_video(state,start_frame_idx=seed_idx,
                    max_frame_num_to_track=len(rawframes)-1,reverse=reverse):
                raw=rawframes[index]
                if raw in records:
                    assert raw==fid;continue
                assert obj_ids==ids
                scores=logits[:,0].float().cpu().numpy()
                maximum=scores.max(0);winner=scores.argmax(0)
                labels=np.where(maximum>0,np.asarray(ids,np.uint16)[winner],0).astype(np.uint16)
                if raw==fid:labels=immutable.copy()
                output=OUT/f'f{raw:06d}.png';Image.fromarray(labels).save(output)
                counts=np.bincount(labels.ravel(),minlength=8)
                records[raw]={'frame':raw,'direction':'reverse' if reverse else 'forward','label_file':str(output),
                    'label_sha256':sha(output),'seed':raw==fid,'mapping_frame':raw%5==0,
                    'areas':{str(oid):int(counts[oid]) for oid in ids}}
                if len(records)%100==0:
                    status.update(completed_frames=len(records),current_frame=raw,elapsed_seconds=round(time.monotonic()-start,2))
                    dump(OUT/'status.json',status)
                    print(json.dumps({k:status[k] for k in ['completed_frames','total_frames','elapsed_seconds']}),flush=True)
            del state;gc.collect();torch.cuda.empty_cache()
    assert sorted(records)==list(range(2000))
    np.testing.assert_array_equal(np.array(Image.open(OUT/f'f{fid:06d}.png')),immutable)
    assert sha(file)==seed['seed_sha256']
    report={**status,'status':'PASS','completed_frames':2000,'elapsed_seconds':round(time.monotonic()-start,2),
        'seed_preserved_exactly_for_both_targets':True,'directions_use_independent_seed_only_states':True,
        'frames':[records[f] for f in sorted(records)]}
    dump(OUT/'complete.json',report)
    dump(OUT/'status.json',{k:v for k,v in report.items() if k!='frames'})
    print(json.dumps({'status':'PASS','frames':2000,'objects':2,'seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        dump(OUT/'failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
