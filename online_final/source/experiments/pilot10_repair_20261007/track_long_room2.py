"""Full-sequence timing/quality pilot, preserving the same eight human seeds."""
from pathlib import Path
import gc, hashlib, json, os, sys, time
import numpy as np
from PIL import Image
import torch

ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
OUT=ROOT/'long_tracking_20261007/room2-14047feb0df3aaa5'
REPO=Path('/home/chenkejun/CVPR/repos/sam2_1')
CKPT=Path('/data/chenkejun/CVPR/models/sam2.1_hiera_large.pt')
sys.path.insert(0,str(REPO))
from sam2.build_sam import build_sam2_video_predictor

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'complete.json').exists(),'Existing completed experiment is preserved'
    manifest=json.loads((ROOT/'human_seeds_20ad6139511b/seed_manifest.json').read_text())
    seed=next(s for s in manifest['seeds'] if s['case_uid']=='room2-14047feb0df3aaa5')
    seedfile=ROOT/'human_seeds_20ad6139511b'/seed['seed_file']
    assert sha(seedfile)==seed['seed_sha256']
    immutable=np.array(Image.open(seedfile),np.uint16)
    ids=[o['track_id'] for o in seed['objects']]
    cfg=json.loads(Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2/configs/raw.json').read_text())
    rgbroot=Path(cfg['source']['scene_root'])
    poses=np.loadtxt(rgbroot/cfg['source']['trajectory']).reshape(-1,4,4)
    total=len(poses);fid=seed['source_choice']['frame'];records={}
    assert total==2000 and ids==list(range(1,9))
    assert sha(CKPT)=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
    start=time.monotonic();torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(CKPT),device='cuda',apply_postprocessing=False)
    status={'status':'RUNNING','case_uid':seed['case_uid'],'scene':'room2','objects':8,'seed_frame':fid,
            'total_frames':total,'completed_frames':0,'GT_used':False,'3D_repair_applied':False,
            'seed_sha256':sha(seedfile),'code_sha256':sha(__file__),'checkpoint_sha256':sha(CKPT),
            'GPU_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,
            'same_seed_ids_as_short_run':True,'partition':'largest logit; smaller track ID breaks ties'}
    dump(OUT/'status.json',status)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for reverse,rawframes in [(False,list(range(fid,total))),(True,list(range(0,fid+1)))]:
            video=OUT/('rgb_reverse' if reverse else 'rgb_forward');video.mkdir(exist_ok=True)
            for i,raw in enumerate(rawframes):
                source=rgbroot/cfg['source']['rgb_pattern'].format(frame=raw)
                assert source.is_file()
                link=video/f'{i:05d}.jpg'
                if link.is_symlink():assert link.resolve()==source.resolve()
                else:link.symlink_to(source.resolve())
            seed_idx=rawframes.index(fid)
            init_start=time.monotonic()
            state=predictor.init_state(str(video),offload_video_to_cpu=True,offload_state_to_cpu=True)
            status['last_init_state_seconds']=round(time.monotonic()-init_start,2)
            for oid in ids:predictor.add_new_mask(state,seed_idx,oid,immutable==oid)
            for index,obj_ids,logits in predictor.propagate_in_video(state,start_frame_idx=seed_idx,max_frame_num_to_track=len(rawframes)-1,reverse=reverse):
                raw=rawframes[index]
                if raw in records:
                    assert raw==fid;continue
                assert obj_ids==ids
                scores=logits[:,0].float().cpu().numpy()
                maximum=scores.max(0);winner=scores.argmax(0)
                labels=np.where(maximum>0,np.asarray(ids,np.uint16)[winner],0).astype(np.uint16)
                if raw==fid:labels=immutable.copy()
                file=OUT/f'f{raw:06d}.png';Image.fromarray(labels).save(file)
                counts=np.bincount(labels.reshape(-1),minlength=9)
                records[raw]={'frame':raw,'offset':raw-fid,'direction':'reverse' if reverse else 'forward',
                              'label_file':str(file),'label_sha256':sha(file),'seed':raw==fid,
                              'mapping_frame':raw%5==0,'areas':{str(oid):int(counts[oid]) for oid in ids}}
                if len(records)%50==0:
                    status.update(completed_frames=len(records),current_frame=raw,elapsed_seconds=round(time.monotonic()-start,2))
                    dump(OUT/'status.json',status)
                    print(json.dumps({k:status[k] for k in ['completed_frames','total_frames','current_frame','elapsed_seconds']}),flush=True)
            del state;gc.collect();torch.cuda.empty_cache()
    assert sorted(records)==list(range(total))
    np.testing.assert_array_equal(np.array(Image.open(OUT/f'f{fid:06d}.png')),immutable)
    assert sha(seedfile)==seed['seed_sha256']
    report={**status,'status':'PASS','completed_frames':len(records),'elapsed_seconds':round(time.monotonic()-start,2),
            'seed_preserved_exactly':True,'directions_use_independent_seed_only_states':True,
            'objects_metadata':seed['objects'],'frames':[records[f] for f in sorted(records)],'source_config':cfg,
            'tracked_entire_sequence_to_avoid_missing_reappearances':True}
    dump(OUT/'complete.json',report);dump(OUT/'status.json',{k:v for k,v in report.items() if k not in ['frames','source_config']})
    print(json.dumps({'status':'PASS','frames':total,'objects':8,'seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
