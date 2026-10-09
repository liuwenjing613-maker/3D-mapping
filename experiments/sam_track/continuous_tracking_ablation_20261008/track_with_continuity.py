"""SAM2.1 offline full-sequence propagation with independent two directions."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
from PIL import Image
import torch
from continuity_policy import DirectionalGate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def main(config_path, output_dir, max_empty_gap):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    assert cfg['seed_mode'] != 'original_control'
    assert cfg['experiment_mode'] == 'offline_retrospective'
    assert cfg['tracking']['directions'] == ['forward', 'reverse']
    case = Path(cfg['output_dir'])
    out = Path(output_dir).resolve() / 'tracking'
    assert 'CVPR' in out.parts and out != (case / 'tracking').resolve(), 'Use a separate CVPR output directory'
    out.mkdir(parents=True, exist_ok=True)
    association = json.loads((case / 'frozen_association.json').read_text())
    file = Path(association['seed_file'])
    assert sha(file) == association['seed_sha256']
    immutable = np.array(Image.open(file), np.uint16)
    ids = [obj['track_id'] for obj in cfg['objects']]
    assert ids == sorted(ids), 'Sorted track IDs implement the same smaller-ID logit tiebreak'
    assert set(np.unique(immutable)).issubset({0, *ids})
    assert all(np.any(immutable == oid) for oid in ids)
    checkpoint = Path(cfg['tracking']['checkpoint'])
    assert sha(checkpoint) == cfg['tracking']['checkpoint_sha256']
    if (out / 'complete.json').exists():
        previous = json.loads((out / 'complete.json').read_text())
        assert previous['code_sha256'] == sha(__file__)
        assert previous['case_config_sha256'] == sha(config_path)
        assert previous['seed_sha256'] == sha(file)
        assert previous['max_empty_gap'] == max_empty_gap
        return
    sys.path.insert(0, cfg['tracking']['repo'])
    from sam2.build_sam import build_sam2_video_predictor
    raw = json.loads(Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main', cfg['scene'], 'configs/raw.json').read_text())
    source = raw['source']
    rgbroot = Path(source['scene_root'])
    poses = np.loadtxt(rgbroot / source['trajectory']).reshape(-1, 4, 4)
    total = len(poses)
    fid = cfg['seed_frame']
    assert 0 <= fid < total
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started = time.monotonic()
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml', str(checkpoint),
                                          device='cuda', apply_postprocessing=False)
    records = {}
    status = {'status': 'RUNNING', 'case_uid': cfg['case_uid'], 'seed_frame': fid, 'track_ids': ids,
              'seed_mode': cfg['seed_mode'], 'seed_sha256': sha(file), 'case_config_sha256': sha(config_path),
              'code_sha256': sha(__file__), 'checkpoint_sha256': sha(checkpoint), 'GT_used': False,
              'total_frames': total, 'completed_frames': 0,
              'partition': 'largest logit; smaller track ID breaks ties',
              'GPU_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'max_empty_gap':max_empty_gap, 'stop_policy':'native post-partition area==0; each original frame; per-target independent permanent stop',
              'directional_stops':{}, 'inferred_nonseed_frames':0,
              'inactive_targets_remain_in_internal_model_state_while_other_targets_continue':True}
    assert max_empty_gap in (0,1)
    dump(out / 'status.json', status)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for reverse, rawframes in [(False, list(range(fid, total))), (True, list(range(fid + 1)))]:
            video = out / ('rgb_reverse' if reverse else 'rgb_forward')
            video.mkdir(exist_ok=True)
            for index, raw_id in enumerate(rawframes):
                original_rgb = (rgbroot / source['rgb_pattern'].format(frame=raw_id)).resolve()
                link = video / ('%05d.jpg' % index)
                if link.is_symlink():
                    assert link.resolve() == original_rgb
                else:
                    link.symlink_to(original_rgb)
            seed_index = rawframes.index(fid)
            gate = DirectionalGate(ids,fid,-1 if reverse else 1,max_empty_gap)
            direction_name = 'reverse' if reverse else 'forward'
            # A fresh state prevents forward results from leaking into reverse memory.
            state = predictor.init_state(str(video), offload_video_to_cpu=True, offload_state_to_cpu=True)
            for oid in ids:
                predictor.add_new_mask(state, seed_index, oid, immutable == oid)
            for index, object_ids, logits in predictor.propagate_in_video(state, start_frame_idx=seed_index,
                    max_frame_num_to_track=len(rawframes) - 1, reverse=reverse):
                raw_id = rawframes[index]
                if raw_id in records:
                    assert raw_id == fid
                    continue
                assert object_ids == ids
                scores = logits[:, 0].float().cpu().numpy()
                labels = np.where(scores.max(0) > 0, np.asarray(ids, np.uint16)[scores.argmax(0)], 0).astype(np.uint16)
                if raw_id == fid:
                    labels = immutable.copy()
                raw_areas = {oid:int(np.sum(labels == oid)) for oid in ids}
                if raw_id != fid:
                    alive = gate.consume(raw_id,raw_areas)
                    labels = np.where(np.isin(labels,alive),labels,0).astype(np.uint16)
                    status['inferred_nonseed_frames'] += 1
                else: alive = ids
                status['directional_stops'][direction_name] = dict(gate.stops)
                output = out / ('f%06d.png' % raw_id)
                Image.fromarray(labels).save(output)
                records[raw_id] = {'frame': raw_id, 'direction': 'reverse' if reverse else 'forward',
                    'label_file': str(output), 'label_sha256': sha(output), 'seed': raw_id == fid,
                    'mapping_frame': raw_id % raw['frame_selection']['stride'] == 0,
                    'areas': {str(oid): int(np.sum(labels == oid)) for oid in ids},
                    'inference_run':True,'raw_areas_pre_policy':raw_areas,'stopped_track_ids':sorted(gate.stops)}
                if len(records) % 100 == 0:
                    status.update(completed_frames=len(records), current_frame=raw_id,
                                  elapsed_seconds=round(time.monotonic() - started, 2))
                    dump(out / 'status.json', status)
                    print(json.dumps({k: status[k] for k in ['completed_frames', 'total_frames', 'elapsed_seconds']}), flush=True)
                if not alive:
                    # The whole group is stopped: no more SAM calls in this direction.
                    empty = np.zeros_like(immutable)
                    remaining = [v for v in rawframes if v < raw_id] if reverse else [v for v in rawframes if v > raw_id]
                    for remaining_id in remaining:
                        output = out / ('f%06d.png' % remaining_id)
                        Image.fromarray(empty).save(output)
                        records[remaining_id] = {'frame':remaining_id,'direction':direction_name,
                            'label_file':str(output),'label_sha256':sha(output),'seed':False,
                            'mapping_frame':remaining_id % raw['frame_selection']['stride'] == 0,
                            'areas':{str(oid):0 for oid in ids},'inference_run':False,
                            'raw_areas_pre_policy':None,'stopped_track_ids':sorted(gate.stops),
                            'empty_is_policy_abstention_not_model_absence':True}
                    break
            del state
            gc.collect(); torch.cuda.empty_cache()
    assert sorted(records) == list(range(total))
    np.testing.assert_array_equal(np.array(Image.open(out / ('f%06d.png' % fid))), immutable)
    assert sha(file) == association['seed_sha256']
    report = {**status, 'status': 'PASS', 'completed_frames': total,
              'elapsed_seconds': round(time.monotonic() - started, 2),
              'seed_preserved_exactly_for_both_targets': True,
              'directions_use_independent_seed_only_states': True,
              'frames': [records[frame] for frame in sorted(records)]}
    dump(out / 'complete.json', report)
    dump(out / 'status.json', {key: value for key, value in report.items() if key != 'frames'})
    print(json.dumps({'status': 'PASS', 'tracking_frames': total, 'seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--max-empty-gap',type=int,choices=[0,1],required=True)
    args=parser.parse_args()
    main(args.config,args.output_dir,args.max_empty_gap)
