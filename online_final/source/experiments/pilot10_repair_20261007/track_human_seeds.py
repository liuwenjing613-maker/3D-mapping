"""Propagate immutable human seed masks on raw RGB frames, without GT."""
from pathlib import Path
import gc, hashlib, json, os, sys, time, traceback
import numpy as np
from PIL import Image
import torch

ROOT = Path('/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007')
SEEDS = ROOT / 'human_seeds_20ad6139511b'
REPO = Path('/home/chenkejun/CVPR/repos/sam2_1')
CKPT = Path('/data/chenkejun/CVPR/models/sam2.1_hiera_large.pt')
sys.path.insert(0, str(REPO))
from sam2.build_sam import build_sam2_video_predictor


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def run_case(predictor, seed):
    started = time.monotonic()
    out = ROOT / 'tracking' / seed['case_uid']
    out.mkdir(parents=True, exist_ok=True)
    source = SEEDS / seed['seed_file']
    assert sha(source) == seed['seed_sha256']
    immutable_seed = np.array(Image.open(source), dtype=np.uint16)
    ids = [o['track_id'] for o in seed['objects']]
    assert np.array_equal(np.unique(immutable_seed[immutable_seed > 0]), ids)
    raw_frames = seed['raw_tracking_frames']
    fid = seed['source_choice']['frame']
    seed_idx = raw_frames.index(fid)
    video = out / 'rgb_symlinks'
    video.mkdir(exist_ok=True)
    cfg = json.loads(Path(f'/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/{seed["scene"]}/configs/raw.json').read_text())
    rgb_root = Path(cfg['source']['scene_root'])
    input_rgb = []
    for index, raw in enumerate(raw_frames):
        src = rgb_root / cfg['source']['rgb_pattern'].format(frame=raw)
        assert src.is_file()
        link = video / f'{index:05d}.jpg'
        if link.is_symlink():
            assert link.resolve() == src.resolve()
        else:
            assert not link.exists()
            link.symlink_to(src.resolve())
        input_rgb.append({'frame': raw, 'path': str(src), 'sha256': sha(src)})
    records = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        # Each direction starts with only the human seed, avoiding incidental
        # forward-memory reuse in the reverse pass.
        for reverse in (False, True):
            if reverse and seed_idx == 0:
                continue
            state = predictor.init_state(str(video), offload_video_to_cpu=True,
                                         offload_state_to_cpu=True)
            for oid in ids:
                predictor.add_new_mask(state, seed_idx, oid, immutable_seed == oid)
            limit = seed_idx if reverse else len(raw_frames) - seed_idx - 1
            for index, obj_ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=seed_idx, max_frame_num_to_track=limit, reverse=reverse):
                raw = raw_frames[index]
                if raw in records:
                    assert raw == fid
                    continue
                assert obj_ids == ids
                scores = logits[:, 0].float().cpu().numpy()
                assert scores.shape == (len(ids), *immutable_seed.shape)
                winner = np.argmax(scores, axis=0)
                maximum = np.max(scores, axis=0)
                labels = np.where(maximum > 0, np.asarray(ids, np.uint16)[winner], 0).astype(np.uint16)
                if raw == fid:
                    labels = immutable_seed.copy()
                label_file = out / f'f{raw:06d}.png'
                Image.fromarray(labels).save(label_file)
                areas = {str(oid): int(np.sum(labels == oid)) for oid in ids}
                overlap = int(np.sum(np.sum(scores > 0, axis=0) > 1))
                row = {'frame': raw, 'direction': 'reverse' if reverse else 'forward',
                       'label_file': str(label_file), 'label_sha256': sha(label_file),
                       'area_pixels': areas, 'overlap_pixels_before_partition': overlap,
                       'missing_track_ids': [oid for oid in ids if areas[str(oid)] == 0],
                       'seed_restored_exactly': raw == fid,
                       'eligible_for_mapping_votes': raw in seed['mapping_vote_frames']}
                # Preserve every raw binary proposal at mapping frames, including
                # overlaps that the deterministic max-logit partition resolves.
                if row['eligible_for_mapping_votes']:
                    raw_file = out / f'f{raw:06d}_raw_proposals.npz'
                    np.savez_compressed(raw_file, object_ids=np.asarray(ids, np.int32),
                        shape=np.asarray(scores.shape, np.int32),
                        positive_mask_bits=np.packbits((scores > 0).reshape(len(ids), -1), axis=1),
                        logit_min=scores.min(axis=(1, 2)), logit_max=scores.max(axis=(1, 2)))
                    row['raw_proposals_sha256'] = sha(raw_file)
                records[raw] = row
            del state
            gc.collect()
            torch.cuda.empty_cache()
    assert sorted(records) == raw_frames
    np.testing.assert_array_equal(np.array(Image.open(out / f'f{fid:06d}.png')), immutable_seed)
    assert sha(source) == seed['seed_sha256']
    report = {'status': 'PASS', 'case_uid': seed['case_uid'], 'scene': seed['scene'],
              'seed_sha256': seed['seed_sha256'], 'source_manifest_sha256': sha(SEEDS / 'seed_manifest.json'),
              'code_sha256': sha(__file__), 'ground_truth_used': False,
              'raw_frames': len(raw_frames), 'mapping_frames': len(seed['mapping_vote_frames']),
              'objects': len(ids), 'seed_preserved_exactly': True,
              'directions_use_independent_seed_only_states': True,
              'mask_threshold_logit': 0, 'overlap_partition': 'largest logit; smaller track ID breaks ties',
              'input_rgb': input_rgb, 'frames': [records[f] for f in raw_frames],
              'seconds': round(time.monotonic() - started, 2)}
    dump(out / 'complete.json', report)
    return report


def main():
    out = ROOT / 'tracking'
    out.mkdir(exist_ok=True)
    seeds = json.loads((SEEDS / 'seed_manifest.json').read_text())
    assert seeds['status'] == 'PASS' and seeds['target_objects'] == 55
    assert sha(CKPT) == '2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml', str(CKPT),
                                         device='cuda', apply_postprocessing=False)
    status = {'status': 'RUNNING', 'completed_cases': [], 'total_cases': 9,
              'GT_used': False, 'checkpoint_sha256': sha(CKPT),
              'code_sha256': sha(__file__), 'manifest_sha256': sha(SEEDS / 'seed_manifest.json'),
              'sam_source_sha256': {f: sha(REPO / f) for f in ['sam2/sam2_video_predictor.py', 'sam2/build_sam.py']},
              'GPU_visible': os.environ.get('CUDA_VISIBLE_DEVICES'), 'torch': torch.__version__,
              'GPU_name': torch.cuda.get_device_name(0)}
    dump(out / 'status.json', status)
    for seed in seeds['seeds']:
        if seed['status'] != 'READY':
            continue
        status['current_case'] = seed['case_uid']
        dump(out / 'status.json', status)
        report_path = out / seed['case_uid'] / 'complete.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            assert report['code_sha256'] == sha(__file__) and report['seed_sha256'] == seed['seed_sha256']
        else:
            report = run_case(predictor, seed)
        status['completed_cases'].append({k: report[k] for k in ['case_uid', 'objects', 'raw_frames', 'mapping_frames', 'seconds']})
        dump(out / 'status.json', status)
        print(json.dumps(status['completed_cases'][-1]), flush=True)
    status['status'] = 'PASS'
    status.pop('current_case', None)
    dump(out / 'status.json', status)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (ROOT / 'tracking').mkdir(exist_ok=True)
        dump(ROOT / 'tracking/failure.json', {'status': 'FAIL', 'traceback': traceback.format_exc()})
        raise
