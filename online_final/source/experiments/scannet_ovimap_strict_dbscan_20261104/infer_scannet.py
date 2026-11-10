"""Frozen P1-A1 CropFormer frontend for ScanNet, without mask refinement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

E = Path(__file__).resolve().parent
C = Path('/data/chenkejun/ovimap_runtime_20260908/detectron2/projects/CropFormer')
sys.path[:0] = [str(C), str(C / 'demo_cropformer'), '/home/chenkejun/CVPR/experiments/p1a1_scannet_20261008/snapshot/revisable_instance_map/src']
from demo_from_dirs import setup_cfg
from predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image
from revisable_instance_map.scannet_frame_io import color_coordinates


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def predict(model, image):
    pred = model.run_on_image(image)['instances']
    keep = pred.scores >= .5
    score, masks = pred.scores[keep], pred.pred_masks[keep]
    result = np.zeros(image.shape[:2], dtype=np.uint16)
    if len(score) >= 65536:
        raise ValueError('Too many masks for uint16')
    for ix in torch.argsort(score):
        result[masks[ix].cpu().numpy() == 1] = int(ix) + 1
    return result, score.cpu().numpy().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root, out = Path(config['source']['scene_root']), Path(config['source']['mask_root'])
    (out / 'native').mkdir(parents=True, exist_ok=True)
    (out / 'records').mkdir(exist_ok=True)
    weights = Path('/data/chenkejun/CVPR/models/CropFormer_hornet_3x_03823a.pth')
    yaml = C / 'configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml'
    weights_hash = sha(weights)
    assert weights_hash == '71915b7e9dc63fc662b32c59dff513530a06221c3b22f6de7ff4d6ef82df1128'
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    cfg = setup_cfg(argparse.Namespace(config_file=str(yaml), opts=['MODEL.WEIGHTS', str(weights)]))
    model = VisualizationDemo(cfg)
    (out / 'model_config.yaml').write_text(cfg.dump())
    control_rgb = Path('/data/chenkejun/CVPR/datasets/Replica/room0/results/frame000000.jpg')
    if not control_rgb.is_file():
        control_rgb = Path('/home/chenkejun/beauty/conceptgraphs/data/Replica/room0/results/frame000000.jpg')
    control_mask = Path('/data/chenkejun/CVPR/datasets/cropformer_frozen/room0/frame000000.png')
    predicted, _ = predict(model, read_image(str(control_rgb), format='BGR'))
    expected = cv2.imread(str(control_mask), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(predicted, expected), 'Frontend differs from frozen P1-A1 Replica control'
    settings = dict(status='PASS', method='P1-A1', weights_sha256=weights_hash,
                    model_config_sha256=sha(yaml), predictor_sha256=sha(C / 'demo_cropformer/predictor.py'),
                    score_threshold=.5, painting='ascending score, original selected-mask IDs',
                    replica_control_pixel_identical=True, depth_mask_refinement=False,
                    alignment='K_color @ inv(K_depth), nearest label sampling; identity sensor extrinsics',
                    gpu=torch.cuda.get_device_name(0), gpu_visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write(out / 'model_provenance.json', settings)
    map_x, map_y = color_coordinates(root, config['camera']['width'], config['camera']['height'])
    records = []
    started = time.perf_counter()
    for no, i in enumerate(config['frame_ids'], 1):
        image_path = root / 'color' / f'{i}.jpg'
        native, aligned, record_path = out / 'native' / f'{i}.png', out / f'{i}.png', out / 'records' / f'{i}.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert sha(image_path) == record['input_sha256'] and sha(native) == record['native_mask_sha256'] and sha(aligned) == record['mask_sha256']
        else:
            start = time.perf_counter()
            result, scores = predict(model, read_image(str(image_path), format='BGR'))
            warped = cv2.remap(result, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            assert cv2.imwrite(str(native), result) and cv2.imwrite(str(aligned), warped)
            record = dict(frame=i, instances=int(len(np.unique(warped[warped > 0]))),
                          native_instances=int(len(np.unique(result[result > 0]))), selected_scores=scores,
                          input_sha256=sha(image_path), native_mask_sha256=sha(native), mask_sha256=sha(aligned),
                          seconds=time.perf_counter() - start)
            write(record_path, record)
        records.append(record)
        if no % 10 == 0 or no == 1 or no == len(config['frame_ids']):
            progress = dict(scene=config['scene'], frames_processed=no, frames_total=len(config['frame_ids']), seconds=time.perf_counter() - started)
            write(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    temporary = out / 'frames.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(x) + '\n' for x in records))
    temporary.replace(out / 'frames.jsonl')
    write(out / 'complete.json', dict(status='PASS', frames=len(records), config_sha256=sha(args.config), metadata_sha256=sha(out / 'frames.jsonl'), seconds=time.perf_counter() - started, **{k:v for k,v in settings.items() if k != 'status'}))


if __name__ == '__main__':
    main()
