"""Make an input view containing exactly the twenty processed RGB-D/mask frames."""
from itertools import islice
from online_common import *

def main():
    assert len(FIDS) == 20 and FIDS[-1] == 95
    if (R / 'inputs_manifest.json').exists():
        manifest = read(R / 'inputs_manifest.json')
        assert manifest['frame_ids'] == list(FIDS)
        for item in manifest['files']:
            assert sha(item['path']) == item['sha256']
        print('PREFIX_INPUTS_VERIFIED', flush=True)
        return
    source_config = Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/configs/room0.json')
    config = read(source_config)
    old_scene = Path(config['source']['scene_root']).resolve()
    old_masks = Path(config['source']['mask_root']).resolve()
    frozen = Path('/data/chenkejun/CVPR/datasets/cropformer_frozen/room0').resolve()
    scene = R / 'inputs/scene'
    masks = R / 'inputs/masks'
    (scene / 'results').mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    records = []
    for frame in FIDS:
        for kind, source, destination in (
            ('rgb', old_scene / f'results/frame{frame:06d}.jpg', scene / f'results/frame{frame:06d}.jpg'),
            ('depth', old_scene / f'results/depth{frame:06d}.png', scene / f'results/depth{frame:06d}.png'),
            ('mask', old_masks / f'frame{frame:06d}.png', masks / f'frame{frame:06d}.png'),
        ):
            source = source.resolve(strict=True)
            if not destination.exists():
                destination.symlink_to(source)
            assert destination.resolve() == source
            digest = sha(source)
            if kind == 'mask':
                assert digest == sha(frozen / f'frame{frame:06d}.png'), 'Frontend cache differs from mapping mask'
            records.append(dict(frame=frame, kind=kind, path=str(destination), source=str(source), sha256=digest))
    with (old_scene / config['source']['trajectory']).open() as f:
        pose_lines = list(islice(f, 96))
    assert len(pose_lines) == 96
    (scene / 'traj.txt').write_text(''.join(pose_lines), encoding='utf-8')
    metadata = []
    with (frozen / 'frames.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            if row['frame'] in FIDS:
                metadata.append(row)
    assert sorted(row['frame'] for row in metadata) == list(FIDS)
    (masks / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metadata), encoding='utf-8')
    config['frame_selection']['stop_exclusive'] = 96
    config['source'].update(scene_root=str(scene), mask_root=str(masks))
    config['purpose'] = 'causal_room0_prefix20_strict_vote_ROI_selection'
    write(C / 'config.json', config)
    write(R / 'inputs_manifest.json', dict(status='PASS', scene='room0', frame_ids=list(FIDS), processed_frames=20,
        last_frame_id=95, files=records, pose_rows=96, trajectory_sha256=sha(scene / 'traj.txt'),
        source_config=str(source_config), source_config_sha256=sha(source_config), config=str(C / 'config.json'),
        config_sha256=sha(C / 'config.json'), use_GT=False, use_future_frames=False,
        mask_cache_provenance='per-frame frozen CropFormer, verified equal to raw baseline mask cache'))
    write(R / 'input_guard.json', dict(root=str(R.resolve()), allowed_source_files=[x['source'] for x in records],
        restricted_input_roots=[str(old_scene), str(old_masks), str(frozen)],
        restricted_result_roots=['/data/chenkejun/CVPR/results', '/data/chenkejun/CVPR/revisable_instance_map',
                                 '/data/chenkejun/CVPR/evaluation_results'], frame_ids=list(FIDS)))
    print('PREFIX_INPUTS_READY', len(records), flush=True)

if __name__ == '__main__':
    main()
