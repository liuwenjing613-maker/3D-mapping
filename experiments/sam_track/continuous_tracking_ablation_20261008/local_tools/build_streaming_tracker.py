"""Prepare a separate SAM runner with the same streaming mask gate; never edit the frozen tracker."""
from pathlib import Path
import hashlib
import json

WORK = Path(__file__).resolve().parent
DONOR = WORK.parents[1] / 'outputs/pilot10_repair_20261007/objectwise_repair_20261007/track_case.py'


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    original = DONOR.read_bytes()
    source = original.decode('utf-8')
    source = replace_once(source, 'import torch\n', 'import torch\nfrom continuity_policy import DirectionalGate\n')
    source = replace_once(source, 'def main(config_path):', 'def main(config_path, output_dir, max_empty_gap):')
    source = replace_once(source, "    out = case / 'tracking'", "    out = Path(output_dir).resolve() / 'tracking'\n    assert 'CVPR' in out.parts and out != (case / 'tracking').resolve(), 'Use a separate CVPR output directory'")
    source = replace_once(source, "        assert previous['seed_sha256'] == sha(file)", "        assert previous['seed_sha256'] == sha(file)\n        assert previous['max_empty_gap'] == max_empty_gap")
    source = replace_once(source, "    assert set(np.unique(immutable)).issubset({0, *ids})", "    assert set(np.unique(immutable)).issubset({0, *ids})\n    assert all(np.any(immutable == oid) for oid in ids)")
    source = replace_once(source, "              'GPU_visible': os.environ.get('CUDA_VISIBLE_DEVICES')}", "              'GPU_visible': os.environ.get('CUDA_VISIBLE_DEVICES'),\n              'max_empty_gap':max_empty_gap, 'stop_policy':'native post-partition area==0; each original frame; per-target independent permanent stop',\n              'directional_stops':{}, 'inferred_nonseed_frames':0,\n              'inactive_targets_remain_in_internal_model_state_while_other_targets_continue':True}\n    assert max_empty_gap in (0,1)")
    source = replace_once(source, "            seed_index = rawframes.index(fid)", "            seed_index = rawframes.index(fid)\n            gate = DirectionalGate(ids,fid,-1 if reverse else 1,max_empty_gap)\n            direction_name = 'reverse' if reverse else 'forward'")
    source = replace_once(source, "                output = out / ('f%06d.png' % raw_id)", "                raw_areas = {oid:int(np.sum(labels == oid)) for oid in ids}\n                if raw_id != fid:\n                    alive = gate.consume(raw_id,raw_areas)\n                    labels = np.where(np.isin(labels,alive),labels,0).astype(np.uint16)\n                    status['inferred_nonseed_frames'] += 1\n                else: alive = ids\n                status['directional_stops'][direction_name] = dict(gate.stops)\n                output = out / ('f%06d.png' % raw_id)")
    source = replace_once(source, "                    'areas': {str(oid): int(np.sum(labels == oid)) for oid in ids}}", "                    'areas': {str(oid): int(np.sum(labels == oid)) for oid in ids},\n                    'inference_run':True,'raw_areas_pre_policy':raw_areas,'stopped_track_ids':sorted(gate.stops)}")
    source = replace_once(source, '            del state\n', '''                if not alive:
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
''')
    source = replace_once(source, "    main(parser.parse_args().config)", "    parser.add_argument('--output-dir',required=True)\n    parser.add_argument('--max-empty-gap',type=int,choices=[0,1],required=True)\n    args=parser.parse_args()\n    main(args.config,args.output_dir,args.max_empty_gap)")
    target = WORK / 'track_with_continuity.py'
    target.write_text(source,encoding='utf-8')
    assert DONOR.read_bytes() == original
    receipt = {'status':'GENERATED_NOT_RUN_ON_GPU','donor_sha256':hashlib.sha256(original).hexdigest(),
        'generated_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'frozen_original_tracker_modified':False,'requires_separate_output_dir':True,
        'one_object_stopping_does_not_modify_other_object_logits_or_internal_SAM_state':True,
        'whole_group_stopping_breaks_the_propagation_generator':True,
        'policy_generated_empty_frames_explicitly_distinguished_from_model_absence':True}
    (WORK/'streaming_tracker_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
