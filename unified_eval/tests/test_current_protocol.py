"""The main default must select the audited baseline and reject accidental mixing."""
import hashlib
import json
from pathlib import Path

import pytest

from unified_eval import current_protocol as current
from unified_eval.cli import build_parser, validate_current_command
from unified_eval.schema import EvaluationError


def scene_args(*extra):
    return build_parser().parse_args(['eval-scene', '--gt', 'unused-gt.npz',
        '--pred', 'unused-pred.npz', '--out', 'unused-output', *extra])


def test_current_registry_locks_the_accepted_scoring_and_350_gt():
    assert current.require_current_protocol()
    assert current.CURRENT_PROTOCOL['profile_revision'] == 2
    assert current.CURRENT_PROTOCOL['scoring_commit'] == 'cd260569e89de1e9525f82c45ee10842b80616b0'
    assert current.CURRENT_PROTOCOL['audit_archive_commit'] == 'd035f68770cefab27c9751e97062c72faa07367c'
    assert sum(x['evaluable_GT_count'] for x in current.CURRENT_PROTOCOL['canonical_GT'].values()) == 350
    assert json.loads(current.CURRENT_CONFIG.read_text())['frozen'] is False


def test_omitted_config_uses_revision_two_without_historical_opt_in():
    args = scene_args()
    assert args.config == current.CURRENT_CONFIG
    assert not args.historical_protocol


def test_explicit_old_config_requires_historical_opt_in_before_gt_read():
    old = current.CURRENT_CONFIG.with_name('replica_ca_v3.object_observed_repair.development.json')
    with pytest.raises(EvaluationError, match='revision 2'):
        validate_current_command(scene_args('--config', str(old)))


def test_explicit_historical_reproduction_remains_available():
    old = current.CURRENT_CONFIG.with_name('replica_ca_v3.object_observed_repair.development.json')
    validate_current_command(scene_args('--config', str(old), '--historical-protocol'))


def test_modified_revision_two_config_is_not_the_current_baseline(tmp_path):
    raw = json.loads(current.CURRENT_CONFIG.read_text())
    raw['geometry_mapping']['max_distance_m'] = .02
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(EvaluationError, match='revision 2'):
        current.require_current_protocol(path)


def test_changed_scoring_cannot_keep_the_same_current_identity(monkeypatch):
    monkeypatch.setattr(current, 'repair_evaluator_code_hashes', lambda: {})
    with pytest.raises(EvaluationError, match='cd260569'):
        current.require_current_protocol()


def test_debug_overrides_are_rejected_before_output_or_gt_access():
    with pytest.raises(EvaluationError, match='parameters are locked'):
        validate_current_command(scene_args('--debug-max-distance-m', '.02'))


def test_copied_locked_gt_is_valid_but_edited_copy_is_rejected(tmp_path, monkeypatch):
    original = tmp_path / 'locked.npz'
    original.write_bytes(b'canonical-file-bytes-for-identity-test')
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    monkeypatch.setitem(current.CURRENT_PROTOCOL, 'canonical_GT', {'test_scene': {'file_sha256': digest}})
    copied = tmp_path / 'copied.npz'
    copied.write_bytes(original.read_bytes())
    assert current.require_current_gt(copied) == 'test_scene'
    copied.write_bytes(copied.read_bytes() + b'changed')
    with pytest.raises(EvaluationError, match='350-GT'):
        current.require_current_gt(copied)


def test_unlocked_gt_does_not_fall_back_to_a_historical_scope(tmp_path):
    gt = tmp_path / 'different-scope.npz'
    gt.write_bytes(b'unknown-GT-file')
    with pytest.raises(EvaluationError, match='350-GT'):
        current.require_current_gt(gt)


def test_legacy_gt_export_requires_an_explicit_historical_request():
    args = build_parser().parse_args(['export-replica-gt', '--reference-root', 'unused',
        '--scene', 'room0', '--out', 'unused.npz'])
    with pytest.raises(EvaluationError, match='legacy GT export'):
        validate_current_command(args)


def test_final_400_frame_scope_is_not_used_for_online_prefixes():
    args = build_parser().parse_args(['eval-online-prefix', '--gt', 'unused.npz',
        '--online-manifest', 'unused.json', '--out', 'unused'])
    with pytest.raises(EvaluationError, match='online prefixes'):
        validate_current_command(args)


def test_historical_flag_cannot_disable_locks_on_the_current_config():
    assert current.require_current_protocol(historical_protocol=True)
    args = build_parser().parse_args(['eval-online-prefix', '--historical-protocol',
        '--gt', 'unused.npz', '--online-manifest', 'unused.json', '--out', 'unused'])
    with pytest.raises(EvaluationError, match='online prefixes'):
        validate_current_command(args)
