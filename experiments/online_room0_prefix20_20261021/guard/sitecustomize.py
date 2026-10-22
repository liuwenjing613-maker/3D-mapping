"""Reject full-sequence experiment data and frames beyond the active cutoff."""
import json
import os
import re
import sys

current_cutoff = 95
settings_path = os.environ.get('CVPR_ONLINE_GUARD')
if settings_path:
    with open(settings_path, encoding='utf-8') as stream:
        settings = json.load(stream)
    root = settings['root']
    allowed = set(settings['allowed_source_files'])
    input_roots = settings['restricted_input_roots']
    old_results = settings['restricted_result_roots']
    audit_dir = root + '/io_audit'
    os.makedirs(audit_dir, exist_ok=True)
    log_fd = os.open(audit_dir + '/' + str(os.getpid()) + '.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    handling = False

    def beneath(path, directory):
        return path == directory or path.startswith(directory + '/')

    def guard(event, args):
        global handling
        if event != 'open' or handling or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        handling = True
        try:
            original = os.fsdecode(args[0])
            path = os.path.realpath(original)
            if beneath(path, audit_dir):
                return
            mode = args[1] if len(args) > 1 else None
            reason = None
            tracked = beneath(path, root) or path in allowed
            if '/ground_truth/' in path or os.path.basename(path) in ['gt.npz','info_semantic.json','semantic.ply']:
                reason = 'GT_is_not_an_input_to_online_selection'
            if any(beneath(path, r) for r in input_roots) and path not in allowed:
                reason = 'outside_prefix_input_allowlist'
            if any(beneath(path, r) for r in old_results) and not beneath(path, root) and path not in allowed:
                reason = 'historical_full_sequence_result'
            if tracked:
                match = re.search(r'(?:frame|depth)(\d{6})', os.path.basename(path))
                if match and int(match.group(1)) > current_cutoff:
                    reason = 'future_frame_after_current_mapping_step'
            if tracked or reason:
                record = dict(path=path, requested=original, mode=str(mode), cutoff=current_cutoff,
                              blocked=bool(reason), reason=reason, probe=os.environ.get('CVPR_GUARD_TEST')=='1')
                os.write(log_fd, (json.dumps(record, ensure_ascii=False) + '\n').encode())
            if reason:
                raise PermissionError('ONLINE_INPUT_FENCE: ' + reason + ': ' + original)
        finally:
            handling = False

    sys.addaudithook(guard)
