"""Add the playback link while preserving the existing repair result page."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
ROOT = WORK / 'results/固定案例_三模型对比_20261006'
assert json.loads((HERE / 'local_validation.json').read_text())['status'] == 'PASS'
entry = '<a href="../room2_tracking_playback_20261008/index.html">按帧观看全部实例追踪（可循环）</a> · '
marker = '<a href="human_selection.json">'
paths = [ROOT / 'room2_top30_repair_20261008/index.html', WORK / 'outputs/room2_top30_repair_20261008/build_review.py']
changes = []
for path in paths:
    before = path.read_bytes()
    if entry.encode('utf-8') not in before:
        assert before.count(marker.encode('utf-8')) == 1, path
        after = before.replace(marker.encode('utf-8'), (entry + marker).encode('utf-8'), 1)
        assert after.replace(entry.encode('utf-8'), b'', 1) == before
        path.write_bytes(after)
    after = path.read_bytes()
    changes.append({'file': str(path), 'before_sha256': hashlib.sha256(before).hexdigest(),
                    'after_sha256': hashlib.sha256(after).hexdigest(), 'change': 'one playback link'})
(HERE / 'entry_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PASS', 'playback_entries_added': 2, 'tracking_repair_and_v3_modified': False}))
