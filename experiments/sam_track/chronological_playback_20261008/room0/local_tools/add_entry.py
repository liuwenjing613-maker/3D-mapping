"""Add one playback entry to the existing viewer and its source template."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
ROOT=WORK/'results'/'固定案例_三模型对比_20261006'
assert json.loads((HERE/'local_validation.json').read_text(encoding='utf-8'))['status']=='PASS'
paths=[ROOT/'room0_repair_batch_20261007/index.html',WORK/'outputs/room0_human_repair_20261007/build_review.py']
entry='<a href="../room0_tracking_playback_20261008/index.html">按帧观看全部实例追踪（可循环）</a> · '
marker='<a href="human_selection.json">'
changes=[]
for path in paths:
    before=path.read_bytes();text=before.decode('utf-8')
    if entry not in text:
        assert text.count(marker)==1,path
        path.write_bytes(before.replace(marker.encode('utf-8'),(entry+marker).encode('utf-8'),1))
    after=path.read_bytes()
    changes.append({'file':str(path),'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),
                     'change':'one playback link; original result content preserved'})
(HERE/'entry_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','added_playback_entry':2,'tracking_repair_and_v3_modified':False}))
