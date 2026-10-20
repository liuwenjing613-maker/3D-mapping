set -eu
python3 - <<'PY'
from pathlib import Path
import hashlib,json
code=Path('/home/chenkejun/CVPR/experiments/room0_repair_20261007/selection')
result=Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/selection')
review=Path('/data/chenkejun/CVPR/revisable_instance_map/roi_threeway_fixed_20261006/selection-20261006-f49eb927/review')
assert (review/'viewer_data.json').is_file()
before={name:hashlib.sha256((review/name).read_bytes()).hexdigest() if (review/name).is_file() else None
        for name in ['viewer_data.json','projection_annotations.json','select.html','index.html']}
code.mkdir(parents=True,exist_ok=True)
result.mkdir(parents=True,exist_ok=True)
assert not any(result.iterdir()),'Existing room0 selection files require inspection before any overwrite'
link=review/'room0_selection_20261007'
if link.exists() or link.is_symlink():
    assert link.is_symlink() and link.resolve()==result.resolve(),str(link)
else:
    link.symlink_to(result,target_is_directory=True)
(code/'source_before_publish.json').write_text(json.dumps(before,indent=2)+'\n')
print('ROOM0_SELECTION_DESTINATIONS_READY')
PY
