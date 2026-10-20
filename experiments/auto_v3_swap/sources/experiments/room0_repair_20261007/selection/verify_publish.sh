set -eu
python3 - <<'PY'
from pathlib import Path
import hashlib,json
result=Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/selection')
code=Path('/home/chenkejun/CVPR/experiments/room0_repair_20261007/selection')
review=Path('/data/chenkejun/CVPR/revisable_instance_map/roi_threeway_fixed_20261006/selection-20261006-f49eb927/review')
expected=json.loads((code/'publish_hashes.json').read_text())
for relative,digest in expected.items():
    path=code/relative[5:] if relative.startswith('code:') else result/relative
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,relative
before=json.loads((code/'source_before_publish.json').read_text())
assert all((hashlib.sha256((review/name).read_bytes()).hexdigest() if (review/name).is_file() else None)==digest
           for name,digest in before.items())
asset_hashes=json.loads((code/'asset_sha256.json').read_text())
for name,digest in asset_hashes.items():
    assert hashlib.sha256((review/name).read_bytes()).hexdigest()==digest,name
assert (review/'room0_selection_20261007').resolve()==result.resolve()
setup=json.loads((result/'selection_setup.json').read_text())
data=json.loads((result/'viewer_data.json').read_text())
assert len(data['cases'])==14 and sum(len(c['views']) for c in data['cases'])==42
assert {c['scene'] for c in data['cases']}=={'room0'}
assert not setup['tracking_started'] and not setup['map_modified'] and setup['human_records_created']==0
for crop in data['crops'].values():
    assert len(crop['models'])==3
    for asset in [crop['RGB']]+[m[k] for m in crop['models'] for k in ('color','boundary','ids_file')]:
        assert asset.startswith('../assets/') and (review/asset[3:]).is_file(),asset
receipt={'status':'PASS','scene':'room0','cases':14,'views':42,'candidate_crops':len(data['crops']),
         'verified_files':len(expected),'verified_assets':len(asset_hashes),
         'assets_reused_without_copy':True,'original_source_unchanged':True,
         'code_directory':str(code),'selection_directory':str(result),
         'tracking_started':False,'map_modified':False,'human_records_created':0}
(result/'server_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY
