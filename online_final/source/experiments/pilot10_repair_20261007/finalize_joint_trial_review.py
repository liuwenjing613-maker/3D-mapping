"""Record observed browser checks and prepare a hash-verified server delivery."""
from pathlib import Path
import hashlib,json,shutil
from PIL import Image

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
OUT=ROOT/'results/固定案例_三模型对比_20261006/joint_mask_replacement_20261007'
SERVER='/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/joint_mask_replacement_20261007/room2_mask14_mask24'
CODE='/home/chenkejun/CVPR/experiments/pilot10_repair_20261007'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    screenshot=OUT/'joint_browser_screenshot.jpg'
    with Image.open(screenshot) as im:
        im.verify()
    proof=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    proof['browser_checks']={
        'status':'PASS','local_PLY_loaded_points':159994,'full_PLY_loaded_points':1229303,
        'mask24_ID355_counts_by_condition':{'baseline':0,'pixel_control':2981,'whole_mask':3082},
        'three_states_have_correct_display_names':True,
        'ID355_explicit_checkbox_hiding_observed':True,'both_IDs_restored':True,
        'instance_colors_rendered':True,'console_errors_after_final_reload':[],
        'proof_screenshot':screenshot.name,'proof_screenshot_sha256':sha(screenshot)}
    write(OUT/'verification.json',proof)
    shutil.copy2(OUT/'geometry_diagnostic.json',HERE/'server_reports/joint_geometry_diagnostic.json')
    assets=[p for p in OUT.iterdir() if p.is_file()]
    assets += sorted((OUT/'vendor').glob('*')) + [OUT/'ply_models/manifest.json']
    assets=[p for p in assets if p.is_file()]
    pairs=[[str(p),SERVER+'/review/'+p.relative_to(OUT).as_posix()] for p in assets]
    sources=['build_joint_trial_review.py','finalize_joint_trial_review.py','diagnose_joint_geometry.sh']
    pairs += [[str(HERE/name),CODE+'/'+name] for name in sources]
    files={SERVER+'/review/'+p.relative_to(OUT).as_posix():sha(p) for p in assets}
    native=json.loads((OUT/'ply_models/manifest.json').read_text(encoding='utf-8'))['cases'][0]
    for scope in ['local','full']:
        files[SERVER+'/review/'+native[scope]['path']]=native[scope]['sha256']
    write(HERE/'joint_final_upload.json',pairs)
    write(HERE/'joint_delivery_hashes.json',files)
    pairs.append([str(HERE/'joint_delivery_hashes.json'),SERVER+'/review/delivery_hashes.json'])
    write(HERE/'joint_final_upload.json',pairs)
    prep=f"""set -eu
/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python - <<'PY'
from pathlib import Path
root=Path('{SERVER}/review')
assert root.is_relative_to(Path('/data/chenkejun/CVPR'))
root.mkdir(parents=True,exist_ok=True)
(root/'vendor').mkdir(exist_ok=True)
(root/'ply_models').mkdir(exist_ok=True)
print('Review destination ready')
PY
"""
    (HERE/'prepare_joint_review_publish.sh').write_text(prep,encoding='utf-8')
    verify=f"""set -eu
/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('{SERVER}')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=json.loads((root/'review/delivery_hashes.json').read_text())
for path,digest in files.items():assert sha(path)==digest,path
repair=json.loads((root/'repair_summary.json').read_text())
for path,digest in repair['base_source_hashes'].items():assert sha(path)==digest,path
freeze=json.loads((root/'evaluation_freeze.json').read_text())
for arm,record in freeze['predictions'].items():assert sha(root/arm/'final/instance_surface.npz')==record['sha256'],arm
proof=json.loads((root/'review/verification.json').read_text())
assert proof['browser_checks']['status']=='PASS' and proof['strict_success'] is False
print(json.dumps({{'status':'PASS','delivery_files_verified':len(files),'baseline_and_trial_maps_unchanged':True,'review':str(root/'review/index.html'),'strict_success':False}}))
PY
"""
    (HERE/'verify_joint_review_publish.sh').write_text(verify,encoding='utf-8')
    print(json.dumps({'status':'PASS','upload_files':len(pairs),'hash_files':len(files),'browser_QA':proof['browser_checks']['status']},ensure_ascii=False))

if __name__=='__main__':main()
