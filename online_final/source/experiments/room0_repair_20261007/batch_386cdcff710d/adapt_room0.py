"""Create the room0 wrapper while preserving the validated joint vote algorithm."""
from pathlib import Path
import hashlib,json

WORK=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE=WORK/'outputs/room2_human_repair_20261007'
annotation=WORK/'room0_修复候选选择_20261007.json'
digest=hashlib.sha256(annotation.read_bytes()).hexdigest()
batch='batch_'+digest[:12]
ROOT='/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/'+batch
CODE='/home/chenkejun/CVPR/experiments/room0_repair_20261007/'+batch
oldroot='/data/chenkejun/CVPR/revisable_instance_map/room2_repair_20261007/batch_8e890bff216d'


def replace_once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new,1)


def load(name):
    return (SOURCE/name).read_text(encoding='utf-8').replace(oldroot,ROOT).replace('room2','room0')


def save(name,text):
    with (HERE/name).open('w',encoding='utf-8',newline='\n') as stream:stream.write(text)


HERE.mkdir(parents=True,exist_ok=True)
local=load('prepare_local.py')
local=local.replace("== 5", "== 14")
local=replace_once(local,"selection = json.loads(SOURCE.read_text(encoding='utf8'))", "selection = json.loads(SOURCE.read_text(encoding='utf-8-sig'))")
local=replace_once(local,"assert set(selection['pilot_case_uids']) == {x['case_uid'] for x in selection['selections']}",
 "assert set(selection['pilot_case_uids']) == {x['case_uid'] for x in selection['selections']} == {c['uid'] for c in view['cases']}\n    assert len({s['case_uid'] for s in selection['selections']}) == 14\n    case_map = {c['uid']:c for c in view['cases']}")
local=replace_once(local,"assert choice['scene'] == 'room0' and not choice.get('is_qa', False)",
 "assert choice['scene'] == 'room0' and not choice.get('is_qa', False)\n        case = case_map[choice['case_uid']]\n        assert case['ROI'] == choice['ROI']\n        selected_view = [v for v in case['views'] if v['frame']==choice['frame'] and v['number']==choice['view_number']]\n        assert len(selected_view)==1 and choice['crop_key'] in selected_view[0]['variants']\n        assert selected_view[0]['source_RGB']==choice['source_rgb']")
save('prepare_local.py',local)
server=load('prepare_server.py')
server=replace_once(server,"gpus=['GPU-d31841e8-d3ae-e7f2-f581-8dc866e5c129','GPU-b003c3ac-453c-dd7b-90b0-31ad9adb3163','GPU-9a11b2b0-1816-1e33-9dd1-395cf0aaf9dd','GPU-d3f2e8e9-db27-643e-ad00-330e778f72c5']",
 "gpus=['GPU-d31841e8-d3ae-e7f2-f581-8dc866e5c129','GPU-b003c3ac-453c-dd7b-90b0-31ad9adb3163','GPU-9a11b2b0-1816-1e33-9dd1-395cf0aaf9dd']")
server=server.replace("gpus[len(diagnostics)]","gpus[len(diagnostics)%len(gpus)]")
server=server.replace("'repair_cases':4,'selected_masks':13", "'repair_cases':len(diagnostics),'selected_masks':len(supports)")
server=server.replace("canvas.save(ROOT/'seed_contact.jpg',quality=92)",
 "canvas.save(ROOT/'seed_contact.jpg',quality=92)\n    for start in range(0,len(contact),6):\n        sheet=Image.new('RGB',(1800,680),(245,245,245))\n        for k,im in enumerate(contact[start:start+6]):sheet.paste(im,(600*(k%3),340*(k//3)))\n        sheet.save(ROOT/('seed_contact_%02d.jpg'%(start//6+1)),quality=95)")
save('prepare_server.py',server)
save('cross_seed_diagnostics.py',load('cross_seed_diagnostics.py'))
for name in ['joint_batch_ops.py','test_joint_batch_ops.py']:
    save(name,(SOURCE/name).read_text(encoding='utf-8'))
run=load('run_batch_v2.py')
run=run.replace("REPAIR_ROOT=ROOT/'validated_v2'","REPAIR_ROOT=ROOT/'validated'")
run=replace_once(run,"canonical=j.canonical_map([obj['track_id'] for case in cases for obj in case['objects']],{9:5})",
 "review=json.loads((ROOT/'alias_review.json').read_text())\n    assert review['status']=='FROZEN' and not review['GT_used']\n    aliases={int(a):int(b) for a,b in review['aliases'].items()}\n    canonical=j.canonical_map([obj['track_id'] for case in cases for obj in case['objects']],aliases)")
run=run.replace("scene.maximum_id+14","scene.maximum_id+max(objects)+1")
a=run.index("            'aliases':{'9':5},'alias_evidence':")
b=run.index("'threshold_dice':.5,",a)
run=run[:a]+"            'aliases':json.loads((ROOT/'alias_review.json').read_text())['aliases'],\n            'alias_evidence':json.loads((ROOT/'alias_review.json').read_text()),"+run[b:]
run=run.replace("'label_files_checked':8000","'label_files_checked':2000*len(cases)")
run=run.replace("ROOT/'input_freeze.json',ROOT/'global_association.json'", "ROOT/'input_freeze.json',ROOT/'alias_review.json',ROOT/'global_association.json'")
run=replace_once(run,"'implementation':'run_batch_v2.py','prior_failed_attempt_preserved':'seed_only','repair_cases':4,'no_reliable_seed_cases':1,'selected_masks':13",
 "'implementation':'run_batch.py','repair_cases':len(cases),'no_reliable_seed_cases':14-len(cases),'selected_masks':len(objects)")
save('run_batch.py',run)
save('task_paths.json',json.dumps({'scene':'room0','annotation_sha256':digest,'batch':batch,'ROOT':ROOT,'CODE':CODE},indent=2)+'\n')
data=json.loads((WORK/'results/固定案例_三模型对比_20261006/room0_selection_20261007/viewer_data.json').read_text(encoding='utf-8'))
save('selection_views.json',json.dumps({'scene':'room0','cases':data['cases']},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'ROOT':ROOT,'CODE':CODE,'annotation_sha256':digest}))
