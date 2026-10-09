"""Adapt the existing room0 chronological playback for the frozen room2 batch."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
DONOR = WORK / 'outputs/room0_tracking_playback_20261008'
DONOR_WEB = WORK / 'results/固定案例_三模型对比_20261006/room0_tracking_playback_20261008'
WEB = WORK / 'results/固定案例_三模型对比_20261006/room2_tracking_playback_20261008'
SOURCE = WORK / 'outputs/room2_top30_repair_20261008'


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


exporter = (DONOR / 'export_playback.py').read_text(encoding='utf-8-sig')
exporter = replace_once(exporter, 'room0_repair_20261007/batch_386cdcff710d', 'room2_top30_repair_20261008/batch_6b3f32f5dc1c')
exporter = exporter.replace('room0', 'room2')
exporter = exporter.replace('all 30 raw', 'all 25 raw')
exporter = replace_once(exporter, 'def main():', 'def main():')
exporter = replace_once(exporter, 'started=time.monotonic();PACK.mkdir(exist_ok=True)', 'started=time.monotonic();OUT.mkdir(exist_ok=True);PACK.mkdir(exist_ok=True)')
exporter = exporter.replace('range(1,31)', 'range(1,26)').replace('areas=[0]*30', 'areas=[0]*25')
exporter = exporter.replace("'all_tracking_masks':30,'unique_repair_identities':29", "'all_tracking_masks':25,'unique_repair_identities':23")
exporter = exporter.replace("'tracks':30,'verified_tracking_PNGs':24000", "'tracks':25,'verified_tracking_PNGs':28000")
exporter = replace_once(exporter, 'ThreadPoolExecutor(max_workers=4)', 'ThreadPoolExecutor(max_workers=8)')
# If all proposals are rejected after overlap trimming, run_batch.py does not
# store a conflict count in that frame row. Its used foreground is empty.
exporter = replace_once(exporter,
    "if decision:assert int(conflicts.sum())==decision.get('cross_case_conflict_pixels',0),(fid,'Conflict reconstruction mismatch')",
    "if decision and 'cross_case_conflict_pixels' in decision:assert int(conflicts.sum())==decision['cross_case_conflict_pixels'],(fid,'Conflict reconstruction mismatch')\n                    elif decision:assert not accepted or int(conflicts.sum())==0,(fid,'Unrecorded accepted conflict')")
exporter = replace_once(exporter,
    "'unreliable_cases':[{'ROI':r['source_choice']['ROI'],'status':r['status']} for r in seeds if r['status']!='READY'],",
    "'unreliable_cases':[{'ROI':r['source_choice']['ROI'],'status':r['status']} for r in seeds if r['status']!='READY'],\n          'unannotated_cases':json.loads((ROOT/'seed_manifest.json').read_text()).get('unannotated_cases',[]),\n          'verified_tracking_PNGs':2000*len(reports),")
assert not re.search(r'\b(30|31|29|24000)\b', exporter), 'Unadapted room0 counts'
compile(exporter, str(HERE / 'export_playback.py'), 'exec')
(HERE / 'export_playback.py').write_text(exporter, encoding='utf-8', newline='\n')

client = (DONOR_WEB / 'playback.js').read_text(encoding='utf-8-sig').replace('room0', 'room2')
client = client.replace('i<=30', 'i<=data.tracks.length').replace('value>>>30', 'value>>>data.tracks.length')
client = client.replace('oid<=30', 'oid<=data.tracks.length').replace('${selected.size}/30', '${selected.size}/${data.tracks.length}')
client = client.replace('data.tracks.length!==30', 'data.tracks.length!==25')
client = client.replace('24000张追踪PNG', '28000张追踪PNG')
client = replace_once(client,
    "el('blinds').onclick=()=>choose(data.tracks.filter(t=>t.ROI==='ROI-C0001').map(t=>t.track));",
    "el('blinds').onclick=()=>choose(data.tracks.filter(t=>['ROI-C0001','ROI-C0008'].includes(t.ROI)).map(t=>t.track));\n  el('floor').onclick=()=>choose(data.tracks.filter(t=>[7,8,13,14,15].includes(t.track)).map(t=>t.track));")
assert not re.search(r'\b(30|24000)\b', client)
document = (DONOR_WEB / 'index.html').read_text(encoding='utf-8-sig').replace('room0', 'room2')
document = document.replace('30个原始追踪mask · 29个修复身份', '25个原始追踪mask · 23个修复身份')
document = document.replace('../room2_repair_batch_20261007/index.html', '../room2_top30_repair_20261008/index.html')
document = document.replace('../room2_blinds_rootcause_20261008/index.html', '../room2_top30_repair_20261008/root_cause_20261008/index.html')
document = document.replace('百叶窗灰点根因', '本轮修复根因分析')
document = replace_once(document,
    '<button id="blinds">只看百叶窗</button>',
    '<button id="blinds">只看百叶窗</button><button id="floor">只看地面局部</button>')
manifest = json.loads((SOURCE / 'seed_manifest.json').read_text(encoding='utf-8-sig'))
unreliable = [row['source_choice']['ROI'] for row in manifest['seeds'] if row['status'] != 'READY']
assert len(unreliable) == 4 and manifest['repair_cases'] == 14 and manifest['selected_masks'] == 25
document = replace_once(document,
    'ROI-C0027、ROI-U0019 标为不可靠，没有独立追踪任务。其余12组的30条实际轨迹全部可看。',
    '、'.join(unreliable) + ' 标为不可靠，没有独立追踪任务。另有12处未标注。本轮14组的25条实际轨迹全部可看。')
document = document.replace('默认展示完整SAM结果', '本页对应 room2 前30选择的本轮修复。默认展示完整SAM结果')
WEB.mkdir(exist_ok=True)
(WEB / 'index.html').write_text(document, encoding='utf-8', newline='\n')
(WEB / 'playback.js').write_text(client, encoding='utf-8', newline='\n')
receipt = {'status': 'PREPARED', 'donor_exporter_sha256': sha(DONOR / 'export_playback.py'),
           'donor_HTML_sha256': sha(DONOR_WEB / 'index.html'), 'donor_JS_sha256': sha(DONOR_WEB / 'playback.js'),
           'source_batch': '/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c',
           'annotation_sha256': manifest['annotation_sha256'], 'tracking_masks': 25, 'identities': 23,
           'changes': ['room2 source batch and source counts', 'show C0001/C0008 blinds and five floor selections',
                       'room2 result and diagnosis links', 'record missing conflict count only when no accepted foreground',
                       '8 CPU readers for presentation export only'],
           'tracking_repair_and_v3_modified': False}
(HERE / 'adaptation_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
