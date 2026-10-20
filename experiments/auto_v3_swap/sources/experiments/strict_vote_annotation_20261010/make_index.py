from pathlib import Path
import json
H=Path(__file__).resolve().parent; W=H.parents[1]
web=W/'results/固定案例_三模型对比_20261006';out=web/'strict_vote_annotation_20261010';out.mkdir(exist_ok=True)
aud=json.loads((H/'catalog_audit.json').read_text())
cards=''
for s in ['room0','room2']:
 setup=json.loads((web/(s+'_strict_vote_annotation_20261010')/'selection_setup.json').read_text())
 assert setup['status']=='READY_FOR_HUMAN_CHOICES'
 cards+=f'<a class="card" href="../{s}_strict_vote_annotation_20261010/index.html"><h2>{s} <span>开始标注 →</span></h2><p>30 个 ROI · 90 个视角 · {setup["candidate_crop_count"]} 组裁剪</p></a>'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>strict-vote · 新一轮人工标注</title><style>body{font:16px/1.7 system-ui,sans-serif;margin:0;background:#f3f6fa;color:#24354c}main{max-width:880px;margin:60px auto;padding:24px}h1{line-height:1.3;font-size:32px}.sub{color:#64748b}.card{display:block;background:white;border:1px solid #dce4ef;border-radius:14px;padding:12px 28px;margin:20px 0;text-decoration:none;color:inherit}.card:hover{border-color:#3275c6}h2 span{float:right;font-size:15px;font-weight:500;color:#2865ad}.note{background:#e7eff9;border-radius:12px;padding:18px 24px}li{margin:8px 0}</style><main><p class="sub">2026-10-10 · 独立新标注</p><h1>P1-A1-strict-vote<br>按新 ROI 排名重新标注</h1><p>room0 与 room2 均使用<strong>尚未人工修复</strong>的严格一票地图，按 CONFLICT 点数从多到少排列。旧轮次选择不会自动载入。</p>'''+cards+'''<div class="note"><strong>每个区域的操作</strong><ol><li>比较三个视角与三组模型，选一个最合适的视角、裁剪和模型。</li><li>点选需要修复的完整目标 Mask，可选多个；没有可靠方案时选“都不可靠”。</li><li>若多个 ROI 对应同一个物体，在目标说明中写清 ROI 和 Mask 编号。</li><li>保存并下一个；每个场景完成后导出 JSON，放回本地 CVPR 文件夹。</li></ol></div><p class="sub">后续评估：object_observed_repair / revision 2。此页面只准备人工选择，未使用 GT 选择 ROI，未提交新修复。</p></main></html>'''
(out/'index.html').write_text(page,encoding='utf-8')
print(out/'index.html')
