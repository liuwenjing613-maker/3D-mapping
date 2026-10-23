"""Summarize completed evidence without changing any experiment outputs."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / 'comparison_page/repair_window_only'
s = json.loads((PAGE / 'summary.json').read_text(encoding='utf-8'))
lines = [
    'room0 A/B 第400帧：仅窗户保留 SAM 特殊来源规则',
    '完成日期：2026-10-23；完整追踪、Gate、partial＋E1、修复提交及独立审计均通过。',
    '每条分支固定其原有修复前400帧状态与300帧已提交历史。只取消柜子 SAM 直通／撤下 CropFormer 的特殊来源规则；P/R 豁免和其余数值规则不变。不是从第1帧按新规则重跑整个场景。',
    '评价：服务器 main 当前 object_observed_repair / revision 2，room0 72 GT、全协议350 GT、frozen=false。三份表面逐点同几何，沿用各分支原 fixed_correspondence；原方案指标重新计算并精确复现。',
    '',
    '分支/阶段 | AP50 | AP | AP25 | F1 | PQ | mCov | 正确归属覆盖 | TP/FP/FN（百分制）',
]
for branch, d in s['branches'].items():
    e, t = d['evaluation'], d['tracking']
    for stage, title in [('before', '共同修复前'), ('original', '原方案修复后'), ('new', '新方案修复后')]:
        m = e['metrics'][stage]
        values = [m[k] for k in ['CA_AP50_uniform', 'CA_AP_uniform', 'CA_AP25_uniform']]
        values += [m['CA_PRF1_0_5']['F1'], m['CA_PQ']['PQ'], m['diagnostics']['mean_best_gt_iou'], m['diagnostics']['Correct_owner_Coverage']]
        counts = '/'.join(str(m['CA_PRF1_0_5'][k]) for k in ['TP', 'FP', 'FN'])
        lines.append(f'{branch}/{title} | ' + ' | '.join(f'{v*100:.2f}' for v in values) + ' | ' + counts)
    cab = next(x for x in e['per_gt'] if x['gt_id'] == 2)
    lines += ['', f'{branch} 柜子 GT2：IoU 修复前 {cab["before"]["best_iou"]*100:.3f}% → 原方案 {cab["original"]["best_iou"]*100:.3f}% → 新方案 {cab["new"]["best_iou"]*100:.3f}%。新方案主ID为 {cab["new"]["pid"]}。',
              f'新方案相对原方案：AP50 {(e["metrics"]["new"]["CA_AP50_uniform"]-e["metrics"]["original"]["CA_AP50_uniform"])*100:+.3f} 个百分点；相对共同修复前 {(e["metrics"]["new"]["CA_AP50_uniform"]-e["metrics"]["before"]["CA_AP50_uniform"])*100:+.3f} 个百分点。',
              f'逐GT最佳IoU变化（含极小边界变化）：{e["GT_changes"]}。']
    for row in e['per_gt']:
        if row['change'] == 'degraded':
            lines.append(f'  小幅退化 GT{row["gt_id"]} {row["semantic_class"]}: {row["original"]["best_iou"]*100:.4f}% → {row["new"]["best_iou"]*100:.4f}%（{row["delta_iou"]*100:+.4f} pp）')
    lost = [r['gt_id'] for r in e['per_gt'] if r['original']['best_iou'] > .5 and r['new']['best_iou'] <= .5]
    lines += [f'相对原方案，从最佳IoU>0.5跌至≤0.5的GT：{lost}。此项为逐GT诊断，TP/FP/FN以正式评估器匹配结果为准。',
              f'共同种子的Gate接受决定有变化：{sum(bool(r["Gate_decision_changed_frames"]) for r in t["rows"])} 个；没有人为统一决定。',
              f'执行 {d["repair_pipeline"]["seconds"]/60:.2f} 分钟，{t["cases"]}组SAM2中严格指纹复用{t["reused_cases"]}组，实际推理{t["inferred_cases"]}组。Gate、E1及提交均重算；时间不含前置候选筛选/R2导出，不能当作从头推理耗时。']
    for p in d['export']['pairs']:
        lines.append(f'PLY {p["key"]}: 每侧{p["points"]:,}点；真实标签差异{p["changed"]:,}点；按同来源种子对齐新生ID后{p["aligned_changed"]:,}点。')
    lines.append('')
lines += [
    '解释：A 原三个柜子局部种子不再进入修复，柜体保留为PID7；原先严重误拆基本消失。但柜体并非与修复前逐元素完全一致（GT正确交集11929→11927）。B主要减少柜子边界的小片改标。',
    '仍有限制：新规则并未修复所有已有问题；A有上述4个小幅IoU退化，B相对原方案其余71个GT的最佳IoU不变。两分支新AP50都超过本次修复前，但B新AP25仍低于本次修复前。',
    '下载PLY保留全部原始坐标与真实持久ID；页面颜色对齐只解释同来源种子的重新编号，不修改PLY。标签变化数不能当作质量提升量。',
    '本次为固定修复前400帧状态的候选规则对照，不是降低strict_surface调用频率的等价测试，不能据此消除此前Open3D重复点/Gate阈值翻转导致的2点差异限制。',
    '页面：http://127.0.0.1:8801/repair_window_only/index.html',
    '完整数据、来源和核验：summary.json；PLY哈希：manifest.json。',
]
(PAGE / 'REPORT.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(PAGE / 'REPORT.txt')
