"""Render a concise report from completed, independently checked run records."""
from pathlib import Path
import json

OUT = Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
HERE = Path(__file__).resolve().parent

def read(p):
    return json.loads(Path(p).read_text())

def pct(x):
    return 'N/A' if x is None else f'{100*x:.2f}%'

def main():
    audit = read(OUT / 'verification/causal_audit.json')
    assert audit['status'] == 'PASS'
    for group in ['A', 'B']:
        assert read(OUT / f'status_{group}.json')['status'] == 'COMPLETE'
        assert read(OUT / 'evaluation_r2' / group / 'evaluation_complete.json')['status'] == 'PASS'
    rows = audit['checkpoints']
    lines = ['# room0 全场景 A B 运行结果', '',
        'A、B均完成400个建图帧及第400帧修复提交。原始序列为0～1999，建图步长5，最后建图输入为原始帧1995。A从全部已到达历史帧寻找候选，B从上次检查点后的新增帧寻找候选；两条均使用完整历史严格投票。', '',
        '此次沿用已核验的真实检查点续建：A接续已修复200帧状态；B接续已有优化版200帧地图，独立完成B200修复后再续建。前50、100帧的已提交修复全部保留。没有新增点排序，也没有修改Gate、SAM2或partial＋E1的数值规则。', '',
        '| 分支 | 检查点 | 种子数 | SAM2新推理任务 | 严格缓存复用任务 | 提交核验 |',
        '|---|---:|---:|---:|---:|---|']
    for r in rows:
        t = r['tracking']
        lines.append(f"| {r['branch']} | {r['checkpoint']} | {r['retained_seeds']} | {t['inferred_cases']} | {t['reused_cases']} | 通过 |")
    lines += ['', '各段续建只在目标检查点重建一次完整表面。A、B的第201帧及第301帧均核验了自己的新提交检查点哈希，并在关联前恢复该状态；历史来源像素、持久ID、逐帧严格票、增量回滚与检查点保存恢复核验通过。算法输入访问审计未发现非探针越界读取。', '',
        '| 分支与检查点 | 主建图 | 完整建图进程 | 候选生产 | 追踪与Gate | partial＋E1 | 提交 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m = r['mapping_timing']
        lines.append(f"| {r['branch']} {r['checkpoint']} | {m['main_mapping_seconds']:.1f} | {m['total_seconds']:.1f} | {r['selection_production_seconds']:.1f} | {r['tracking']['seconds']:.1f} | {r['repair_summary']['seconds']:.1f} | {r['commit_summary']['seconds']:.1f} |")
    lines += ['', '时间单位为秒。主建图包含逐帧读取、关联、TSDF及检查点历史投票；完整建图另含恢复与导出。候选生产排除复核图、QA和GPU等待，追踪编排包含等待。B200建图时间来自此前已核验运行，本次复用其地图。A、B并行且缓存命中情况不同，这些是阶段执行记录，不能相加当作从零开始400帧的冷启动耗时。', '',
        '前次同一100帧已修复起点的对照中，101～200主建图由3235.885秒降至98.909秒，提速32.72倍；完整建图由3290.492秒降至158.050秒，提速20.82倍。该结论仅适用于相应建图计时范围。', '',
        '最终质量评分从main工作区运行默认object_observed_repair revision 2：全协议350个GT，其中room0为72个；frozen=false。下表的“修复前”是到达400帧、已包含此前检查点修复的地图，“修复后”是再完成第400帧修复的地图。每条分支的前后表面坐标逐元素相同，使用同一对应索引评分；未对在线前缀套用400帧GT。', '',
        '| 分支 | 第400帧阶段 | AP50 | F1@0.5 | mCov | 正确归属覆盖率 |',
        '|---|---|---:|---:|---:|---:|']
    for group in ['A', 'B']:
        for stage, label in [('before', '修复前'), ('after', '修复后')]:
            s = read(OUT / 'evaluation_r2' / group / stage / 'metrics.json')['summary']
            assert s['profile_revision'] == 2 and s['scene_count'] == 1
            lines.append(f"| {group} | {label} | {pct(s['CA_AP50_uniform'])} | {pct(s['CA_PRF1_0_5']['F1'])} | {pct(s['CA_mCov'])} | {pct(s['owner_surface']['Correct_owner_Coverage'])} |")
    lines += ['', '| 分支 | 第400帧修复后实例数 | 表面点数 | GT最佳IoU改善 | 下降 | 不变 |',
              '|---|---:|---:|---:|---:|---:|']
    for group in ['A', 'B']:
        r = next(r for r in rows if r['branch'] == group and r['checkpoint'] == 400)
        pair = read(OUT / 'evaluation_r2' / group / 'pair/paired_repair_metrics.json')
        assert sum(pair[k] for k in ['corrected_instances', 'degraded_instances', 'unchanged_instances']) == 72
        s = r['final_surface']
        lines.append(f"| {group} | {s['positive_instances']} | {s['surface_points']:,} | {pair['corrected_instances']} | {pair['degraded_instances']} | {pair['unchanged_instances']} |")
    lines += ['', 'A/B是两种候选搜索策略，不能把两者差异当作优化前后差异。此前100→200真实闭环对照仍有一次Gate阈值翻转，严格逐元素等价未通过；本次保留真实优化结果继续运行，没有手工统一候选或用固定表面对照替换真实输出。', '',
        '完整服务器结果：`/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023`。各分支最终地图以`400/A/repair/commit_complete.json`和`400/B/repair/commit_complete.json`记录的final_map为准。', '',
        '核验记录见[连续状态核验](verification/causal_audit.json)、[证据文件哈希](evidence_manifest.json)；最终评分见[A评分](evaluation_r2/A/after/metrics.json)和[B评分](evaluation_r2/B/after/metrics.json)。实例着色PLY仅用于查看，保留全部坐标和持久ID。', '']
    text = '\n'.join(lines)
    review = OUT / 'review'
    review.mkdir(exist_ok=True)
    (review / 'REPORT.md').write_text(text)
    (HERE / 'REPORT.md').write_text(text)
    print(str(review / 'REPORT.md'))

if __name__ == '__main__':
    main()
