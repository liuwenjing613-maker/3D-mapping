"""Measure local map effects without claiming counterfactual AP/PQ gains."""
from pathlib import Path
import numpy as np

DEFAULT_ROOT = Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main')

def enrich(payload, root=DEFAULT_ROOT):
    from unified_eval.io import load_gt, load_prediction
    root = Path(root)
    for scene in sorted({c['scene'] for c in payload['cases']}):
        gt = load_gt(root / scene / 'ground_truth/gt.npz')
        pred = load_prediction(root / scene / 'evaluation/adapter/canonical_prediction.npz')
        owner = np.full(len(gt.instance_id), -1, dtype=np.int32)
        uid_index = {}
        for i, inst in enumerate(pred.instances):
            indices = np.asarray(inst.vertex_indices)
            if np.any(owner[indices] != -1):
                raise ValueError('Impact counting requires partition predictions')
            owner[indices] = i
            uid_index[inst.instance_uid] = i
        for c in [c for c in payload['cases'] if c['scene'] == scene]:
            target = gt.instance_id == c['gt_id']
            idx = uid_index.get(c['pred_uid'], -2)
            selected = owner == idx
            hit = int(np.count_nonzero(target & selected))
            elsewhere = int(np.count_nonzero(target & (owner >= 0) & ~selected))
            unassigned = int(np.count_nonzero(target & (owner == -1)))
            others = selected & (gt.instance_id >= 0) & ~target
            background = selected & (gt.instance_id < 0)
            ids, counts = np.unique(owner[target & (owner >= 0) & ~selected], return_counts=True)
            order = np.argsort(-counts)
            competing = [{'pred_uid':pred.instances[int(ids[j])].instance_uid,'points':int(counts[j])} for j in order[:5]]
            gids, gcounts = np.unique(gt.instance_id[others], return_counts=True)
            order = np.argsort(-gcounts)
            contamination = [{'gt_id':int(gids[j]),'points':int(gcounts[j])} for j in order[:5]]
            impact = {
                'protocol':'Replica-CA-v1', 'domain':'shared reference vertices after 5cm adapter',
                'gt_points':int(target.sum()), 'pred_points':int(selected.sum()),
                'target_in_selected_pred':hit, 'target_in_other_preds':elsewhere,
                'target_without_pred_label':unassigned,
                'pred_from_other_valid_gt':int(others.sum()),
                'pred_from_noninstance_reference':int(background.sum()),
                'other_pred_count_on_target':len(ids), 'other_gt_count_in_pred':len(gids),
                'largest_other_preds':competing, 'largest_other_gt':contamination,
                'scope':'Exact local surface counts; causal propagation remains a hypothesis.',
            }
            assert hit + elsewhere + unassigned == int(target.sum())
            assert hit + impact['pred_from_other_valid_gt'] + impact['pred_from_noninstance_reference'] == int(selected.sum())
            union = int(target.sum()) + int(selected.sum()) - hit
            assert abs((hit/union if union else 0)-c['iou']) < 1e-8
            c['impact'] = impact
    payload['impact_version'] = 1
    return payload


JS = r'''
function impactPanel(x){const m=x.impact;if(!m)return '<p>此案例影响统计待补充。</p>';const t=x.trace||{},l=t.prediction_lineage||{};const part=(n,d)=>pct(d?n/d:0);let steps=[];
if(t.raw_coverage!==undefined){const delta=t.raw_coverage-t.final_coverage;steps.push(`前端 → 几何细化：目标可见采样的前景支持从 ${pct(t.raw_coverage)} 变为 ${pct(t.final_coverage)}，净变化 ${delta>=0?'减少':'增加'} ${(100*Math.abs(delta)).toFixed(1)} 个百分点。这是对任意前景的采样统计，不能直接认定为误删或边界改善。`)}
if(l.total_observations!==undefined)steps.push(`观测 → 实例关联：该 Pred 累计 ${l.total_observations} 条观测，抽查 ${l.sampled_observations} 条。${l.raw_mixed_observations} 条抽样原始 mask 符合混合阈值，${l.final_mixed_observations} 条细化后仍符合；这些次数不能当成全部观测的错误率。`);
if(t.persistent_ids?.length)steps.push(`实例关联 → 表面投票：目标可见投影的支持出现于 ${t.persistent_ids.length}${t.persistent_ids.length>=8?' 个已列出的主要':' 个'} Pred。它提示身份分散的可能性，但只列出了主要 ID，且不同帧的采样会重复计数；上面的最终表面统计才是不重复的点数。`);
if(t.surface)steps.push(`共享几何 → 实例输出：${pct(t.surface.geometry_coverage)} 的 GT 参考点在 3 cm 内有 TSDF 对应。该值描述几何接近程度，与上方 5 cm 评估标签覆盖的含义不同。`);
const propagation={cropformer_mask_merge:'若单帧确实把多个真实对象放在同一 mask，后端可能把它们作为同一个观测关联并投给同一个 ID，最后表现为粉色多余表面。要确认这一链条，需要追到首次混合帧并做替换观测的重放。',association_merge:'若原本分属不同对象的干净观测被关联到同一 ID，后续表面投票可能继承该身份，形成合并。需要核查具体关联决策及当时的候选分数；当前抽样不能证明首次错误位置。',association_fragmentation:'若同一对象在跨帧关联时不断产生新 ID，最终目标表面可能分散在多个实例里，导致每个实例都覆盖不全。需要比较这些 ID 的出生帧与对应观测，排除真实的前端分割差异。',depth_refinement_erasure:'若细化删掉了目标的有效像素，后续关联和表面投票可用的支持会减少；最终可能表现为无标签或覆盖不足。但净前景减少本身还不能证明删错，需要检查被删像素的深度和对象边界。',geometry_gap:'若输入可见表面未进入共享 TSDF，仅修改实例身份无法恢复这部分几何。当前距离统计也可能受未观测面、几何偏移或位姿误差影响，需要先区分。',surface_evidence_gap:'若已存在的表面收到的实例证据稀少或冲突，发布策略可能留下无标签区域。需要检查点级证据和发布前后的状态；参考点无标签不等于相机没见过。'};
const otherPred=m.largest_other_preds.map(z=>`Pred ${esc(z.pred_uid)}：${z.points} 点`).join('；')||'没有其他 Pred 覆盖目标';
const otherGT=m.largest_other_gt.map(z=>`GT ${z.gt_id}：${z.points} 点`).join('；')||'没有其他有效 GT 混入此 Pred';
let visual='';if(m.target_in_other_preds)visual+='目标的一部分已被分配到其他 ID，可能表现为同一物体多色或碎片。';if(m.target_without_pred_label)visual+='目标还存在未获任何预测标签的参考点，可能表现为缺口或灰区；需结合原始 TSDF 判断是否真的缺几何。';if(m.pred_from_other_valid_gt)visual+='选中 Pred 包含其他 GT 的表面，可能表现为多个物体共用一个身份。';if(m.pred_from_noninstance_reference)visual+='它还覆盖了非有效实例参考表面，应检查墙地面等背景和评估排除区域。';
return `<h3>③ 对最终地图质量的影响 · 可直接计数</h3><p>以下统计在 <b>v1 的公共参考表面</b>上完成，单位为顶点数（不是平方米，也不是 RGB 像素）。它衡量最终标签解释，不直接等于原生 PLY 的灰色面积。</p><div class="impact-grid"><div class="fact"><b>目标 GT ${x.gt_id} 去了哪里</b><ul><li>选中 Pred 覆盖：${m.target_in_selected_pred} 点 · ${part(m.target_in_selected_pred,m.gt_points)}</li><li>被其他 Pred 覆盖：${m.target_in_other_preds} 点 · ${part(m.target_in_other_preds,m.gt_points)}</li><li>任何 Pred 都未覆盖：${m.target_without_pred_label} 点 · ${part(m.target_without_pred_label,m.gt_points)}</li></ul><p class="muted">其他身份共 ${m.other_pred_count_on_target} 个；主要是 ${otherPred}。</p></div><div class="fact"><b>选中 Pred ${esc(x.pred_uid||'无')} 多带入了什么</b><ul><li>来自目标 GT：${m.target_in_selected_pred} 点 · ${part(m.target_in_selected_pred,m.pred_points)}</li><li>来自其他有效 GT：${m.pred_from_other_valid_gt} 点 · ${part(m.pred_from_other_valid_gt,m.pred_points)}</li><li>来自非实例参考区域：${m.pred_from_noninstance_reference} 点 · ${part(m.pred_from_noninstance_reference,m.pred_points)}</li></ul><p class="muted">其他 GT 共 ${m.other_gt_count_in_pred} 个；主要是 ${otherGT}。</p></div></div><p><b>可视地图中的可能表现：</b>${visual||'该对象对未显示可计数的多余或遗漏表面。'}</p><p><b>对评价的直接影响：</b>${x.error_type==='FN'?'此 GT 在 v1 的 IoU>0.5 一对一匹配下未成功匹配，计为一个 FN。':'此预测在 v1 的 IoU>0.5 一对一匹配下未成功匹配，计为一个 FP。'}上面的 ID 数包含任意点级重叠；少量边界点不等于整个对象被错误合并。同一对象对可能另有 FN/FP 卡片，不能把卡片数当成独立错误数。这些局部统计不能直接换算为全局 AP/PQ 损失或修复收益。</p><h3>④ 中间过程的影响与传播 · 已测事实 + 待验证链条</h3><ul class="evidence">${steps.map(s=>'<li>'+esc(s)+'</li>').join('')||'<li>缺少该对象的完整阶段追溯；目前只能确认最终表面影响。</li>'}</ul><div class="hypothesis"><b>可能的传播过程（未做因果验证）</b><p>${esc(propagation[x.root_cause]||'现有证据不足以把最终误差归到单一阶段。应依次核对目标观测、跨帧 ID、表面证据与发布结果，找出误差首次出现的位置。')}</p><p>下一步验证应固定输入，针对疑似阶段做局部替换或重放，再观察这些点的归属与统一指标是否改变；当前没有运行此干预实验。</p></div>`}
'''


def add_impact_ui(page):
    page = page.replace('function facts(x)', JS + '\nfunction facts(x)', 1)
    page = page.replace('<h3>③ 待验证的阶段线索</h3>', '${impactPanel(x)}<h3>⑤ 待验证的阶段线索</h3>', 1)
    page = page.replace('</style>', '.impact-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.impact-grid ul{padding-left:20px}.impact-grid li{margin:5px 0}@media(max-width:1050px){.impact-grid{grid-template-columns:1fr}}</style>', 1)
    return page


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import review_error_dashboard as review
    from analyze_replica8_errors import read_json, write_json
    out=DEFAULT_ROOT/'error_dashboard'
    payload=enrich(read_json(out/'error_cases.json'))
    write_json(out/'error_cases.json',payload)
    (out/'index.html').write_text(review.dashboard_html(payload),encoding='utf-8')
    print('Impact analysis added:',len(payload['cases']),'cases')
