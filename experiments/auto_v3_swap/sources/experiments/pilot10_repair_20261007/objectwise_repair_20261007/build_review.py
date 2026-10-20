"""Build a concise Chinese comparison with five exact PLY states."""
from pathlib import Path
from html import escape
import hashlib
import json
import shutil
import tarfile

HERE = Path(__file__).parent.parent
ROOT = HERE.parents[1]
DELIVERY = ROOT / 'results/固定案例_三模型对比_20261006'
OUT = DELIVERY / 'objectwise_repair_20261007'
CODE = HERE / 'objectwise_repair_20261007'
TITLES = {'baseline': '原基线', 'gated_strict': '旧：两个对象同时合格', 'ungated_strict': '旧：无质量门槛',
          'sam_objectwise': '新：SAM2 候选逐对象', 'original_objectwise': '新：原 mask 逐对象'}


def pct(value):
    return '%.2f%%' % (100 * value)


def main():
    record = json.loads((CODE / 'server_reports/review_bundle.json').read_text(encoding='utf-8'))
    bundle = HERE / 'objectwise_review_bundle.tar.gz'
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == record['sha256']
    with tarfile.open(bundle) as archive:
        for item in archive.getmembers():
            assert (DELIVERY / item.name).resolve().is_relative_to(OUT.resolve())
            assert not item.issym() and not item.islnk()
        archive.extractall(DELIVERY, filter='data')
    data = json.loads((OUT / 'comparison_data.json').read_text(encoding='utf-8'))
    shutil.copytree(DELIVERY / 'tracking_review_20261007/vendor', OUT / 'vendor', dirs_exist_ok=True)
    shutil.copy2(DELIVERY / 'tracking_review_20261007/instance_palettes.json', OUT / 'instance_palettes.json')
    shutil.copy2(CODE / 'comparison_viewer.js', OUT / 'comparison_viewer.js')
    shutil.copy2(CODE / 'server_reports/remaining_vote_audit.json', OUT / 'remaining_vote_audit.json')
    sam, original = data['reports']['sam'], data['reports']['original']
    metrics = data['previous_strict_summary']['metrics']
    metrics.update({'sam_objectwise': sam['evaluation']['objectwise_v3_metrics'],
                    'original_objectwise': original['evaluation']['objectwise_v3_metrics']})
    diag = sam['evaluation']['fixed_surface_diagnostics']
    diag['sam_objectwise'] = diag.pop('objectwise')
    diag['original_objectwise'] = original['evaluation']['fixed_surface_diagnostics']['objectwise']
    target_rows = []
    for name in TITLES:
        for target in diag[name]['targets']:
            target_rows.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
                TITLES[name], '椅子A / 4001' if target['GT_id'] == 4001 else '椅子B / 4002',
                pct(target['expected_identity_fixed_IoU']), pct(target['expected_identity_full_GT_coverage']),
                pct(target['expected_identity_reference_purity']),
                target['reachable_label_counts'].get('52', 0), target['unassigned_reachable_reference_points']))
    global_rows = []
    for name in TITLES:
        m = metrics[name]
        global_rows.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s / %s</td></tr>' % (
            TITLES[name], pct(m['CA_AP50_uniform']), pct(m['CA_PRF1_0_5']['F1']), pct(m['CA_PQ']['PQ']),
            m['structure']['merge_prediction_count'], m['structure']['split_gt_count']))
    target_v3 = {'baseline': [row['before'] for row in sam['evaluation']['target_v3_metrics']],
                 'sam_objectwise': [row['after'] for row in sam['evaluation']['target_v3_metrics']],
                 'original_objectwise': [row['after'] for row in original['evaluation']['target_v3_metrics']]}
    for name in ('gated_strict', 'ungated_strict'):
        target_v3[name] = data['previous_strict_summary']['non_target_v3_audit'][name]['target_v3_metrics']
    v3_target_rows = []
    for name in TITLES:
        a = next(row for row in target_v3[name] if row['GT_id'] == 4001)
        b = next(row for row in target_v3[name] if row['GT_id'] == 4002)
        v3_target_rows.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
            TITLES[name], pct(a['best_IoU']), pct(b['best_IoU']),
            pct(b['completeness_best_single_instance']), pct(b['purity_of_best_IoU_instance'])))
    counts_rows = []
    for title, value in [('SAM2 候选逐对象', sam), ('原 mask 逐对象', original)]:
        repair = value['complete']['repair']; counts = repair['counts']; commit = repair['strict_commit']
        counts_rows.append('<tr><td>%s</td><td>%s / %s</td><td>%s</td><td>%s / %s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
            title, counts['updated_mapping_frames'], counts['related_mapping_frames'], counts['single_object_frames'],
            counts['whole_observations'], counts['partial_observations'], commit['allowed_surface_points'],
            commit['final_inside_changes'], commit['final_outside_changes']))
    figures = []
    for view in data['views']:
        figures.append('<section><h2>固定视角 f%s · 相同几何和深度可见性</h2><div class="views">%s</div></section>' % (
            view['frame'], ''.join('<figure><figcaption>%s</figcaption><a href="%s" target="_blank"><img src="%s" alt="%s f%s"></a></figure>' %
                (TITLES[name], view['assets'][name], view['assets'][name], TITLES[name], view['frame'])
                for name in ('baseline', 'gated_strict', 'sam_objectwise', 'original_objectwise'))))
    timelines = []
    for source, title in [('sam', 'SAM2候选'), ('original', '原mask')]:
        for oid, label in [(4, '椅子 A / ID52'), (7, '椅子 B / ID355')]:
            cells = []
            for row in data['timeline'][source]:
                accepted = oid in row.get('accepted_track_ids', [])
                check = row.get('objects', {}).get(str(oid), {})
                tip = 'f%s %s' % (row['frame'], '使用可靠证据' if accepted else ', '.join(check.get('reasons', ['保留旧证据'])))
                cells.append('<span class="%s" title="%s"></span>' % ('accepted' if accepted else 'retained', escape(tip)))
            timelines.append('<p>%s · %s</p><div class="timeline">%s</div>' % (title, label, ''.join(cells)))
    audit = json.loads((OUT / 'remaining_vote_audit.json').read_text(encoding='utf-8'))
    audit_rows = []
    for source, title in [('room2_sam_seed', 'SAM2候选'), ('room2_original_seed', '原mask')]:
        row = audit['cases'][source]
        states = row['retired_support_raw_states_and_final_labels']
        audit_rows.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
            title, row['retired_support_unique_native_points'], sum(s['final_unassigned'] for s in states),
            next(s['final_unassigned'] for s in states if s['state'] == 3),
            row['target_B_ID52_votes_after'], row['target_B_ID355_votes_after']))
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐对象修复 · 实际结果</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5fa;color:#172b46;font:16px/1.65 system-ui,"Microsoft Yahei",sans-serif}main{max-width:1380px;margin:auto;padding:24px}h1{font-size:28px;margin:0 0 10px}h2{font-size:20px;margin:0 0 12px}section{background:#fff;padding:22px;border-radius:12px;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #dce4ef}th{background:#edf3fa}a{color:#166bab}.muted{font-size:14px;color:#60738c}.good{color:#17754c}.caution{color:#a14d14}.scroll{overflow-x:auto}.views{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;border-radius:8px}figcaption{font-size:14px;font-weight:600;margin-bottom:8px}.timeline{display:grid;grid-template-columns:repeat(100,1fr);gap:2px}.timeline span{height:11px}.accepted{background:#30ae76}.retained{background:#d8e0ec}button,select{padding:6px 8px;border:1px solid #aebcd1;border-radius:5px;background:white;cursor:pointer}.float{position:fixed;right:18px;bottom:18px;width:540px;height:560px;z-index:20;background:#f7fbff;box-shadow:0 8px 34px #10294355;border-radius:12px;overflow:hidden;display:flex;flex-direction:column;border:1px solid #afbed0}.float[hidden]{display:none}.float header{display:flex;align-items:center;justify-content:space-between;padding:9px 12px;background:#dceafa;cursor:move;touch-action:none}.float .tools{padding:8px 10px;font-size:12px;display:flex;gap:6px;flex-wrap:wrap}.float .tools select{max-width:230px}#stage{flex:1;min-height:120px;position:relative}canvas{width:100%;height:100%;display:block}#busy{position:absolute;left:15px;top:15px;color:white}#modelStatus{padding:6px 10px;font-size:12px;min-height:42px}.mini{height:46px}.mini .tools,.mini #stage,.mini #modelStatus{display:none}#open{position:fixed;right:20px;bottom:20px;z-index:30}@media(max-width:760px){main{padding:10px}.views{grid-template-columns:repeat(2,1fr)}.float{width:calc(100vw - 20px);right:10px;height:480px}.timeline{grid-template-columns:repeat(50,1fr)}}</style><main>
<section><h1>逐对象修复 · room2 双椅实际结果</h1><p><b>已按最新建议实现三项改动，SAM2 仍然向前、向后各自独立追踪完整 2000 帧。</b></p><ol><li><b>原 mask 只改关联：</b>原始 CropFormer mask8、7 的像素完全保留，ID52、52 改为 ID52、355。</li><li><b>逐对象接收证据：</b>A可靠就更新A；B不可靠就保留未解释的旧贡献。全部对象可靠、旧观察在范围内时，才整张撤回。</li><li><b>案例配置：</b>场景、种子、对象、旧家族与输出路径从 case.json 读取；GT身份只在单独评价配置中使用。</li></ol><p class="muted">数字门槛、确认规则（至少2票且比例≥0.67）、TSDF几何、原后处理设置及v3参数均保持。新增保护：另一个追踪对象的可见种子也受保护，避免单对象追踪侵入它。允许范围仍按撤回旧贡献与可靠新支持的并集计算，不做额外几何扩张；接收帧变化会使允许点集合自然变化。</p><p class="good">无操作对照：400帧重算后，票数、最终标签和v3完全等于基线。两组新版均通过完整重算、准确回滚和范围外零改动检查。</p><p class="caution">以下为当前 v3 调试协议（pending、未冻结），并非正式主表结论。更保守的换票能保留未知证据，也可能保留尚未解决的错误ID；请同时看覆盖、纯度和残留。</p></section>
<section><h2>实际更新了多少证据</h2><div class="scroll"><table><tr><th>新版</th><th>更新/相关建图帧</th><th>仅一个对象可靠的帧</th><th>整张/部分观察</th><th>允许范围点数</th><th>标签改动</th><th>范围外改动</th></tr>__COUNTS__</table></div><p class="muted">允许范围＝被撤回旧贡献的表面 ∪ 可靠新mask表面。保留的未知残余不会单独扩张范围。旧门控只用了107/210帧；无门槛用了210/210帧。</p></section>
<section><h2>两把椅子：统一v3实际效果</h2><div class="scroll"><table><tr><th>版本</th><th>A最佳实例IoU</th><th>B最佳实例IoU</th><th>B完整覆盖率</th><th>B纯度</th></tr>__V3TARGETS__</table></div><p class="muted">基线两个目标的最佳预测都是错误合并的ID52；修复后B的最佳预测为ID355。IoU没有超过0.5，因此尚未产生新的正确匹配实例。</p></section>
<section><h2>固定表面：身份、覆盖与灰点</h2><div class="scroll"><table><tr><th>版本</th><th>目标</th><th>预期身份IoU</th><th>完整GT覆盖率</th><th>纯度</th><th>ID52点数</th><th>未分配点数</th></tr>__TARGETS__</table></div><p class="muted">各版使用同一份完整TSDF→GT对应；灰点仍占据原来的几何位置。预期身份固定为A→ID52、B→ID355；基线没有ID355，所以B的预期身份IoU为0，并非上表最佳合并实例的IoU为0。ID52对于椅子B表示旧合并身份残留，对于椅子A表示正确身份。最后两列统计1cm内可对应的GT参考点；IoU与覆盖率分母包括完整GT。当前几何可对应A 3747/9124、B 3901/9503，约41%，它不等于实际几何缺失比例。</p></section>
<section><h2>统一v3：整个场景</h2><div class="scroll"><table><tr><th>版本</th><th>AP50</th><th>F1</th><th>PQ</th><th>合并预测/分裂GT数量</th></tr>__GLOBAL__</table></div><p class="muted">v3 与上面的固定表面辅助诊断分别列出，避免混用数值。GT没有参与种子关联、筛帧或换票。</p></section>
<section><h2>逐对象用帧情况</h2><p class="muted">从f0到f1995，每格一个建图帧，绿色＝该对象可靠证据已使用，灰色＝保留旧证据。悬停查看帧号和原因。</p>__TIMELINES__</section>
<section><h2>剩余问题：为什么还不完整</h2><div class="scroll"><table><tr><th>来源</th><th>撤回涉及原生表面点</th><th>其中最终灰点</th><th>其中原生CONFLICT灰点</th><th>B表面ID52票</th><th>B表面ID355票</th></tr>__AUDIT__</table></div><p>第二把椅子仍有约5万次ID52投票。相关历史帧中，51帧没有达到锚点覆盖要求；原因统计可重叠。两种种子最终几乎相同，说明仅更换种子来源并未解决这些历史证据。</p><p class="muted">表面点与投票次数是不同单位；投票按帧、原生TSDF点、实例去重。上述ID52票包含最终更新后的同身份贡献，不能都当成被保留的原始错误票。灰点主要来自相互冲突的证据；只有部分解释旧观察时保留未知，不保证整张替换或最终投票后灰点减少。<a href="remaining_vote_audit.json">逐帧剩余票和状态审计</a></p></section>
<section><h2>彩色悬浮3D</h2><p>右下角切换五个版本，镜头保持不动。蓝色 ID52、绿色 ID355，可单独隐藏；灰色为未分配点。支持旋转、缩放、拖动和完整场景查看。</p><p><a href="ply_models/objectwise_full.ply">完整五版本PLY</a> · <a href="ply_models/objectwise_local.ply">局部五版本PLY</a> · <a href="comparison_data.json">所有结果与校验</a></p></section>__FIGURES__
<section><h2>你下一步需要做什么</h2><p>选一个场景，人工挑10–15个案例：原mask正确但关联错误、新mask才分得开、状态异常但无需修复的对照。记录各对象的完整可见mask和种子帧；按这版固定配置运行，暂不增加自动ROI或VLM。</p><p class="muted">本轮只验证同一个双椅案例及无操作对照，尚未证明整场景稳定收益。保留原baseline及全部旧试验结果。</p></section></main>
<aside class="float" id="float"><header id="drag"><b>实际PLY · 五版本对比</b><span><button id="minimize">收起/展开</button> <button id="close">关闭</button></span></header><div class="tools"><select id="state"><option value="0">原基线</option><option value="1">旧：两个对象同时合格</option><option value="2">旧：无质量门槛</option><option value="3" selected>新：SAM2候选逐对象</option><option value="4">新：原mask逐对象</option></select><select id="scope"><option value="local">局部真实密度</option><option value="full">完整场景</option></select><select id="colorMode"><option value="1">实例颜色</option><option value="0">真实RGB</option><option value="2">改动位置</option></select><button id="reset">定位双椅</button><label><input id="blue" type="checkbox" checked>蓝ID52</label><label><input id="green" type="checkbox" checked>绿ID355</label><label><input id="unknown" type="checkbox" checked>灰点</label><label><input id="only" type="checkbox">只看两ID</label><label>点大小<input id="pointSize" type="range" min="1" max="5" value="2" step=".5"></label></div><div id="stage"><canvas id="canvas"></canvas><div id="busy">加载中…</div></div><div id="modelStatus"></div></aside><button id="open" hidden>打开3D</button><script type="application/json" id="modelData">__MODEL__</script><script type="module" src="comparison_viewer.js"></script></html>'''
    values = {'__COUNTS__': ''.join(counts_rows), '__TARGETS__': ''.join(target_rows), '__GLOBAL__': ''.join(global_rows),
              '__V3TARGETS__': ''.join(v3_target_rows),
              '__AUDIT__': ''.join(audit_rows),
              '__TIMELINES__': ''.join(timelines), '__FIGURES__': ''.join(figures),
              '__MODEL__': json.dumps(data['model'], ensure_ascii=False).replace('</', '<\\/')}
    for token, value in values.items():
        html = html.replace(token, value)
    (OUT / 'index.html').write_text(html, encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'page': str(OUT / 'index.html'), 'PLY_points': data['model']['local']['points']}))


if __name__ == '__main__':
    main()
