"""Report paired repair effects relative to the baseline of the SAME vote rule."""
from pathlib import Path
import csv,hashlib,json,zipfile
import numpy as np

ROOT=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009')
CODE=Path('/home/chenkejun/CVPR/experiments/p1a1_strict_repair_20261009')
NAMES={'original':'原语义','strict':'严格一票+弃权'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    complete=json.loads((ROOT/'evaluation_complete.json').read_text());assert complete['status']=='PASS'
    independent=json.loads((ROOT/'independent_correctness.json').read_text());assert independent['status']=='PASS'
    comparison=json.loads((ROOT/'comparison.json').read_text());assert comparison['all_baseline_metric_controls_exact']
    policy=json.loads((ROOT/'policy.json').read_text())
    mapping=json.loads((ROOT/'mapping_complete.json').read_text());assert mapping['full_replay_delta_rollback_and_scope_pass']
    tests=json.loads((ROOT/'tests_complete.json').read_text());assert tests['status']=='PASS'
    assert sha(ROOT/'comparison.json')==complete['comparison_sha256']
    effects={};rows=[]
    lines=['P1-A1 严格一票+弃权：冻结修复效果复验','',
        '目的：验证严格计票下修复是否仍有效，并区分计票规则改变与修复自身的收益。',
        '范围：room0最新12个修复案例；room2 Top30中实际执行的14个修复案例。',
        '条件：未修复、仅种子帧、完整追踪；原语义与严格语义全部采用同一当前v3开发协议重评。',
        'GT范围、对应关系、代码、配置完全一致；非正式冻结基准。预测全部在本轮读取GT前冻结。','',
        '1. 本轮决定与控制']
    lines.extend(f'{i+1}. {value}' for i,value in enumerate(policy['decisions']))
    lines.extend(['8. 严格修复的最终范围外标签，相对于严格规则未修复基线保持不变。',
                  '9. 每个条件保留其原语义对照的导出实例清单，严格规则激活的既有实例可追加；不会因弃权隐式删除原实例。','',
                  '2. 两场景合并：最终结果','语义 | 条件 | F1% | PQ% | mCov% | 正确参考顶点 | 错误参考顶点 | 未分配TSDF点'])
    totals={}
    for semantic in ('original','strict'):
        effects[semantic]={};base=comparison['pooled'][semantic+'_baseline_final']
        for cond in ('baseline','seed_only','full_track'):
            name=semantic+'_'+cond+'_final';v=comparison['pooled'][name]
            counts=np.sum([s[name]['counts_U_T_assigned_C'] for s in comparison['states'].values()],axis=0)
            totals[name]=counts
            lines.append(f"{NAMES[semantic]} | {cond} | {v['F1']*100:.6f} | {v['PQ']*100:.6f} | {v['mCov']*100:.6f} | {v['correct_owner_vertices']:,} | {v['wrong_owner_vertices']:,} | {int(counts[[0,1,3]].sum()):,}")
            if cond!='baseline':
                data={k:v[k]-base[k] for k in ('F1','PQ','mCov','macro_best_GT_recall','correct_owner_vertices','wrong_owner_vertices','unassigned_with_geometry_target_vertices','TP','FP','FN')}
                effects[semantic][cond]=data
                rows.append({'scope':'pooled_room0_room2','semantic':semantic,'condition':cond,**data})
    lines.extend(['','3. 修复增益必须相对于同规则未修复基线比较'])
    for semantic in ('original','strict'):
        for cond,d in effects[semantic].items():
            lines.append(f"{NAMES[semantic]} {cond}：ΔF1 {d['F1']*100:+.6f}个百分点，ΔPQ {d['PQ']*100:+.6f}个百分点，ΔmCov {d['mCov']*100:+.6f}个百分点；正确参考顶点 {d['correct_owner_vertices']:+,}，错误参考顶点 {d['wrong_owner_vertices']:+,}。")
    interaction={cond:{k:effects['strict'][cond][k]-effects['original'][cond][k] for k in effects['strict'][cond]} for cond in ('seed_only','full_track')}
    for cond,d in interaction.items():
        lines.append(f"{cond} 采用严格规则后的额外修复增益（差中差）：PQ {d['PQ']*100:+.6f}个百分点，F1 {d['F1']*100:+.6f}个百分点；正确参考顶点增益变化 {d['correct_owner_vertices']:+,}。")
    lines.extend(['','4. 最终三状态数量与逐点进出（TSDF点）','场景 | 语义 | 条件 | 灰U | 黄T | 红CONFLICT | 三状态总数'])
    scenes={}
    for scene,states in comparison['states'].items():
        run=json.loads((ROOT/scene/'complete.json').read_text());scenes[scene]={}
        for semantic in ('original','strict'):
            for cond in ('baseline','seed_only','full_track'):
                c=states[semantic+'_'+cond+'_final']['counts_U_T_assigned_C']
                lines.append(f"{scene} | {NAMES[semantic]} | {cond} | {c[0]:,} | {c[1]:,} | {c[3]:,} | {sum(c[i] for i in (0,1,3)):,}")
        for cond,data in run['conditions'].items():
            assert data['strict_commit']['final_outside_changes']==0
            old=data['old_vote_rule_controls'];new=data['strict_vote_rule']
            scenes[scene][cond]={'original':old,'strict':new}
            lines.append(f"{scene} {cond}：原语义新增未分配 {old['assigned_to_unassigned']:,}、恢复分配 {old['unassigned_to_assigned']:,}；严格规则新增未分配 {new['assigned_to_unassigned']:,}、恢复分配 {new['unassigned_to_assigned']:,}。")
            lines.append(f"  {data['projection_checked_transactions']}个修改帧的原始投影重放完全一致；共享原ID支持 {data['raw_shared_vote_keys_preserved']:,} 个键保留；原先弃权的保留旧ID票被释放 {data['released_previously_abstained_retained_votes']:,} 个。")
            lines.append(f"  范围外最终标签变化0；严格规则与原语义修复结果不同的点 {data['final_changed_points_vs_original_semantics']:,}。")
    lines.extend(['灰U表示无生效身份票；弃权后的U可能仍有原始观察。三状态点总数只统计最终未分配点。',
        '身份分配不等于正确归属；以下正确性采用同一v3最大交集一对一对齐，仅用于评分。','',
        '5. 完整追踪修复的归属变化（GT参考顶点）'])
    pair_details={}
    for semantic in ('original','strict'):
        matrix=np.zeros((4,4),np.int64);alignment_changes=0;same_identity_quality_changes=0
        detail={'selected_targets':[],'other_targets':[]}
        for scene in comparison['per_scene']:
            name=semantic+'_full_track_final'
            paired=json.loads((ROOT/'evaluation'/scene/name/'paired_vs_same_rule_baseline.json').read_text())
            matrix+=np.asarray(paired['quality_transition_matrix'])
            alignment_changes+=len(paired['owner_alignment_changes']);same_identity_quality_changes+=paired['quality_changed_on_same_identity_vertices']
            for group,contents in paired['groups'].items():detail[group]+=contents['rows']
        pair_details[semantic]={'quality_matrix':matrix.tolist(),'groups':{}}
        lines.append(f"{NAMES[semantic]}：原未分配→正确 {matrix[1,2]:,}；→错误 {matrix[1,3]:,}；原正确→未分配 {matrix[2,1]:,}；原错误→未分配 {matrix[3,1]:,}；正确→错误 {matrix[2,3]:,}；错误→正确 {matrix[3,2]:,}。")
        lines.append(f"  评分身份对齐变化 {alignment_changes} 项；身份标签未变但评分类别变化 {same_identity_quality_changes:,} 个参考顶点。该数值用于区分标签改变与全局评分对齐变化。")
        for group,rr in detail.items():
            data={'objects':len(rr),'mean_IoU_delta':float(np.mean([x['delta_iou'] for x in rr])) if rr else None,
                'mean_recall_delta':float(np.mean([x['delta_best_recall'] for x in rr])) if rr else None,
                'improved_over_0_01':sum(x['delta_iou']>.01 for x in rr),'degraded_over_0_01':sum(x['delta_iou']<-.01 for x in rr)}
            pair_details[semantic]['groups'][group]=data
            if rr:
                lines.append(f"  {group}：{data['objects']}个GT对象，平均IoU变化 {data['mean_IoU_delta']*100:+.6f}个百分点；IoU改善超过0.01的 {data['improved_over_0_01']} 个，退化超过0.01的 {data['degraded_over_0_01']} 个。")
            else:
                lines.append(f"  {group}：本轮没有合格GT对象。")
    lines.extend(['选择目标沿用原批次冻结人工种子对应的目标列表，并与本轮合格观测GT取交集；不据此改变修复。',
        '目标完整度、结构合并/碎片、全场景PQ/F1都保存在comparison.json和逐GT paired文件中。','',
        '6. 正确性验收'])
    lines.extend([f"10项语义测试和100组随机撤销/重放通过；800原始帧重新投影；{mapping['fresh_projection_transactions']}个真实修复事务与冻结原始Mask修改完全一致。",
        f"独立审计覆盖{independent['real_frame_conditions_checked']}个真实帧条件：每帧每点最多一票；从零累加与增量票表完全一致；逐点票总数不超过400。",
        '所有修复条件精确回滚；无修复严格控制与上一轮逐数组一致；原语义修复票表和原生状态逐数组复现。',
        '局部范围、几何、RGB、原始观察、追踪Mask、种子、身份和门槛均冻结；范围外票/证据/最终标签不变。',
        '24份原生/最终预测采用同一v3开发协议评分；所有基线指标精确复现；输入及预测哈希评分前后不变。',
        '实现过程中的图像只读数组错误、进度回调兼容错误已保留记录并修正；最终提交补齐冻结RGB校验输入，完整重跑后验收。',
        '独立审计分别给出真正碰撞的(frame,point)数和歧义候选键数。主重放receipt内ambiguous_point_frames_before/after字段保存的是候选键数，不能作为碰撞点帧数。','',
        '7. 本轮结论'])
    e=effects['strict']['full_track'];extra=interaction['full_track']
    lines.append(f"严格规则下完整追踪修复相对严格未修复：PQ {e['PQ']*100:+.6f}个百分点，F1 {e['F1']*100:+.6f}个百分点，正确参考顶点 {e['correct_owner_vertices']:+,}，错误参考顶点 {e['wrong_owner_vertices']:+,}。")
    lines.append(f"相对原计票规则，完整追踪修复增益的额外变化：PQ {extra['PQ']*100:+.6f}个百分点，正确参考顶点 {extra['correct_owner_vertices']:+,}。")
    lines.append('严格一票消除了同帧多ID生效票，但跨历史帧不同身份的竞争仍然允许存在；不能把弃权或少红点当成身份关联已经修复。')
    lines.append('本轮不覆盖或替换冻结基线；保留严格修复实验和可复现代码，再根据实际收益决定后续历史重关联。')
    report=ROOT/'严格一票修复复验结果.txt';assert not report.exists();report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result={'status':'PASS','repair_effect_vs_same_rule_baseline':effects,'additional_repair_gain_from_strict_voting':interaction,
        'point_state_effects':scenes,'full_track_quality_and_groups':pair_details,'policy_decisions':policy['decisions'],
        'baseline_replaced':False,'original_sources_modified':False,'GT_used_for_repair':False}
    dump(ROOT/'effect_analysis.json',result)
    with (ROOT/'repair_effect.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    members={}
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.csv','.txt','.log') and p.name not in ('report_complete.json','bundle_manifest.json'):
            members['results/'+str(p.relative_to(ROOT))]=p
    for p in CODE.glob('*.py'):members['code/'+p.name]=p
    member_hashes={name:sha(path) for name,path in members.items()}
    dump(ROOT/'bundle_manifest.json',member_hashes)
    bundle=ROOT/'严格一票修复复验_证据包.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in members.items():z.write(path,name)
        z.writestr('member_sha256.json',json.dumps(member_hashes,ensure_ascii=False,indent=2))
    done={'status':'PASS','path':str(bundle),'sha256':sha(bundle),'bytes':bundle.stat().st_size,'member_count':len(members),'report_sha256':sha(report),
        'includes':'code, tests, protocol/source freezes, full metrics and paired per-GT audits; large prediction maps and masks remain on server'}
    dump(ROOT/'report_complete.json',done);print(json.dumps(done),flush=True)

if __name__=='__main__':main()
