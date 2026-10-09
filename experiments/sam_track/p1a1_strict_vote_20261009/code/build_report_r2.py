"""Report paired repair effects relative to the baseline of the SAME vote rule."""
from pathlib import Path
import csv,hashlib,json,zipfile
import numpy as np

SOURCE=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009')
ROOT=SOURCE/'evaluation_r2'
CODE=Path('/home/chenkejun/CVPR/experiments/p1a1_strict_repair_20261009')
EVALUATOR=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
NAMES={'original':'原语义','strict':'严格一票+弃权'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    complete=json.loads((ROOT/'evaluation_complete.json').read_text());assert complete['status']=='PASS'
    independent=json.loads((SOURCE/'independent_correctness.json').read_text());assert independent['status']=='PASS'
    comparison=json.loads((ROOT/'comparison.json').read_text());assert comparison['all_baseline_metric_controls_exact']
    assert comparison['profile_revision']==2 and complete['current_protocol_locked_and_verified']
    protocol_freeze=json.loads((ROOT/'evaluation_protocol_freeze.json').read_text())
    attribution=json.loads((ROOT/'analysis/targets_and_attribution_complete.json').read_text());assert attribution['status']=='PASS'
    assert attribution['main_metrics_sha256_unchanged']==sha(ROOT/'comparison.json')
    policy=json.loads((SOURCE/'policy.json').read_text())
    mapping=json.loads((SOURCE/'mapping_complete.json').read_text());assert mapping['full_replay_delta_rollback_and_scope_pass']
    tests=json.loads((SOURCE/'tests_complete.json').read_text());assert tests['status']=='PASS'
    assert sha(ROOT/'comparison.json')==complete['comparison_sha256']
    effects={};rows=[]
    lines=['P1-A1 严格一票+弃权：冻结修复效果复验（最新 v3 R2）','',
        '目的：验证严格计票下修复是否仍有效，并区分计票规则改变与修复自身的收益。',
        '范围：room0最新12个修复案例；room2 Top30中实际执行的14个修复案例。',
        '条件：未修复、仅种子帧、完整追踪；原语义与严格语义全部采用同一当前v3修订版R2开发协议重评。',
        '协议：object_observed_repair / profile_revision=2；评分源码cd260569，当前主分支入口29304d8。',
        '配置哈希：'+comparison['profile_config_sha256'],
        'GT范围、对应关系、代码、配置完全一致；非正式冻结基准。预测全部在本轮读取GT前冻结。','',
        '1. 本轮决定与控制']
    decisions=['使用room0最新12个、room2实际14个冻结案例，分别复验仅种子帧和完整追踪。',
        '冻结SAM结果、Mask编辑、身份关联、接受门槛和修复范围；逐个重投影验证原事务。',
        '先合并整帧保留旧支持与新支持，再按每点唯一ID归约；不同ID不能分别计有效票。',
        '同ID共享原始支持保留；撤去歧义支持后允许原先弃权的保留旧票重新生效。',
        '严格修复相对严格未修复作比较；另外计算差中差，扣除规则改变本身的基线收益。',
        '要求400帧全量等于增量、精确回滚、范围外票/证据/最终标签不变。',
        '预测先冻结再评分；不使用GT重新选择种子、案例、身份或门槛。',
        '用户要求最新协议后核对服务器，全部24份预测统一改用主分支锁定的R2；R1评分仅留历史记录。',
        '共用固定GT和同一R2几何缓存，精确重合坐标按R2一致归属规则处理；GT范围不重建。',
        '保留各条件原语义对照的导出实例清单；严格规则新激活的既有ID可追加，弃权不隐式删除实例。',
        '冻结P1-A1与原修复结果均不覆盖，所有新产物保存在独立实验目录。']
    lines.extend(f'{i+1}. {value}' for i,value in enumerate(decisions))
    lines.extend(['',
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
        run=json.loads((SOURCE/scene/'complete.json').read_text());scenes[scene]={}
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
        changed_identity_gain=0;unchanged_identity_gain=0
        detail={'selected_targets':[],'other_targets':[]}
        for scene in comparison['per_scene']:
            name=semantic+'_full_track_final'
            paired=json.loads((ROOT/'analysis/corrected_pairs'/scene/name/'paired_vs_same_rule_baseline.json').read_text())
            matrix+=np.asarray(paired['quality_transition_matrix'])
            alignment_changes+=len(paired['owner_alignment_changes']);same_identity_quality_changes+=paired['quality_changed_on_same_identity_vertices']
            changed_identity_gain+=paired['correct_owner_gain_at_changed_identity_vertices']
            unchanged_identity_gain+=paired['correct_owner_gain_at_unchanged_identity_vertices']
            for group,contents in paired['groups'].items():detail[group]+=contents['rows']
        pair_details[semantic]={'quality_matrix':matrix.tolist(),'groups':{},
            'correct_gain_at_changed_identity_vertices':changed_identity_gain,
            'correct_gain_at_unchanged_identity_vertices':unchanged_identity_gain}
        lines.append(f"{NAMES[semantic]}：原未分配→正确 {matrix[1,2]:,}；→错误 {matrix[1,3]:,}；原正确→未分配 {matrix[2,1]:,}；原错误→未分配 {matrix[3,1]:,}；正确→错误 {matrix[2,3]:,}；错误→正确 {matrix[3,2]:,}。")
        lines.append(f"  评分身份对齐变化 {alignment_changes} 项；身份标签未变但评分类别变化 {same_identity_quality_changes:,} 个参考顶点。该数值用于区分标签改变与全局评分对齐变化。")
        lines.append(f"  标签实际发生改变的位置净增加正确归属 {changed_identity_gain:,} 个参考顶点；标签未变、仅评分对齐发生变化的位置净增加 {unchanged_identity_gain:,} 个。两项之和等于总正确归属增量。")
        for group,rr in detail.items():
            data={'objects':len(rr),'mean_IoU_delta':float(np.mean([x['delta_iou'] for x in rr])) if rr else None,
                'mean_recall_delta':float(np.mean([x['delta_best_recall'] for x in rr])) if rr else None,
                'improved_over_0_01':sum(x['delta_iou']>.01 for x in rr),'degraded_over_0_01':sum(x['delta_iou']<-.01 for x in rr)}
            pair_details[semantic]['groups'][group]=data
            if rr:
                lines.append(f"  {group}：{data['objects']}个GT对象，平均IoU变化 {data['mean_IoU_delta']*100:+.6f}个百分点、平均best-GT recall变化 {data['mean_recall_delta']*100:+.6f}个百分点；IoU改善超过0.01的 {data['improved_over_0_01']} 个，退化超过0.01的 {data['degraded_over_0_01']} 个。")
            else:
                lines.append(f"  {group}：本轮没有合格GT对象。")
    lines.extend(['选择目标沿用原批次冻结目标列表，通过完全相同的参考顶点一一映射到R2物理GT，再与固定合格观测GT取交集；不据此改变修复。',
        '旧列表使用按类别编号的compact evaluation ID，不能直接与R2物理ID比较，也不能对compact ID取模猜测。首次分组为空的诊断已保留；analysis/corrected_pairs内28份校验后的分组替代原分组，主指标和预测未变。',
        '目标完整度、结构合并/碎片、全场景PQ/F1都保存在comparison.json和逐GT paired文件中。','',
        '6. 正确性验收'])
    lines.extend([f"10项语义测试和100组随机撤销/重放通过；800原始帧重新投影；{mapping['fresh_projection_transactions']}个真实修复事务与冻结原始Mask修改完全一致。",
        f"独立审计覆盖{independent['real_frame_conditions_checked']}个真实帧条件：每帧每点最多一票；从零累加与增量票表完全一致；逐点票总数不超过400。",
        '所有修复条件精确回滚；无修复严格控制与上一轮逐数组一致；原语义修复票表和原生状态逐数组复现。',
        '局部范围、几何、RGB、原始观察、追踪Mask、种子、身份和门槛均冻结；范围外票/证据/最终标签不变。',
        '24份原生/最终预测采用同一最新v3 R2评分；8个原规则场景控制与R2审核归档逐字段精确复现，3个最终合并控制亦一致。',
        '全部24个参考归属分区经过独立坐标组归约复核；输入、预测、GT、缓存、协议及评分源码哈希评分前后不变。',
        '实现过程中的图像只读数组错误、进度回调兼容错误已保留记录并修正；最终提交补齐冻结RGB校验输入，完整重跑后验收。',
        '独立审计分别给出真正碰撞的(frame,point)数和歧义候选键数。主重放receipt内ambiguous_point_frames_before/after字段保存的是候选键数，不能作为碰撞点帧数。','',
        '7. 本轮结论'])
    e=effects['strict']['full_track'];extra=interaction['full_track']
    lines.append(f"严格规则下完整追踪修复相对严格未修复：PQ {e['PQ']*100:+.6f}个百分点，F1 {e['F1']*100:+.6f}个百分点，正确参考顶点 {e['correct_owner_vertices']:+,}，错误参考顶点 {e['wrong_owner_vertices']:+,}。")
    lines.append(f"相对原计票规则，完整追踪修复增益的额外变化：PQ {extra['PQ']*100:+.6f}个百分点，正确参考顶点 {extra['correct_owner_vertices']:+,}。")
    native_delta={k:comparison['pooled']['strict_full_track_native'][k]-comparison['pooled']['strict_baseline_native'][k] for k in ('F1','PQ','correct_owner_vertices','wrong_owner_vertices')}
    lines.append(f"严格规则原生结果：完整追踪修复相对未修复PQ {native_delta['PQ']*100:+.6f}个百分点、F1 {native_delta['F1']*100:+.6f}个百分点；正确归属 {native_delta['correct_owner_vertices']:+,}、错误归属 {native_delta['wrong_owner_vertices']:+,} 个参考顶点。收益在后处理前已存在。")
    new_unknown=int(totals['strict_full_track_final'][[0,1,3]].sum()-totals['strict_baseline_final'][[0,1,3]].sum())
    old_unknown=int(totals['original_full_track_final'][[0,1,3]].sum()-totals['original_baseline_final'][[0,1,3]].sum())
    lines.append(f"完整追踪修复的三状态净增：原语义 {old_unknown:,}、严格规则 {new_unknown:,}；严格规则只减少该净增的 {old_unknown-new_unknown:,} 个（{(old_unknown-new_unknown)/old_unknown*100:.4f}%）。")
    lines.append(f"完整追踪修复的平均best-GT recall变化 {e['macro_best_GT_recall']*100:+.6f}个百分点；主结构merge {comparison['pooled']['strict_baseline_final']['merge_predictions']}→{comparison['pooled']['strict_full_track_final']['merge_predictions']}，split {comparison['pooled']['strict_baseline_final']['split_GT']}→{comparison['pooled']['strict_full_track_final']['split_GT']}。质量提升伴随完整度/碎片风险，不能宣称全面改善。")
    lines.append('仅种子帧条件的F1未提升且正确归属略减；主要质量收益来自完整追踪。严格计票的额外收益较小，不是大规模新冲突的解决办法。')
    lines.append('严格一票消除了同帧多ID生效票，但跨历史帧不同身份的竞争仍然允许存在；不能把弃权或少红点当成身份关联已经修复。')
    lines.append('本轮不覆盖或替换冻结基线；保留严格修复实验和可复现代码，再根据实际收益决定后续历史重关联。')
    report=ROOT/'严格一票修复复验结果_R2.txt';assert not report.exists();report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result={'status':'PASS','repair_effect_vs_same_rule_baseline':effects,'additional_repair_gain_from_strict_voting':interaction,
        'point_state_effects':scenes,'full_track_quality_and_groups':pair_details,'policy_decisions':decisions,
        'profile_revision':2,'scoring_commit':comparison['scoring_commit'],'entry_commit':comparison['evaluator_commit'],
        'profile_config_sha256':comparison['profile_config_sha256'],'strict_native_full_track_effect':native_delta,
        'strict_full_track_new_three_state_points':new_unknown,'original_full_track_new_three_state_points':old_unknown,
        'target_mapping_and_attribution_audit':str(ROOT/'analysis/targets_and_attribution_complete.json'),
        'baseline_replaced':False,'original_sources_modified':False,'GT_used_for_repair':False}
    dump(ROOT/'effect_analysis.json',result)
    with (ROOT/'repair_effect.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    members={}
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.csv','.txt','.log') and p.name not in ('report_complete.json','bundle_manifest.json'):
            members['results/'+str(p.relative_to(ROOT))]=p
    for p in CODE.glob('*.py'):members['code/'+p.name]=p
    for name in ('policy.json','mapping_complete.json','predictions_freeze.json','independent_correctness.json','tests_complete.json',
                 'latest_protocol_check.json','correctness_tests.log','failure_attempt01.json','failure_attempt02.json','interrupted_attempt03.json'):
        p=SOURCE/name
        if p.exists():members['mapping/'+name]=p
    for scene in ('room0','room2'):members['mapping/'+scene+'/complete.json']=SOURCE/scene/'complete.json'
    for name,digest in protocol_freeze['current_protocol']['evaluator_code_sha256'].items():
        p=EVALUATOR/'unified_eval'/name;assert sha(p)==digest;members['protocol_r2/unified_eval/'+name]=p
    for rel in ('AGENTS.md','unified_eval/current_protocol.py','unified_eval/configs/current_protocol.json',
                'unified_eval/configs/replica_ca_v3.object_observed_repair.audit_r2.json','docs/OBJECT_OBSERVED_REPAIR_V3_CN.md'):
        members['protocol_r2/'+rel]=EVALUATOR/rel
    member_hashes={name:sha(path) for name,path in members.items()}
    dump(ROOT/'bundle_manifest.json',member_hashes)
    bundle=ROOT/'严格一票修复复验_R2_证据包.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in members.items():z.write(path,name)
        z.writestr('member_sha256.json',json.dumps(member_hashes,ensure_ascii=False,indent=2))
    done={'status':'PASS','path':str(bundle),'sha256':sha(bundle),'bytes':bundle.stat().st_size,'member_count':len(members),'report_sha256':sha(report),
        'includes':'code, tests, protocol/source freezes, full metrics and paired per-GT audits; large prediction maps and masks remain on server'}
    dump(ROOT/'report_complete.json',done);print(json.dumps(done),flush=True)

if __name__=='__main__':main()
