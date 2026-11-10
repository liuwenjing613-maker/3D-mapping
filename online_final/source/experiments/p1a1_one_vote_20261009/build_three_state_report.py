"""Create a compact Chinese audit report from verified paired counts."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile
import numpy as np

ROOT=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009')
DEST=ROOT/'three_state_analysis'
CODE=Path('/home/chenkejun/CVPR/experiments/p1a1_one_vote_20261009')
NAMES={'original':'冻结原始','abstain':'严格一票：歧义弃权','depth':'严格一票：深度消歧'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_csv(path,rows):
    assert not path.exists()
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    d=json.loads((DEST/'three_state_analysis.json').read_text())
    comp=json.loads((ROOT/'comparison.json').read_text())
    assert d['status']=='PASS'
    totals={p:{stage:np.zeros(4,np.int64) for stage in ('native','final')} for p in NAMES}
    rows=[];flows=[];qflows=[]
    for scene,r in d['scenes'].items():
        for policy,v in r['policies'].items():
            for stage in ('native','final'):
                vals=np.array(v[stage+'_counts'])
                totals[policy][stage]+=vals
                assert int(vals.sum())==r['surface_points']
                rows.append({'scene':scene,'policy':policy,'stage':stage,'U':int(vals[0]),'T':int(vals[1]),
                    'assigned_or_confirmed':int(vals[2]),'CONFLICT':int(vals[3]),'three_state_total':int(vals[[0,1,3]].sum())})
        for policy,v in r['paired_vs_original'].items():
            for stage in ('native','final'):
                mat=np.array(v[stage+'_matrix'])
                assert np.array_equal(mat.sum(1),r['policies']['original'][stage+'_counts'])
                assert np.array_equal(mat.sum(0),r['policies'][policy][stage+'_counts'])
                for i,a in enumerate(d['state_order']):
                    for j,b in enumerate(d['state_order']):
                        flows.append({'scene':scene,'policy':policy,'stage':stage,'before':a,'after':b,'TSDF_points':int(mat[i,j])})
                qm=np.array(v['quality'][stage]['matrix'])
                assert np.array_equal(qm.sum(1),r['policies']['original']['quality_'+stage+'_counts'])
                assert np.array_equal(qm.sum(0),r['policies'][policy]['quality_'+stage+'_counts'])
                assert v['quality'][stage]['owner_alignment_changes']=={},'Document any changed scoring alignment'
                for i,a in enumerate(d['quality_order']):
                    for j,b in enumerate(d['quality_order']):
                        qflows.append({'scene':scene,'policy':policy,'stage':stage,'before':a,'after':b,'GT_reference_vertices':int(qm[i,j])})
    for p in NAMES:
        for stage in ('native','final'):
            q=np.sum([r['policies'][p]['quality_'+stage+'_counts'] for r in d['scenes'].values()],axis=0)
            assert q[2]==comp['pooled'][p+'_'+stage]['correct_owner_vertices']
            assert q[3]==comp['pooled'][p+'_'+stage]['wrong_owner_vertices']
    write_csv(DEST/'state_counts.csv',rows)
    write_csv(DEST/'state_transitions.csv',flows)
    write_csv(DEST/'quality_transitions.csv',qflows)
    lines=['P1-A1 严格一帧一点一票：三状态点变化分析','',
        '结论：本轮歧义弃权整体比深度消歧更稳健，但新增身份的目标参考顶点中，多数归属错误。',
        '保留冻结 P1-A1；两种严格方案作为消融证据，不直接替换底座。',
        '范围：room0、room2；原几何、原关联、原确认阈值（至少2票且占比至少0.67）、原后处理。',
        '评分：相同 v3 object_observed_repair 开发协议，非正式冻结基准；所有预测读取GT前已冻结。','',
        '1. 统计口径',
        '最终三状态点 = 后处理后 instance_id <= 0，再按该策略自己的原生 state 区分。',
        'U（灰）：零生效身份票；T（黄）：身份支持不足；CONFLICT（红）：多身份且头名占比不足0.67。',
        '已分配/原生确认不属于最终三状态点。弃权后的U可能有原始观察，不能称为从未被观测。',
        'TSDF点数与GT参考顶点数分别报告，二者不能直接相除。','',
        '2. 最终未分配三状态数量（TSDF点）','场景 | 策略 | 灰U | 黄T | 红CONFLICT | 三状态合计']
    for row in rows:
        if row['stage']=='final':lines.append(f"{row['scene']} | {NAMES[row['policy']]} | {row['U']:,} | {row['T']:,} | {row['CONFLICT']:,} | {row['three_state_total']:,}")
    original_total=int(totals['original']['final'][[0,1,3]].sum())
    for p in ('abstain','depth'):
        delta=totals[p]['final']-totals['original']['final'];newtotal=int(totals[p]['final'][[0,1,3]].sum())
        lines.append(f"合并 {NAMES[p]}：灰 {delta[0]:+,}；黄 {delta[1]:+,}；红 {delta[3]:+,}；合计 {newtotal-original_total:+,}（减少{(original_total-newtotal)/original_total*100:.4f}%）。")
    lines.extend(['','3. 逐点迁移：红点减少是否等于获得身份'])
    for p in ('abstain','depth'):
        n=np.asarray(d['pooled']['native_matrices'][p]);f=np.asarray(d['pooled']['final_matrices'][p])
        exit=int(f[[0,1,3],2].sum());entry=int(f[2,[0,1,3]].sum())
        lines.extend([f"{NAMES[p]}：",
            f"  原生冲突→确认 {n[3,2]:,}；→证据不足 {n[3,1]:,}；→零票 {n[3,0]:,}；原生确认→冲突 {n[2,3]:,}。",
            f"  最终原冲突→已分配 {f[3,2]:,}；→黄 {f[3,1]:,}；→灰 {f[3,0]:,}；原已分配→冲突 {f[2,3]:,}。",
            f"  最终原三状态→已分配 {exit:,}；原已分配→三状态 {entry:,}；净增加身份分配 {exit-entry:,}。"])
    lines.extend(['弃权没有改变任何原生确认点的身份；但固定后处理仍可使少量原已分配点重新未分配。',
        '深度消歧保留更多票，仍有254个原生确认点退到冲突；不能仅凭保留票数判断更优。','',
        '4. 新增身份是否正确（同一v3评分，GT参考顶点）'])
    for p in ('abstain','depth'):
        m=np.asarray(d['pooled']['quality_matrices']['final'][p]);new=int(m[1,2]+m[1,3]);v=comp['pooled'][p+'_final'];old=comp['pooled']['original_final']
        lines.extend([f"{NAMES[p]}：原未分配→正确 {m[1,2]:,}，→错误 {m[1,3]:,}；新增分配正确比例 {m[1,2]/new*100:.2f}%。",
            f"  正确→未分配 {m[2,1]:,}；错误→未分配 {m[3,1]:,}；正确→错误 {m[2,3]:,}；错误→正确 {m[3,2]:,}。",
            f"  正确覆盖净增 {v['correct_owner_vertices']-old['correct_owner_vertices']:,}；错误覆盖净增 {v['wrong_owner_vertices']-old['wrong_owner_vertices']:,}。",
            f"  F1 {v['F1']*100:.6f}%；PQ {v['PQ']*100:.6f}%（相对原始{(v['PQ']-old['PQ'])*100:+.6f}个百分点）。"])
    lines.extend(['正确/错误沿用v3最大交集一对一身份对齐，仅用于评分；本轮各策略对齐映射完全相同。',
        '深度方案F1下降已由既有ID220新增1个目标参考顶点触发FP解释，TP/FN未变，不能解读为主要对象普遍退化。','',
        '5. 原因与判断'])
    untouched=sum(r['original_conflict_without_any_same_frame_collision'] for r in d['scenes'].values())
    totalconf=int(totals['original']['native'][3])
    lines.extend([f"原生冲突 {totalconf:,} 点中，{untouched:,}（{untouched/totalconf*100:.2f}%）从未发生同帧多ID碰撞，因此同帧修正无法直接改变这些点的票表。",
        '严格规则消除同帧重复证据，主要让既有头名占比跨过0.67阈值，未新增观察，也未重关联历史身份。',
        '弃权转确认队列：room0的头名置信占比中位数0.656→0.700，room2为0.653→0.714；头名支持票中位数同时26→23、13→11。',
        '这是证据归约后的比例变化，不能当作额外观测验证；最终新增分配正确比例约43%也说明确认不等于正确。',
        '灰U不是几何缺失。原几何完全相同，原始mask和投影关系保留；弃权新增的118个原生零票点仍有原始碰撞观察。',
        '若下一步做修复，优先检查新确认但错误的原冲突点和跨帧旧/新ID竞争，回到完整历史观察处理关联。',
        '无需仅为减少红点放宽确认阈值；两场景结果不足以保证其它六场景同样受益。','',
        '6. 正确性与文件',
        f"逐点迁移矩阵行列守恒；最终原room0三状态计数精确复现既有PLY审计；48个输入哈希执行前后不变。",
        '每个策略的原生/最终预测精确匹配自己的证据状态；GT参考身份精确匹配固定几何对应后的原生标签。',
        '正确、错误、未分配参考顶点计数精确复现前一轮v3结果；没有重跑预测或改写冻结文件。',
        'three_state_analysis.json：逐场景迁移矩阵、队列分析、每项决定、输入哈希。',
        'state_counts.csv：原生与最终数量；state_transitions.csv：TSDF点迁移；quality_transitions.csv：参考顶点正确性迁移。'])
    report=DEST/'三状态点变化分析.txt';assert not report.exists();report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    members={p.name:p for p in DEST.iterdir() if p.is_file()}
    for name in ('analyze_three_states.py','build_three_state_report.py'):members['code/'+name]=CODE/name
    members['previous_comparison.json']=ROOT/'comparison.json'
    member_hashes={n:sha(p) for n,p in members.items()}
    bundle=DEST/'三状态点变化_审计证据.zip';assert not bundle.exists()
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for n,p in members.items():z.write(p,n)
        z.writestr('member_sha256.json',json.dumps(member_hashes,ensure_ascii=False,indent=2))
    done={'status':'PASS','sha256':sha(bundle),'bytes':bundle.stat().st_size,'member_sha256':member_hashes}
    (DEST/'report_complete.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(done,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
