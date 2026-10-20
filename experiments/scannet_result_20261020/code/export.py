"""Audit the six final outputs and export original-index PLYs and comparisons."""
from pathlib import Path
import json,hashlib,colorsys,csv,sys
import numpy as np
R=Path('/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<i4'),('state','u1'),('surface_index','<i4')])
def main():
    scenes=sys.argv[1:] or ['scene0011_00','scene0608_00']
    assert scenes and len(scenes)==len(set(scenes)) and set(scenes)<={'scene0011_00','scene0608_00'}
    rows=[];exports=[];audits=[]
    for scene in scenes:
        comparison=read(R/scene/'evaluation_r2/comparison.json')
        assert comparison['status']=='PASS' and len(comparison['rows'])==6
        assert {x['method'] for x in comparison['rows']}=={'baseline','E1','swap_partial'}
        for p,h in comparison['fixed_input_sha256'].items():assert sha(p)==h
        rows+=comparison['rows']
        for row in comparison['historical_rows']:rows.append(dict(row,stage='historical_final'))
        roots={'baseline':R/'baseline/ablation'/scene/'abstain',
               'E1':R/scene/'e1_repair/e1/E1_expanded_history'/scene,
               'swap_partial':R/scene/'partial/whole_to_partial'/scene}
        out=R/scene/'ply';out.mkdir(exist_ok=True)
        reference=None
        for method,root in roots.items():
            source=root/'final/instance_surface.npz'
            evidence=root/('surface_evidence.npz' if method=='baseline' else 'native/surface_evidence.npz')
            with np.load(source) as z,np.load(evidence) as e:
                xyz=z['xyz_m'];labels=z['instance_id'];state=e['state']
                np.testing.assert_array_equal(xyz,e['xyz_m'])
                if reference is None:reference=xyz.copy()
                else:np.testing.assert_array_equal(xyz,reference)
                metric=next(r for r in comparison['rows'] if r['method']==method and r['stage']=='final')
                assert metric['gray_total']==int((labels<=0).sum())
                for kind,selected in [('instances',labels>0),('three_states_black_yellow_red',labels<=0)]:
                    idx=np.flatnonzero(selected);a=np.empty(len(idx),dtype)
                    for k,n in enumerate(['x','y','z']):a[n]=xyz[idx,k]
                    a['instance_id']=labels[idx];a['state']=state[idx];a['surface_index']=idx
                    colors=np.zeros((len(idx),3),np.uint8)
                    if kind=='instances':
                        for pid in np.unique(labels[idx]):
                            colors[labels[idx]==pid]=np.array(colorsys.hsv_to_rgb((int(pid)*.61803398875)%1,.67,.94))*255
                    else:
                        assert set(np.unique(state[idx]))<={0,1,3}
                        colors[state[idx]==1]=[255,255,0];colors[state[idx]==3]=[255,0,0]
                    for k,n in enumerate(['red','green','blue']):a[n]=colors[:,k]
                    p=out/f'{scene}_{method}_{kind}.ply'
                    header=('ply\nformat binary_little_endian 1.0\ncomment original fixed TSDF surface_index preserved\n'
                        f'element vertex {len(a)}\nproperty float x\nproperty float y\nproperty float z\n'
                        'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty int instance_id\n'
                        'property uchar state\nproperty int surface_index\nend_header\n').encode()
                    if not p.exists():
                        with p.open('xb') as f:f.write(header);a.tofile(f)
                    b=np.memmap(p,dtype=dtype,mode='r',offset=len(header),shape=(len(a),))
                    np.testing.assert_array_equal(a,b)
                    assert p.stat().st_size==len(header)+len(a)*24
                    exports.append(dict(file=str(p.relative_to(R)),sha256=sha(p),vertices=len(a),method=method,
                        source_sha256=sha(source),readback_exact=True))
        pair=read(R/scene/'evaluation_r2/paired/swap_partial_final.json')
        e1=read(R/scene/'evaluation_r2/paired/E1_final.json')
        selection=read(R/scene/'e1_repair/confirmed_selection.json')
        source_report=read(R/scene/'v1/semantic_exclusions.json')
        audits.append(dict(scene=scene,GT=comparison['GT'],partial_corrected=pair['corrected_instances'],
            partial_degraded=pair['degraded_instances'],E1_corrected=e1['corrected_instances'],E1_degraded=e1['degraded_instances'],
            automatic_selected_seeds=len(read(R/scene/'v2/selection.json')['retained_group_ids']),
            executed_seeds=len(selection['included']),
            zero_projection_skipped=selection.get('execution_skipped_seeds',[]),
            post_crop_source_exclusions=source_report.get('post_crop_exclusions',[]),
            post_noise_source_exclusions=source_report.get('post_noise_exclusions',[]),
            partial_replay_exact=read(R/scene/'partial/predictions_freeze.json')['checks']))
    assert sum(a['GT'] for a in audits)==sum({'scene0011_00':27,'scene0608_00':34}[s] for s in scenes)
    assert len(exports)==6*len(scenes)
    deltas=[]
    for scene in scenes:
        baseline=next(r for r in rows if r['scene']==scene and r['stage']=='final' and r['method']=='baseline')
        for method in ['E1','swap_partial']:
            row=next(r for r in rows if r['scene']==scene and r['stage']=='final' and r['method']==method)
            deltas.append(dict(scene=scene,method=method,baseline='strict_one_vote_no_repair',
                **{k+'_pp':100*(row[k]-baseline[k]) for k in ['AP','AP50','F1','PQ','mCov']},
                **{k:row[k]-baseline[k] for k in ['TP','FP','FN','gray_total']}))
    summary=dict(status='PASS',completed_scenes=scenes,all_requested_scenes_complete=len(scenes)==2,
        primary_baseline='strict_one_vote_no_repair',rows=rows,
        deltas_vs_strict_no_repair=deltas,audits=audits,exports=exports,
        scoring='main locked R2 with archived ScanNet scope',frozen=False)
    (R/'comparison_all.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    fields=['scene','method','stage','GT','AP','AP50','F1','PQ','mCov','TP','FP','FN','gray_total']
    with (R/'comparison_all.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    lines=['ScanNet：严格一票无修复 / E1 / auto-v3-swap partial 对比',
        '当前完成场景：'+', '.join(scenes),'',
        '唯一主基线：严格一票无修复。同帧同点不同实例有冲突则弃票；无冲突时每帧每点最多一票。',
        'E1与partial均报告相对于该主基线的变化；历史结果仅作参考，不作为修复收益基线。',
        '固定步长10；scene0011_00为238帧/27 GT，scene0608_00为339帧/34 GT。',
        '新三组共用已修正的历史TSDF、原始关联、固定GT与对应索引；R2规则和评分源码不变。',
        'E1与partial均使用当前swap顺序自动选择；partial复用同批种子、追踪、ID和门控，只替换whole撤票。',
        '旧P1-A1为反向投影投票，新无修复采用当前E1正向投影与严格一票；旧新差值包含基线架构变化。',
        '追踪只使用服务器已有的步长10序列；保留原始帧号，没有补造中间帧。','',
        '场景 | 方法 | AP% | AP50% | F1% | PQ% | TP/FP/FN']
    names={'baseline':'严格一票无修复','E1':'E1（swap种子）','swap_partial':'auto-v3-swap partial',
           'P1-A1_original':'历史P1-A1原版','P1-A1_engineering_fixed':'历史P1-A1工程修正版','MaskClustering':'历史MaskClustering'}
    for row in rows:
        if row['stage']!='final':continue
        lines.append(f"{row['scene']} | {names[row['method']]} | {row['AP']*100:.2f} | {row['AP50']*100:.2f} | {row['F1']*100:.2f} | {row['PQ']*100:.2f} | {row['TP']}/{row['FP']}/{row['FN']}")
    lines+=['','相对严格一票无修复的变化（百分点）','场景 | 方法 | ΔAP | ΔAP50 | ΔF1 | ΔPQ']
    for row in deltas:
        lines.append(f"{row['scene']} | {names[row['method']]} | {row['AP_pp']:+.2f} | {row['AP50_pp']:+.2f} | {row['F1_pp']:+.2f} | {row['PQ_pp']:+.2f}")
    lines+=['','历史参考（包含其他架构差异，不用于计算修复收益）','场景 | 方法 | AP% | AP50% | F1% | PQ% | TP/FP/FN']
    for row in rows:
        if row['stage']!='historical_final':continue
        lines.append(f"{row['scene']} | {names[row['method']]} | {row['AP']*100:.2f} | {row['AP50']*100:.2f} | {row['F1']*100:.2f} | {row['PQ']*100:.2f} | {row['TP']}/{row['FP']}/{row['FN']}")
    lines+=['','各场景目录的paired文件包含逐GT变化；PLY保留实例ID、原生状态和原始表面索引。']
    lines+=['','经用户确认：零有效正向投影的种子不执行修复；原自动筛选结果保留，E1与partial同步跳过。']
    for audit in audits:
        skipped=audit['zero_projection_skipped']
        lines.append(f"{audit['scene']}：自动选中{audit['automatic_selected_seeds']}，执行{audit['executed_seeds']}，跳过{len(skipped)}。")
        for s in skipped:
            lines.append(f"  {s['group_id']} / {s['mask_id']} / frame {s['frame']}：{s['reason']}；有效深度像素{s['valid_depth_pixels']}。")
        for stage,key in [('裁剪后','post_crop_source_exclusions'),('噪声过滤后','post_noise_source_exclusions')]:
            exclusions=audit[key]
            if exclusions:
                dropped=[e['id'] for e in exclusions if e['entire_mask_excluded']]
                lines.append(f"  {stage}重新分组沿用已有SAM-only规则：排除整掩码{', '.join(dropped) or '无'}；移除{sum(len(e['removed_aliases']) for e in exclusions)}条CropFormer来源记录，阈值和掩码像素不变。")
    lines+=['','部分原审计字段沿用all_400_frames名称；实际完整回放帧数分别为238和339，以本报告和固定frame_ids为准。']
    (R/'对比说明.txt').write_text('\n'.join(lines),encoding='utf-8')
    manifest={str(p.relative_to(R)):sha(p) for p in [R/'comparison_all.json',R/'comparison_all.csv',R/'对比说明.txt']}
    for record in exports:manifest[record['file']]=record['sha256']
    for scene in scenes:
        for p in (R/scene/'evaluation_r2').rglob('*.json'):manifest[str(p.relative_to(R))]=sha(p)
        for name in ['v2/selection.json','v2/seed_manifest.json','e1_repair/confirmed_selection.json',
                     'status.json','adapter_preflight.json','partial/policy.json','partial/predictions_freeze.json',
                     'grouping_r3/semantic_exclusions.json','v1/semantic_exclusions.json']:
            p=R/scene/name;manifest[str(p.relative_to(R))]=sha(p)
        for name in ['regroup_source_approval.json','regroup_source_diagnostic.json']:
            p=R/scene/name
            if p.exists():manifest[str(p.relative_to(R))]=sha(p)
    (R/'delivery_manifest.json').write_text(json.dumps(dict(status='PASS',completed_scenes=scenes,sha256=manifest),indent=2))
    print(json.dumps(dict(status='PASS',scenes=scenes,GT=sum(a['GT'] for a in audits),PLY=len(exports),rows=len(rows),audits=audits)),flush=True)
if __name__=='__main__':main()
