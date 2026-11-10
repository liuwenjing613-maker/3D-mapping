"""Scientific plots, all-case seed views and a self-contained Chinese report."""
from pathlib import Path
import base64,csv,hashlib,html,json,sys,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT=Path('/data/chenkejun/CVPR/results/p1a1_history_core_reassociation_20261009')
CODE=Path('/home/chenkejun/CVPR/experiments/p1a1_history_core_reassociation_20261009')
STRICT=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009')
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
from run_repairs import Scene
ROOTS={'room0':Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d'),
       'room2':Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')}
FIG=OUT/'figures';CONDITIONS=('strict_baseline_final','strict_full_track_final','history_core_final')
TITLES=('Strict baseline','Strict full-track repair','+ History reassociation')
COLORS=('#7d8794','#3674b4','#179585')

def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,value):
    p=Path(p);assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def pct(x):return f'{x*100:.2f}%'
def signed(x):return f'{x*100:+.2f}'
def image(name,caption):
    b=base64.b64encode((FIG/name).read_bytes()).decode()
    return '<figure><img src="data:image/png;base64,'+b+'" alt="'+html.escape(caption)+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'
def palette(labels):
    # The same persistent ID always has the same color across all conditions.
    result=plt.get_cmap('hsv')(((labels.astype(np.int64)*.61803398875)%1)).copy()
    result[labels<=0]=(.60,.60,.60,1);return result
def charts(result):
    FIG.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(14,8),dpi=160)
    for i,scene in enumerate(('room0','room2')):
        for j,key in enumerate(('F1','PQ')):
            ax=axes[i,j];values=[100*result['per_scene'][scene][c][key] for c in CONDITIONS]
            ax.bar(np.arange(3),values,color=COLORS,width=.63);ax.set_ylim(0,100)
            ax.set_xticks(range(3),['Baseline','Repair','+History']);ax.set_ylabel(key+' (%)')
            ax.set_title(f'{scene}: final {key}; delta {values[2]-values[1]:+.2f} pp')
            for x,v in enumerate(values):ax.text(x,v+1.8,f'{v:.2f}',ha='center',fontsize=10)
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax=axes[i,2];bottom=np.zeros(3)
        for k,title,col in [(0,'Unobserved','#6f7c89'),(1,'Tentative','#e7b44c'),(3,'Conflict','#cc5b65')]:
            values=np.array([result['states'][scene][c]['counts_U_T_assigned_C'][k] for c in CONDITIONS])
            ax.bar(range(3),values,bottom=bottom,color=col,width=.63,label=title);bottom+=values
        ax.set_xticks(range(3),['Baseline','Repair','+History']);ax.set_title(scene+': final unassigned TSDF points')
        ax.set_ylim(0,bottom.max()*1.23)
        for x,v in enumerate(bottom):ax.text(x,v+bottom.max()*.025,f'{int(v):,}',ha='center',fontsize=10)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0,2].legend(fontsize=9,ncol=1,loc='upper right')
    fig.suptitle('Fixed geometry and R2 scope | 26 cases, 55 seed masks | strict one vote + abstention',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(FIG/'two_scene_overview.png');fig.savefig(FIG/'two_scene_overview.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,9),dpi=165)
    for i,scene in enumerate(('room0','room2')):
        rows=result['paired'][scene]['history_core_final_vs_strict_full_track']['per_gt']
        for j,(key,label) in enumerate([('iou','Best GT IoU'),('best_recall','Best single-instance GT recall')]):
            ax=axes[i,j];ax.plot([0,100],[0,100],c='#333',ls='--',lw=.8)
            for selected,col,name in [(False,'#7c8999','Other GT'),(True,'#159788','Selected GT')]:
                rr=[r for r in rows if r['selected_target']==selected]
                old='before_best_iou' if key=='iou' else 'before_best_recall';new='after_best_iou' if key=='iou' else 'after_best_recall'
                ax.scatter([100*r[old] for r in rr],[100*r[new] for r in rr],s=29,c=col,label=name,alpha=.8)
                for r in rr:
                    if abs(r[new]-r[old])>.08:
                        ax.annotate('GT'+str(r['raw_gt_id']),(100*r[old],100*r[new]),xytext=(4,4),textcoords='offset points',fontsize=8)
            ax.set_xlim(-3,103);ax.set_ylim(-3,103);ax.set_aspect('equal');ax.set_xlabel('Strict full-track repair (%)');ax.set_ylabel('+ History (%)')
            ax.set_title(scene+': '+label,fontsize=10);ax.grid(alpha=.14)
    axes[0,0].legend(fontsize=9,loc='upper left')
    fig.suptitle('All 124 qualified physical GT; no target selection changes after scoring',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(FIG/'all_GT_comparison.png');fig.savefig(FIG/'all_GT_comparison.pdf');plt.close(fig)
def case_views(result,parent):
    views=[]
    for scene_id in ('room0','room2'):
        scene=Scene(scene_id);records=parent['scenes'][scene_id]['predictions'];xyz=scene.xyz
        labels=[load(records[name]['path'])['instance_id'] for name in CONDITIONS[:2]]
        labels.append(load(OUT/scene_id/'final/instance_surface.npz')['instance_id'])
        cases=read(ROOTS[scene_id]/'seed_diagnostics.json')['cases']
        for index,case in enumerate(cases,1):
            uid=case['case_uid'];cfg=read(case['config_path']);frame=scene.source.load_frame(cfg['seed_frame'])
            support=load(ROOTS[scene_id]/'cases'/uid/'seed_projected_support.npz');p=np.unique(support['surface_point_index'])
            inv=np.linalg.inv(frame.camera_to_world);cam=xyz[p].astype(np.float64)@inv[:3,:3].T+inv[:3,3]
            u=frame.camera.fx*cam[:,0]/cam[:,2]+frame.camera.cx;v=frame.camera.fy*cam[:,1]/cam[:,2]+frame.camera.cy
            order=np.argsort(cam[:,2])[::-1]
            fig,axes=plt.subplots(1,3,figsize=(13.5,4.7),dpi=145,sharex=True,sharey=True)
            for ax,name,lab in zip(axes,TITLES,labels):
                gray=int(np.sum(lab[p]<=0));ax.scatter(u[order],v[order],c=palette(lab[p][order]),s=.65,linewidths=0,rasterized=True)
                ax.set_xlim(u.min()-8,u.max()+8);ax.set_ylim(v.max()+8,v.min()-8);ax.set_aspect('equal')
                ax.set_title(name+'\n'+f'gray {gray:,} / {len(p):,}',fontsize=10);ax.set_xlabel('Seed camera u (px)')
            axes[0].set_ylabel('Seed camera v (px)')
            targetids=sorted({o['persistent_id'] for o in result['mapping'][scene_id]['objects'] if o['case_uid']==uid})
            legend=[Patch(color=(.6,.6,.6),label='Unassigned')]+[Patch(color=palette(np.array([pid]))[0],label='Target ID'+str(pid)) for pid in targetids]
            fig.legend(handles=legend,loc='lower center',ncol=min(len(legend),6),fontsize=8)
            fig.suptitle(scene_id+' case '+str(index)+' | fixed seed-frame view f'+str(cfg['seed_frame'])+' | all selected seed points',fontsize=11)
            fig.tight_layout(rect=(0,.07,1,.92));name=scene_id+'_case_%02d.png'%index;fig.savefig(FIG/name);plt.close(fig)
            views.append({'scene':scene_id,'case_uid':uid,'ROI':case['source_choice']['ROI'],'figure':name,
                'seed_frame':cfg['seed_frame'],'target_IDs':targetids,'points':len(p)})
        scene.verify_unchanged()
    return views
def main():
    done=read(OUT/'evaluation_r2/evaluation_complete.json');result=read(OUT/'evaluation_r2/comparison.json')
    assert done['status']==result['status']=='PASS' and result['profile_revision']==2
    assert sha(OUT/'evaluation_r2/comparison.json')==done['comparison_sha256']
    diagnosis=read(OUT/'evaluation_r2/analysis/remaining_case_diagnosis.json');assert diagnosis['status']=='PASS'
    freeze=read(OUT/'predictions_freeze.json');parent=read(STRICT/'predictions_freeze.json')
    for p,d in freeze['output_sha256'].items():assert sha(p)==d,p
    for p,d in result['protected_sha256'].items():assert sha(p)==d,p
    charts(result);views=case_views(result,parent)
    pool=result['pooled'];p=result['pooled_paired']['history_core_final_vs_strict_full_track']
    targets=p['groups']['selected_targets'];other=p['groups']['other_targets']
    exclude=[r for r in p['per_gt'] if r['selected_target'] and (r['scene_id'],r['raw_gt_id'])!=('room0',58)]
    excluded_delta={k:float(np.mean([r[k] for r in exclude])) for k in ('delta_iou','delta_best_recall')}
    gray_before=result['pooled_states']['strict_full_track_final']['three_state_total'];gray_after=result['pooled_states']['history_core_final']['three_state_total']
    summary={'status':'PASS','profile_revision':2,'rows':[],'selected_targets':{k:v for k,v in targets.items() if k!='rows'},
        'other_targets':{k:v for k,v in other.items() if k!='rows'},'excluding_original_diagnosis_GT58_selected_objects':len(exclude),
        'excluding_original_diagnosis_mean_deltas':excluded_delta,'gray_before':gray_before,'gray_after':gray_after,
        'gray_reduction':gray_before-gray_after,'gray_reduction_fraction':(gray_before-gray_after)/gray_before,
        'net_correct_owner_gain':p['correct_owner_vertex_gain'],
        'net_wrong_owner_gain':pool['history_core_final']['wrong_owner_vertices']-pool['strict_full_track_final']['wrong_owner_vertices'],
        'implementation_correctness_passed':True,'same_physical_target_can_have_multiple_frozen_target_IDs':True,
        'automatic_baseline_replacement_recommended':False,'all_case_views':views}
    table=[]
    for scene in ('room0','room2','pooled'):
        source=result['per_scene'][scene] if scene!='pooled' else pool
        states=result['states'][scene] if scene!='pooled' else result['pooled_states']
        for cond in CONDITIONS:
            s=source[cond];counts=states[cond]['counts_U_T_assigned_C']
            row={'scene':scene,'condition':cond,'F1':s['F1'],'PQ':s['PQ'],'AP':s['AP'],
                'macro_best_GT_recall':s['macro_best_GT_recall'],'U':counts[0],'T':counts[1],'C':counts[3],
                'final_gray':states[cond]['three_state_total'],'correct_owner_vertices':s['correct_owner_vertices'],'wrong_owner_vertices':s['wrong_owner_vertices']}
            summary['rows'].append(row)
            cname={'strict_baseline_final':'严格一票基线','strict_full_track_final':'严格一票＋既有追踪修复','history_core_final':'再加核心区域历史重关联'}[cond]
            table.append('<tr><td>'+scene+'</td><td>'+cname+'</td>'+''.join('<td>'+x+'</td>' for x in
                (pct(s['F1']),pct(s['PQ']),pct(s['macro_best_GT_recall']),f'{counts[0]:,}',f'{counts[1]:,}',f'{counts[3]:,}',f'{row["final_gray"]:,}'))+'</tr>')
    dump(OUT/'two_scene_result_summary.json',summary)
    with (OUT/'two_scene_comparison.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(summary['rows'][0]));w.writeheader();w.writerows(summary['rows'])
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>两场景历史重关联验证</title><style>body{max-width:1180px;margin:26px auto;padding:0 20px;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#223041;background:#fcfcfd}h1{font-size:28px}h2{font-size:21px;margin-top:28px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border:1px solid #cad3dd;padding:8px;text-align:right}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}thead{background:#eaf1f8}aside{padding:14px 18px;border-left:4px solid #437cb6;background:#edf4fa}figure{margin:18px 0}img{width:100%;height:auto}figcaption,.small{font-size:14px;color:#526172}code{font-size:13px;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}a{color:#24679c}</style><body>
<h1>room0、room2：严格一票＋弃权下的历史观察重关联</h1>
<p><strong>结果：两个场景的F1、PQ提高，冲突灰点减少；但仍存在同一物体被多个修复目标身份拆分的退化。</strong>这说明局部历史重关联有效，也说明不能直接把当前规则作为完整修复方案。</p>
<aside>本实验使用固定10像素收缩的人工种子核心，扩展到所有26个既有案例、55个种子Mask（52个规范目标身份）。保留原始几何、Mask、追踪结果、确认阈值与后处理。两个新预测先一起冻结，再读取GT评分；GT不决定像素、帧、ID或参数。此前已经诊断过room0 GT58，因此属于诊断驱动的开发验证，不能当作预注册泛化基准。</aside>
<h2>1. 固定规则与最新协议</h2>
<p>对每个原始保留的深度像素，若最近TSDF点属于唯一目标的种子核心，就把该像素的旧身份重新解释为目标身份；检索所有旧ID，不受old_family_ids限制。只拆分相关源像素，不整体改名旧ID，不恢复已经撤销的观察，也不修改已接受的新追踪Mask。重新投影残留Mask与新增片段，联合重建整帧候选：同身份共享支持只计一次，不同身份仍弃权。同一点有多个目标核心认领时保留原观察。</p>
<p>主协议为main当前默认入口 <code>python -m unified_eval.cli</code>，<code>object_observed_repair / revision 2</code>；评分源码 <code>cd260569e89de1e9525f82c45ee10842b80616b0</code>，审核归档 <code>d035f68770cefab27c9751e97062c72faa07367c</code>。固定350个可评价GT，两个场景实际评价124个（room0 72、room2 52）。6个极小待审GT、168个UNKNOWN和既有GT范围保持原状；frozen=false，属于开发评价。</p>
<h2>2. 全场景最终结果</h2><div style="overflow:auto"><table><thead><tr><th>场景</th><th>条件</th><th>F1</th><th>PQ</th><th>最佳单实例召回</th><th>U灰点</th><th>T灰点</th><th>C灰点</th><th>总灰点</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div>
<p class="small">U/T/C分别为未观察、证据不足、身份冲突；这里统计最终仍未归属的TSDF点。R2质量指标是在固定可观察GT参考表面评分，与全场景TSDF点数不是同一分母。最佳单实例召回取各GT被任意单个预测覆盖的最大比例，因此错误的大合并实例也可能得到高召回，必须和IoU及正确归属一起判断。</p>'''
    text+='<p>相对既有严格一票追踪修复，合并F1 '+pct(pool['strict_full_track_final']['F1'])+' → '+pct(pool['history_core_final']['F1'])+'（'+signed(pool['history_core_final']['F1']-pool['strict_full_track_final']['F1'])+'个百分点），PQ '+pct(pool['strict_full_track_final']['PQ'])+' → '+pct(pool['history_core_final']['PQ'])+'（'+signed(pool['history_core_final']['PQ']-pool['strict_full_track_final']['PQ'])+'个百分点）。灰点 '+f'{gray_before:,} → {gray_after:,}'+'，减少 '+f'{gray_before-gray_after:,}'+'（'+pct(summary['gray_reduction_fraction'])+'）。U不变，T仅减少2个，主要恢复的是C冲突点。</p>'
    text+=image('two_scene_overview.png','相同几何、GT范围和对应索引下，三个条件的最终指标与三状态灰点。')
    text+='<h2>3. 目标效果与收益来源</h2><p>45个已选物理GT的平均最佳IoU '+pct(targets['mean_before_iou'])+' → '+pct(targets['mean_after_iou'])+'；16个提高、5个下降、24个不变，其中9个提高和2个下降超过1个百分点。平均最佳单实例召回 '+pct(targets['mean_before_best_recall'])+' → '+pct(targets['mean_after_best_recall'])+'；4个目标下降超过1个百分点。其余79个GT的最佳召回全部不变，没有最佳IoU下降超过1个百分点的物体。</p>'
    text+='<p>收益含有明显的原诊断案例贡献：room0 GT58百叶窗IoU 3.44% → 82.32%。排除这个已诊断物体，其余44个选中GT的平均IoU变化 '+signed(excluded_delta['delta_iou'])+'个百分点、平均最佳召回变化 '+signed(excluded_delta['delta_best_recall'])+'个百分点。因此两个场景总分提高，仍不足以证明对每类受损目标都有稳定完整度收益。</p>'
    text+='<p>固定参考表面的正确归属净增 '+f'{summary["net_correct_owner_gain"]:,}'+' 个，错误归属也净增 '+f'{summary["net_wrong_owner_gain"]:,}'+' 个。room0正确归属增加9,837，全部来自实际身份修改；room2净增6,155，其中修改身份处增加8,599，身份未改变处因全局一对一匹配变化净减2,444。不能把全局匹配带来的变化解释为那些位置的地图标签也被修改了。</p>'
    text+=image('all_GT_comparison.png','每个物理GT的IoU和最佳单实例召回，包含全部124个可评价物体。对角线上方为提高。')
    text+='''<h2>4. 逐个解释明显下降</h2>
<p><strong>room0 GT60地毯：</strong>同一物体的核心被ID13、ID15分别认领。1,610个参考顶点从13改为15，另有2,625个灰点改为15；主实例的召回84.20% → 81.26%，IoU33.09% → 31.95%。核心颜色统一了，但整个物体仍有多个身份。</p>
<p><strong>room2 GT1地毯：</strong>三个目标ID355、356、361的核心均落在同一物理GT上，原ID1也仍保留其他部分。重关联把2,665个原ID1参考顶点分给三个目标身份。该GT最佳IoU31.25% → 22.80%，最佳召回43.61% → 31.81%；R2全局匹配还将ID1从GT1改配GT9，356改配GT1。这是修复目标身份未统一造成的真实拆分问题，不能靠消灰点判断成功。</p>
<p><strong>room0 GT36门：</strong>ROI-C0005 mask2的核心ID801同时覆盖335个门GT36参考顶点和570个GT61参考顶点。333个原ID10门顶点被改为801，门的召回98.03% → 95.45%。向内收缩10像素不能保证人工Mask或目标身份已经正确。</p>
<p><strong>room2 GT38百叶窗：</strong>最佳单实例召回76.60% → 62.25%，但这不是同样的退化。覆盖它的大旧ID10匹配的是其他GT32；936个错误旧ID10顶点和666个灰点改为正确目标358。目标358自身召回3.24% → 27.80%，GT最佳IoU15.28% → 27.80%，正确归属增加1,602。这里最大召回下降反映了旧大合并被局部拆开，不能单独据此判为错误。</p>
<p class="small">全部超过1个百分点的IoU或最佳召回下降均已列出并保存于 <code>remaining_case_diagnosis.json</code>。GT只用于事后解释；本次没有按GT修改目标ID、扩展Mask或调阈值。</p>
<h2>5. 正确性、覆盖与边界</h2>
<p>room0 383帧、room2 394帧发生原始源像素重关联，共4,388,606个采样像素。核心区域分别为88,446、97,352点，两个场景均无不同目标核心重叠。room0 10/30、room2 3/25个种子Mask收缩后为空，本次按固定规则保留空核心，没有临时放宽；因此薄物体和小Mask的覆盖有限。最终修改只提交到有效票变化的局部范围，范围外票据、证据字段和最终标签逐项一致。</p>
<p>8项单元验证（含100轮随机独立字典对照）通过；两场景各400帧票据回放与增量修改完全一致、撤销精确恢复；原始投影缓存与重新投影一致，既有残留/新追踪Mask投影与冻结修复一致；源像素账目与新片段投影一致；共享支持、严格一帧一点最多一票、旧观察不复活、新Mask保持、范围外一致均通过。原单案例85帧事务精确复现。两个新场景在评分前一起冻结；评分使用main默认R2、相同固定GT与同一对应缓存；所有冻结控制分数精确复现，原输入/协议/预测哈希不变。</p>
<p>第一次回放在无有效追踪提议帧缺少审核字段时停止，已根据原程序直接保留原帧的逻辑修正；评分前的数量断言也已改为核对gt_scope的可评价集合。两个失败尝试及原代码均归档，均未产生有效评分。正式结果在完整检查通过后生成。</p>
<h2>6. 下一步判断</h2>
<p>保留严格一票＋弃权和源观察重建机制。历史重关联能有效纠正同一表面的跨帧身份矛盾，但还需要先保证<strong>多个修复种子指向同一物理物体时使用一致目标身份，并检查混合Mask</strong>，再扩大历史重关联覆盖。当前结果适合作为发展方向证据，不能直接覆盖冻结P1-A1。本次所有结果独立保存，未变更基线。</p>
<h2>7. 全部26个案例的固定种子视角</h2><p>下面保留所有案例，避免只展示成功者。颜色固定对应持久ID，灰色为未归属；图只显示该案例原种子投影支持，视角和点集一致，不使用GT选择展示位置。</p>'''
    for v in views:
        caption=v['scene']+' '+v['ROI']+'，种子帧 '+str(v['seed_frame'])+'，目标身份 '+', '.join(map(str,v['target_IDs']))
        text+='<details><summary>'+html.escape(caption)+'</summary>'+image(v['figure'],caption)+'</details>'
    text+='<h2>8. 数据与复验</h2><p><a href="two_scene_comparison.csv">三条件对比CSV</a> · <a href="evaluation_r2/per_gt_vs_current_repair.csv">124个物体逐项CSV</a> · <a href="two_scene_result_summary.json">摘要JSON</a> · <a href="two_scene_history_reassociation_audit.zip">代码与证据包</a>。完整场景预测和逐帧事务保留在服务器 <code>'+str(OUT)+'</code>；包内包含源路径和SHA256，基线与大体积事务不重复复制。</p></body></html>'
    (OUT/'two_scene_report_zh.html').write_text(text)
    # Include all small evidence, plots, source and selected surfaces. Large
    # full-scene maps/transactions stay on the server with the frozen hashes.
    files={}
    for p in CODE.glob('*.py'):files['code/'+p.name]=p
    for p in OUT.rglob('*'):
        if not p.is_file() or p.suffix not in ('.json','.csv','.html','.png','.pdf','.log','.py'):continue
        if p.name in ('delivery_complete.json','bundle_manifest.json'):continue
        if 'transactions' in p.parts:continue
        files['results/'+str(p.relative_to(OUT))]=p
    for s in ('room0','room2'):
        for name in ('selected_surface_comparison.npz','core_claims.npz','allowed_surface_ids.npz'):files['results/'+s+'/'+name]=OUT/s/name
    manifest={'status':'PASS','profile_revision':2,'full_maps_and_transactions_on_server':str(OUT),
        'files':{name:{'source':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for name,p in files.items()}}
    dump(OUT/'bundle_manifest.json',manifest)
    package=OUT/'two_scene_history_reassociation_audit.zip';assert not package.exists()
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.write(OUT/'bundle_manifest.json','bundle_manifest.json')
        for name,p in files.items():z.write(p,name)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        for name,item in manifest['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==item['sha256']
    for p,d in freeze['output_sha256'].items():assert sha(p)==d,p
    for p,d in result['protected_sha256'].items():assert sha(p)==d,p
    dump(OUT/'delivery_complete.json',{'status':'PASS','report_sha256':sha(OUT/'two_scene_report_zh.html'),
        'archive_sha256':sha(package),'archive_bytes':package.stat().st_size,'archive_members':len(files)+1,
        'all_archive_member_hashes_verified':True,'all_26_case_views_included':True,
        'output_sha256':{str(p):sha(p) for p in (OUT/'two_scene_report_zh.html',OUT/'two_scene_result_summary.json',OUT/'two_scene_comparison.csv',package,OUT/'bundle_manifest.json',*FIG.glob('*'))}})
    print(json.dumps({'status':'PASS','archive_bytes':package.stat().st_size,'case_views':len(views),'summary':{k:v for k,v in summary.items() if k not in ('rows','selected_targets','other_targets','all_case_views')}}),flush=True)

if __name__=='__main__':main()
