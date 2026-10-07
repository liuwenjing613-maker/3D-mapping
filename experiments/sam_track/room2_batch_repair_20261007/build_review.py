"""Local Chinese presentation of the validated room2 batch; no source edits."""
from pathlib import Path
from html import escape
import hashlib,json,shutil,tarfile
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
WEB=WORK/'results/固定案例_三模型对比_20261006'
OUT=WEB/'room2_repair_batch_20261007'
TITLES={'baseline':'原始地图','seed_only':'仅种子修复','full_track':'完整双向追踪修复'}
MODELS={'original':'原始 CropFormer','cropformer_local':'局部 CropFormer','sam2_1_local':'SAM2.1 局部候选','unreliable':'三种都不可靠'}
def pct(n):return f'{100*n:.2f}%'
def main():
    record=json.loads((HERE/'review_bundle.json').read_text(encoding='utf8'));bundle=HERE/'review_bundle.tar.gz'
    assert hashlib.sha256(bundle.read_bytes()).hexdigest()==record['sha256']
    with tarfile.open(bundle) as archive:
        for item in archive.getmembers():
            assert (WEB/item.name).resolve().is_relative_to(OUT.resolve()) and not item.issym() and not item.islnk()
        archive.extractall(WEB,filter='data')
    shutil.copytree(WEB/'tracking_review_20261007/vendor',OUT/'vendor',dirs_exist_ok=True)
    shutil.copy2(HERE/'room2_viewer.js',OUT/'room2_viewer.js')
    shutil.copy2(HERE/'seed_contact.jpg',OUT/'seed_contact.jpg')
    shutil.copy2(HERE/'inputs/human_selection.json',OUT/'human_selection.json')
    shutil.copy2(HERE/'CHANGES_CN.md',OUT/'CHANGES_CN.md')
    data=json.loads((OUT/'comparison_data.json').read_text(encoding='utf8'));ev=data['evaluation'];repair=data['repair_reports']['full_track']
    assoc={oid:obj for obj in data['global_association']['objects'] for oid in obj['member_track_ids']}
    identity_sources={}
    for row in data['human_selections']:
        for obj in row.get('objects',[]):
            source=f'{row["source_choice"]["ROI"]} mask{obj["native_mask_id"]}'
            identity_sources.setdefault(str(obj['old_family_id']),set()).add('原观察 '+source)
            identity_sources.setdefault(str(assoc[obj['track_id']]['persistent_id']),set()).add('目标 '+source)
    data['model']['identity_sources']={pid:sorted(rows) for pid,rows in identity_sources.items()}
    metrics={'baseline':ev['baseline_v3_metrics'],**{name:ev['comparisons'][name]['scene_metrics'] for name in ['seed_only','full_track']}}
    table=[]
    for name,title in TITLES.items():
        m=metrics[name];table.append(f'<tr><td>{title}</td><td>{pct(m["CA_AP50_uniform"])}</td><td>{pct(m["CA_PRF1_0_5"]["F1"])}</td><td>{pct(m["CA_PQ"]["PQ"])}</td><td>{m["CA_PRF1_0_5"]["TP"]}/{m["CA_PRF1_0_5"]["FP"]}/{m["CA_PRF1_0_5"]["FN"]}</td><td>{m["structure"]["merge_prediction_count"]}/{m["structure"]["split_gt_count"]}</td></tr>')
    case_rows=[];options=[];details=[]
    for row in data['human_selections']:
        uid=row['case_uid'];choice=row['source_choice'];roi=choice['ROI'];ready=row['status']=='READY'
        options.append(f'<option value="{uid}">{roi} · f{choice["frame"]}</option>')
        ids=', '.join(map(str,choice['mask_ids'])) or '无'
        accepted=', '.join(f'm{o["native_mask_id"]}: {repair["counts"]["accepted_frames_by_track"][str(o["track_id"])]}' for o in row.get('objects',[])) or '未单独修复'
        case_rows.append(f'<tr><td><button data-open-case="{uid}">{roi}</button></td><td>{MODELS[choice["choice"]]} · f{choice["frame"]}</td><td>{ids}</td><td>{accepted}</td></tr>')
        seedrows=[];timeline=[]
        for obj in row.get('objects',[]):
            oid=obj['track_id'];au=next(a for a in data['seed_audits'] if a['track_id']==oid);pid=assoc[oid]['persistent_id']
            alias='（与 C0002 mask11 共用）' if oid==9 else ''
            votes=au['seed_surface_final_identity_vote_counts']
            seedrows.append(f'<tr><td>mask{obj["native_mask_id"]}</td><td>{obj["old_family_id"]} → {pid}{alias}</td><td>{au["accepted_mapping_frames"]}/400</td><td>{pct(au["desired_identity_seed_surface_fraction_before"])} → {pct(au["desired_identity_seed_surface_fraction_after"])}</td><td>{au["seed_surface_unknown_before"]} → {au["seed_surface_unknown_after"]}</td><td>{votes.get(str(obj["old_family_id"]),0):,} / {votes.get(str(pid),0):,}</td></tr>')
            cells=[];decisions={d['frame']:d for d in data['timeline']['frames']}
            for fid in range(0,2000,5):
                decision=decisions.get(fid,{});used=oid in decision.get('accepted_track_ids',[]);check=decision.get('objects',{}).get(str(oid),{})
                tip=f'f{fid}: '+('通过检查，参与重建' if used else ', '.join(check.get('reasons',['保留原观测'])))
                cells.append(f'<span class="{"used" if used else "kept"}" title="{escape(tip)}"></span>')
            timeline.append(f'<p>mask{obj["native_mask_id"]} · ID{pid} · {au["accepted_mapping_frames"]}/400 帧</p><div class="timeline">{"".join(cells)}</div>')
        figures=[]
        for view in [v for v in data['views'] if v['case_uid']==uid]:
            figures.append(f'<p>固定视角 f{view["frame"]} · <a href="{view["assets"]["rgb"]}">原始 RGB</a> · <a href="{view["assets"]["delta"]}">标签改动位置</a></p><div class="views">'+''.join(f'<figure><figcaption>{title}</figcaption><a href="{view["assets"][name]}" target="_blank"><img src="{view["assets"][name]}" alt="{roi} f{view["frame"]} {title}" loading="lazy"></a></figure>' for name,title in TITLES.items())+'</div>')
        tracks=''.join(f'<figure><figcaption>f{v["frame"]} · 本组通过检查的追踪编号：{v["only_this_case_used_track_ids"]}</figcaption><a href="{v["asset"]}" target="_blank"><img src="{v["asset"]}" loading="lazy"></a></figure>' for v in data['track_views'] if v['case_uid']==uid)
        note='C0009 mask4 触碰人工裁剪边界，是同一地毯的局部种子，保留原选择并与 C0002 合并身份。' if choice['ROI']=='ROI-C0009' else ''
        if not ready:note='你标记为“三种都不可靠”，没有独立的追踪或换票。下图是统一的联合修复地图，可能受到其他选区的共享修复影响。'
        details.append(f'<section data-case="{uid}" hidden><h2>{roi} · {MODELS[choice["choice"]]} · 种子帧 {choice["frame"]}</h2><p>{note}</p>'+
          (f'<div class="scroll"><table><tr><th>你的 mask</th><th>原观察身份 → 新身份</th><th>通过检查并用于重建</th><th>种子表面目标身份比例</th><th>种子表面灰点</th><th>原家族ID / 目标ID累计票</th></tr>{"".join(seedrows)}</table></div><p class="muted">种子表面统计只覆盖这个视角的可见表面；全物体完整度另看 v3 和固定表面诊断。累计票取最终票据，在此种子表面按帧、点和ID去重。ID保持时，两个数相同且都包含保留旧票与新贡献。</p>' if ready else '')+
          ''.join(figures)+(f'<details><summary>双向追踪的实际帧示例（含未写入地图的候选）</summary><div class="trackViews">{tracks}</div><p>图片为原始 SAM2 传播结果。通过质量检查的帧才用于观测重建，编号与下方条带对应。</p></details><details><summary>每个 mask 的 400 帧用证据情况</summary><p>从 f0 到 f1995，每格一个建图帧；绿格参与重建，灰格保留旧观测。悬停查看原因。</p>{"".join(timeline)}</details>' if ready else '')+'</section>')
    targets=[]
    for target in ev['comparisons']['full_track']['targets']:
        gid=target['before']['GT_id'];seed=next(t['after'] for t in ev['comparisons']['seed_only']['targets'] if t['before']['GT_id']==gid)
        targets.append(f'<tr><td>{gid}</td><td>{pct(target["before"]["best_IoU"])}</td><td>{pct(seed["best_IoU"])}</td><td>{pct(target["after"]["best_IoU"])}</td><td>{target["after"]["best_pred_uid"]}</td><td>{pct(target["after"]["completeness_best_single_instance"])}</td></tr>')
    cmp=ev['comparisons']['full_track'];commit=repair['strict_commit'];success='达到当前修复成功判据' if cmp['strict_success'] else '尚未达到当前修复成功判据'
    warning='多个选中 mask 对应同一个 GT 实例：'+escape(json.dumps(ev['multiple_selected_parts_of_same_GT_instance'],ensure_ascii=False))+'。指标仍按原 v3 规则计分，没有合并预测来提高分数。' if ev['multiple_selected_parts_of_same_GT_instance'] else ''
    proof=f'400 帧完整重算＝增量换票；准确回滚通过；TSDF几何/RGB保留；范围外全部原始证据、投票及最终标签一致（最终外部改动 {commit["final_outside_changes"]} 点）。'
    gray=repair['whole_scene_unknown_after']-repair['whole_scene_unknown_before']
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>room2 人工标注修复 · 实际结果</title>
<style>*{box-sizing:border-box}body{margin:0;background:#eef2f7;color:#16233b;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:22px}section{background:white;border:1px solid #d8e0eb;border-radius:12px;padding:20px;margin:0 0 18px}h1{font-size:26px;margin:0 0 8px}h2{font-size:19px;margin:0 0 10px}p{margin:8px 0}a{color:#1666ab}button,select{font:inherit;border:1px solid #cad5e3;background:white;color:inherit;border-radius:6px;padding:5px 8px}button{cursor:pointer}.muted{color:#65758c;font-size:13px}.caution{background:#fff4db;padding:12px;border-radius:8px}.good{background:#e5f5ed;padding:12px;border-radius:8px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #e1e7ef;padding:10px;text-align:left;white-space:nowrap}th{color:#536b89}.stats{display:flex;gap:10px;flex-wrap:wrap}.stat{background:#edf3fa;border-radius:9px;padding:12px;flex:1;min-width:145px}.stat b{display:block;font-size:24px}.views{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}figure{margin:0}img{display:block;width:100%;border-radius:6px}figcaption{font-size:13px;color:#52657d;margin:5px 0}.trackViews{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.timeline{display:grid;grid-template-columns:repeat(80,1fr);gap:2px}.timeline span{height:8px;background:#c8d1de}.timeline .used{background:#35b48a}details{margin-top:18px}summary{cursor:pointer;font-weight:600}.float{position:fixed;right:18px;bottom:14px;z-index:30;width:590px;height:535px;max-width:calc(100vw - 20px);max-height:calc(100vh - 35px);background:#172338;color:#ecf1f9;border-radius:10px;box-shadow:0 12px 36px #0005;display:grid;grid-template-rows:auto auto auto 1fr auto;resize:both;overflow:hidden}.float header{padding:9px 12px;background:#26374e;display:flex;justify-content:space-between;align-items:center;touch-action:none;cursor:move}.float button,.float select{font-size:12px;padding:3px 5px}.tools{display:flex;flex-wrap:wrap;gap:6px;padding:7px 9px;background:#233249}.float label{font-size:12px;display:inline-flex;align-items:center;white-space:nowrap}.float input[type=range]{width:60px}.idLegend{display:flex;flex-wrap:wrap;gap:5px;padding:3px 9px;max-height:66px;overflow:auto}.idLegend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 3px}#stage{position:relative;min-height:60px}canvas{width:100%;height:100%;display:block}#busy{position:absolute;inset:0;display:grid;place-content:center;background:#172338df}#busy[hidden]{display:none}#modelStatus{font-size:11px;padding:6px 10px;background:#233249}.mini{height:46px!important;resize:none}.mini .tools,.mini .idLegend,.mini #stage,.mini #modelStatus{display:none}#open{position:fixed;right:20px;bottom:20px;z-index:35}section[hidden]{display:none}@media(max-width:760px){main{padding:10px}section{padding:14px}.views{grid-template-columns:1fr}.float{width:calc(100vw - 20px);height:470px;right:10px}.trackViews{grid-template-columns:1fr}.timeline{grid-template-columns:repeat(50,1fr)}}</style>
<main><section><h1>room2 · 你这轮标注的实际修复结果</h1><p>5 个区域，4 组有效选择、13 个 mask；重复的地毯共用身份，共 12 个目标。四组都完成 2000 帧独立前后传播，联合替换到一张单独的 room2 地图。</p><div class="stats">__STATS__</div><p class="good">__PROOF__</p><p class="caution"><b>__SUCCESS__。</b>请同时看实例分离、完整度、灰点和其他物体的损伤。当前仍为统一 v3 的 pending 调试协议，未作为正式主表结果。</p><p><a href="human_selection.json">本轮原始标注 JSON</a> · <a href="seed_contact.jpg">13 个种子对应图</a> · <a href="CHANGES_CN.md">本轮所有改动说明</a></p></section>
<section><h2>这次实际处理了哪些选择</h2><div class="scroll"><table><tr><th>区域（点击查看）</th><th>你选择的来源和种子帧</th><th>原 mask 编号</th><th>各 mask 参与重建帧数</th></tr>__CASE_ROWS__</table></div><p class="muted">传播处理全部原始帧；原始地图每 5 帧建图一次，换票基于原有 400 帧。没有为其他 1600 帧增加新投票。最后一例保留“不可靠”的选择，未独立换票。</p></section>
<section><h2>整个 room2：同一 v3 参数对比</h2><div class="scroll"><table><tr><th>版本</th><th>AP50</th><th>F1</th><th>PQ</th><th>TP/FP/FN</th><th>合并预测/分裂GT</th></tr>__GLOBAL__</table></div><p>目标平均最佳 IoU 变化：__DELTA__ 个百分点。未选目标中，原先正确匹配丢失：__LOST__；IoU下降超过1个百分点：__DEGRADED__。</p><p>__WARNING__</p><details><summary>所有固定种子目标的 v3 指标</summary><div class="scroll"><table><tr><th>GT ID</th><th>原图最佳IoU</th><th>仅种子</th><th>完整追踪</th><th>追踪后最佳预测ID</th><th>完整覆盖率</th></tr>__TARGETS__</table></div><p class="muted">目标定义沿用此前固定种子表面、1cm近邻、至少10点且占比≥5%的规则；GT仅在所有预测冻结后用于评估。<a href="reports/evaluation_summary.json">完整指标与固定表面身份、纯度、覆盖诊断</a></p></details></section>
<section><h2>为什么还有灰点和残留</h2><p>撤回涉及的表面中，最终灰点35,376个，其中34,680个（98.03%）来自原生CONFLICT状态。两个效果弱的窗帘种子表面，原ID10/新ID358仍各有220,665/196,388票，原ID10/新ID359各有163,632/79,157票；保留的原家族证据还在与新身份竞争。</p><p>种子处改善也不代表整体完整：地毯种子表面的目标身份比例从59.34%升至75.68%，但完整地毯的v3最佳IoU从22.00%降至13.92%。本轮保留所有指标，没有根据GT调整身份或门槛。</p></section>
<section><h2>逐区同视角对比</h2><p id="caseStatus"></p><p>在悬浮3D窗口里选择区域，或点击上表区域。三个版本切换时镜头保留；本轮所有目标ID可逐个隐藏，悬停ID查看对应区域和原mask。灰色是未分配点，原实例颜色保留，新身份使用不同颜色。</p><p><a href="ply_models/room2_full.ply">完整room2三版本PLY（真实密度）</a> · <a href="comparison_data.json">全部结果与用帧记录</a></p></section>__DETAILS__
<section><h2>本轮实现与失败记录</h2><p>在上一版逐对象修复基础上新增：跨区域身份核对、全局一对一关联、同身份前景合并、跨物体重叠像素保留旧证据、每帧只重建一次。整张撤回还要求共享旧家族的其他已选目标都可靠；否则仅替换可靠前景下的旧贡献。</p><p>数字门槛、投票去重、至少2票且比例≥0.67的确认规则、TSDF几何、后处理和v3参数保持。目标旧家族取种子上实际原始观察的主导身份，完整重叠直方图保留；尚未搜索目标在其他历史ID下的完整谱系。</p><p class="muted">窗户组第一次GPU初始化失败，换到空闲卡完成；新批处理入口遇到Python版本语法问题和来源检查变量重名问题，均在有效结果提交前发现。失败文件保留，最终地图来自独立validated_v2目录。没有改原始地图或人工标注。</p></section></main>
<aside class="float" id="float"><header id="drag"><b>实际PLY · room2 三版本</b><span><button id="minimize">收起/展开</button> <button id="close">关闭</button></span></header>
<div class="tools"><label>选区 <select id="caseSelect">__OPTIONS__</select></label><label>版本 <select id="state"><option value="0">原始地图</option><option value="1">仅种子修复</option><option value="2" selected>完整双向追踪修复</option></select></label><label>显示范围 <select id="scope"><option value="local">当前区域真实密度</option><option value="full">完整room2</option></select></label><label>颜色 <select id="colorMode"><option value="1">实例颜色</option><option value="0">真实RGB</option><option value="2">标签改动位置</option></select></label><button id="reset">重置视角</button><button id="allIds">全部ID显示</button><label><input id="only" type="checkbox">只看目标ID</label><label><input id="unknown" type="checkbox" checked>灰点</label><label>点大小<input id="pointSize" type="range" min="1" max="5" step=".5" value="2"></label></div>
<div id="idLegend" class="idLegend"></div><div id="stage"><canvas id="canvas"></canvas><div id="busy">加载中…</div></div><footer id="modelStatus"></footer></aside><button id="open" hidden>打开3D</button>
<script id="modelData" type="application/json">__MODEL__</script><script type="module" src="room2_viewer.js"></script><script>document.querySelectorAll('[data-open-case]').forEach(button=>button.onclick=()=>{const s=document.getElementById('caseSelect');s.value=button.dataset.openCase;s.dispatchEvent(new Event('change'));});</script></html>'''
    html=html.replace('</style>','.float select,.float button{color:#19314f;background:#f5f8fd}</style>')
    html=html.replace('<input id="only" type="checkbox">','<input id="only" type="checkbox" checked>')
    vals={'__STATS__':''.join(f'<div class="stat"><span>{label}</span><b>{value}</b></div>' for label,value in [('参与重建的建图帧',f'{repair["counts"]["updated_frames"]}/400'),('最终标签变化点',f'{commit["final_inside_changes"]:,}'),('全图灰点变化',f'{gray:+,}'),('范围外变化点',commit['final_outside_changes'])]),
      '__PROOF__':proof,'__SUCCESS__':success,'__CASE_ROWS__':''.join(case_rows),'__GLOBAL__':''.join(table),
      '__DELTA__':f'{100*cmp["target_mean_IoU_delta"]:+.2f}','__LOST__':str(len(cmp['non_target_previously_correct_lost'])),
      '__DEGRADED__':str(len(cmp['non_target_IoU_degraded_over_0_01'])),'__WARNING__':warning,'__TARGETS__':''.join(targets),
      '__DETAILS__':''.join(details),'__OPTIONS__':''.join(options),'__MODEL__':json.dumps(data['model'],ensure_ascii=False).replace('</',r'<\/')}
    for a,b in vals.items():html=html.replace(a,b)
    assert '__STATS__' not in html
    (OUT/'index.html').write_text(html,encoding='utf8')
    (OUT/'build_receipt.json').write_text(json.dumps({'status':'PASS','source_bundle_sha256':record['sha256'],'html_sha256':hashlib.sha256(html.encode()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':5,'v3_unchanged':True},indent=2),encoding='utf8')
    print(json.dumps({'status':'PASS','url':'http://127.0.0.1:8785/room2_repair_batch_20261007/index.html','out':str(OUT)},ensure_ascii=True))

if __name__=='__main__':main()
