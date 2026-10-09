"""Build two exact-mask continuity previews, sharing the existing verified RGB/mask packs."""
from pathlib import Path
import hashlib
import html
import json

WORK = Path(__file__).resolve().parent
BASE = WORK.parents[1] / 'results' / '固定案例_三模型对比_20261006'
WEB = BASE / 'continuous_tracking_ablation_20261008'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Expected a unique donor fragment: ' + old[:100])
    return text.replace(old, new)


def first_reappearance(meta, window):
    oid = window['track']
    result = []
    for name, step in [('forward', 1), ('reverse', -1)]:
        stop = window['directions'][name]['stop_frame']
        if stop is None:
            continue
        fid = next((f for f in range(stop + step, 2000 if step > 0 else -1, step)
                    if meta['frames'][f]['areas'][oid - 1] > 0), None)
        if fid is not None:
            result.append({'direction': name, 'stop_frame': stop, 'reappearance_frame': fid,
                           'raw_area': meta['frames'][fid]['areas'][oid - 1]})
    return result


def band(meta, window):
    oid = window['track']
    runs = []
    start = None
    for fid in range(2001):
        exists = fid < 2000 and meta['frames'][fid]['areas'][oid - 1] > 0
        if exists and start is None:
            start = fid
        if not exists and start is not None:
            runs.append((start, fid))
            start = None
    a, b = window['allowed_frame_range']
    raw = ''.join(f'<rect x="{s/4:.2f}" y="3" width="{max(.25,(e-s)/4):.2f}" height="4" fill="#90a5b9"/>' for s, e in runs)
    kept = ''.join(f'<rect x="{max(s,a)/4:.2f}" y="10" width="{max(.25,(min(e,b+1)-max(s,a))/4):.2f}" height="5" fill="#1a9471"/>' for s, e in runs if max(s,a)<min(e,b+1))
    return f'<svg viewBox="0 0 500 20" role="img" aria-label="灰色为原始非空mask帧，绿色为保留帧，红线为种子"><rect width="500" height="20" fill="#eef2f6"/>{raw}{kept}<path d="M{window["seed_frame"]/4} 0v20" stroke="#cb4764" stroke-width="1.5"/></svg>'


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    all_summary = {}
    sections = []
    for scene in ['room0', 'room2']:
        donor = BASE / (scene + '_tracking_playback_20261008')
        source = donor / 'playback_data.json'
        meta = json.loads(source.read_text(encoding='utf-8'))
        policies = {f'gap{gap}': json.loads((WORK / f'{scene}_gap{gap}_windows.json').read_text(encoding='utf-8')) for gap in (0, 1)}
        assert meta['status'] == 'PASS' and meta['frames_count'] == 2000
        for chunk in meta['chunks']:
            chunk['path'] = '../' + donor.name + '/' + chunk['path']
        meta['continuity_policies'] = policies
        meta['source_metadata_sha256'] = sha(source)
        meta['preview_only'] = True
        meta['repair_or_evaluation_rerun'] = False
        dump(WEB / (scene + '_preview_data.json'), meta)
        stats = {}
        for key, report in policies.items():
            windows = report['tracks']
            raw = sum(w['raw_nonempty_frames'] for w in windows)
            retained = sum(w['retained_nonempty_frames'] for w in windows)
            stats[key] = {'raw_nonempty_mask_frames': raw, 'retained_nonempty_mask_frames': retained,
                          'retained_fraction': retained/raw,
                          'previously_accepted_mapping_mask_frames_removed': sum(len(w['previously_accepted_mapping_frames_removed']) for w in windows),
                          'original_accepted_mapping_mask_frames': sum(len(f['accepted']) for f in meta['frames'])}
        all_summary[scene] = stats
        rows = []
        for w in policies['gap1']['tracks']:
            reappear = first_reappearance(meta, w)
            chosen = next((r for r in reappear if r['direction'] == 'forward'), reappear[0] if reappear else None)
            frame = chosen['reappearance_frame'] if chosen else w['seed_frame']
            url = f'playback.html?scene={scene}&track={w["track"]}&frame={frame}&policy=unrestricted'
            events = '；'.join(f'{"前向" if r["direction"] == "forward" else "后向"} f{r["stop_frame"]} 停止，f{r["reappearance_frame"]} 又出现' for r in reappear) or '停止方向之后没有非空mask'
            rows.append(f'<tr><td><a href="{url}">{html.escape(w["ROI"])}<br>mask{w["mask"]} → ID{w["persistent_id"]}</a></td><td>f{w["seed_frame"]}</td><td>{band(meta,w)}<span>保留 f{w["allowed_frame_range"][0]}–f{w["allowed_frame_range"][1]}</span></td><td>{w["raw_nonempty_frames"]} → <strong>{w["retained_nonempty_frames"]}</strong></td><td>{len(w["previously_accepted_mapping_frames_removed"])}</td><td>{events}</td></tr>')
        s0, s1 = stats['gap0'], stats['gap1']
        sections.append(f'''<section><div class="scene-head"><h2>{scene} · {len(meta['tracks'])} 条轨迹</h2><a class="button" href="playback.html?scene={scene}&policy=gap1">打开该场景播放对照</a></div>
<div class="cards"><div>原轨迹非空 mask 帧<strong>{s1['raw_nonempty_mask_frames']:,}</strong></div><div>一空帧即停<strong>{s0['retained_nonempty_mask_frames']:,}（{s0['retained_fraction']:.1%}）</strong></div><div>允许空一帧<strong>{s1['retained_nonempty_mask_frames']:,}（{s1['retained_fraction']:.1%}）</strong></div><div>截除原已通过的mask×建图帧<strong>{s1['previously_accepted_mapping_mask_frames_removed']} / {s1['original_accepted_mapping_mask_frames']}</strong></div></div>
<p class="muted">下面显示“允许空一帧”的结果。灰条＝原始非空帧；绿条＝保留非空帧；红线＝人工种子。点击目标直接检查断档后重新出现的画面，再切换策略对照。</p>
<div class="table-wrap"><table><thead><tr><th>目标</th><th>种子帧</th><th>全部2000帧中的非空记录与保留区间</th><th>非空帧数量</th><th>截除已通过帧</th><th>断档后重新出现（未经正确性判定）</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>''')

    template = (BASE / 'room2_tracking_playback_20261008/index.html').read_text(encoding='utf-8')
    start = template.index('<header class="head">')
    end = template.index('</header>', start) + len('</header>')
    template = template[:start] + '''<header class="head"><h1 id="pageTitle">连续空帧截断 · mask播放对照</h1>
<p>同一张原始RGB，切换原轨迹与截断方案；全部2000帧按原顺序保留。</p>
<p class="notice" id="experimentNote">这是实际保存mask的精确截断预览。尚未重新换票或得到新v3指标；被截除的mask不能自动判为错误。</p>
<div class="row"><label>场景 <select id="scene"><option value="room0">room0</option><option value="room2">room2</option></select></label>
<label>策略 <select id="policy"><option value="unrestricted">原轨迹：允许断档后继续</option><option value="gap0">严格连续：1帧空就停</option><option value="gap1" selected>允许空1帧：连续2帧空就停</option></select></label><a href="index.html">返回全部目标与截断统计</a></div>
<p id="policySummary" class="muted"></p></header>''' + template[end:]
    template = replace_once(template, '<title>room2 · 全部实例实际追踪回放</title>', '<title>room0 / room2 · 连续空帧截断对照</title>')
    template = replace_once(template, 'id="skipEmpty" checked', 'id="skipEmpty"')
    template = replace_once(template, '<option value="used">本帧用于换票的前景mask</option>', '')
    template = replace_once(template, '<option value="raw">全部实际追踪mask</option>', '<option value="raw">当前策略保留的追踪mask</option>')
    screen = '<div class="screen"><canvas id="canvas" width="1200" height="680" aria-label="当前原始RGB帧与实际追踪mask"></canvas><div id="busy">读取回放记录…</div><div id="hover"></div></div>'
    template = replace_once(template,screen,'<div class="comparison" id="comparison"><div id="rawPanel"><p class="compare-label" id="rawLabel">原始SAM轨迹</p><div class="screen"><canvas id="rawCanvas" width="1200" height="680" aria-label="同一帧原始SAM轨迹"></canvas></div></div><div><p class="compare-label" id="filteredLabel">当前策略</p>'+screen+'</div></div>')
    template = replace_once(template,'<div class="row"><label>显示','<div class="row"><label><input id="compare" type="checkbox" checked>同帧并排对比</label><label>显示')
    template = replace_once(template,'</style>','.comparison{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}.comparison .screen{height:auto;aspect-ratio:1200/680}.comparison .screen canvas{width:100%;height:100%}.compare-label{font-size:13px;margin:0;padding:5px 8px;background:#edf3f6;min-height:48px}.comparison.large,.comparison.single{grid-template-columns:1fr}#rawPanel[hidden]{display:none}@media(max-width:650px){.comparison{grid-template-columns:1fr}}</style>')
    template = replace_once(template, '颜色对应修复地图的实例ID。多条轨迹重叠时用条纹显示所有被勾选的mask；共用同一身份的轨迹保留各自开关。换票视图只显示通过检查且扣除跨物体冲突像素的前景，真正投票还要求深度有效并投影到固定表面。', '每个目标从自己的人工种子分别向前、向后判定。达到空帧限制后，该方向永久停止；其他目标独立继续。保留mask的像素、颜色和身份不变，重叠用条纹显示。此页没有新投票结果。')
    template = replace_once(template, '<button id="blinds">只看百叶窗</button><button id="floor">只看地面局部</button>', '<button id="blinds">只看百叶窗</button><button id="floor">只看地面局部</button>')
    # This donor-specific note becomes a scene-aware note in JavaScript.
    note = 'ROI-C0004、ROI-C0006、ROI-C0010、ROI-C0014 标为不可靠，没有独立追踪任务。另有12处未标注。本轮14组的25条实际轨迹全部可看。'
    template = replace_once(template, note, '<span id="sceneNote"></span>')
    template = replace_once(template, 'src="playback.js"', 'src="continuity_playback.js"')
    (WEB / 'playback.html').write_text(template, encoding='utf-8')

    js = (BASE / 'room2_tracking_playback_20261008/playback.js').read_text(encoding='utf-8')
    js = replace_once(js, "const STORE='room2_tracking_playback_20261008';", "const query=new URLSearchParams(location.search);\nconst SCENE=['room0','room2'].includes(query.get('scene'))?query.get('scene'):'room2';\nconst STORE='continuous_tracking_preview_'+SCENE;")
    js = replace_once(js, 'function selectedBits(){let bits=0;for(const oid of selected)bits|=1<<(oid-1);return bits>>>0;}', '''function currentWindow(oid){const name=el('policy').value;if(name==='unrestricted')return null;return data.continuity_policies[name].tracks.find(t=>t.track===oid);}
function allowed(oid,fid){const w=currentWindow(oid);return !w||(fid>=w.allowed_frame_range[0]&&fid<=w.allowed_frame_range[1]);}
function selectedBits(){let bits=0;for(const oid of selected)if(allowed(oid,current))bits|=1<<(oid-1);return bits>>>0;}''')
    js = replace_once(js, "function activeArea(frame){const used=el('mode').value==='used';return [...selected].reduce((sum,oid)=>sum+(used?frame.used_areas:frame.areas)[oid-1],0);}", "function activeArea(frame){return [...selected].reduce((sum,oid)=>sum+(allowed(oid,frame.frame)?frame.areas[oid-1]:0),0);}")
    js = replace_once(js, "mode:el('mode').value,loop:", "mode:el('mode').value,policy:el('policy').value,loop:")
    js = replace_once(js, "for(const id of ['fps','mode','opacity'])", "for(const id of ['fps','mode','opacity','policy'])")
    js = replace_once(js, "const present=data.tracks.filter(t=>selected.has(t.track)&&((mode==='used'?meta.used_areas:meta.areas)[t.track-1]>0));", "const present=data.tracks.filter(t=>selected.has(t.track)&&allowed(t.track,current)&&meta.areas[t.track-1]>0);")
    a = js.index("  el('frameState').textContent=")
    b = js.index("  for(const t of data.tracks){", a)
    js = js[:a] + '''  const stopped=data.tracks.filter(t=>selected.has(t.track)&&!allowed(t.track,current));
  el('frameState').textContent=el('policy').value==='unrestricted'?'原始轨迹 · 无空帧截断':`截断预览 · 本帧已排除 ${stopped.length} 条`;
  el('frameState').className='badge';
  el('frameNote').textContent='空帧指原始分辨率保存mask的像素数为0。停止从各自种子向外计数，不按建图每5帧计数；不同ROI或共用身份的轨迹各自独立停止。';
  const report=data.continuity_policies[el('policy').value];
  el('policySummary').textContent=report?`${SCENE} · ${data.tracks.length}条轨迹 · 当前规则最多容许连续${report.max_empty_gap}空帧 · 到达停止条件后该方向不再恢复。`:`${SCENE} · ${data.tracks.length}条原轨迹 · 可查看消失以后重新出现的实际结果。`;
''' + js[b:]
    js = replace_once(js, "if(!area){tag.textContent='本帧无mask';", "if(!allowed(t.track,current)){const w=currentWindow(t.track);tag.textContent='断轨截除'+(area?' · 原有mask':'');tag.className='tag rejected';tag.title=`保留 f${w.allowed_frame_range[0]}–f${w.allowed_frame_range[1]}；原始本帧${area}像素。此方向已停止，不重新关联。`;}\n    else if(!area){tag.textContent='本帧无mask';")
    js = replace_once(js, "tag.textContent='本帧参与换票'", "tag.textContent='保留 · 原方案曾通过门槛'")
    js = replace_once(js, "tag.textContent='有mask · 未用于换票'", "tag.textContent='保留 · 原方案未通过门槛'")
    js = replace_once(js, "el('play').onclick=play;", "el('scene').value=SCENE;el('scene').onchange=()=>{pause();location.href='playback.html?scene='+el('scene').value+'&policy='+el('policy').value;};el('policy').onchange=()=>{pause();updateEligible();paint();prefetch();};\n  el('play').onclick=play;")
    js = replace_once(js, "['ROI-C0001','ROI-C0008'].includes(t.ROI)", "(SCENE==='room2'?['ROI-C0001','ROI-C0008']:['ROI-C0011']).includes(t.ROI)")
    js = replace_once(js, "[7,8,13,14,15].includes(t.track)", "(SCENE==='room2'?[7,8,13,14,15]:[]).includes(t.track)")
    js = replace_once(js, '`room2_${formattedFrame(current)}_${el(\'mode\').value}.png`', '`'+ '${SCENE}_${formattedFrame(current)}_${el(\'policy\').value}.png`')
    js = replace_once(js, "fetch('playback_data.json')", "fetch(SCENE+'_preview_data.json')")
    js = replace_once(js, "data.tracks.length!==25", "data.tracks.length!==(SCENE==='room0'?30:25)")
    js = replace_once(js, "restorePreferences();makeLegend();", "restorePreferences();if(['gap0','gap1','unrestricted'].includes(query.get('policy')))el('policy').value=query.get('policy');if(query.has('frame'))current=safeFrame(query.get('frame'));if(query.has('track')&&data.tracks.some(t=>t.track===Number(query.get('track'))))selected=new Set([Number(query.get('track'))]);el('pageTitle').textContent=SCENE+' · 连续空帧截断对照';el('floor').hidden=SCENE==='room0';el('sceneNote').textContent='只使用本轮实际保存的原始轨迹；没有增加目标、重新分配像素或更改原结果。';makeLegend();")
    js = replace_once(js, "el('verification').textContent='已核对2000张原始RGB、28000张追踪PNG；mask无损还原，重叠保留。所有400个建图帧的冲突扣除与修复记录一致。播放速度是演示设置，不代表采集帧率。';", "el('verification').textContent='复用已经校验的2000张原RGB和原始分辨率mask。只按冻结的连续性区间筛除轨迹位；保留像素与原结果完全相同。此页不代表新的投票、地图或v3结果。';")
    js = replace_once(js,"const canvas = el('canvas'), ctx = canvas.getContext('2d', {alpha:false});","const canvas = el('canvas'), ctx = canvas.getContext('2d', {alpha:false});\nconst rawCanvas=el('rawCanvas'),rawCtx=rawCanvas.getContext('2d',{alpha:false});")
    a = js.index('function paint(){')
    b = js.index('  const present=data.tracks.filter',a)
    paint_prefix = js[a:b]
    render = replace_once(paint_prefix,"function paint(){\n  if(!currentImage)return;const {image,meta}=currentImage;ctx.drawImage(image,0,0);const mode=el('mode').value;","function renderComparison(destination,pixels,image,bits){\n  destination.drawImage(image,0,0);const mode=el('mode').value;")
    render = replace_once(render,"const pixels=mode==='used'?currentImage.used:currentImage.raw,bits=selectedBits(),targetColors=new Map();","const targetColors=new Map();")
    render = render.replace('ctx.drawImage(overlay,0,0)','destination.drawImage(overlay,0,0)')
    render += '''}
function rawSelectionBits(){let bits=0;for(const oid of selected)bits|=1<<(oid-1);return bits>>>0;}
function paint(){
  if(!currentImage)return;const {image,meta}=currentImage,mode=el('mode').value;
  el('rawPanel').hidden=!el('compare').checked;el('comparison').classList.toggle('single',!el('compare').checked);
  renderComparison(ctx,currentImage.raw,image,selectedBits());
  if(el('compare').checked)renderComparison(rawCtx,currentImage.raw,image,rawSelectionBits());
'''
    js = js[:a] + render + js[b:]
    js = replace_once(js,"el('visibleCount').textContent=`当前勾选 ${selected.size}/${data.tracks.length} 条 · 本帧非空 ${present.length} 条`;","el('visibleCount').textContent=`当前勾选 ${selected.size}/${data.tracks.length} 条 · 本帧非空 ${present.length} 条`;\n  el('rawLabel').textContent='原始SAM轨迹 · 本帧'+data.tracks.filter(t=>selected.has(t.track)&&meta.areas[t.track-1]>0).length+'个非空mask';\n  el('filteredLabel').textContent=el('policy').selectedOptions[0].textContent+' · 本帧'+present.length+'个非空mask';")
    js = replace_once(js,"el('enlarge').onclick=()=>{const screen=canvas.parentElement;screen.classList.toggle('large');el('enlarge').textContent=screen.classList.contains('large')?'适应窗口':'放大画面';};","el('compare').onchange=paint;el('enlarge').onclick=()=>{const pair=el('comparison');pair.classList.toggle('large');el('enlarge').textContent=pair.classList.contains('large')?'并排适应窗口':'上下放大画面';};")
    js = replace_once(js,"selected=new Set(data.tracks.map(t=>t.track));restorePreferences();","try{const response=await fetch('experiment_status.json');if(response.ok){const state=await response.json();if(state.status==='RUNNING')el('experimentNote').textContent='正在重建投票与地图，完整v3待完成。当前画面是原轨迹与截断mask同帧对照；被截除结果不能自动判为错误。';else if(state.status==='PASS')el('experimentNote').textContent='完整换票、扩散和统一v3对照已完成，地图及指标见总览页。当前只比较同一帧的mask，左边原轨迹、右边当前策略。';}}catch{}\n  selected=new Set(data.tracks.map(t=>t.track));restorePreferences();")
    (WEB / 'continuity_playback.js').write_text(js, encoding='utf-8')

    overview = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>room0 / room2 · 连续空帧截断实验</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf2f6;color:#193044;font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:20px}h1{font-size:26px;margin:0}h2{font-size:21px;margin:0}p{margin:8px 0}a{color:#146d86}.notice{padding:12px 15px;background:#fff0d1;border:1px solid #e8c677;border-radius:9px}.rules{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}.rules>div,section{background:white;border:1px solid #d0dce6;border-radius:10px;padding:15px}section{margin-top:20px}.scene-head{display:flex;justify-content:space-between;align-items:center;gap:10px}.button{display:inline-block;background:#176c83;color:white;padding:7px 12px;border-radius:6px;text-decoration:none}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}.cards>div{background:#edf4f7;padding:12px;border-radius:8px;font-size:13px}.cards strong{display:block;font-size:23px;margin-top:4px}.muted{font-size:13px;color:#607688}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dfE7ed;padding:9px;text-align:left;vertical-align:middle}th{background:#f2f5f8}td:nth-child(3){min-width:260px;width:34%}svg{display:block;width:100%;height:24px}td:nth-child(3)>span{display:block;color:#497065}td:first-child{min-width:130px}td:last-child{min-width:200px}code{font-size:13px}@media(max-width:900px){.cards,.rules{grid-template-columns:1fr 1fr}.cards strong{font-size:19px}}</style>
<main><h1>room0 / room2 · 连续空帧截断实验</h1><p>检查消失后重新出现的mask，并比较“严格连续”和“允许空1帧”。</p>
<p class="notice"><strong>当前阶段：mask截断预览已完成；新换票、扩散、3D和v3尚未运行。</strong>完整地图对照待执行，当前已有地图没有被改动。下表统计的是被筛除的原mask，不能把所有被截除结果都判为追错物体。</p>
<div class="rules"><div><strong>严格连续</strong><p>从种子向前、向后独立检查；遇到第一帧空mask，永久停止该方向。</p></div><div><strong>允许空1帧</strong><p>允许单帧缺失；出现连续两帧空mask，永久停止该方向。别的mask继续。</p></div></div>
<p class="muted">空mask＝原始1200×680保存mask的像素数为0，沿用原来多目标最大logit分配后的结果。按所有原始帧计数，未按每5帧建图间隔计数。共用身份的多条轨迹仍独立判断。规则与阈值在任何新GT评估前冻结。</p>
''' + ''.join(sections) + '''<p class="muted">判断：适合阻断断档后的无依据续接，不能检测“始终非空但已经追偏”。2帧是工程上的保守阈值，不等于身份正确性证据。覆盖损失很大，需与原方案比较统一v3、目标覆盖、分裂/合并和新增未分配点，再决定是否采用。</p>
<p class="muted"><a href="experiment_summary.json">截断统计JSON</a> · <a href="../room0_tracking_playback_20261008/index.html">room0原轨迹</a> · <a href="../room2_tracking_playback_20261008/index.html">room2原轨迹</a></p></main></html>'''
    (WEB / 'index.html').write_text(overview, encoding='utf-8')
    dump(WEB / 'experiment_summary.json', {'status':'MASK_PREVIEW_COMPLETE_REPAIR_AND_V3_NOT_RUN',
        'definition': 'Original saved native mask area == 0; original consecutive frames; per-track independent seed-outward directions; permanent stop without reacquisition',
        'conditions': ['unrestricted_reference', 'gap0', 'gap1'], 'scenes': all_summary,
        'GT_used_for_filter': False, 'SAM_rerun': False, 'old_maps_modified': False,
        'replacement_repair_and_v3_executed': False})
    dump(WORK / 'preview_delivery.json', {'status':'PASS','URL':'http://127.0.0.1:8785/continuous_tracking_ablation_20261008/index.html',
         'source_files': {str(BASE/(s+'_tracking_playback_20261008')/'playback_data.json'):sha(BASE/(s+'_tracking_playback_20261008')/'playback_data.json') for s in ['room0','room2']},
         'new_web_files': {p.name:sha(p) for p in WEB.iterdir() if p.is_file()},
         'old_maps_modified':False,'repair_and_v3_executed':False})
    print(json.dumps({'status':'PASS','web':str(WEB),'summary':all_summary},ensure_ascii=False))


if __name__ == '__main__':
    main()
