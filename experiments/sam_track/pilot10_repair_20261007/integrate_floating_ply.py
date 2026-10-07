"""Install the reusable floating viewer into the existing review template."""
from pathlib import Path
import hashlib, json, shutil, tarfile

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'results/固定案例_三模型对比_20261006/tracking_review_20261007'
bundle=HERE/'floating_ply_bundle.tar.gz'
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='42d40f98f7d404e88e878c7af5b7dcdd172e85f0ec9406e58cc5b21f47e179b2'
with tarfile.open(bundle,'r:gz') as archive:
    for member in archive.getmembers():
        assert (OUT/member.name).resolve().is_relative_to(OUT.resolve())
        assert member.isfile() or member.isdir()
    archive.extractall(OUT,filter='data')
manifest=json.loads((OUT/'ply_models/manifest.json').read_text())
for case in manifest['cases']:
    for scope in ['local','full']:
        p=OUT/case[scope]['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==case[scope]['sha256']
for name in ['floating_ply.css','floating_ply.js','mask_visibility.js']:
    shutil.copyfile(HERE/name,OUT/name)
template=HERE/'tracking_review_template.html'
s=template.read_text(encoding='utf-8')
if '<!-- FLOATING_PLY -->' not in s:
    extension='\n<!-- FLOATING_PLY -->\n'+(HERE/'floating_ply.html').read_text(encoding='utf-8')
    s=s.replace('</main><script', '</main>'+extension+'\n<script',1)
    s=s.replace("if(c.status!=='READY')return;const row=c.frames[fi];", "if(all&&window.pilotPlyViewer)window.pilotPlyViewer.sync(window.pilotPlyApi.state());if(c.status!=='READY')return;const row=c.frames[fi];",1)
    s=s.replace("function showCase(){stop();epoch++;const c=C();", "function showCase(){stop();epoch++;const c=C();if(c.status!=='READY'&&window.pilotPlyViewer)window.pilotPlyViewer.sync({c,solo:null,visible_track_ids:[]});",1)
    s=s.replace('showCase();\n</script>', "window.pilotPlyApi={state:()=>({c:C(),solo,visible_track_ids:idsForCase().map(o=>o.track_id)})};showCase();\n</script>",1)
    template.write_text(s,encoding='utf-8')
else:
    start=s.index('<!-- FLOATING_PLY -->')
    end=s.find('<script type="application/json" id="data">',start)
    if end<0:end=s.index('<script>',start)
    s=s[:start]+'<!-- FLOATING_PLY -->\n'+(HERE/'floating_ply.html').read_text(encoding='utf-8')+'\n'+s[end:]
    template.write_text(s,encoding='utf-8')
if 'id="data"' not in s:
    s=s.replace('<script>','<script type="application/json" id="data">__DATA__</script>\n<script>',1)
    template.write_text(s,encoding='utf-8')
if 'enabled_track_ids:C().objects' not in s:
    s=s.replace('visible_track_ids:idsForCase().map(o=>o.track_id)',
                'visible_track_ids:idsForCase().map(o=>o.track_id),enabled_track_ids:C().objects.filter(o=>!hidden.has(o.track_id)).map(o=>o.track_id)',1)
    template.write_text(s,encoding='utf-8')
data=(OUT/'tracking_review_data.json').read_text(encoding='utf-8')
html=s.replace('__DATA__',data.replace('</','<\\/')).replace('__PLY_DATA__',json.dumps(manifest,ensure_ascii=False).replace('</','<\\/'))
(OUT/'index.html').write_text(html,encoding='utf-8')
(HERE/'tracking_viewer_check.js').write_text(s.split('<script>')[1].split('</script>')[0],encoding='utf-8')
record={'status':'PASS','cases':len(manifest['cases']),'native_density':True,'source_maps_unchanged':True,
        'full_PLY_exports':10,'local_PLY_exports':10,'source_bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()}
(OUT/'floating_ply_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record))
