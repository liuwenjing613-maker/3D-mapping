"""Vendor a pinned official Three.js build for an offline PLY viewer."""
from pathlib import Path
import hashlib, json, urllib.request, sys

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[2]/'results/固定案例_三模型对比_20261006/tracking_review_20261007/vendor'
ROOT.mkdir(exist_ok=True)
record={}
for source,target in [('build/three.module.js','three.module.js'),('build/three.core.js','three.core.js'),('examples/jsm/controls/OrbitControls.js','OrbitControls.js'),('LICENSE','THREE_LICENSE.txt')]:
    url='https://raw.githubusercontent.com/mrdoob/three.js/r180/'+source
    data=urllib.request.urlopen(url,timeout=30).read()
    if target=='OrbitControls.js':
        data=data.replace(b"from 'three';",b"from './three.module.js';")
    (ROOT/target).write_bytes(data)
    record[target]={'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
(ROOT/'provenance.json').write_text(json.dumps({'version':'r180','source':'official mrdoob/three.js','files':record},indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','files':len(record),'bytes':sum(r['bytes'] for r in record.values())}))
