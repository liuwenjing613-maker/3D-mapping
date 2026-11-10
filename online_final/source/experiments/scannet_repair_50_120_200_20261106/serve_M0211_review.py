from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import functools,json,os
root=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106/review_M0211')
handler=functools.partial(SimpleHTTPRequestHandler,directory=str(root))
s=ThreadingHTTPServer(('127.0.0.1',0),handler)
(root/'http_state.json').write_text(json.dumps({'status':'RUNNING','bind':'127.0.0.1','port':s.server_port,'pid':os.getpid()}))
s.serve_forever()
