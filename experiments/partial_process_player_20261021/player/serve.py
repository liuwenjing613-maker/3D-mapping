"""Read-only local HTTP server. No external listening address or mutation API."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import argparse,json
ROOT=Path(__file__).resolve().parent
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT),**kwargs)
 def do_GET(self):
  if self.path=='/api/health':
   data=json.dumps(dict(app='partial frame inspector',root=str(ROOT))).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
  else:super().do_GET()
 def log_message(self,fmt,*args):
  if len(args)>1 and str(args[1]) not in ('200','304'):super().log_message(fmt,*args)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8917);a=p.parse_args();print(f'http://127.0.0.1:{a.port}/',flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
