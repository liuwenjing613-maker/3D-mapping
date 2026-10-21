from pathlib import Path
import argparse,json,os,socket,subprocess,sys,time,urllib.request,webbrowser
ROOT=Path(__file__).resolve().parent
def healthy(port):
 try:
  with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f'http://127.0.0.1:{port}/api/health',timeout=.5) as r:
   x=json.load(r);return x.get('app')=='partial frame inspector' and x.get('root')==str(ROOT)
 except (OSError,ValueError):return False
def main():
 p=argparse.ArgumentParser();p.add_argument('--no-browser',action='store_true');args=p.parse_args()
 for port in range(8917,8927):
  if healthy(port):break
  with socket.socket() as probe:
   try:probe.bind(('127.0.0.1',port))
   except OSError:continue
  with (ROOT/'server.log').open('ab') as log:
   kwargs=dict(stdin=subprocess.DEVNULL,stdout=log,stderr=log,cwd=str(ROOT))
   if os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS
   else:kwargs['start_new_session']=True
   proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'serve.py'),'--port',str(port)],**kwargs);(ROOT/'server.pid').write_text(str(proc.pid))
  for _ in range(40):
   if healthy(port):break
   time.sleep(.15)
  if not healthy(port):raise RuntimeError('Local player did not start; inspect server.log')
  break
 else:raise RuntimeError('No free local port')
 url=f'http://127.0.0.1:{port}/?run=manual_room1&id=417&frame=390&stage=0&region=9';print(url)
 if not args.no_browser:webbrowser.open(url)
if __name__=='__main__':main()
