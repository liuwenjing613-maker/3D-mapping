"""Download a pinned NVIDIA checkpoint; verify official weight SHA256 before use."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    root=Path(cfg['model_directory']); root.mkdir(parents=True,exist_ok=True)
    assert root.resolve().is_relative_to(Path('/data/chenkejun/CVPR/models').resolve())
    start=time.perf_counter(); files={}
    for name in ['config.json','preprocessor_config.json','README.md','pytorch_model.bin']:
        source=f"https://huggingface.co/{cfg['model_id']}/resolve/{cfg['model_revision']}/{name}"
        url=source.replace('https://huggingface.co','https://hf-mirror.com')+'?download=true'
        target=root/name
        if target.exists():
            if name=='pytorch_model.bin': assert digest(target)==cfg['weight_sha256']
            print('EXISTING',name,flush=True)
        else:
            for attempt in range(3):
                try:
                    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
                    count=0; tick=0
                    with urllib.request.urlopen(req,timeout=45) as response, target.with_suffix(target.suffix+'.partial').open('wb') as out:
                        expected=response.headers.get('Content-Length')
                        while True:
                            block=response.read(1024*1024)
                            if not block: break
                            out.write(block); count+=len(block)
                            if count//(32*1024*1024)>tick:
                                tick=count//(32*1024*1024); print('DOWNLOAD',name,round(count/1e6,1),'MB',flush=True)
                    if expected is not None: assert count==int(expected),'Incomplete response'
                    tmp=target.with_suffix(target.suffix+'.partial')
                    if name=='pytorch_model.bin': assert digest(tmp)==cfg['weight_sha256'],'Official SHA256 mismatch'
                    elif name.endswith('.json'): json.loads(tmp.read_text())
                    tmp.replace(target); break
                except Exception as e:
                    print('DOWNLOAD_ERROR',name,attempt+1,type(e).__name__,str(e)[:220],flush=True)
                    if attempt==2: raise
                    time.sleep(2)
        files[name]={'sha256':digest(target),'bytes':target.stat().st_size,'official_url':source,'download_url':url}
        print('VERIFIED',name,files[name]['bytes'],files[name]['sha256'],flush=True)
    assert files['pytorch_model.bin']['sha256']==cfg['weight_sha256']
    metadata={'model_id':cfg['model_id'],'revision':cfg['model_revision'],'files':files,
              'weight_verification':'SHA256 matches official Hugging Face NVIDIA repository file page',
              'seconds':time.perf_counter()-start}
    (root/'download_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('DOWNLOAD_COMPLETE',json.dumps(metadata),flush=True)


if __name__=='__main__': main()
