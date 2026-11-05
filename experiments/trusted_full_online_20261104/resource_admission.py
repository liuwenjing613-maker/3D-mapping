"""RAM reservations for concurrent inference; GPU reservation remains unchanged."""
from pathlib import Path
import contextlib,fcntl,json,os,time,uuid,threading
import psutil
ROOT=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104/resource_leases')
LOCAL=threading.local()
GIB=1024**3

def current_lease():
    return getattr(LOCAL,'key','')

def live(row):
    try:
        return abs(psutil.Process(row['pid']).create_time()-row['created'])<1
    except psutil.Error:return False

def tagged_rss(rows):
    result={key:0 for key in rows}
    for pid in {row['pid'] for row in rows.values()}:
        try:children=psutil.Process(pid).children(recursive=True)
        except psutil.Error:continue
        for child in children:
            try:
                key=child.environ().get('CVPR_RAM_LEASE')
                if key in result:result[key]+=child.memory_info().rss
            except (psutil.Error,OSError):pass
    return result

@contextlib.contextmanager
def inference_memory(checkpoint,status_root,required_free):
    ROOT.mkdir(parents=True,exist_ok=True)
    key=uuid.uuid4().hex
    # Dense tracking retains the arrived RGB prefix. Keep headroom for CPU mapping/repair.
    is_tracking='tracking' in Path(status_root).parts
    reserve=((10+.021*(checkpoint*5)) if is_tracking else 12)*GIB
    margin=40*GIB
    started=time.time()
    while True:
        with (ROOT/'memory.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=ROOT/'leases.json'
            rows=json.loads(path.read_text()) if path.exists() else {}
            rows={k:r for k,r in rows.items() if live(r)}
            rss=tagged_rss(rows)
            pending=sum(max(0,r['reserve_bytes']-rss.get(k,0)) for k,r in rows.items())
            available=psutil.virtual_memory().available
            accepted=available>=reserve+pending+margin
            if accepted:
                rows[key]=dict(pid=os.getpid(),created=psutil.Process().create_time(),reserve_bytes=reserve,checkpoint=checkpoint,started=time.time())
            path.write_text(json.dumps(rows))
            snapshot=dict(status='ACQUIRED' if accepted else 'WAITING',available_gib=round(available/GIB,2),reserved_unallocated_gib=round(pending/GIB,2),request_gib=round(reserve/GIB,2),headroom_gib=40,checkpoint=checkpoint,seconds=time.time()-started)
            target=Path(status_root)/('ram_wait_'+str(threading.get_ident())+'.json')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(snapshot))
        if accepted:break
        time.sleep(10)
    LOCAL.key=key
    try:yield
    finally:
        LOCAL.key=''
        with (ROOT/'memory.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            rows=json.loads((ROOT/'leases.json').read_text())
            rows.pop(key,None)
            (ROOT/'leases.json').write_text(json.dumps(rows))
