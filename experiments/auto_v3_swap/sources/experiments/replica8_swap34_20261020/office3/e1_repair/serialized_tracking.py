from pathlib import Path
import fcntl, subprocess, sys
C=Path(__file__).resolve().parent
with open('/data/chenkejun/CVPR/results/replica8_swap34_20261020/tracking_queue.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    subprocess.run([sys.executable,'-B','-u',str(C/'tracking_queue_v2.py')],check=True,cwd=C)
