from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

C = Path(__file__).resolve().parent
R = Path('/data/chenkejun/CVPR/results/room1_swap34_20261020')

def save(name, data):
    p = R / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)

def main():
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4',
           'PYTHONDONTWRITEBYTECODE': '1'}
    stages = [('selection', C / 'v2/select_reused.py'),
              ('original_E1', C / 'e1_repair/run_selected.py'),
              ('partial', C / 'partial/controller.py')]
    for name, script in stages:
        save('status.json', dict(status='RUNNING', stage=name, unix=time.time()))
        with (R / (name + '.controller.log')).open('x') as stream:
            process = subprocess.Popen([sys.executable, '-B', '-u', str(script)], cwd=script.parent,
                env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            assert process.wait() == 0, name + ' failed'
    save('complete.json', dict(status='PASS', unix=time.time(), scope='room1'))
    save('status.json', dict(status='PASS', stage='complete', unix=time.time()))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        save('failure.json', dict(status='FAIL', traceback=traceback.format_exc()))
        raise
