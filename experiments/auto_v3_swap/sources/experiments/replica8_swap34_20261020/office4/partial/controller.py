from pathlib import Path
import json
import os
import subprocess
import sys
import time
import traceback

from bootstrap import C, R


def dump(name, value):
    p = R / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(p)


def main():
    for stage, args in [('tests', ['-m', 'unittest', 'test_partial_policy', '-v']),
                        ('mapping', [str(C / 'run_partial_replay.py')]),
                        ('evaluation', [str(C / 'evaluate_r2.py')])]:
        dump('run_status.json', {'status': 'RUNNING', 'stage': stage, 'unix': time.time()})
        with (R / (stage + '.log')).open('x') as log:
            p = subprocess.Popen([sys.executable, '-B', '-u', *args], cwd=C, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
            assert p.wait() == 0, stage + ' failed; inspect ' + str(R / (stage + '.log'))
        if stage == 'tests':
            dump('tests_complete.json', {'status': 'PASS', 'tests': 3})
    dump('run_status.json', {'status': 'PASS', 'stage': 'complete', 'unix': time.time()})
    dump('run_complete.json', {'status': 'PASS', 'unix': time.time()})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        dump('run_failure.json', {'traceback': traceback.format_exc()})
        raise
