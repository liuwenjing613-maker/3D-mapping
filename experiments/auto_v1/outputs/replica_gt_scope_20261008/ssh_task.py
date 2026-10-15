"""Run task files on the authorized CVPR server; never store credentials."""
import argparse
from pathlib import Path
import re
import shlex
import sys
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import paramiko

ROOT = Path(__file__).resolve().parents[2]


def connect():
    password = re.search(r'密码([^。\s]+)', (ROOT / 'AGENTS.md').read_text(encoding='utf-8')).group(1)
    client = paramiko.SSHClient()
    known = ROOT / 'known_hosts'
    if known.exists():
        client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('frp-van.com', port=64906, username='chenkejun', password=password,
                   timeout=25, banner_timeout=25, auth_timeout=25)
    return client


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['exec', 'python', 'put', 'get'])
    p.add_argument('source')
    p.add_argument('target', nargs='?')
    args = p.parse_args()
    client = connect()
    try:
        if args.action in ('put', 'get'):
            with client.open_sftp() as sftp:
                if args.action == 'put':
                    sftp.put(args.source, args.target)
                else:
                    Path(args.target).parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(args.source, args.target)
            print(f'{args.action}: {args.target}')
            return
        command = args.source
        if args.action == 'python':
            code = Path(args.source).read_text(encoding='utf-8')
            command = 'python3 -c ' + shlex.quote(code)
        _, out, err = client.exec_command(command)
        for line in out:
            print(line, end='', flush=True)
        errors = err.read().decode('utf-8', errors='replace')
        if errors:
            print(errors, file=sys.stderr, end='')
        raise SystemExit(out.channel.recv_exit_status())
    finally:
        client.close()


if __name__ == '__main__':
    main()
