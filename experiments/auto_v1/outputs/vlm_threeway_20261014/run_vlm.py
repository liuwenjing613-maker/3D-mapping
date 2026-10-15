"""Three independent, single-image requests to the existing local 35B VLM."""
from pathlib import Path
import base64, hashlib, json, time, urllib.request, traceback

ROOT = Path('/data/chenkejun/CVPR/results/vlm_threeway_20261014')
INPUTS = ROOT / 'inputs'
OUT = ROOT / 'outputs'
BASE = 'http://127.0.0.1:11465'
MODEL = 'qwen3.6:35b-a3b-mtp-q4_K_M'
OPTIONS = {'temperature': 0, 'seed': 42, 'num_ctx': 32768, 'num_predict': 4096}

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def request(route, payload=None, timeout=900):
    req = urllib.request.Request(BASE + route,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((INPUTS / 'input_manifest.json').read_text(encoding='utf-8'))
    prompt = (INPUTS / 'vlm_prompt.txt').read_text(encoding='utf-8').strip()
    info = request('/api/show', {'model': MODEL}, timeout=30)
    assert 'vision' in info.get('capabilities', []), info.get('capabilities')
    write(OUT / 'model_info.json', info)
    write(OUT / 'run_config.json', {'model': MODEL, 'endpoint': BASE, 'options': OPTIONS,
        'think': False, 'format': 'json', 'independent_requests': True,
        'images_per_request': 1, 'base_prompt': prompt,
        'version': request('/api/version', timeout=15),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    rows = []
    for entry in manifest['models']:
        method = entry['method']
        img = INPUTS / (method + '_vlm_input.png')
        image_bytes = img.read_bytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        assert digest == entry['input_image_sha256'], (method, digest)
        message = prompt + '\n本图全部候选 mask 编号：' + json.dumps(entry['mask_ids']) + '。'
        payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': message,
            'images': [base64.b64encode(image_bytes).decode('ascii')]}],
            'stream': False, 'think': False, 'format': 'json', 'options': OPTIONS, 'keep_alive': -1}
        write(OUT / (method + '_request_metadata.json'), {'model': MODEL, 'messages': [
            {'role': 'user', 'content': message, 'images': [{'file': str(img), 'sha256': digest}]}],
            'stream': False, 'think': False, 'format': 'json', 'options': OPTIONS, 'keep_alive': -1})
        print('START', method, time.strftime('%Y-%m-%dT%H:%M:%S%z'), flush=True)
        start = time.monotonic()
        try:
            raw = request('/api/chat', payload)
            write(OUT / (method + '_raw_response.json'), raw)
            content = raw.get('message', {}).get('content', '')
            (OUT / (method + '_output.txt')).write_text(content, encoding='utf-8')
            parsed = json.loads(content)
            expected = set(entry['mask_ids'])
            groups = {k: parsed.get(k) for k in ['accepted_ids', 'uncertain_ids', 'rejected_ids']}
            types_ok = all(isinstance(v, list) and all(type(x) is int for x in v) for v in groups.values())
            all_ids = sum(groups.values(), []) if types_ok else []
            validation = {'integer_lists': types_ok, 'valid_partition': types_ok and set(all_ids) == expected and len(all_ids) == len(set(all_ids)),
                'missing_ids': sorted(expected - set(all_ids)), 'unexpected_ids': sorted(set(all_ids) - expected),
                'done': raw.get('done'), 'done_reason': raw.get('done_reason')}
            row = {'method': method, 'seconds': round(time.monotonic()-start, 3), 'input_sha256': digest,
                'output': parsed, 'validation': validation,
                'runtime': {k: raw[k] for k in ['total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration'] if k in raw}}
            write(OUT / (method + '_result.json'), row)
            print('RESULT', json.dumps(row, ensure_ascii=False), flush=True)
            rows.append(row)
        except Exception as exc:
            row = {'method': method, 'error': str(exc), 'traceback': traceback.format_exc(),
                'seconds': round(time.monotonic()-start, 3)}
            write(OUT / (method + '_error.json'), row)
            rows.append(row)
            print('ERROR', json.dumps(row, ensure_ascii=False), flush=True)
        write(OUT / 'summary.json', {'model': MODEL, 'results': rows})
    print('COMPLETE', len(rows), flush=True)

if __name__ == '__main__':
    main()

