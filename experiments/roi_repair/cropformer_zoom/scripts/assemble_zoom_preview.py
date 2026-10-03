from pathlib import Path
p=Path(__file__).parent
markup=(p/'zoom_preview_template.html').read_text(encoding='utf-8');data=(p/'zoom_inline_data.json').read_text(encoding='utf-8')
content=markup.replace('__ZOOM_DATA__',data)
assert len(content.encode('utf-8'))<1_000_000
out=Path('D:/codex/home/visualizations/2026/10/02/01a0fb2d-4a26-7950-bad3-be6d50b1b315/room0-cropformer-zoom.html')
out.write_text(content,encoding='utf-8');print('inline_bytes',out.stat().st_size)
