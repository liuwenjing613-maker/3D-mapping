from pathlib import Path
p=Path(__file__).parent
out=Path('D:/codex/home/visualizations/2026/10/02/01a0fb2d-4a26-7950-bad3-be6d50b1b315/room0-roi-priority.html')
markup=(p/'roi_preview_template.html').read_text(encoding='utf-8')
data=(p/'inline_data_annotated.json').read_text(encoding='utf-8')
content=markup.replace('__ROI_DATA__',data)
assert len(content.encode('utf-8'))<1_000_000
assert '<!doctype' not in content.lower() and '<html' not in content.lower()
out.write_text(content,encoding='utf-8')
print('inline_bytes',out.stat().st_size)
