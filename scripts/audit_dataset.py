"""Read-only detailed source inventory and Python environment probe (stdlib only)."""
import collections
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path.home() / 'Desktop' / 'yolov5data'
REPORTS = ROOT / 'reports/datasets/pingpong_v1'
REPORTS.mkdir(parents=True, exist_ok=True)
extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
groups = {}
xml_classes = collections.Counter()
xml_failures = []
samples = {}
configs = {}
for directory, _, names in os.walk(SOURCE):
    folder = Path(directory)
    group = {'extensions': collections.Counter(), 'bytes': 0, 'examples': []}
    for name in sorted(names):
        p = folder / name
        group['extensions'][p.suffix.lower()] += 1
        group['bytes'] += p.stat().st_size
        if len(group['examples']) < 5:
            group['examples'].append(name)
        if p.suffix.lower() in {'.xml', '.txt'}:
            key = str(folder.relative_to(SOURCE)) + '/' + p.suffix.lower()
            if key not in samples:
                samples[key] = {'file': str(p.relative_to(SOURCE)), 'content': p.read_text(encoding='utf-8-sig', errors='replace')[:4000]}
        if p.suffix.lower() == '.xml':
            try:
                tree = ET.parse(p)
                xml_classes.update(obj.findtext('name', '') for obj in tree.findall('object'))
            except Exception as exc:
                xml_failures.append({'path': str(p.relative_to(SOURCE)), 'error': str(exc)})
        if p.suffix.lower() in {'.yaml', '.yml'} or (p.suffix.lower() == '.py' and p.name != 'change.py'):
            configs[str(p.relative_to(SOURCE))] = p.read_text(encoding='utf-8-sig', errors='replace')[:10000]
    if names:
        groups[str(folder.relative_to(SOURCE))] = group
environments = []
probe = "import sys,importlib.util,json;print(json.dumps({'python':sys.version,'modules':{m:bool(importlib.util.find_spec(m)) for m in ['torch','ultralytics','PIL','tensorrt','onnx','cv2']}}))"
executables = [Path(sys.executable), Path('D:/ProgramData/anaconda3/python.exe')]
executables += sorted(Path('D:/ProgramData/anaconda3/envs').glob('*/python.exe'))
for exe in executables:
    if not exe.is_file():
        continue
    try:
        proc = subprocess.run([str(exe), '-c', probe], capture_output=True, text=True, timeout=30)
        environments.append({'executable': str(exe), 'returncode': proc.returncode, 'stdout': proc.stdout.strip(), 'stderr': proc.stderr.strip()[:1000]})
    except Exception as exc:
        environments.append({'executable': str(exe), 'error': str(exc)})
report = {'source': str(SOURCE), 'groups': groups, 'xml_classes': xml_classes, 'xml_failures': xml_failures, 'samples': samples, 'configs': configs, 'environments': environments}
target = REPORTS / 'dataset_detail.json'
target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(report, indent=2, ensure_ascii=True))
print('Report:', target)
