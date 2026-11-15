"""Explain unmatched annotations without touching source files."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path.home() / 'Desktop/yolov5data'
images = defaultdict(list)
for p in (SOURCE / 'data').rglob('*.png'):
    images[p.name.lower()].append(p)
sets = [
    ('data/060101', 'data/060101/text'),
    ('data/img_data/images', 'data/img_data/Annotations'),
    ('data/img_data.0527/img_data/images', 'data/img_data.0527/img_data/labels'),
    ('data/标记2/images/20240527_2', 'data/标记2/annotations/0527_2'),
    ('data/标记2/images/20240527_3', 'data/标记2/annotations/0527_3'),
]
report = []
for image_relative, label_relative in sets:
    image_dir = SOURCE / image_relative
    annotations = sorted((SOURCE / label_relative).glob('*.xml'))
    missing = []
    for annotation in annotations:
        if (image_dir / (annotation.stem + '.png')).is_file():
            continue
        xml = ET.parse(annotation).getroot()
        declared = Path(xml.findtext('filename', '')).name
        matches = images.get(declared.lower(), []) + images.get((annotation.stem + '.png').lower(), [])
        missing.append({'annotation': str(annotation.relative_to(SOURCE)), 'declared_filename': declared, 'found_elsewhere': sorted(set(str(p.relative_to(SOURCE)) for p in matches))})
    report.append({'image_directory': image_relative, 'annotations': len(annotations), 'unmatched_annotations': len(missing), 'details': missing})
(ROOT / 'reports/datasets/pingpong_v1/pairing_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
for group in report:
    print(json.dumps({**group, 'details': group['details'][:8]}, ensure_ascii=True, indent=2), flush=True)
missing_stems = {Path(item['annotation']).stem.lower() for group in report for item in group['details']}
archives = []
for archive in sorted(SOURCE.rglob('*.zip')):
    with zipfile.ZipFile(archive) as contents:
        image_entries = [item for item in contents.infolist() if Path(item.filename).suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        matches = [item.filename for item in image_entries if Path(item.filename).stem.lower() in missing_stems]
        archives.append({'archive': str(archive.relative_to(SOURCE)), 'image_count': len(image_entries), 'uncompressed_image_bytes': sum(item.file_size for item in image_entries), 'missing_annotation_stems_found': matches})
(ROOT / 'reports/datasets/pingpong_v1/archive_inventory.json').write_text(json.dumps(archives, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'archives': archives}, ensure_ascii=True, indent=2), flush=True)
