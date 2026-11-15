"""Read-only audit of newly located legacy datasets; never alter active training data."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [Path('D:/yolov5-master/VOCData'), Path('D:/yolov5-master/history_VOCOdata')]
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def xml_label(path, size):
    xml = ET.parse(path).getroot()
    width, height = int(xml.findtext('size/width', '0')), int(xml.findtext('size/height', '0'))
    if (width, height) != size:
        raise ValueError('XML dimensions differ from decoded image')
    boxes = []
    for obj in xml.findall('object'):
        if obj.findtext('name', '').strip() != 'ball':
            raise ValueError('Unexpected class: ' + obj.findtext('name', ''))
        x1, y1, x2, y2 = [float(obj.findtext('bndbox/' + key, 'nan')) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Invalid XML box')
        box = (0, ((x1 + x2) / 2 - 1) / width, ((y1 + y2) / 2 - 1) / height, (x2 - x1) / width, (y2 - y1) / height)
        _, x, y, w, h = box
        if not (-1e-9 <= x - w / 2 and x + w / 2 <= 1 + 1e-9 and -1e-9 <= y - h / 2 and y + h / 2 <= 1 + 1e-9):
            raise ValueError('XML box crosses boundary under legacy conversion')
        boxes.append(box)
    return boxes

def txt_label(path):
    boxes = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError('TXT is not a five-column detection annotation')
        cls = int(fields[0])
        x, y, w, h = map(float, fields[1:])
        if cls != 0 or not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError('Unexpected class or non-finite TXT coordinates')
        if not (0 < w <= 1 and 0 < h <= 1 and -1e-6 <= x - w / 2 and x + w / 2 <= 1 + 1e-6 and -1e-6 <= y - h / 2 and y + h / 2 <= 1 + 1e-6):
            raise ValueError('Invalid TXT box')
        boxes.append((cls, x, y, w, h))
    return boxes

def labels_agree(left, right, tolerance=1e-7):
    """Compare unordered boxes without rounding-tie artifacts or box reuse."""
    if len(left) != len(right):
        return False
    matches = {}

    def assign(index, visited):
        for other, box in enumerate(right):
            if other in visited or left[index][0] != box[0]:
                continue
            if not all(math.isclose(a, b, rel_tol=0, abs_tol=tolerance)
                       for a, b in zip(left[index][1:], box[1:])):
                continue
            visited.add(other)
            if other not in matches or assign(matches[other], visited):
                matches[other] = index
                return True
        return False

    return all(assign(index, set()) for index in range(len(left)))

def inspect_image(task):
    image, label_paths = task
    result = {'image': str(image), 'bytes': image.stat().st_size, 'annotations': [str(p) for p in label_paths]}
    try:
        with Image.open(image) as decoded:
            size = decoded.size
            decoded.load()
        result['width'], result['height'] = size
        result['sha256'] = sha256(image)
        parsed = []
        annotation_issues = []
        for label in label_paths:
            try:
                boxes = xml_label(label, size) if label.suffix.lower() == '.xml' else txt_label(label)
                parsed.append((label, boxes))
            except Exception as exc:
                annotation_issues.append({'path': str(label), 'error': str(exc)})
        result['annotation_issues'] = annotation_issues
        if not parsed:
            result['status'] = 'invalid_annotations' if label_paths else 'missing_annotation'
        elif not all(labels_agree(parsed[0][1], boxes) for _, boxes in parsed[1:]):
            result['status'] = 'annotation_conflict'
            result['candidate_labels'] = [{'path': str(p), 'boxes': boxes} for p, boxes in parsed]
        else:
            chosen, boxes = next(((p, b) for p, b in parsed if p.suffix.lower() == '.xml'), parsed[0])
            result.update(status='verified_pair' if boxes else 'empty_annotation_needs_review', annotation=str(chosen), boxes=boxes, objects=len(boxes))
    except Exception as exc:
        result.update(status='invalid_image', error=str(exc))
    return result

def main():
    reports = ROOT / 'reports/datasets/pingpong_additional_v1'
    reports.mkdir(parents=True, exist_ok=True)
    inventory, files = [], []
    for source in SOURCES:
        if not source.is_dir():
            raise FileNotFoundError(source)
        for directory, subdirs, names in os.walk(source, followlinks=False):
            folder = Path(directory)
            subdirs[:] = [name for name in subdirs if not (folder / name).is_symlink() and not (folder / name).is_junction()]
            paths = [folder / name for name in sorted(names) if not (folder / name).is_symlink()]
            files.extend(paths)
            if paths:
                inventory.append({'directory': str(folder), 'files': len(paths), 'bytes': sum(p.stat().st_size for p in paths), 'extensions': dict(Counter(p.suffix.lower() for p in paths)), 'examples': [p.name for p in paths[:5]]})
    images = [p for p in files if p.suffix.lower() in EXTENSIONS]
    xmls = [p for p in files if p.suffix.lower() == '.xml']
    print(f'Enumerated {len(files)} files, {len(images)} images, {len(xmls)} XML annotations', flush=True)
    (reports / 'additional_directory_inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding='utf-8')
    paths_by_lower = {str(p).lower(): p for p in files}
    xml_stats = Counter()
    xml_errors = []
    for xml_index, xml in enumerate(xmls, 1):
        try:
            tree = ET.parse(xml).getroot()
            objects = tree.findall('object')
            xml_stats['xml_files'] += 1
            xml_stats['empty_xml_files'] += not objects
            for obj in objects:
                xml_stats['class:' + obj.findtext('name', '').strip()] += 1
        except Exception as exc:
            xml_errors.append({'path': str(xml), 'error': str(exc)})
        if xml_index % 2000 == 0:
            print(f'Parsed {xml_index}/{len(xmls)} XML annotations', flush=True)
    tasks = []
    for image in images:
        candidates = []
        base = image.parent.parent if image.parent.name.lower() in {'images', 'jpegimages', 'processed_images'} else image.parent
        for directory in [image.parent, base / 'Annotations', base / 'annotations', base / 'labels', base / 'text']:
            for extension in ('.xml', '.txt'):
                candidate = paths_by_lower.get(str(directory / (image.stem + extension)).lower())
                if candidate is not None and candidate not in candidates:
                    candidates.append(candidate)
        tasks.append((image, candidates))
    active = [json.loads(line) for line in (ROOT / 'datasets/pingpong_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    active_by_hash = {r['image_sha256']: r for r in active}
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, result in enumerate(pool.map(inspect_image, tasks), 1):
            old = active_by_hash.get(result.get('sha256'))
            if old:
                result['active_dataset_match'] = {'image': old['image'], 'split': old['split']}
                if 'boxes' in result:
                    previous = [tuple(map(float, line.split())) for line in old['label'].splitlines()]
                    result['active_annotation_agrees'] = labels_agree(result['boxes'], previous)
            results.append(result)
            if index % 500 == 0:
                print(f'Checked and hashed {index}/{len(tasks)} images', flush=True)
    by_hash = defaultdict(list)
    for record in results:
        if record.get('sha256'):
            by_hash[record['sha256']].append(record)
    fresh, duplicate_conflicts = [], []
    for checksum, records in by_hash.items():
        if checksum in active_by_hash:
            continue
        valid = [r for r in records if r['status'] == 'verified_pair']
        if not valid:
            continue
        alternatives = [r['boxes'] for r in valid]
        alternatives.extend(c['boxes'] for r in records for c in r.get('candidate_labels', []))
        if not all(labels_agree(alternatives[0], boxes) for boxes in alternatives[1:]):
            duplicate_conflicts.append({'sha256': checksum, 'images': [r['image'] for r in records]})
            continue
        fresh.append(next((r for r in valid if not r.get('annotation_issues')), valid[0]))
    # Recover previously orphaned annotations only on exact stem+annotation bytes.
    xml_by_stem = defaultdict(list)
    for p in xmls:
        xml_by_stem[p.stem.lower()].append(p)
    image_by_stem = defaultdict(list)
    for r in results:
        image_by_stem[Path(r['image']).stem.lower()].append(r)
    preparation = json.loads((ROOT / 'reports/datasets/pingpong_v1/dataset_preparation.json').read_text(encoding='utf-8'))
    old_source = Path.home() / 'Desktop/yolov5data'
    recovery = []
    for relative in preparation['orphan_annotations']:
        orphan = old_source / relative
        matched_images = image_by_stem.get(orphan.stem.lower(), [])
        if matched_images:
            checksum = sha256(orphan)
            recovery.append({'previous_annotation': str(orphan), 'images': [r['image'] for r in matched_images], 'matching_annotation_copies': [str(p) for p in xml_by_stem[orphan.stem.lower()] if sha256(p) == checksum]})
    archives = []
    for archive in [p for p in files if p.suffix.lower() == '.zip']:
        try:
            with zipfile.ZipFile(archive) as zf:
                counts = Counter(Path(info.filename).suffix.lower() for info in zf.infolist() if not info.is_dir())
                archives.append({'path': str(archive), 'extensions': dict(counts), 'uncompressed_bytes': sum(i.file_size for i in zf.infolist()), 'examples': zf.namelist()[:12]})
        except Exception as exc:
            archives.append({'path': str(archive), 'error': str(exc)})
    free = shutil.disk_usage(ROOT).free
    summary = {'source_roots': [str(p) for p in SOURCES], 'total_files': len(files), 'total_bytes': sum(p.stat().st_size for p in files), 'images': len(images), 'xml_summary': dict(xml_stats), 'image_statuses': dict(Counter(r['status'] for r in results)), 'images_matching_active_dataset': sum('active_dataset_match' in r for r in results), 'active_annotation_conflicts': sum(r.get('active_annotation_agrees') is False for r in results), 'unique_verified_additional_images': len(fresh), 'unique_additional_image_bytes': sum(r['bytes'] for r in fresh), 'new_duplicate_annotation_conflict_groups': len(duplicate_conflicts), 'previously_orphaned_xml_with_image_candidates': len(recovery), 'free_bytes': free, 'estimated_free_after_new_images_bytes': free - sum(r['bytes'] for r in fresh), 'negative_sample_note': 'Empty annotations are candidates only; missing annotation is not a verified negative.', 'active_dataset_changed': False, 'source_files_changed': False}
    full = {'summary': summary, 'directories': inventory, 'xml_errors': xml_errors, 'images': results, 'additional_candidates': fresh, 'duplicate_conflicts': duplicate_conflicts, 'orphan_recovery': recovery, 'archives': archives}
    (reports / 'additional_data_audit.json').write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding='utf-8')
    (reports / 'additional_data_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    print(json.dumps({'directories': inventory, 'archives': archives}, indent=2, ensure_ascii=True), flush=True)

if __name__ == '__main__':
    main()
