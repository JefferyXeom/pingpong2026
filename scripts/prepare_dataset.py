"""Build a verified, independent YOLO dataset without changing the desktop source.

Uses the source project's VOC conversion convention (center minus one pixel).
Requires Pillow; can run with the existing Anaconda base Python.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import xml.etree.ElementTree as ET

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path.home() / 'Desktop' / 'yolov5data'
GIB = 1024**3
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def inspect(task):
    batch, split, image, annotation = task
    try:
        xml = ET.parse(annotation).getroot()
        width = int(xml.findtext('size/width', '0'))
        height = int(xml.findtext('size/height', '0'))
        if min(width, height) <= 0:
            raise ValueError('Invalid XML image dimensions')
        with Image.open(image) as decoded:
            if decoded.size != (width, height):
                raise ValueError('Image dimensions differ from XML')
            decoded.load()
        boxes = []
        difficult = 0
        pixels = []
        for obj in xml.findall('object'):
            if obj.findtext('name', '').strip() != 'ball':
                raise ValueError('Unexpected class: ' + obj.findtext('name', ''))
            coords = [float(obj.findtext('bndbox/' + key, 'nan')) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
            xmin, ymin, xmax, ymax = coords
            if not all(math.isfinite(v) for v in coords):
                raise ValueError('Non-finite bounding box')
            if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                raise ValueError('Out-of-bounds or zero-area bounding box')
            xc, yc = ((xmin + xmax) / 2 - 1) / width, ((ymin + ymax) / 2 - 1) / height
            bw, bh = (xmax - xmin) / width, (ymax - ymin) / height
            if not (bw / 2 <= xc <= 1 - bw / 2 and bh / 2 <= yc <= 1 - bh / 2):
                raise ValueError('Box crosses boundary after legacy VOC coordinate conversion')
            boxes.append((0, xc, yc, bw, bh))
            pixels.append((xmax - xmin, ymax - ymin))
            difficult += int(obj.findtext('difficult', '0') == '1')
        label = ''.join('0 ' + ' '.join(f'{x:.10f}' for x in box[1:]) + '\n' for box in boxes)
        return {
            'batch': batch, 'split': split,
            'source_image': str(image.relative_to(SOURCE)),
            'source_annotation': str(annotation.relative_to(SOURCE)),
            'image_sha256': digest(image), 'annotation_sha256': digest(annotation),
            'label': label, 'width': width, 'height': height,
            'objects': len(boxes), 'difficult_objects': difficult,
            'box_pixels': pixels, 'bytes': image.stat().st_size,
        }
    except Exception as exc:
        return {'batch': batch, 'split': split, 'source_image': str(image.relative_to(SOURCE)), 'source_annotation': str(annotation.relative_to(SOURCE)), 'error': str(exc)}

def copy_verified(source, dest, expected_hash):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if digest(dest) != expected_hash:
            raise RuntimeError('Existing destination differs; refusing overwrite: ' + str(dest))
        return
    if shutil.disk_usage(ROOT).free - source.stat().st_size < 5 * GIB + 512 * 1024**2:
        raise RuntimeError('Copy stopped before violating the 5 GiB free-space floor.')
    temporary = dest.with_name(dest.name + '.partial')
    if temporary.exists():
        raise RuntimeError('Incomplete copy requires review: ' + str(temporary))
    shutil.copy2(source, temporary)
    if digest(temporary) != expected_hash:
        raise RuntimeError('Copy verification failed: ' + str(temporary))
    temporary.rename(dest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='datasets/pingpong_v1')
    args = parser.parse_args()
    target = (ROOT / args.output).resolve()
    if not target.is_relative_to(ROOT / 'datasets'):
        raise RuntimeError('Dataset output must be inside workspace/datasets.')
    if target.exists():
        raise RuntimeError('Output already exists. Review it before creating a new version; nothing overwritten.')
    reports = ROOT / 'reports/datasets/pingpong_v1'
    reports.mkdir(parents=True, exist_ok=True)
    # Whole recording batches are held out. Left/right views stay together.
    # The two 0527 recordings are distinct sequences, not random frame subsets.
    batches = [
        ('may27_recording2', 'test', 'data/标记2/images/20240527_2', 'data/标记2/annotations/0527_2'),
        ('may27_recording3', 'val', 'data/标记2/images/20240527_3', 'data/标记2/annotations/0527_3'),
        ('june01_recording1', 'train', 'data/060101', 'data/060101/text'),
        ('june01_recording2', 'train', 'data/img_data/images', 'data/img_data/Annotations'),
        ('may27_recording1', 'train', 'data/img_data.0527/img_data/images', 'data/img_data.0527/img_data/labels'),
    ]
    tasks, excluded, orphan_annotations = [], [], []
    for batch, split, images_relative, labels_relative in batches:
        image_dir, label_dir = SOURCE / images_relative, SOURCE / labels_relative
        images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        annotations = {p.stem: p for p in label_dir.glob('*.xml')}
        for stem, image in sorted(images.items()):
            if stem not in annotations:
                excluded.append({'source_image': str(image.relative_to(SOURCE)), 'reason': 'missing_annotation', 'batch': batch})
            else:
                tasks.append((batch, split, image, annotations[stem]))
        orphan_annotations.extend(str(p.relative_to(SOURCE)) for stem, p in annotations.items() if stem not in images)
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(inspect, tasks), 1):
            results.append(result)
            if index % 500 == 0:
                print(f'Validated {index}/{len(tasks)} image/XML pairs', flush=True)
    valid = [r for r in results if 'error' not in r]
    issues = [r for r in results if 'error' in r]
    by_hash = defaultdict(list)
    for result in valid:
        by_hash[result['image_sha256']].append(result)
    selected, duplicates = [], []
    for checksum, records in by_hash.items():
        signatures = {tuple(sorted(r['label'].splitlines())) for r in records}
        if len(signatures) > 1:
            for record in records:
                issues.append({**record, 'error': 'Exact duplicate image has conflicting annotations'})
            continue
        selected.append(records[0])
        duplicates.extend({'source_image': r['source_image'], 'retained_source': records[0]['source_image'], 'image_sha256': checksum} for r in records[1:])
    # Check the duplicated VOCData copy against original source names without copying it.
    mirrors = []
    original_batch = {Path(r['source_image']).name: r for r in valid if r['batch'] == 'june01_recording2'}
    for name, original in sorted(original_batch.items()):
        mirror = SOURCE / 'VOCData/images' / name
        if mirror.is_file():
            mirrors.append({'source_image': str(mirror.relative_to(SOURCE)), 'canonical_source': original['source_image'], 'matches': digest(mirror) == original['image_sha256']})
    if any(not m['matches'] for m in mirrors):
        raise RuntimeError('VOCData mirror differs from source. Review before copying.')
    planned_bytes = sum(r['bytes'] for r in selected)
    if shutil.disk_usage(ROOT).free - planned_bytes < 15 * GIB:
        raise RuntimeError('Insufficient space for dataset plus environment/training headroom.')
    counts = Counter(r['split'] for r in selected)
    if any(counts[split] == 0 for split in ('train', 'val', 'test')):
        raise RuntimeError('Every split must contain verified samples.')
    # Persist the full inspection before materializing a dataset.
    preparation = {'paired_candidates': len(tasks), 'selected': len(selected), 'split_counts': counts, 'image_bytes': planned_bytes, 'excluded_missing_labels': excluded, 'invalid_pairs': issues, 'duplicates': duplicates, 'verified_vocdata_mirrors': len(mirrors), 'orphan_annotations': orphan_annotations, 'excluded_directories': ['waste_data', 'VOCData/processed_images'], 'excluded_archives': ['data/img_data.0527.zip', 'VOCData/images.zip'], 'coordinate_convention': 'Legacy project VOC: center=((min+max)/2-1)/size, extent=(max-min)/size; invalid boxes are excluded for review, never silently clipped.', 'split_policy': 'Entire recording 20240527_3 is validation; entire recording 20240527_2 is test. All other canonical batches are training. Exact duplicate content is retained once, preferring held-out splits. Background variants must follow the original split and only augment training.', 'heldout_limit': 'Recording boundaries inferred from existing directory names; site separation is not established.'}
    (reports / 'dataset_preparation.json').write_text(json.dumps(preparation, ensure_ascii=False, indent=2), encoding='utf-8')
    target.mkdir(parents=True)
    marker = target / 'PREPARATION_IN_PROGRESS'
    marker.write_text('Use only after dataset_summary.json exists and this marker is gone.\n', encoding='utf-8')
    manifest_path = target / 'manifest.jsonl'
    with manifest_path.open('w', encoding='utf-8') as manifest:
        for index, record in enumerate(selected, 1):
            image = SOURCE / record['source_image']
            name = record['batch'] + '__' + image.stem
            relative_image = Path('images') / record['split'] / (name + image.suffix.lower())
            relative_label = Path('labels') / record['split'] / (name + '.txt')
            relative_annotation = Path('annotations/voc') / (name + '.xml')
            copy_verified(image, target / relative_image, record['image_sha256'])
            copy_verified(SOURCE / record['source_annotation'], target / relative_annotation, record['annotation_sha256'])
            (target / relative_label).parent.mkdir(parents=True, exist_ok=True)
            (target / relative_label).write_text(record['label'], encoding='utf-8')
            output_record = {**record, 'image': relative_image.as_posix(), 'label_path': relative_label.as_posix(), 'annotation': relative_annotation.as_posix()}
            if record['batch'] == 'june01_recording2':
                background = SOURCE / 'VOCData/processed_images' / image.name
                if background.is_file():
                    output_record['future_background_source'] = str(background.relative_to(SOURCE))
            manifest.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            if index % 500 == 0:
                manifest.flush()
                print(f'Copied and SHA256-verified {index}/{len(selected)} images; free {shutil.disk_usage(ROOT).free / GIB:.2f} GiB', flush=True)
    config = '# Root is inferred from this YAML location; no source-machine paths.\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ball\n'
    (target / 'data.yaml').write_text(config, encoding='utf-8')
    all_pixels = [min(box) for r in selected for box in r['box_pixels']]
    small_counts = {str(size): sum(p * size / 1280 < 4 for p in all_pixels) for size in (640, 960, 1280)}
    summary = {**{k: preparation[k] for k in ('selected', 'split_counts', 'image_bytes', 'coordinate_convention', 'split_policy', 'heldout_limit')}, 'objects': sum(r['objects'] for r in selected), 'negative_images': sum(r['objects'] == 0 for r in selected), 'difficult_objects': sum(r['difficult_objects'] for r in selected), 'image_dimensions': dict(Counter(f"{r['width']}x{r['height']}" for r in selected)), 'box_min_side_pixels_median': statistics.median(all_pixels) if all_pixels else None, 'boxes_under_four_pixels_after_resize': small_counts, 'batch_counts': dict(Counter(r['batch'] for r in selected)), 'free_gib_after_copy': shutil.disk_usage(ROOT).free / GIB, 'independent_copies': True, 'background_variants_included': False, 'exact_duplicate_images_removed': len(duplicates), 'invalid_pairs_excluded': len(issues), 'missing_annotation_images_excluded': len(excluded)}
    (target / 'dataset_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    marker.unlink()
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)

if __name__ == '__main__':
    main()
