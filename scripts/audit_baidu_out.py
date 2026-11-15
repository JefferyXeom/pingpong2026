"""Read-only content/annotation audit of the 2026-11-13 Baidu source; cached by stat."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

from PIL import Image
from audit_additional_data import EXTENSIONS, xml_label, txt_label, labels_agree, sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('E:/BaiduNetdiskDownload/pingpongdata----out')
VERSION = 'pingpong_baidu_out_v1'
REPORT = ROOT / 'reports/datasets' / VERSION


def old_records():
    records = []
    for ds in ['pingpong_v1', 'pingpong_additional_v1']:
        for line in (ROOT / 'datasets' / ds / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            r['dataset'] = ds
            r['absolute_image'] = str(ROOT / 'datasets' / ds / r['image'])
            r['old_boxes'] = [list(map(float, x.split())) for x in r['label'].splitlines()] if ds == 'pingpong_v1' else r['boxes']
            r['effective_split'] = r['split'] if ds == 'pingpong_v1' else 'train'
            records.append(r)
    return records


def inspect(task):
    path, annotations = task
    stat = path.stat()
    result = {'source_image': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
              'sha256': sha256(path), 'annotations': [str(a) for a in annotations]}
    try:
        with Image.open(path) as im:
            im.load()
            result['width'], result['height'] = im.size
            rgb = im.convert('RGB')
            result['pixel_sha256'] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        parsed = [(a, xml_label(a, im.size) if a.suffix.lower() == '.xml' else txt_label(a)) for a in annotations]
        if not parsed:
            result['status'] = 'missing_annotation'
        elif not all(labels_agree(parsed[0][1], boxes) for _, boxes in parsed[1:]):
            result['status'] = 'annotation_conflict'
        else:
            result.update(status='verified_pair' if parsed[0][1] else 'empty_annotation_needs_review',
                          boxes=parsed[0][1], objects=len(parsed[0][1]), annotation=str(parsed[0][0]))
    except Exception as exc:
        result.update(status='invalid_image_or_annotation', error=str(exc))
    return result


def old_pixel(record):
    path = Path(record['absolute_image'])
    stat = path.stat()
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        checksum = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'pixel_sha256': checksum}


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in SOURCE.rglob('*') if p.is_file())
    images = [p for p in paths if p.suffix.lower() in EXTENSIONS]
    lookup = {str(p).lower(): p for p in paths}
    tasks = [(p, [lookup[str(p.with_suffix(ext)).lower()] for ext in ['.xml', '.txt'] if str(p.with_suffix(ext)).lower() in lookup]) for p in images]
    inventory = {'source_root': str(SOURCE), 'files': len(paths), 'extensions': dict(Counter(p.suffix.lower() for p in paths)),
                 'bytes': sum(p.stat().st_size for p in paths), 'images': len(images), 'source_files_modified': False}
    (REPORT / 'source_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    print(json.dumps(inventory), flush=True)
    cache_path = REPORT / 'source_audit.jsonl'
    cached = {r['source_image']: r for line in cache_path.read_text(encoding='utf-8').splitlines() if (r := json.loads(line))} if cache_path.exists() else {}
    pending = [t for t in tasks if str(t[0]) not in cached or cached[str(t[0])]['mtime_ns'] != t[0].stat().st_mtime_ns]
    with cache_path.open('a', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=2) as pool:
        for i, result in enumerate(pool.map(inspect, pending), 1):
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            cached[result['source_image']] = result
            if i % 500 == 0:
                stream.flush()
                print(json.dumps({'stage': 'hash_decode_labels', 'checked': i, 'pending': len(pending)}), flush=True)
    results = [cached[str(p)] for p in images]
    previous = old_records()
    by_hash, by_stem, by_pixel = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in previous:
        by_hash[r['image_sha256']].append(r)
        by_stem[Path(r['source_image']).stem.lower()].append(r)
    pixels_path = REPORT / 'existing_pixel_hashes.jsonl'
    pixels = {r['path']: r for line in pixels_path.read_text(encoding='utf-8').splitlines() if (r := json.loads(line))} if pixels_path.exists() else {}
    pending_old = [r for r in previous if r['absolute_image'] not in pixels or pixels[r['absolute_image']]['mtime_ns'] != Path(r['absolute_image']).stat().st_mtime_ns]
    with pixels_path.open('a', encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=2) as pool:
        for i, result in enumerate(pool.map(old_pixel, pending_old), 1):
            stream.write(json.dumps(result) + '\n')
            pixels[result['path']] = result
            if i % 500 == 0:
                stream.flush()
                print(json.dumps({'stage': 'existing_pixel_hashes', 'checked': i, 'pending': len(pending_old)}), flush=True)
    for r in previous:
        by_pixel[pixels[r['absolute_image']]['pixel_sha256']].append(r)
    seen = set()
    classifications, accepted = [], []
    for r in results:
        matches = by_hash.get(r['sha256'], [])
        pixel_matches = by_pixel.get(r.get('pixel_sha256'), [])
        stem = Path(r['source_image']).stem.lower()
        ancestor_stems = {stem}
        name = stem
        while re.search(r'(_\d+|_white|_processed)$', name):
            name = re.sub(r'(_\d+|_white|_processed)$', '', name)
            ancestor_stems.add(name)
        parents = [p for s in ancestor_stems for p in by_stem.get(s, [])]
        item = {**r, 'existing_matches': [{'dataset': p['dataset'], 'image': p['image'], 'split': p['effective_split'],
                    'label_agrees': labels_agree(r['boxes'], p['old_boxes']) if 'boxes' in r else None} for p in (matches or pixel_matches)],
                'filename_parent_candidates': [{'dataset': p['dataset'], 'image': p['image'], 'split': p['effective_split'],
                                                'source_image': p['source_image']} for p in parents]}
        if matches or pixel_matches:
            item['decision'] = 'already_present_bytes' if matches else 'already_present_decoded_pixels'
        elif r['status'] != 'verified_pair':
            item['decision'] = 'exclude_' + r['status']
        elif (r['width'], r['height']) != (1280, 1024):
            item['decision'] = 'exclude_non_native_shape'
        elif any(p['effective_split'] in ('val', 'test') for p in parents):
            item['decision'] = 'exclude_possible_holdout_derivative'
        elif r['pixel_sha256'] in seen:
            item['decision'] = 'duplicate_within_new_source'
        else:
            item['decision'] = 'new_train_candidate'
            accepted.append(item)
            seen.add(r['pixel_sha256'])
        classifications.append(item)
    # Duplicate new pixels must have consistent labels before admitting any representative.
    new_groups = defaultdict(list)
    for r in classifications:
        if r.get('pixel_sha256') and r.get('boxes'):
            new_groups[r['pixel_sha256']].append(r)
    conflicts = {k for k, group in new_groups.items() if any(not labels_agree(group[0]['boxes'], r['boxes']) for r in group[1:])}
    for r in classifications:
        if r['decision'] in ('new_train_candidate', 'duplicate_within_new_source') and r['pixel_sha256'] in conflicts:
            r['decision'] = 'exclude_new_duplicate_label_conflict'
    accepted = [r for r in classifications if r['decision'] == 'new_train_candidate']
    with (REPORT / 'audit_decisions.jsonl').open('w', encoding='utf-8') as stream:
        for r in classifications:
            stream.write(json.dumps(r, ensure_ascii=False) + '\n')
    summary = {**inventory, 'decisions': dict(Counter(r['decision'] for r in classifications)),
               'existing_overlap_by_dataset_split': dict(Counter(p['dataset']+'/'+p['split'] for r in classifications for p in r['existing_matches'])),
               'existing_duplicate_label_disagreements': sum(p['label_agrees'] is False for r in classifications for p in r['existing_matches']),
               'new_images': len(accepted), 'new_objects': sum(r['objects'] for r in accepted),
               'new_image_bytes': sum(r['bytes'] for r in accepted), 'new_with_filename_parent': sum(bool(r['filename_parent_candidates']) for r in accepted),
               'free_disk_gib': shutil.disk_usage(ROOT).free/1024**3,
               'limits': ['Byte hashes and decoded RGB hashes checked against all prior dataset images.',
                          'Filename ancestry is a conservative candidate relationship, not proof of independent capture.',
                          'No verified new negative images; unknown historical recording independence is not established.']}
    (REPORT / 'audit_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (REPORT / 'README.md').write_text('# Baidu pingpongdata----out audit (2026-11-13)\n\nSource is read-only. source_inventory.json: counts; source_audit.jsonl: source hashes/labels; existing_pixel_hashes.jsonl: previous decoded RGB hashes; audit_decisions.jsonl: per-image overlap/decision; audit_summary.json: summary. No training performed by this script.\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
