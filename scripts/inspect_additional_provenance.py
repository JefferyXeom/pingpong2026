"""Review file families and background parents; verify archive contents without extraction."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from audit_additional_data import ROOT, labels_agree


def family(stem):
    if stem.isdigit():
        return 'numeric'
    if re.fullmatch(r'\d+_img_\d+_\d+', stem):
        return 'indexed_img'
    if stem.endswith('_white'):
        return 'white_background'
    return re.sub(r'-?\d+', '#', stem)


def main():
    audit = json.loads((ROOT / 'reports/datasets/pingpong_additional_v1/additional_data_audit.json').read_text(encoding='utf-8'))
    active = [json.loads(line) for line in (ROOT / 'datasets/pingpong_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    by_stem = defaultdict(list)
    for record in active:
        by_stem[Path(record['source_image']).stem.lower()].append(record)
    groups = defaultdict(list)
    variants = []
    for record in audit['additional_candidates']:
        source = Path(record['image'])
        groups[(str(source.parent), family(source.stem))].append(record)
        if source.stem.endswith('_white'):
            parents = by_stem.get(source.stem[:-6].lower(), [])
            matches = [p for p in parents if labels_agree(record['boxes'], [list(map(float, line.split())) for line in p['label'].splitlines()])]
            variants.append({'image': record['image'], 'parent_matches': [
                {'image': p['image'], 'split': p['split'], 'sha256': p['image_sha256']} for p in matches]})
    summaries = []
    for (directory, name_family), records in sorted(groups.items()):
        examples = []
        for r in records[:3]:
            item = {'image': Path(r['image']).name, 'dimensions': [r['width'], r['height']]}
            if Path(r['annotation']).suffix.lower() == '.xml':
                xml = ET.parse(r['annotation']).getroot()
                item['xml_filename'] = xml.findtext('filename')
                item['xml_path'] = xml.findtext('path')
            examples.append(item)
        summaries.append({'directory': directory, 'family': name_family, 'count': len(records), 'examples': examples})
    known_images = {r['sha256'] for r in audit['images'] if 'sha256' in r}
    archive_reports = []
    for item in audit['archives']:
        stats = Counter()
        unmatched = []
        with zipfile.ZipFile(item['path']) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if Path(info.filename).suffix.lower() == '.png':
                    with archive.open(info) as stream:
                        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                    stats['images'] += 1
                    if digest in known_images:
                        stats['images_matching_loose_file_sha256'] += 1
                    else:
                        unmatched.append(info.filename)
        archive_reports.append({'path': item['path'], **stats, 'unmatched_images': unmatched})
    result = {'file_families': summaries,
              'white_background_images': len(variants),
              'white_with_exactly_one_active_parent_and_agreeing_labels': sum(len(v['parent_matches']) == 1 for v in variants),
              'white_parent_splits': dict(Counter(p['split'] for v in variants for p in v['parent_matches'])),
              'white_parent_review_failures': [v for v in variants if len(v['parent_matches']) != 1],
              'archives': archive_reports,
              'note': 'Filename families and XML paths are provenance clues, not proof of independent recordings.'}
    (ROOT / 'reports/datasets/pingpong_additional_v1/additional_data_provenance.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
