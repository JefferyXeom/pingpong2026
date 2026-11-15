"""Back up verified, distinct additional pairs; leave split assignment pending review."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

from audit_additional_data import ROOT, sha256, labels_agree, txt_label, xml_label
from inspect_additional_provenance import family

GIB = 1024**3
TARGET = ROOT / 'datasets/pingpong_additional_v1'

def copy_verified(source, dest, expected):
    if shutil.disk_usage(ROOT).free - source.stat().st_size < 5.5 * GIB:
        raise RuntimeError('Copy stopped to preserve the 5 GiB floor.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + '.partial')
    if dest.exists() or temporary.exists():
        raise RuntimeError('Destination already exists; refusing overwrite: ' + str(dest))
    shutil.copy2(source, temporary)
    if sha256(temporary) != expected:
        raise RuntimeError('Source changed or copy verification failed: ' + str(source))
    temporary.rename(dest)

def main():
    audit = json.loads((ROOT / 'reports/datasets/pingpong_additional_v1/additional_data_audit.json').read_text(encoding='utf-8'))
    active = [json.loads(line) for line in (ROOT / 'datasets/pingpong_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    active_hashes = {r['image_sha256'] for r in active}
    by_stem = defaultdict(list)
    for r in active:
        by_stem[Path(r['source_image']).stem.lower()].append(r)
    candidates = [r for r in audit['additional_candidates'] if r['status'] == 'verified_pair' and not r.get('annotation_issues')]
    if not candidates:
        raise RuntimeError('No clean additional pairs available for backup.')
    if TARGET.exists():
        raise RuntimeError('Supplementary backup already exists; nothing overwritten.')
    required = sum(r['bytes'] for r in candidates)
    if shutil.disk_usage(ROOT).free - required < 8 * GIB:
        raise RuntimeError('Insufficient space for independent copies and required reserve.')
    TARGET.mkdir(parents=True)
    marker = TARGET / 'PREPARATION_IN_PROGRESS'
    marker.write_text('Unassigned backup; do not train until complete and split provenance reviewed.\n', encoding='utf-8')
    seen, counts = set(), Counter()
    with (TARGET / 'manifest.jsonl').open('w', encoding='utf-8') as manifest:
        for index, record in enumerate(candidates, 1):
            checksum = record['sha256']
            if checksum in active_hashes or checksum in seen:
                raise RuntimeError('Unexpected duplicate in additional selection.')
            seen.add(checksum)
            source = Path(record['image'])
            annotation_source = Path(record['annotation'])
            annotation_hash = sha256(annotation_source)
            current_boxes = (xml_label(annotation_source, (record['width'], record['height']))
                             if annotation_source.suffix.lower() == '.xml' else txt_label(annotation_source))
            if not labels_agree(current_boxes, record['boxes']):
                raise RuntimeError('Source annotation changed after audit: ' + str(annotation_source))
            name = checksum[:16] + '__' + source.stem[:80]
            image = Path('images/unassigned') / (name + source.suffix.lower())
            label = Path('labels/unassigned') / (name + '.txt')
            annotation = Path('annotations/original') / (name + annotation_source.suffix.lower())
            copy_verified(source, TARGET / image, checksum)
            copy_verified(annotation_source, TARGET / annotation, annotation_hash)
            content = ''.join('0 ' + ' '.join(f'{v:.10f}' for v in box[1:]) + '\n' for box in record['boxes'])
            (TARGET / label).parent.mkdir(parents=True, exist_ok=True)
            (TARGET / label).write_text(content, encoding='utf-8')
            if (TARGET / label).read_text(encoding='utf-8') != content:
                raise RuntimeError('Output label verification failed.')
            parent_stem = source.stem[:-6] if source.stem.endswith('_white') else source.stem
            parent_candidates = by_stem.get(parent_stem.lower(), [])
            parents = [r for r in parent_candidates if labels_agree(record['boxes'],
                       [list(map(float, line.split())) for line in r['label'].splitlines()])]
            parent_info = [{'image': r['image'], 'split': r['split'], 'sha256': r['image_sha256']} for r in parents]
            manifest.write(json.dumps({'source_image': str(source), 'source_annotation': str(annotation_source), 'image': image.as_posix(), 'label': label.as_posix(), 'annotation': annotation.as_posix(), 'image_sha256': checksum, 'annotation_sha256': annotation_hash, 'split': 'unassigned', 'width': record['width'], 'height': record['height'], 'objects': record['objects'], 'boxes': record['boxes'], 'filename_family': family(source.stem), 'background_variant': source.stem.endswith('_white'), 'active_parent_matches_for_review': parent_info}, ensure_ascii=False) + '\n')
            counts[str(source.parent)] += 1
            if index % 500 == 0:
                manifest.flush()
                print(f'Copied and SHA256-verified {index}/{len(candidates)} supplemental images; free {shutil.disk_usage(ROOT).free / GIB:.2f} GiB', flush=True)
    assert len(list((TARGET / 'images/unassigned').iterdir())) == len(candidates)
    assert len(list((TARGET / 'labels/unassigned').iterdir())) == len(candidates)
    assert len(list((TARGET / 'annotations/original').iterdir())) == len(candidates)
    summary = {'images': len(candidates), 'objects': sum(r['objects'] for r in candidates), 'image_bytes': required, 'split': 'unassigned', 'source_directory_counts': dict(counts), 'image_and_annotation_copy_sha256_verified': True, 'labels_written_and_verified': True, 'active_dataset_unchanged': True, 'sources_unchanged': True, 'free_gib_after_copy': shutil.disk_usage(ROOT).free / GIB, 'excluded': 'Existing exact image duplicates, conflicting/invalid/missing annotations and unreviewed empty annotations. Archives are inventoried only.', 'usage': 'Independent supplementary backup only. No data.yaml generated; review recording provenance, background variants and held-out overlap before training.'}
    (TARGET / 'dataset_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (TARGET / 'README.md').write_text('# 新发现历史数据的独立补充备份\n\n本目录只保存与 pingpong_v1 内容不同、标注通过检查的样本。split 暂为 unassigned，尚未加入当前训练。\n\n训练前需核实采集批次、背景变体及与现有验证/测试画面的关系；不能将 images/unassigned 全量直接并入训练。manifest.jsonl 保留原始路径、标注、内容哈希和同名画面线索。\n', encoding='utf-8')
    marker.unlink()
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)

if __name__ == '__main__':
    main()
