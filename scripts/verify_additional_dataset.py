"""Check supplemental manifest, converted labels, provenance and final file inventory."""
from collections import Counter
import json
from pathlib import Path
import shutil

from audit_additional_data import ROOT, labels_agree, txt_label


def main():
    dataset = ROOT / 'datasets/pingpong_additional_v1'
    if (dataset / 'PREPARATION_IN_PROGRESS').exists():
        raise RuntimeError('Backup is incomplete.')
    summary = json.loads((dataset / 'dataset_summary.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (dataset / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    active = [json.loads(line) for line in (ROOT / 'datasets/pingpong_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    active_hashes = {r['image_sha256'] for r in active}
    hashes = [r['image_sha256'] for r in records]
    assert len(hashes) == len(set(hashes)) == summary['images']
    assert not active_hashes.intersection(hashes)
    expected = {key: set() for key in ('image', 'label', 'annotation')}
    for record in records:
        assert record['split'] == 'unassigned'
        for key in expected:
            path = (dataset / record[key]).resolve()
            assert path.is_relative_to(dataset.resolve()) and path.is_file()
            expected[key].add(path)
        assert labels_agree(txt_label(dataset / record['label']), record['boxes'])
        if record['background_variant']:
            parents = record['active_parent_matches_for_review']
            assert len(parents) == 1 and parents[0]['split'] == 'train'
    for key, directory in [('image', 'images/unassigned'), ('label', 'labels/unassigned'), ('annotation', 'annotations/original')]:
        actual = {p.resolve() for p in (dataset / directory).iterdir()}
        assert actual == expected[key]
    assert not (dataset / 'data.yaml').exists()
    assert shutil.disk_usage(ROOT).free >= 5 * 1024**3
    result = {'verified': True, 'images': len(records), 'objects': sum(r['objects'] for r in records),
              'filename_family_counts': dict(Counter(r['filename_family'] for r in records)),
              'background_variants': sum(r['background_variant'] for r in records),
              'exact_content_overlap_with_active_dataset': 0,
              'label_and_inventory_checks': 'All records passed',
              'copy_integrity': 'Every image and original annotation SHA256 checked during preparation; no redundant full rehash.',
              'free_gib': shutil.disk_usage(ROOT).free / 1024**3}
    (ROOT / 'reports/datasets/pingpong_additional_v1/additional_dataset_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
