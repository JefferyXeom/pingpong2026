"""Correct cached comparisons without decoding or hashing all source images again."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

from audit_additional_data import ROOT, labels_agree


def main():
    path = ROOT / 'reports/datasets/pingpong_additional_v1/additional_data_audit.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    active = [json.loads(line) for line in
              (ROOT / 'datasets/pingpong_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    active_by_hash = {r['image_sha256']: r for r in active}
    corrected = 0
    for record in report['images']:
        candidates = record.get('candidate_labels', [])
        if candidates and all(labels_agree(candidates[0]['boxes'], c['boxes']) for c in candidates[1:]):
            chosen = next((c for c in candidates if Path(c['path']).suffix.lower() == '.xml'), candidates[0])
            boxes = chosen['boxes']
            record.update(status='verified_pair' if boxes else 'empty_annotation_needs_review',
                          annotation=chosen['path'], boxes=boxes, objects=len(boxes))
            del record['candidate_labels']
            corrected += 1
        old = active_by_hash.get(record.get('sha256'))
        if old and 'boxes' in record:
            previous = [tuple(map(float, line.split())) for line in old['label'].splitlines()]
            record['active_annotation_agrees'] = labels_agree(record['boxes'], previous)
    by_hash = defaultdict(list)
    for record in report['images']:
        if record.get('sha256'):
            by_hash[record['sha256']].append(record)
    fresh, conflicts = [], []
    for checksum, records in by_hash.items():
        if checksum in active_by_hash:
            continue
        valid = [r for r in records if r['status'] == 'verified_pair']
        if not valid:
            continue
        alternatives = [r['boxes'] for r in valid]
        alternatives.extend(c['boxes'] for r in records for c in r.get('candidate_labels', []))
        if not all(labels_agree(alternatives[0], b) for b in alternatives[1:]):
            conflicts.append({'sha256': checksum, 'images': [r['image'] for r in records]})
            continue
        fresh.append(next((r for r in valid if not r.get('annotation_issues')), valid[0]))
    free = shutil.disk_usage(ROOT).free
    report['additional_candidates'] = fresh
    report['duplicate_conflicts'] = conflicts
    report['summary'].update(
        image_statuses=dict(Counter(r['status'] for r in report['images'])),
        active_annotation_conflicts=sum(r.get('active_annotation_agrees') is False for r in report['images']),
        unique_verified_additional_images=len(fresh),
        unique_additional_image_bytes=sum(r['bytes'] for r in fresh),
        new_duplicate_annotation_conflict_groups=len(conflicts),
        free_bytes=free,
        estimated_free_after_new_images_bytes=free - sum(r['bytes'] for r in fresh),
        annotation_comparison='Unordered one-to-one boxes; exact class; absolute normalized coordinate tolerance 1e-7.',
    )
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (ROOT / 'reports/datasets/pingpong_additional_v1/additional_data_summary.json').write_text(
        json.dumps(report['summary'], indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'corrected_rounding_conflicts': corrected, 'summary': report['summary']}, indent=2))


if __name__ == '__main__':
    main()
