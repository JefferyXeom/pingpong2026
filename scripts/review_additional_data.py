"""Summarize the audit and draw representative newly found annotated images."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from PIL import Image, ImageDraw
from inspect_additional_provenance import family

ROOT = Path(__file__).resolve().parents[1]

def main():
    report = json.loads((ROOT / 'reports/datasets/pingpong_additional_v1/additional_data_audit.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for record in report['additional_candidates']:
        source = Path(record['image'])
        groups[(str(source.parent), family(source.stem))].append(record)
    preview = Image.new('RGB', (4 * 480, max(1, len(groups)) * 430), '#202020')
    summaries = []
    for row, ((directory, name_family), records) in enumerate(sorted(groups.items())):
        summaries.append({'directory': directory, 'filename_family': name_family, 'new_images': len(records), 'bytes': sum(r['bytes'] for r in records), 'examples': [Path(r['image']).name for r in records[:5]]})
        for column in range(4):
            record = records[min(len(records) - 1, int((column + .5) * len(records) / 4))]
            with Image.open(record['image']) as image:
                image = image.convert('RGB')
                scale = min(480 / image.width, 384 / image.height)
                tile = image.resize((round(image.width * scale), round(image.height * scale)))
                draw = ImageDraw.Draw(tile)
                for _, x, y, w, h in record['boxes']:
                    draw.rectangle(((x - w / 2) * tile.width, (y - h / 2) * tile.height, (x + w / 2) * tile.width, (y + h / 2) * tile.height), outline='#00ff00', width=2)
                preview.paste(tile, (column * 480, row * 430))
                draw = ImageDraw.Draw(preview)
                draw.text((column * 480 + 5, row * 430 + 389), Path(directory).parent.name + '/' + Path(directory).name, fill='white')
                draw.text((column * 480 + 5, row * 430 + 408), Path(record['image']).name[:65], fill='white')
    if groups:
        preview.save(ROOT / 'reports/datasets/pingpong_additional_v1/additional_data_preview.jpg', quality=90)
    statuses = defaultdict(Counter)
    for record in report['images']:
        statuses[str(Path(record['image']).parent)][record['status']] += 1
    conflicts = [r for r in report['images'] if r.get('active_annotation_agrees') is False or r['status'] in ('annotation_conflict', 'invalid_annotations', 'invalid_image')]
    result = {'summary': report['summary'], 'new_image_groups': summaries, 'statuses_by_directory': {k: dict(v) for k, v in statuses.items()}, 'conflict_examples': conflicts[:12], 'orphan_recovery_examples': report['orphan_recovery'][:5], 'preview': 'reports/datasets/pingpong_additional_v1/additional_data_preview.jpg' if groups else None}
    (ROOT / 'reports/datasets/pingpong_additional_v1/additional_data_review.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=True), flush=True)

if __name__ == '__main__':
    main()
