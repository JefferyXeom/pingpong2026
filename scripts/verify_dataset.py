"""Check split/label integrity and render representative source annotations."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets/pingpong_v1'
if (DATA / 'PREPARATION_IN_PROGRESS').exists():
    raise RuntimeError('Dataset copy is not complete.')
records = [json.loads(line) for line in (DATA / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
hashes = set()
counts = Counter()
groups = defaultdict(list)
for record in records:
    image, label = DATA / record['image'], DATA / record['label_path']
    assert image.is_file() and label.is_file(), record
    assert record['image_sha256'] not in hashes, 'Duplicate image content across dataset'
    hashes.add(record['image_sha256'])
    assert label.read_text(encoding='utf-8') == record['label']
    rows = label.read_text(encoding='utf-8').splitlines()
    assert len(rows) == record['objects']
    for row in rows:
        fields = row.split()
        assert len(fields) == 5 and fields[0] == '0'
        x, y, w, h = map(float, fields[1:])
        assert 0 < w <= 1 and 0 < h <= 1
        assert -1e-8 <= x - w / 2 and x + w / 2 <= 1 + 1e-8
        assert -1e-8 <= y - h / 2 and y + h / 2 <= 1 + 1e-8
    counts[record['split']] += 1
    groups[record['batch']].append(record)
for split in ('train', 'val', 'test'):
    assert len(list((DATA / 'images' / split).iterdir())) == counts[split]
    assert len(list((DATA / 'labels' / split).iterdir())) == counts[split]
summary = json.loads((DATA / 'dataset_summary.json').read_text(encoding='utf-8'))
assert dict(counts) == summary['split_counts']
canvas = Image.new('RGB', (4 * 480, len(groups) * 430), '#202020')
for row, (batch, items) in enumerate(sorted(groups.items())):
    for col in range(4):
        record = items[min(len(items) - 1, int((col + 0.5) * len(items) / 4))]
        with Image.open(DATA / record['image']) as original:
            original = original.convert('RGB')
            panel = original.resize((480, 384))
            draw = ImageDraw.Draw(panel)
            boxes = [list(map(float, line.split()[1:])) for line in record['label'].splitlines()]
            for x, y, w, h in boxes:
                draw.rectangle(((x - w / 2) * 480, (y - h / 2) * 384, (x + w / 2) * 480, (y + h / 2) * 384), outline='#00ff00', width=2)
            if boxes:
                x, y, w, h = boxes[0]
                cx, cy = x * original.width, y * original.height
                extent = max(24, w * original.width * 2, h * original.height * 2)
                crop = original.crop((int(cx - extent / 2), int(cy - extent / 2), int(cx + extent / 2), int(cy + extent / 2))).resize((112, 112))
                panel.paste(crop, (368, 0))
            canvas.paste(panel, (col * 480, row * 430))
            draw = ImageDraw.Draw(canvas)
            draw.text((col * 480 + 5, row * 430 + 388), batch + ' / ' + record['split'], fill='white')
            draw.text((col * 480 + 5, row * 430 + 407), Path(record['source_image']).name[:62], fill='white')
canvas.save(ROOT / 'reports/datasets/pingpong_v1/dataset_preview.jpg', quality=92)
result = {'verified_images': len(records), 'split_counts': counts, 'exact_image_hashes_disjoint': True, 'label_files_match_manifest': True, 'source_copy_sha256_verification': 'performed during preparation', 'preview': 'reports/datasets/pingpong_v1/dataset_preview.jpg'}
(ROOT / 'reports/datasets/pingpong_v1/dataset_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
