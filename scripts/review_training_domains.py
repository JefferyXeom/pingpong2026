"""Inspect image/ball distributions and render train-versus-validation annotation crops."""
from collections import defaultdict
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))


def main():
    import numpy as np
    from PIL import Image, ImageDraw
    from ultralytics import YOLO
    data = ROOT / 'datasets/pingpong_v1'
    records = [json.loads(line) for line in (data / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = defaultdict(list)
    for r in records:
        if r['split'] != 'test':
            groups[r['batch']].append(r)
    result = {}
    preview = Image.new('RGB', (4 * 480, len(groups) * 330), '#202020')
    for row, (batch, source) in enumerate(sorted(groups.items())):
        sizes, centers = [], []
        for record in source:
            for line in record['label'].splitlines():
                _, x, y, w, h = map(float, line.split())
                sizes.append([w * record['width'], h * record['height']])
                centers.append([x, y])
        brightness = []
        for i in range(32):
            record = source[int(i * len(source) / 32)]
            with Image.open(data / record['image']) as image:
                brightness.append(float(np.asarray(image.resize((160, 128)).convert('L')).mean()))
        result[batch] = {'split': source[0]['split'], 'images': len(source),
                         'box_min_side_pixel_percentiles_10_50_90': np.percentile(np.min(sizes, axis=1), [10, 50, 90]).tolist(),
                         'box_center_xy_mean': np.mean(centers, axis=0).tolist(),
                         'mean_image_luma_32_sample_mean': float(np.mean(brightness))}
        for column in range(4):
            record = source[int((column + .5) * len(source) / 4)]
            with Image.open(data / record['image']) as opened:
                image = opened.convert('RGB')
            tile = image.resize((360, 288))
            draw = ImageDraw.Draw(tile)
            boxes = [list(map(float, line.split())) for line in record['label'].splitlines()]
            for _, x, y, w, h in boxes:
                draw.rectangle(((x-w/2)*360, (y-h/2)*288, (x+w/2)*360, (y+h/2)*288), outline='lime', width=2)
            preview.paste(tile, (column * 480, row * 330))
            _, x, y, w, h = boxes[0]
            cx, cy = x * image.width, y * image.height
            crop = image.crop((round(cx)-24, round(cy)-24, round(cx)+24, round(cy)+24)).resize((120, 120))
            crop_draw = ImageDraw.Draw(crop)
            crop_draw.rectangle((60-w*image.width*1.25, 60-h*image.height*1.25,
                                 60+w*image.width*1.25, 60+h*image.height*1.25), outline='lime', width=2)
            preview.paste(crop, (column * 480 + 360, row * 330))
            draw = ImageDraw.Draw(preview)
            draw.text((column*480+5, row*330+291), f'{record["split"]} / {batch}', fill='white')
            draw.text((column*480+5, row*330+309), Path(record['image']).stem[-65:], fill='white')
    preview.save(ROOT / 'reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/train_val_domain_preview.jpg', quality=92)
    report = {'recording_stats': result, 'luma_note': 'Mean grayscale on 32 uniform samples per recording, 0-255. Descriptive only; not a causal test.',
              'preview': 'reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/train_val_domain_preview.jpg', 'test_set_used': False}
    (ROOT / 'reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/train_val_domain_review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
