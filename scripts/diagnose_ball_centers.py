"""Distinguish missed balls from box-size mismatch using fixed diagnostic samples."""
from collections import defaultdict
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))


def match_pairs(matrix, limit, maximize=False):
    candidates = [(float(value), i, j) for i, row in enumerate(matrix) for j, value in enumerate(row)
                  if (value >= limit if maximize else value <= limit)]
    candidates.sort(reverse=maximize)
    seen_left, seen_right, pairs = set(), set(), []
    for value, i, j in candidates:
        if i not in seen_left and j not in seen_right:
            seen_left.add(i)
            seen_right.add(j)
            pairs.append((i, j, value))
    return pairs


def main():
    import numpy as np
    from PIL import Image, ImageDraw
    from ultralytics import YOLO
    data = ROOT / 'datasets/pingpong_v1'
    records = [json.loads(line) for line in (data / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    record_by_image = {r['image']: r for r in records}
    subsets = {}
    for subset in ('train_subset', 'val_subset'):
        paths = (ROOT / f'reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/baseline_diagnostics/{subset}.txt').read_text(encoding='utf-8').splitlines()
        subsets[subset] = [record_by_image[Path(p).relative_to(data).as_posix()] for p in paths]
    report = {'scope': 'Same fixed diagnostic subsets as head evaluation. Center radius in original 1280x1024 pixels; center matches are a supplemental measure, not mAP or proof of annotation errors.', 'evaluations': []}
    preview = Image.new('RGB', (4*480, 4*300), '#202020')
    selected_preview_indices = [int((i+.5)*len(subsets['val_subset'])/8) for i in range(8)]
    for model_index, checkpoint in enumerate(('best', 'last')):
        model = YOLO(str(ROOT / f'reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/baseline_diagnostics/{checkpoint}.pt'))
        for subset, selected in subsets.items():
            totals = {threshold: defaultdict(list) for threshold in (.05, .25, .5)}
            for offset in range(0, len(selected), 16):
                chunk = selected[offset:offset+16]
                predictions = model.predict([str(data / r['image']) for r in chunk], imgsz=960,
                                            batch=16, device=0, rect=True, conf=.05,
                                            nms=False, verbose=False)
                for index, (record, prediction) in enumerate(zip(chunk, predictions), offset):
                    ground_truth = []
                    for line in record['label'].splitlines():
                        _, x, y, w, h = map(float, line.split())
                        ground_truth.append([(x-w/2)*1280, (y-h/2)*1024, (x+w/2)*1280, (y+h/2)*1024])
                    gt = np.asarray(ground_truth)
                    all_predictions = prediction.boxes.data.cpu().numpy()
                    for threshold, accumulator in totals.items():
                        pred = all_predictions[all_predictions[:,4] >= threshold, :4]
                        distances = np.linalg.norm(((gt[:,None,:2]+gt[:,None,2:])/2) - ((pred[None,:,:2]+pred[None,:,2:])/2), axis=-1)
                        inter = np.maximum(0, np.minimum(gt[:,None,2:], pred[None,:,2:])-np.maximum(gt[:,None,:2], pred[None,:,:2])).prod(axis=-1)
                        union = np.prod(gt[:,2:]-gt[:,:2], axis=-1)[:,None] + np.prod(pred[:,2:]-pred[:,:2], axis=-1)[None,:] - inter
                        iou = inter / np.maximum(union, 1e-9)
                        accumulator['gt'].append(len(gt))
                        accumulator['pred'].append(len(pred))
                        accumulator['center5'].append(len(match_pairs(distances, 5)))
                        pairs = match_pairs(distances, 10)
                        accumulator['center10'].append(len(pairs))
                        accumulator['iou50'].append(len(match_pairs(iou, .5, maximize=True)))
                        for a,b,distance in pairs:
                            accumulator['center_error'].append(distance)
                            accumulator['width_ratio'].append((pred[b,2]-pred[b,0])/(gt[a,2]-gt[a,0]))
                            accumulator['height_ratio'].append((pred[b,3]-pred[b,1])/(gt[a,3]-gt[a,1]))
                    if subset == 'val_subset' and index in selected_preview_indices:
                        tile_index = model_index*8 + selected_preview_indices.index(index)
                        row, column = divmod(tile_index,4)
                        with Image.open(data / record['image']) as opened:
                            image = opened.convert('RGB')
                        tile = image.resize((320,256))
                        draw = ImageDraw.Draw(tile)
                        for box in gt:
                            draw.rectangle(tuple(box*np.array([.25,.25,.25,.25])), outline='lime', width=2)
                        kept = all_predictions[all_predictions[:,4]>=.25]
                        for box in kept[:,:4]:
                            draw.rectangle(tuple(box*.25), outline='magenta', width=1)
                        preview.paste(tile, (column*480,row*300))
                        cx,cy=(gt[0,:2]+gt[0,2:])/2
                        left,top=round(cx)-32,round(cy)-32
                        crop=image.crop((left,top,left+64,top+64)).resize((160,160))
                        draw=ImageDraw.Draw(crop)
                        for boxes,color in [(gt,'lime'),(kept[:,:4],'magenta')]:
                            for box in boxes:
                                coordinates=(box-np.array([left,top,left,top]))*2.5
                                if coordinates[2]>=0 and coordinates[3]>=0 and coordinates[0]<160 and coordinates[1]<160:
                                    draw.rectangle(tuple(coordinates),outline=color,width=2)
                        preview.paste(crop,(column*480+320,row*300))
                        draw=ImageDraw.Draw(preview)
                        draw.text((column*480+4,row*300+260),f'{checkpoint}: green GT; magenta pred conf>=0.25',fill='white')
                        draw.text((column*480+4,row*300+278),Path(record['image']).stem[-62:],fill='white')
            results={}
            for threshold, values in totals.items():
                g,p=sum(values['gt']),sum(values['pred'])
                results[str(threshold)]={'ground_truth_boxes':g,'predictions':p,
                    'iou50_recall':sum(values['iou50'])/g,
                    'center5_recall':sum(values['center5'])/g,
                    'center10_recall':sum(values['center10'])/g,
                    'center10_precision':sum(values['center10'])/max(p,1),
                    'center10_matched_median_error_px':float(np.median(values['center_error'])) if values['center_error'] else None,
                    'center10_matched_median_pred_gt_width_ratio':float(np.median(values['width_ratio'])) if values['width_ratio'] else None,
                    'center10_matched_median_pred_gt_height_ratio':float(np.median(values['height_ratio'])) if values['height_ratio'] else None}
            item={'checkpoint':checkpoint,'subset':subset,'thresholds':results}
            report['evaluations'].append(item)
            (ROOT / 'reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/baseline_center_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print(json.dumps(item),flush=True)
    preview.save(ROOT / 'reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/baseline_prediction_crops.jpg',quality=92)


if __name__=='__main__':
    main()
