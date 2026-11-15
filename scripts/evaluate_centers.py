"""Evaluate ball centers and box overlap on a complete held-out split at fixed thresholds."""
import argparse
import json
import os
from pathlib import Path
import time

from diagnose_ball_centers import match_pairs

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))


def main():
    import numpy as np
    from ultralytics import YOLO
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--report-dir', default='reports/evaluation/baseline_yolo26n_960/center_selection_v1',
                        help='Workspace-relative report batch directory; choose a new directory for a new experiment')
    parser.add_argument('--square', action='store_true', help='Use fixed square letterbox instead of minimal rectangular padding')
    args = parser.parse_args()
    directory = (ROOT / args.report_dir).resolve()
    if not directory.is_relative_to((ROOT / 'reports/evaluation').resolve()):
        raise ValueError('--report-dir must be inside reports/evaluation.')
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / (args.name + '.json')
    if output.exists():
        raise RuntimeError('Evaluation report already exists; use a new --name.')
    dataset = ROOT / 'datasets/pingpong_v1'
    records = [json.loads(line) for line in (dataset / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    records = [r for r in records if r['split'] == args.split]
    report = {'split': args.split, 'images': len(records), 'thresholds': [.05, .25, .5],
              'preprocessing': {'imgsz': 960, 'batch': 16, 'rect': not args.square, 'nms': False},
              'matching': 'One-to-one greedy nearest centers within 5 or 10 original-image pixels; separate greedy IoU>=0.5 box matching.',
              'limits': 'All frames have annotated balls. This does not measure false positives on verified no-ball video, temporal tracking or stereo depth accuracy.',
              'evaluations': []}
    for weights in args.weights:
        model = YOLO(str(ROOT / weights))
        accumulated = {t: {'gt': 0, 'pred': 0, 'center5': 0, 'center10': 0, 'iou50': 0,
                           'errors': [], 'width_ratios': [], 'height_ratios': [],
                           'missed_frames_center5': 0, 'extra_prediction_frames_center5': 0} for t in report['thresholds']}
        started = time.time()
        frame_reviews = []
        for offset in range(0, len(records), 16):
            chunk = records[offset:offset+16]
            predictions = model.predict([str(dataset / r['image']) for r in chunk], imgsz=960,
                                        batch=16, device=0, rect=not args.square, conf=.05, nms=False, verbose=False)
            for record, prediction in zip(chunk, predictions):
                gt=[]
                for line in record['label'].splitlines():
                    _,x,y,w,h=map(float,line.split())
                    gt.append([(x-w/2)*record['width'],(y-h/2)*record['height'],
                               (x+w/2)*record['width'],(y+h/2)*record['height']])
                gt=np.asarray(gt)
                all_predictions=prediction.boxes.data.cpu().numpy()
                review={'image':record['image'], 'gt_boxes':gt.tolist(), 'thresholds':{}}
                for threshold, values in accumulated.items():
                    pred=all_predictions[all_predictions[:,4]>=threshold,:4]
                    distances=np.linalg.norm((gt[:,None,:2]+gt[:,None,2:])/2-(pred[None,:,:2]+pred[None,:,2:])/2,axis=-1)
                    inter=np.maximum(0,np.minimum(gt[:,None,2:],pred[None,:,2:])-np.maximum(gt[:,None,:2],pred[None,:,:2])).prod(axis=-1)
                    union=np.prod(gt[:,2:]-gt[:,:2],axis=-1)[:,None]+np.prod(pred[:,2:]-pred[:,:2],axis=-1)[None,:]-inter
                    iou=inter/np.maximum(union,1e-9)
                    pairs5=match_pairs(distances,5)
                    pairs10=match_pairs(distances,10)
                    values['gt']+=len(gt)
                    values['pred']+=len(pred)
                    values['center5']+=len(pairs5)
                    values['center10']+=len(pairs10)
                    values['iou50']+=len(match_pairs(iou,.5,maximize=True))
                    values['missed_frames_center5']+=len(pairs5)<len(gt)
                    values['extra_prediction_frames_center5']+=len(pred)>len(pairs5)
                    for a,b,distance in pairs10:
                        values['errors'].append(distance)
                        values['width_ratios'].append((pred[b,2]-pred[b,0])/(gt[a,2]-gt[a,0]))
                        values['height_ratios'].append((pred[b,3]-pred[b,1])/(gt[a,3]-gt[a,1]))
                    review['thresholds'][str(threshold)]={'predictions':len(pred), 'matched_center5':len(pairs5),
                        'matched_center10':len(pairs10), 'center10_errors':[p[2] for p in pairs10]}
                frame_reviews.append(review)
            if (offset+len(chunk))%256==0:
                print(f'{weights}: evaluated {offset+len(chunk)}/{len(records)} {args.split} images',flush=True)
        scores={}
        for threshold,values in accumulated.items():
            precision=values['center5']/max(values['pred'],1)
            recall=values['center5']/values['gt']
            scores[str(threshold)]={'ground_truth_boxes':values['gt'], 'predictions':values['pred'],
                'center5_precision':precision, 'center5_recall':recall,
                'center5_f1':2*precision*recall/max(precision+recall,1e-9),
                'center10_precision':values['center10']/max(values['pred'],1),
                'center10_recall':values['center10']/values['gt'], 'iou50_recall':values['iou50']/values['gt'],
                'center10_matched_error_px_median_p95':np.percentile(values['errors'],[50,95]).tolist() if values['errors'] else None,
                'center10_matched_pred_gt_width_ratio_median':float(np.median(values['width_ratios'])) if values['width_ratios'] else None,
                'center10_matched_pred_gt_height_ratio_median':float(np.median(values['height_ratios'])) if values['height_ratios'] else None,
                'missed_frames_center5':values['missed_frames_center5'],
                'extra_prediction_frames_center5':values['extra_prediction_frames_center5']}
        item={'weights':weights, 'seconds':time.time()-started, 'scores':scores}
        frame_path=directory/f'{args.name}_{Path(weights).stem}_frames.json'
        frame_path.write_text(json.dumps(frame_reviews),encoding='utf-8')
        item['frame_review_path']=str(frame_path)
        report['evaluations'].append(item)
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(item,indent=2),flush=True)
    if args.split=='val':
        chosen=max(report['evaluations'],key=lambda r:r['scores']['0.25']['center5_f1'])
        report['selected_for_center_application']={'weights':chosen['weights'],
            'criterion':'Highest full-validation center F1 within 5 original pixels at the fixed conf=0.25; test results not used.',
            'conf':.25, 'center5_f1':chosen['scores']['0.25']['center5_f1']}
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
