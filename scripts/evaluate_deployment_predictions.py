"""Compare portable original-coordinate predictions with GT and source PT caches.

Requires numpy only. No inference, box-IoU ranking, or threshold selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def pairs(distance, radius):
    candidates = sorted((float(d), i, j) for i, row in enumerate(distance) for j, d in enumerate(row) if d <= radius)
    left, right, found = set(), set(), []
    for d, i, j in candidates:
        if i not in left and j not in right:
            found.append((i, j, d)); left.add(i); right.add(j)
    return found


def centers(boxes):
    return (boxes[:, :2] + boxes[:, 2:4]) / 2


def distances(a, b):
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def load_predictions(path, count):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {f'f{i}' for i in range(count)}:
            raise ValueError('Prediction cache must contain f0..fN-1 in image_order order')
        predictions = [archive[f'f{i}'].copy() for i in range(count)]
    for p in predictions:
        if p.ndim != 2 or p.shape[1] != 6 or not np.isfinite(p).all() or np.any(p[:, 5] != 0):
            raise ValueError('Expected finite Nx6 xyxy,conf,class=0 arrays')
        if np.any(p[:, 4] < 0) or np.any(p[:, 4] > 1) or np.any(p[:, 2:4] < p[:, :2]):
            raise ValueError('Invalid confidence or xyxy coordinates')
    return predictions


def score(records, predictions, conf=.25):
    metrics = {}
    for split in ['val', 'test']:
        gt_count = pred_count = tp = tp10 = 0
        errors = []
        for record, prediction in zip(records, predictions):
            if record['split'] != split:
                continue
            gt = np.asarray(record['gt_xyxy_px'], dtype=np.float64).reshape(-1, 4)
            pred = prediction[prediction[:, 4] >= conf]
            d = distances(centers(gt), centers(pred))
            found = pairs(d, 5); found10 = pairs(d, 10)
            gt_count += len(gt); pred_count += len(pred); tp += len(found); tp10 += len(found10)
            errors.extend(d for _, _, d in found10)
        p, r = tp / max(pred_count, 1), tp / max(gt_count, 1)
        metrics[split] = {'images': sum(r['split'] == split for r in records), 'gt': gt_count,
                          'predictions': pred_count, 'tp': tp, 'fp': pred_count-tp, 'fn': gt_count-tp,
                          'precision': p, 'recall': r, 'f1': 2*p*r/max(p+r, 1e-9),
                          'matched_within_10px': tp10,
                          'error_p50_p95_p99_px': np.percentile(errors, [50,95,99]).tolist() if errors else None}
    return metrics


def compare(records, reference, candidate, conf=.25):
    shifts, gaps = [], []
    lost = recovered = unmatched_ref = unmatched_new = 0
    for record, ref, new in zip(records, reference, candidate):
        ref = ref[ref[:,4] >= conf]; new = new[new[:,4] >= conf]
        found = pairs(distances(centers(ref), centers(new)), 10)
        unmatched_ref += len(ref)-len(found); unmatched_new += len(new)-len(found)
        for a,b,d in found:
            shifts.append(d); gaps.append(abs(float(ref[a,4]-new[b,4])))
        gt = np.asarray(record['gt_xyxy_px'], dtype=np.float64).reshape(-1,4)
        before = {i for i,_,_ in pairs(distances(centers(gt),centers(ref)),5)}
        after = {i for i,_,_ in pairs(distances(centers(gt),centers(new)),5)}
        lost += len(before-after); recovered += len(after-before)
    return {'scope': 'val+test, conf>=.25, one-to-one prediction pairing within10px; may include shared false detections',
            'pairs': len(shifts), 'shift_p50_p95_p99_max_px': np.percentile(shifts,[50,95,99,100]).tolist() if shifts else None,
            'score_abs_gap_p50_p95_p99_max': np.percentile(gaps,[50,95,99,100]).tolist() if gaps else None,
            'new_gt_misses_at_5px': lost, 'recovered_gt_at_5px': recovered,
            'unmatched_reference_predictions': unmatched_ref, 'unmatched_candidate_predictions': unmatched_new}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--case', default='h1024_w1280_pytorch_fp32_b2')
    parser.add_argument('--subset', choices=['full','quick128'], default='full')
    parser.add_argument('--predictions', type=Path, help='Target-machine predictions in original coordinates; omit to verify packaged cache')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.core/'package.json').read_text(encoding='utf-8'))
    model = next(r for r in manifest['models'] if r['id'] == args.model)
    records = json.loads((args.core/'testdata/image_order.json').read_text(encoding='utf-8'))
    count=len(records)
    indices=list(range(count)) if args.subset=='full' else json.loads((args.core/'testdata/quick128_indices.json').read_text(encoding='utf-8'))
    case = next(c for c in model['cases'] if c['key'] == args.case)
    path = args.predictions or args.core/case['reference_predictions']
    if args.predictions:
        pred=load_predictions(path,len(indices))
    else:
        cached=load_predictions(path,count)
        pred=[cached[i] for i in indices]
    ref_key = f'h1024_w1280_pytorch_fp32_b{case["batch"]}'
    ref_case = next(c for c in model['cases'] if c['key'] == ref_key)
    ref_all=load_predictions(args.core/ref_case['reference_predictions'],count)
    ref=[ref_all[i] for i in indices]
    records=[records[i] for i in indices]
    result = {'model': args.model, 'case': args.case, 'subset':args.subset,'conf': .25, 'radius_px': 5,
              'metrics': score(records, pred), 'against_source_pytorch': compare(records, ref, pred)}
    if not args.predictions:
        expected=case['expected'] if args.subset=='full' else case['expected_quick128']
        for split in ['val','test']:
            for metric in ['tp','fp','fn','gt','predictions']:
                assert result['metrics'][split][metric] == expected[split][metric]
            assert np.allclose(result['metrics'][split]['error_p50_p95_p99_px'],expected[split]['error_p50_p95_p99_px'],atol=1e-7,rtol=0)
        result['packaged_reference_verified'] = True
    if args.output:
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
