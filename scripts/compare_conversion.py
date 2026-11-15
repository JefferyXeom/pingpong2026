"""Compare saved backend detections by spatial matching, independent of row order."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def match(reference, candidate):
    a = np.asarray(reference, dtype=np.float64).reshape(-1, 6)
    b = np.asarray(candidate, dtype=np.float64).reshape(-1, 6)
    pairs = []
    used = set()
    for i in np.argsort(-a[:, 4]):
        if not len(b):
            break
        lo = np.maximum(a[i, :2], b[:, :2])
        hi = np.minimum(a[i, 2:4], b[:, 2:4])
        intersection = np.maximum(0, hi - lo).prod(axis=1)
        aa = np.maximum(0, a[i, 2:4] - a[i, :2]).prod()
        ba = np.maximum(0, b[:, 2:4] - b[:, :2]).prod(axis=1)
        ious = intersection / np.maximum(aa + ba - intersection, 1e-12)
        ious[(b[:, 5] != a[i, 5]) | (np.abs(b[:, 4] - a[i, 4]) > 0.005)] = -1
        if used:
            ious[list(used)] = -1
        j = int(np.argmax(ious))
        if ious[j] >= 0.95:
            used.add(j)
            pairs.append((i, j, float(ious[j])))
    return {'pytorch_detections': len(a), 'backend_detections': len(b), 'matched_iou_at_least_095_and_score_gap_under_0005': len(pairs), 'unmatched_pytorch': len(a) - len(pairs), 'unmatched_backend': len(b) - len(pairs), 'max_matched_score_difference': max((float(abs(a[i, 4] - b[j, 4])) for i, j, _ in pairs), default=None), 'max_matched_coordinate_difference_pixels': max((float(np.max(np.abs(a[i, :4] - b[j, :4]))) for i, j, _ in pairs), default=None)}

def main():
    output = {}
    for report_path in sorted((ROOT / 'artifacts').glob('*/conversion_report.json')):
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if 'predictions' not in report:
            continue
        results = {}
        for backend in ('onnx_cpu', 'tensorrt_fp16'):
            comparisons = [match(a, b) for a, b in zip(report['predictions']['pytorch'], report['predictions'][backend])]
            matched = sum(r['matched_iou_at_least_095_and_score_gap_under_0005'] for r in comparisons)
            reference_count = sum(r['pytorch_detections'] for r in comparisons)
            results[backend] = {'matched_reference_fraction': matched / reference_count if reference_count else None, 'images': comparisons}
        output[report_path.parent.name] = {'report': str(report_path), 'confidence_threshold': 0.05, 'comparisons': results, 'scope': 'Spatial agreement of saved smoke detections, not ground-truth accuracy. Count differences near confidence cutoff and unmatched boxes require baseline validation.'}
    destination=ROOT / 'reports/evaluation/smoke_conversion_v1/conversion_comparison.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding='utf-8')
    for name, result in output.items():
        print(json.dumps({'artifact': name, 'spatial_match_fraction': {k: v['matched_reference_fraction'] for k, v in result['comparisons'].items()}}, indent=2))

if __name__ == '__main__':
    main()
