"""Record interrupted baseline evidence without starting or changing training."""
import csv
from datetime import datetime, timezone
import json

from audit_additional_data import ROOT


def main():
    import psutil
    import torch
    run = ROOT / 'runs/detect/baseline_yolo26n_960'
    with (run / 'results.csv').open(encoding='utf-8', newline='') as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    best = max(rows, key=lambda r: r['metrics/mAP50-95(B)'])
    checkpoint = torch.load(run / 'weights/last.pt', map_location='cpu', weights_only=False)
    state = json.loads((ROOT / 'reports/training/baseline_yolo26n_960/baseline_process.json').read_text(encoding='utf-8'))
    alive = False
    try:
        process = psutil.Process(state['pid'])
        alive = abs(process.create_time() - state['process_created_at']) < 1
    except psutil.NoSuchProcess:
        pass
    result = {'inspected_at_utc': datetime.now(timezone.utc).isoformat(),
              'recorded_launcher_alive': alive,
              'normal_result_report_exists': (ROOT / 'reports/training/baseline_yolo26n_960/baseline_yolo26n_960_result.json').exists(),
              'completed_epochs_in_csv': len(rows),
              'last_completed_epoch_metrics': rows[-1], 'best_completed_epoch_metrics': best,
              'last_checkpoint_epoch_zero_based': checkpoint.get('epoch'),
              'last_checkpoint_has_optimizer': checkpoint.get('optimizer') is not None,
              'stderr_bytes': (ROOT / 'logs/baseline.stderr.log').stat().st_size,
              'assessment': 'Recorded process absent, no normal completion report, log stopped within epoch 25. Cause not established. Existing last/best checkpoints retained; no automatic restart or data modification.'}
    (ROOT / 'reports/training/baseline_yolo26n_960/baseline_interruption_review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
