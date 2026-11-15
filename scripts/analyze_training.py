"""Summarize full-run metrics, dataset sizes and measured training time; draw exportable learning curves."""
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = ROOT / 'runs/detect/baseline_yolo26n_960'
    with (run / 'results.csv').open(encoding='utf-8', newline='') as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    durations = []
    previous = 0
    completed_session_seconds = 0
    for row in rows:
        duration = row['time'] - previous
        if duration < 0:
            completed_session_seconds += previous
        if duration > 0:
            durations.append(duration)
        previous = row['time']
    total_training_seconds = completed_session_seconds + previous
    # A resume resets the CSV time column. Use positive, later epoch differences for ETA.
    typical = statistics.median(durations[-10:])
    best = max(rows, key=lambda r: r['metrics/mAP50-95(B)'])
    records = [json.loads(line) for line in (ROOT / 'datasets/pingpong_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    datasets = {}
    for split in ('train', 'val', 'test'):
        selected = [r for r in records if r['split'] == split]
        datasets[split] = {'images': len(selected), 'objects': sum(r['objects'] for r in selected),
                           'image_gib': sum(r['bytes'] for r in selected) / 1024**3,
                           'recordings': dict(Counter(r['batch'] for r in selected))}
    completed = int(rows[-1]['epoch'])
    result_path = ROOT / 'reports/training/baseline_yolo26n_960/baseline_yolo26n_960_result.json'
    finished = result_path.exists()
    report = {'updated_at_utc': datetime.now(timezone.utc).isoformat(),
              'run': str(run), 'normal_completion_report_exists': finished, 'completed_epochs': completed,
              'best_epoch': int(best['epoch']), 'best_epoch_metrics': best, 'latest_epoch_metrics': rows[-1],
              'dataset': datasets, 'batch': 16, 'training_batches_per_epoch': 414, 'input_size': 960,
              'median_recent_epoch_seconds': typical, 'configured_max_epochs': 100, 'early_stopping_patience': 25,
              'total_recorded_training_minutes_excluding_pause': total_training_seconds / 60,
              'estimated_remaining_minutes_if_max_epochs': 0 if finished else (100 - completed) * typical / 60,
              'estimated_early_stop_epoch_if_no_improvement': int(best['epoch']) + 25,
              'estimated_remaining_minutes_if_no_improvement': 0 if finished else max(0, int(best['epoch']) + 25 - completed) * typical / 60,
              'time_note': 'Measured epoch time includes training and validation. CSV timer resets on resume; negative differences omitted. Startup, final validation and competing jobs can add time.',
              'manual_interruption': 'User confirmed previous stop was manual. Resumed from completed epoch 24.',
              'interpretation': 'Box AP deteriorates after warmup while training AP improves. Supplemental center checks show strong ball localization with predictions about half the validation box dimensions. Low box AP alone must not be interpreted as failed ball localization. Review box annotation consistency; keep original labels and splits unchanged.'}
    if finished:
        report['final_best_validation'] = json.loads(result_path.read_text(encoding='utf-8'))
        progress_path = ROOT / 'reports/training/baseline_yolo26n_960/baseline_yolo26n_960_progress.json'
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        progress.update(epoch_completed=completed, state='completed', metrics_source='final_best_validation')
        progress_path.write_text(json.dumps(progress, indent=2), encoding='utf-8')
    (ROOT / 'reports/training/baseline_yolo26n_960/baseline_training_analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    epochs = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for key, label in [('train/box_loss', 'Train'), ('val/box_loss', 'Validation')]:
        axes[0].plot(epochs, [r[key] for r in rows], label=label)
    axes[0].set(title='Box loss', ylabel='Loss')
    for key, label in [('train/cls_loss', 'Train'), ('val/cls_loss', 'Validation')]:
        axes[1].plot(epochs, [r[key] for r in rows], label=label)
    axes[1].set(title='Classification loss', ylabel='Loss')
    for key, label in [('metrics/mAP50(B)', 'mAP50'), ('metrics/mAP50-95(B)', 'mAP50-95')]:
        axes[2].plot(epochs, [100 * r[key] for r in rows], label=label)
    axes[2].set(title='Validation average precision', ylabel='AP (%)')
    for ax in axes:
        ax.axvline(best['epoch'], color='gray', linestyle=':', linewidth=1, label=f'Best epoch {int(best["epoch"])}')
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('YOLO26n / 960 / pingpong_v1: full validation per epoch')
    fig.tight_layout()
    fig.savefig(ROOT / 'reports/training/baseline_yolo26n_960/baseline_learning_curves.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
