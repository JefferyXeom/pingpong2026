"""Compare checkpoint heads on fixed train/validation subsets; leave test set untouched."""
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))


def main():
    import torch
    from ultralytics import YOLO
    target = ROOT / 'reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/baseline_diagnostics'
    target.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / 'datasets/pingpong_v1'
    records = [json.loads(line) for line in (dataset / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    train_groups = defaultdict(list)
    for r in records:
        if r['split'] == 'train':
            train_groups[r['batch']].append(r)
    groups = {'train_subset': [], 'val_subset': [r for r in records if r['split'] == 'val']}
    for recording in sorted(train_groups):
        source = train_groups[recording]
        groups['train_subset'].extend(source[int(i * len(source) / 128)] for i in range(128))
    source = groups['val_subset']
    groups['val_subset'] = [source[int(i * len(source) / 256)] for i in range(256)]
    for name, subset in groups.items():
        listing = target / (name + '.txt')
        listing.write_text(''.join((dataset / r['image']).as_posix() + '\n' for r in subset), encoding='utf-8')
        (target / (name + '.yaml')).write_text(
            f'path: {dataset.as_posix()}\ntrain: {listing.as_posix()}\nval: {listing.as_posix()}\nnames:\n  0: ball\n', encoding='utf-8')
    snapshots = {}
    for name in ('best', 'last'):
        source = ROOT / 'runs/detect/baseline_yolo26n_960/weights' / (name + '.pt')
        snapshot = target / (name + '.pt')
        shutil.copy2(source, snapshot)
        checkpoint = torch.load(snapshot, map_location='cpu', weights_only=False)
        snapshots[name] = {'path': str(snapshot), 'completed_epoch': checkpoint['epoch'] + 1,
                           'stored_metrics': checkpoint.get('train_metrics'), 'stopper': checkpoint.get('stopper')}
        del checkpoint
    report = {'scope': 'Fixed diagnostic subsets: 128 images per training recording, 256 uniform validation samples. No test evaluation; subset metrics are not full-set estimates.',
              'subsets': {name: len(subset) for name, subset in groups.items()}, 'snapshots': snapshots, 'evaluations': []}
    report_path = ROOT / 'reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/baseline_diagnostics.json'
    experiments = [(name, subset, False) for name in snapshots for subset in groups]
    experiments.append(('last', 'val_subset', None))
    for name, subset, nms in experiments:
        head = 'nms_free' if nms is False else 'external_nms'
        started = time.time()
        model = YOLO(snapshots[name]['path'])
        result = model.val(data=str(target / (subset + '.yaml')), imgsz=960, batch=16,
                           device=0, workers=0, nms=nms, rect=True, plots=False,
                           conf=0.001, iou=0.7, max_det=300, verbose=False,
                           project=str(ROOT / 'runs/diagnostics'), name=f'{name}_{subset}_{head}',
                           exist_ok=False)
        item = {'checkpoint': name, 'checkpoint_epoch': snapshots[name]['completed_epoch'], 'subset': subset,
                'head': head, 'metrics': {k: float(v) for k, v in result.results_dict.items()},
                'seconds': time.time() - started}
        report['evaluations'].append(item)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(item), flush=True)
        del model, result
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
