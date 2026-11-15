"""Reproducible YOLO26 smoke run or full baseline with disk-space guards."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))

def guard_space(_trainer=None):
    if shutil.disk_usage(ROOT).free < int(5.5 * 1024**3):
        raise RuntimeError('Training stopped to preserve at least 5 GiB free disk space.')

def record_progress(trainer):
    with Path(trainer.csv).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    completed = int(float(rows[-1]['epoch'])) if rows else 0
    final_validation = completed != int(trainer.epoch + 1)
    status = {'pid': os.getpid(), 'run': str(trainer.save_dir), 'epoch_completed': completed, 'epochs_requested': int(trainer.epochs), 'metrics': {k: float(v) for k, v in trainer.metrics.items()}, 'metrics_source': 'final_best_validation' if final_validation else 'latest_epoch_validation', 'state': 'completed' if final_validation else 'training', 'free_gib': shutil.disk_usage(ROOT).free / 1024**3, 'updated_at_unix': time.time()}
    directory = ROOT / 'reports/training' / Path(trainer.save_dir).name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (Path(trainer.save_dir).name + '_progress.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'baseline'], default='smoke')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--name')
    parser.add_argument('--data', help='Optional baseline data YAML, relative to the workspace')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    guard_space()
    from ultralytics import YOLO
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Refusing an accidental CPU training run.')
    dataset = ROOT / 'datasets/pingpong_v1'
    if (dataset / 'PREPARATION_IN_PROGRESS').exists() or not (dataset / 'dataset_summary.json').is_file():
        raise RuntimeError('Verified dataset preparation has not completed.')
    data = dataset / 'data.yaml'
    if args.data:
        if args.stage != 'baseline':
            raise ValueError('--data is supported for baseline runs; smoke uses its fixed subset.')
        data = (ROOT / args.data).resolve()
        if not data.is_relative_to(ROOT) or not data.is_file():
            raise ValueError('--data must name an existing YAML inside the workspace.')
    if args.stage == 'smoke':
        records = [json.loads(line) for line in (dataset / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        for split, limit in [('train', 192), ('val', 64)]:
            selected = [r for r in records if r['split'] == split]
            indices = [min(len(selected) - 1, int(i * len(selected) / min(limit, len(selected)))) for i in range(min(limit, len(selected)))]
            listing = ''.join((dataset / selected[index]['image']).as_posix() + '\n' for index in indices)
            (dataset / f'smoke_{split}.txt').write_text(listing, encoding='utf-8')
        data = dataset / 'data_smoke.yaml'
        data.write_text('train: smoke_train.txt\nval: smoke_val.txt\nnames:\n  0: ball\n', encoding='utf-8')
    name = args.name or f'{args.stage}_yolo26n_{args.imgsz}'
    reports = ROOT / 'reports/training' / name
    reports.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / 'runs/detect' / name
    if run_dir.exists() and not args.resume:
        raise RuntimeError('Run already exists; use --resume or a new --name. Existing results preserved.')
    (ROOT / 'models').mkdir(exist_ok=True)
    model = YOLO(str(run_dir / 'weights/last.pt') if args.resume else str(ROOT / 'models/yolo26n.pt'))
    model.add_callback('on_train_epoch_start', guard_space)
    model.add_callback('on_train_batch_end', guard_space)
    model.add_callback('on_fit_epoch_end', record_progress)
    if args.resume:
        result = model.train(resume=True, device=0)
    else:
        result = model.train(
            data=str(data), epochs=args.epochs or (1 if args.stage == 'smoke' else 100),
            imgsz=args.imgsz, batch=args.batch or (8 if args.stage == 'smoke' else 16),
            device=0, workers=2, cache=False, seed=20261108, deterministic=True,
            project=str(ROOT / 'runs/detect'), name=name, exist_ok=False,
            patience=25, save=True, save_period=-1, plots=True, val=True,
            amp=True, nms=False, optimizer='auto', mosaic=0.25, scale=0.25,
            close_mosaic=0 if args.stage == 'smoke' else 10,
            mixup=0.0, copy_paste=0.0, degrees=0.0, flipud=0.0, fliplr=0.5,
        )
    summary = {'stage': args.stage, 'run': str(model.trainer.save_dir), 'best': str(model.trainer.best), 'last': str(model.trainer.last), 'metrics': {key: float(value) for key, value in result.results_dict.items()}, 'free_gib': shutil.disk_usage(ROOT).free / 1024**3}
    (reports / f'{name}_result.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
