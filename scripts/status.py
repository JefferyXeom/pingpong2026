"""Read workspace progress, persistent training process and current GPU usage."""
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    import psutil
    report = {'free_gib': shutil.disk_usage(ROOT).free / 1024**3}
    for path in sorted((ROOT/'reports/training').glob('native_*/progress.json')):
        report[path.relative_to(ROOT).as_posix()]=json.loads(path.read_text(encoding='utf-8'))
    suite=ROOT/'reports/experiments/native_resolution_suite_v1/progress.json'
    if suite.exists():
        report['native_suite']=json.loads(suite.read_text(encoding='utf-8'))
    paths = ['reports/training/baseline_yolo26n_960/baseline_process.json', 'reports/training/baseline_yolo26n_960/baseline_yolo26n_960_progress.json', 'reports/training/baseline_yolo26n_960/baseline_yolo26n_960_result.json', 'datasets/pingpong_v1/dataset_summary.json', 'datasets/pingpong_additional_v1/dataset_summary.json', 'reports/datasets/pingpong_additional_v1/additional_dataset_verification.json']
    for relative in paths:
        path = ROOT / relative
        if path.is_file():
            report[relative] = json.loads(path.read_text(encoding='utf-8'))
    card_path = ROOT / 'models/pingpong_yolo26n_center_v1/model_card.json'
    if card_path.is_file():
        card = json.loads(card_path.read_text(encoding='utf-8'))
        report['center_model_candidate'] = {
            'weights': 'models/pingpong_yolo26n_center_v1/weights.pt',
            'source_epoch': card['source_epoch'], 'conf': card['selection']['conf'],
            'validation_center5_f1': card['validation']['scores']['0.25']['center5_f1'],
            'test_center5_f1': card['test']['scores']['0.25']['center5_f1'],
            'note': 'Center metrics are supplemental; box AP and no-ball false positives remain separate.'}
    state = report.get('reports/training/baseline_yolo26n_960/baseline_process.json')
    if state:
        try:
            process = psutil.Process(state['pid'])
            report['baseline_process_alive'] = abs(process.create_time() - state['process_created_at']) < 1
        except psutil.NoSuchProcess:
            report['baseline_process_alive'] = False
    running = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_times']):
        try:
            args = process.info['cmdline'] or []
            monitored_scripts = ('export_validate.py', 'train.py', 'audit_additional_data.py', 'prepare_additional_dataset.py',
                                 'run_native_suite.py','train_native_experiment.py','evaluate_conversion_matrix.py','rebenchmark_conversion_matrix.py')
            if process.info['name'] and process.info['name'].lower() == 'python.exe' and any(any(name in arg for name in monitored_scripts) for arg in args):
                cpu = process.info['cpu_times']
                running.append({'pid': process.pid, 'command': args, 'elapsed_seconds': time.time() - process.info['create_time'], 'cpu_seconds': cpu.user + cpu.system})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    report['training_and_export_processes'] = running
    for relative in ['logs/baseline.stdout.log', 'logs/baseline.stderr.log']:
        path = ROOT / relative
        if path.is_file():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 3000))
                report[relative] = stream.read().decode('utf-8', errors='replace')
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv'], capture_output=True, text=True, timeout=15)
    report['gpu'] = result.stdout.strip() or result.stderr.strip()
    print(json.dumps(report, indent=2, ensure_ascii=True), flush=True)

if __name__ == '__main__':
    main()
