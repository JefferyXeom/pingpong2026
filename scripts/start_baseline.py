"""Launch the long baseline with persistent log files and a recorded process ID."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume last.pt and append to existing logs')
    args = parser.parse_args()
    import psutil
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    state_path = ROOT / 'reports/training/baseline_yolo26n_960/baseline_process.json'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        try:
            previous = psutil.Process(state['pid'])
            if abs(previous.create_time() - state['process_created_at']) < 1:
                raise RuntimeError('Recorded baseline process is already running.')
        except psutil.NoSuchProcess:
            pass
    run_dir = ROOT / 'runs/detect/baseline_yolo26n_960'
    if args.resume and not (run_dir / 'weights/last.pt').is_file():
        raise RuntimeError('No baseline last.pt checkpoint to resume.')
    if run_dir.exists() and not args.resume:
        raise RuntimeError('Baseline output already exists. Review or resume it; refusing a duplicate run.')
    for process in psutil.process_iter(['name', 'cmdline']):
        try:
            command = process.info['cmdline'] or []
            if (process.info['name'] or '').lower() == 'python.exe' and any(Path(arg).name == 'train.py' for arg in command) and '--stage' in command and 'baseline' in command:
                raise RuntimeError('A baseline training process already exists; refusing a second GPU job.')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    stdout_path, stderr_path = logs / 'baseline.stdout.log', logs / 'baseline.stderr.log'
    child_environment = os.environ.copy()
    child_environment['PYTHONIOENCODING'] = 'utf-8'
    child_environment['PYTHONUTF8'] = '1'
    command = [sys.executable, '-u', str(ROOT / 'scripts/train.py'), '--stage', 'baseline']
    if args.resume:
        command.append('--resume')
    with stdout_path.open('ab' if args.resume else 'wb') as stdout, stderr_path.open('ab' if args.resume else 'wb') as stderr:
        if args.resume:
            stdout.write(f'\nResuming manually interrupted baseline at {time.strftime("%Y-%m-%d %H:%M:%S")}\n'.encode('utf-8'))
            stdout.flush()
        proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, creationflags=flags, env=child_environment)
    if state_path.exists():
        with (ROOT / 'reports/training/baseline_yolo26n_960/baseline_launch_history.jsonl').open('a', encoding='utf-8') as history:
            history.write(json.dumps(json.loads(state_path.read_text(encoding='utf-8'))) + '\n')
    state = {'pid': proc.pid, 'process_created_at': psutil.Process(proc.pid).create_time(), 'started_at_unix': time.time(), 'command': command, 'resume': args.resume, 'stdout': str(stdout_path), 'stderr': str(stderr_path), 'run_directory': str(run_dir), 'epochs': 100, 'early_stopping_patience': 25, 'imgsz': 960, 'batch': 16, 'nms_free': True}
    state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    time.sleep(5)
    if proc.poll() is not None:
        raise RuntimeError('Baseline exited during startup. Inspect the recorded logs.')
    print(json.dumps(state, indent=2), flush=True)

if __name__ == '__main__':
    main()
