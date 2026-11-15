"""Create an isolated CUDA training/export environment in this workspace."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / '.venv'
REPORTS = ROOT / 'reports/environment/windows_rtx3090'
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

def run(args):
    print('Running:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)

def space_check():
    free = shutil.disk_usage(ROOT).free
    if free < 12 * 1024**3:
        raise RuntimeError('Environment setup requires at least 12 GiB free; preserve the 5 GiB floor.')

space_check()
if not ENV.exists():
    venv.EnvBuilder(with_pip=True).create(ENV)
python = ENV / 'Scripts' / 'python.exe'
if not python.is_file():
    raise RuntimeError('Existing .venv does not contain a Windows Python interpreter; refusing to replace it.')
REPORTS.mkdir(parents=True, exist_ok=True)
run([python, '-m', 'pip', 'install', '--no-cache-dir', '--upgrade', 'pip'])
space_check()
run([python, '-m', 'pip', 'install', '--no-cache-dir', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu128'])
space_check()
run([python, '-m', 'pip', 'install', '--no-cache-dir', 'ultralytics>=8.4.0', 'onnx', 'onnxslim', 'onnxruntime-gpu'])
space_check()
try:
    run([python, '-m', 'pip', 'install', '--no-cache-dir', 'tensorrt-cu12>=10,<11'])
except subprocess.CalledProcessError as exc:
    (REPORTS / 'tensorrt_install_issue.json').write_text(json.dumps({'returncode': exc.returncode, 'note': 'Training environment installed; TensorRT installation needs a separate follow-up.'}, indent=2), encoding='utf-8')
    print('TensorRT installation failed. Training environment remains usable.', flush=True)
freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
(REPORTS / 'environment_freeze.txt').write_text(freeze, encoding='utf-8')
run([python, '-c', "import torch,ultralytics; print('ultralytics',ultralytics.__version__); print('torch',torch.__version__); print('cuda',torch.version.cuda,torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"])
