"""Audit actual staged/tracked blobs without requiring training artifacts."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE = 2 * 1024**2
MAX_TOTAL = 20 * 1024**2
BANNED_DIRS = {'.venv', '.tools', '.cache', '.ultralytics', '.vscode', '.idea', 'datasets', 'runs', 'artifacts', 'logs', 'transfers', 'recordings'}
BANNED_EXT = {'.pt','.pth','.onnx','.engine','.plan','.npz','.npy','.zip','.7z','.png','.jpg','.jpeg','.mp4','.avi','.mov','.mkv','.dll','.pyd','.so','.key','.pem','.safetensors','.bin','.bag'}


def git_binary():
    local = ROOT / '.tools/git/cmd/git.exe'
    if local.exists():
        return str(local)
    binary = shutil.which('git')
    if not binary:
        raise RuntimeError('Git unavailable; install Git or prepare .tools/git')
    return binary


def git(*args):
    return subprocess.check_output([git_binary(), '-C', str(ROOT), *args])


def audit():
    entries = git('ls-files', '--stage', '-z').split(b'\0')
    rows, problems = [], []
    for entry in entries:
        if not entry:
            continue
        header, raw_path = entry.split(b'\t', 1)
        mode, oid, stage = header.decode().split()
        path = raw_path.decode('utf-8')
        if stage != '0':
            problems.append('Unresolved index stage: ' + path)
            continue
        size = int(git('cat-file', '-s', oid))
        rows.append({'path': path, 'bytes': size})
        p = Path(path)
        if p.parts[0] in BANNED_DIRS or p.suffix.lower() in BANNED_EXT:
            problems.append('Excluded content in index: ' + path)
        if p.parts[:2] == ('configs','datasets') or p.name == '.env' or (p.name.startswith('.env.') and p.name != '.env.example'):
            problems.append('Local config/secret candidate in index: ' + path)
        if path.startswith('models/') and p.name != 'model_card.json' and not (p.name.startswith('README') and p.suffix == '.md'):
            problems.append('Model payload is not metadata: ' + path)
        if size > MAX_FILE:
            problems.append(f'File exceeds 2MiB: {path} ({size})')
    total = sum(r['bytes'] for r in rows)
    if not rows:
        problems.append('Git index is empty')
    if total > MAX_TOTAL:
        problems.append(f'Index exceeds 20MiB: {total}')
    result = {'passed': not problems, 'files': len(rows), 'bytes': total, 'mib': round(total/1024**2,3),
              'per_file_limit_mib': 2, 'total_limit_mib': 20,
              'largest_files': sorted(rows,key=lambda r:r['bytes'],reverse=True)[:12], 'problems': problems}
    return result


if __name__ == '__main__':
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
