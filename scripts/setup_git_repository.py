"""Prepare official portable Git if needed, audit the index, and make the first commit.

No remote creation, push, global identity change, or training is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / '.tools'


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'pingpong-yolo26-repository-setup', 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def install_git():
    local = TOOLS / 'git/cmd/git.exe'
    if local.exists():
        return str(local)
    system = shutil.which('git')
    if system:
        return system
    TOOLS.mkdir(exist_ok=True)
    print('Git is unavailable; querying official git-for-windows release metadata.', flush=True)
    release = json.loads(get('https://api.github.com/repos/git-for-windows/git/releases/latest'))
    assets = [a for a in release['assets'] if re.fullmatch(r'MinGit-[\d.]+(?:\([\d]+\))?-64-bit\.zip', a['name'])]
    if len(assets) != 1:
        raise RuntimeError('Expected exactly one standard x64 MinGit ZIP in the official release')
    asset = assets[0]
    url = asset['browser_download_url']
    if not url.startswith('https://github.com/git-for-windows/git/releases/download/'):
        raise RuntimeError('Unexpected release asset origin')
    archive = TOOLS / asset['name']
    if not archive.exists():
        print('Downloading official ' + asset['name'], flush=True)
        # The ZIP is binary; do not request the GitHub JSON content type.
        request = urllib.request.Request(url, headers={'User-Agent': 'pingpong-yolo26-repository-setup'})
        with urllib.request.urlopen(request, timeout=90) as response, archive.open('wb') as stream:
            shutil.copyfileobj(response, stream)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    upstream = asset.get('digest')
    if upstream and upstream != 'sha256:' + checksum:
        raise RuntimeError('GitHub asset digest mismatch; refusing to run the downloaded Git')
    target = TOOLS / 'git'
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            resolved = (target / member.filename).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise RuntimeError('Unsafe archive path')
        package.extractall(target)
    if not local.exists():
        raise RuntimeError('Official ZIP did not contain cmd/git.exe')
    receipt = {'release': release['tag_name'], 'asset': asset['name'], 'url': url,
               'sha256': checksum, 'github_digest_verified': bool(upstream), 'global_install': False}
    (TOOLS / 'git_install.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return str(local)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--email', required=True)
    args = parser.parse_args()
    if not args.name.strip() or '@' not in args.email or '\n' in args.name + args.email:
        raise ValueError('A real user-provided Git author name/email is required')
    binary = install_git()

    def git(*arguments, check=True):
        result = subprocess.run([binary, '-C', str(ROOT), *arguments], capture_output=True, text=True, encoding='utf-8', errors='replace')
        if check and result.returncode:
            raise RuntimeError('Git failed: ' + ' '.join(arguments) + '\n' + result.stderr[-2000:] + result.stdout[-2000:])
        return result

    if not (ROOT / '.git').exists():
        print(git('init', '--initial-branch=main').stdout.strip(), flush=True)
    if git('rev-parse', '--verify', 'HEAD', check=False).returncode == 0:
        raise RuntimeError('Repository already has commits; use normal Git workflow instead of setup')
    for key, value in [('user.name',args.name), ('user.email',args.email), ('core.autocrlf','false'), ('core.quotepath','false')]:
        git('config', '--local', key, value)
    assert not git('remote').stdout.strip(), 'Unexpected remote; review before staging'
    # VS Code setting is machine-local and ignored. Never overwrite an existing settings file.
    vscode = ROOT / '.vscode'
    vscode.mkdir(exist_ok=True)
    if not (vscode / 'settings.json').exists():
        (vscode / 'settings.json').write_text(json.dumps({'git.path': binary}, indent=2) + '\n', encoding='utf-8')
    git('add', '--all')
    from audit_git_content import audit
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result['passed']:
        raise RuntimeError('Index audit failed; no commit created')
    # Matplotlib SVG uses harmless trailing spaces in path coordinates; preserve
    # historical figures. Historical prose may end with an extra blank line.
    git('-c', 'core.whitespace=-blank-at-eof', 'diff', '--cached', '--check', '--', '.', ':(exclude)*.svg')
    # Check representative real and future paths without adding any binary files.
    ignored = ['datasets/pingpong_v1/manifest.jsonl', 'models/yolo26n.pt',
               'reports/evaluation/baseline_yolo26n_960/diagnostics_epoch24/baseline_diagnostics/best.pt',
               'reports/datasets/pingpong_additional_v1/additional_data_audit.json',
               'reports/datasets/pingpong_baidu_out_v1/label_quality_gallery.html',
               'artifacts/example/weights.onnx', 'runs/example/weights/best.pt', '.tools/git/cmd/git.exe',
               'configs/datasets/native_merged_v2_baiduout_20261113/pingpong_merged_v2_train.txt',
               'logs/example.log', 'recordings/example.mp4', '.env']
    for path in ignored:
        assert git('check-ignore', '--quiet', '--no-index', path, check=False).returncode == 0, path
    for path in ['scripts/train_native_experiment.py', 'configs/deployment_target.json', 'docs/status.md',
                 'models/pingpong_yolo26n_merged_v2_baiduout/model_card.json',
                 'reports/training/native_merged_v2_baiduout_yolo26n_1280/epoch_metrics.jsonl']:
        assert git('check-ignore', '--quiet', '--no-index', path, check=False).returncode == 1, path
    print(git('commit', '-m', 'Initialize training and deployment handoff repository').stdout[-1500:], flush=True)
    status = git('status', '--porcelain').stdout.strip()
    assert not status, status
    receipt = {'commit': git('rev-parse', 'HEAD').stdout.strip(), 'branch': git('branch', '--show-current').stdout.strip(),
               'author': git('log', '-1', '--format=%an <%ae>').stdout.strip(), 'git': binary,
               'git_version': git('--version').stdout.strip(), 'tracked_content': result,
               'ignore_rules_checked': len(ignored), 'working_tree_clean': True, 'remote_configured': False, 'pushed': False}
    (ROOT / 'repository_setup.local.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
