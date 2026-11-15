"""Prepare a repository-only SSH identity; never uploads or prints private keys."""
import argparse
import csv
import io
import html
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import urllib.request

from audit_git_content import audit, git_binary

ROOT = Path(__file__).resolve().parents[1]
REMOTE = 'git@github.com:JefferyXeom/pingpong2026.git'
DIRECTORY = ROOT / '.tools/ssh'
KEY = DIRECTORY / 'pingpong2026_training_ed25519'


def run(argv, check=True):
    result = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if check and result.returncode:
        raise RuntimeError(f'{Path(argv[0]).name} failed ({result.returncode}): {result.stderr[-2000:]}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit-handoff', action='store_true', help='Commit only the prepared handoff docs and this script')
    args = parser.parse_args()
    binary = git_binary()

    def git(*arguments, check=True):
        return run([binary, '-C', str(ROOT), *arguments], check=check)

    ssh = shutil.which('ssh')
    keygen = shutil.which('ssh-keygen')
    if not ssh or not keygen:
        raise RuntimeError('OpenSSH ssh and ssh-keygen are required')
    origin = git('remote', 'get-url', 'origin', check=False)
    if origin.returncode == 0 and origin.stdout.strip() not in [REMOTE, 'https://github.com/JefferyXeom/pingpong2026', 'https://github.com/JefferyXeom/pingpong2026.git']:
        raise RuntimeError('Origin points elsewhere; refusing to overwrite it')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    relative_key = KEY.relative_to(ROOT).as_posix()
    if git('check-ignore', '--quiet', '--no-index', relative_key, check=False).returncode != 0:
        raise RuntimeError('Private key path must be ignored before generation')
    # HTTPS metadata authenticates GitHub host keys before any SSH connection.
    known_hosts = DIRECTORY / 'github_known_hosts'
    host_source = 'existing local GitHub known-hosts file'
    if not known_hosts.exists():
        host_keys = []
        sources = ['https://api.github.com/meta',
                   'https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints']
        for source in sources:
            try:
                request = urllib.request.Request(source, headers={'User-Agent': 'pingpong2026-deploy-key-setup'})
                with urllib.request.urlopen(request, timeout=20) as response:
                    content = response.read().decode('utf-8')
                if source == sources[0]:
                    host_keys = json.loads(content).get('ssh_keys', [])
                else:
                    content = html.unescape(re.sub('<[^>]+>', ' ', content))
                    host_keys = [kind + ' ' + value for kind, value in re.findall(r'(ssh-ed25519)\s+(AAAA[A-Za-z0-9+/=]+)', content)]
                if not host_keys or any('\n' in key or '\r' in key for key in host_keys):
                    raise RuntimeError('Unexpected GitHub SSH host key metadata')
                host_source = source
                known_hosts.write_text(''.join('github.com ' + key + '\n' for key in dict.fromkeys(host_keys)), encoding='utf-8')
                break
            except Exception as exc:
                print('Host-key source unavailable: ' + source + ' (' + type(exc).__name__ + ')', flush=True)
        if not known_hosts.exists():
            host_source = 'pending; strict verification remains enabled; SSH will fail closed until host keys are installed'
    if not KEY.exists():
        if KEY.with_suffix('.pub').exists():
            raise RuntimeError('Public key exists without private key; refusing to overwrite')
        run([keygen, '-t', 'ed25519', '-f', str(KEY), '-N', '', '-C', 'pingpong2026-windows-training-deploy', '-q'])
    if os.name == 'nt':
        # Restrict the newly dedicated private key to the current Windows SID.
        identity = run(['whoami', '/user', '/fo', 'csv', '/nh']).stdout.strip()
        sid = next(csv.reader(io.StringIO(identity)))[-1]
        if not sid.startswith('S-1-'):
            raise RuntimeError('Unable to resolve current Windows SID')
        run(['icacls', str(KEY), '/inheritance:r', '/grant:r', '*' + sid + ':(F)'])
    public_fields = run([keygen, '-y', '-f', str(KEY)]).stdout.strip().split()
    if len(public_fields) < 2 or public_fields[0] != 'ssh-ed25519':
        raise RuntimeError('Unexpected public key format')
    public = ' '.join(public_fields[:2])
    public_line = public + ' pingpong2026-windows-training-deploy'
    pub_path = Path(str(KEY) + '.pub')
    if pub_path.exists() and ' '.join(pub_path.read_text().split()[:2]) != public:
        raise RuntimeError('Existing public key differs from private key')
    pub_path.write_text(public_line + '\n', encoding='utf-8')
    fingerprint = run([keygen, '-lf', str(pub_path), '-E', 'sha256']).stdout.strip()
    # All options are repository-local; no edits to personal ~/.ssh/config.
    ssh_command = (f'"{Path(ssh).as_posix()}" -i "{KEY.as_posix()}" '
                   f'-o IdentitiesOnly=yes -o BatchMode=yes '
                   f'-o UserKnownHostsFile="{known_hosts.as_posix()}" -o StrictHostKeyChecking=yes')
    git('config', '--local', 'core.sshCommand', ssh_command)
    if origin.returncode:
        git('remote', 'add', 'origin', REMOTE)
    else:
        git('remote', 'set-url', 'origin', REMOTE)
    if args.commit_handoff:
        changes = ['docs/status.md', 'docs/repository_workflow.zh-CN.md', 'scripts/prepare_github_deploy_key.py']
        # Do not accidentally commit anything the user has staged independently.
        staged = git('diff', '--cached', '--name-only').stdout.splitlines()
        if any(path not in changes for path in staged):
            raise RuntimeError('Unrelated staged files exist; handoff commit not created')
        git('add', '--', *changes)
        review = audit()
        if not review['passed']:
            raise RuntimeError(json.dumps(review, ensure_ascii=False))
        git('diff', '--cached', '--check')
        if git('diff', '--cached', '--quiet', check=False).returncode:
            git('commit', '-m', 'Prepare GitHub deploy key and deployment handoff')
    tracked = git('ls-files', '--', relative_key, str(pub_path.relative_to(ROOT).as_posix()), str(known_hosts.relative_to(ROOT).as_posix())).stdout.strip()
    if tracked:
        raise RuntimeError('Machine-local SSH material must not enter Git')
    result = {'repository': REMOTE, 'public_key_file': str(pub_path.relative_to(ROOT)),
              'public_key': public_line, 'fingerprint': fingerprint,
              'host_keys_source': host_source, 'host_keys_installed': known_hosts.exists(), 'private_key_acl': 'current Windows user only',
              'key_passphrase': 'empty for unattended repository access', 'global_ssh_config_modified': False,
              'github_key_registered': False, 'pushed': False,
              'head': git('rev-parse', '--short', 'HEAD').stdout.strip(),
              'working_tree_clean': not git('status', '--porcelain').stdout.strip()}
    (DIRECTORY / 'github_setup.local.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
