"""Prepare an ignored machine-specific SSH key for deployment-file transfer."""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    key=ROOT/'.tools/ssh/pingpong_transfer_ed25519'
    key.parent.mkdir(parents=True,exist_ok=True)
    keygen=shutil.which('ssh-keygen')
    if not keygen:
        raise RuntimeError('OpenSSH ssh-keygen is required')
    def run(arguments):
        p=subprocess.run(arguments,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if p.returncode:
            raise RuntimeError(p.stderr)
        return p.stdout.strip()
    from audit_git_content import git_binary
    result=subprocess.run([git_binary(),'-C',str(ROOT),'check-ignore','--quiet','--no-index',key.relative_to(ROOT).as_posix()])
    if result.returncode:
        raise RuntimeError('Transfer private key path must be ignored')
    if not key.exists():
        if Path(str(key)+'.pub').exists():
            raise RuntimeError('Public key already exists without private key')
        run([keygen,'-t','ed25519','-f',str(key),'-N','','-C','pingpong2026-training-transfer','-q'])
    if os.name=='nt':
        sid=next(csv.reader(io.StringIO(run(['whoami','/user','/fo','csv','/nh']))))[-1]
        if not sid.startswith('S-1-'):
            raise RuntimeError('Unable to resolve Windows SID')
        run(['icacls',str(key),'/inheritance:r','/grant:r','*'+sid+':(F)'])
    public=' '.join(run([keygen,'-y','-f',str(key)]).split()[:2])+' pingpong2026-training-transfer'
    pub=Path(str(key)+'.pub'); pub.write_text(public+'\n',encoding='utf-8')
    print(json.dumps({'public_key':public,'fingerprint':run([keygen,'-lf',str(pub),'-E','sha256']),
                      'private_key_ignored':True,'github_key_reused':False},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
