"""Send the prepared deployment package using pinned OpenSSH and resumable SFTP.

Destination and fingerprint come from the user's deployment-machine handoff.
Every completed file is SHA256 checked on target before .part is promoted.
"""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PACKAGE='deployment_handoff_20261118_v1'
LOCAL=ROOT/'transfers'/PACKAGE
REMOTE='/home/pingpong/pingpong-transfer/incoming/'+PACKAGE
HOST='10.13.100.37'
USER='pingpong'
FINGERPRINT='SHA256:klk5Qej7Eg/eFMNkpqQFeX3NKLLe61I2huNSM8Vt57I'
SSH=Path('C:/Program Files/OpenSSH/ssh.exe')
SFTP=SSH.with_name('sftp.exe')
KEY=ROOT/'.tools/ssh/pingpong_transfer_ed25519'
KNOWN=ROOT/'.tools/ssh/deployment_host_known_hosts'

REMOTE_HELPER=r'''
import hashlib,json,os,shutil,stat,subprocess,sys
from pathlib import Path
a=ARGS
root=Path(a['directory'])
if not root.is_dir() or not os.access(root,os.W_OK):
    raise RuntimeError('Receiving directory missing or not writable')
def file_path(name):
    if Path(name).name!=name or '/' in name or '\\' in name:
        raise RuntimeError('Unsafe destination filename')
    p=root/name
    if p.is_symlink() or (p.exists() and not p.is_file()):
        raise RuntimeError('Destination must be a regular file')
    return p
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()
action=a['action']
if action=='preflight':
    print(json.dumps({'user':__import__('getpass').getuser(),'directory':str(root.resolve()),
       'free_bytes':shutil.disk_usage(root).free,'python':sys.version.split()[0],
       'entries':[{'name':p.name,'bytes':p.stat().st_size} for p in root.iterdir() if p.is_file()]}))
elif action=='inspect':
    final=file_path(a['name']); part=file_path(a['part'])
    if final.exists():
        print(json.dumps({'final':True,'bytes':final.stat().st_size,'sha256':digest(final)}))
    elif part.exists():
        print(json.dumps({'final':False,'bytes':part.stat().st_size,'sha256':digest(part)}))
    else: print(json.dumps({'final':False,'bytes':0,'sha256':None}))
elif action=='stat':
    p=file_path(a['part']);print(json.dumps({'bytes':p.stat().st_size if p.exists() else 0}))
elif action=='finalize':
    final=file_path(a['name']);part=file_path(a['part'])
    p=final if final.exists() else part
    if not p.exists() or p.stat().st_size!=a['bytes'] or digest(p)!=a['sha256']:
        raise RuntimeError('Target size/SHA256 mismatch; partial file preserved')
    if not final.exists():
        # Link without overwriting an existing destination; same filesystem.
        os.link(part,final);part.unlink()
    print(json.dumps({'verified':True,'name':a['name'],'bytes':a['bytes'],'sha256':a['sha256']}))
elif action=='verify_packages':
    tool=file_path('verify_deployment_package.py')
    results=[]
    for name in ['models_and_references.zip','holdout_images.zip']:
        p=subprocess.run([sys.executable,str(tool),str(file_path(name))],capture_output=True,text=True)
        if p.returncode:raise RuntimeError('Member verification failed: '+p.stderr[-2000:])
        result=json.loads(p.stdout.strip().splitlines()[-1]);results.append(result)
    print(json.dumps({'archive_members_verified':True,'results':results}))
else: raise RuntimeError('Unknown action')
'''


def emit(value):
    print(json.dumps(value,ensure_ascii=False),flush=True)


def options():
    return ['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
            '-o','UserKnownHostsFile='+str(KNOWN),'-o','HostKeyAlgorithms=ssh-ed25519',
            '-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
            '-i',str(KEY)]


def remote(action,**kwargs):
    payload={'directory':REMOTE,'action':action,**kwargs}
    encoded=base64.b64encode(json.dumps(payload).encode()).decode()
    program="import base64,json\nARGS=json.loads(base64.b64decode('"+encoded+"'))\n"+REMOTE_HELPER
    p=subprocess.run([str(SSH),*options(),'-p','22',USER+'@'+HOST,'python3 -'],input=program,
                     capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    if p.returncode:
        raise RuntimeError('SSH '+action+' failed: '+p.stderr[-3000:]+p.stdout[-1000:])
    return json.loads(p.stdout.strip())


def digest(path,count=None):
    h=hashlib.sha256()
    with path.open('rb') as f:
        remaining=count
        while remaining is None or remaining>0:
            block=f.read(4*1024**2 if remaining is None else min(4*1024**2,remaining))
            if not block:break
            h.update(block)
            if remaining is not None:remaining-=len(block)
    return h.hexdigest()


def transfer(path):
    checksum=digest(path);size=path.stat().st_size
    part=path.name+'.'+checksum[:16]+'.part'
    info=remote('inspect',name=path.name,part=part)
    if info['final']:
        if info['bytes']!=size or info['sha256']!=checksum:
            raise RuntimeError('Existing target differs; will not overwrite '+path.name)
        emit({'stage':'already_present_verified','file':path.name,'bytes':size})
        return {'name':path.name,'bytes':size,'sha256':checksum,'remote_verified':True,'skipped_existing':True}
    if info['bytes']>size or (info['bytes'] and info['sha256']!=digest(path,info['bytes'])):
        raise RuntimeError('Existing partial file prefix differs; refusing blind resume: '+part)
    if info['bytes']<size:
        batch=ROOT/'.tools/ssh/sftp_upload.batch'
        command=('reput' if info['bytes'] else 'put')+' "'+path.as_posix()+'" "'+part+'"'
        batch.write_text('cd "'+REMOTE+'"\n'+command+'\nbye\n',encoding='ascii')
        log=LOCAL/'sftp_upload.local.log'
        emit({'stage':'uploading','file':path.name,'total_bytes':size,'resume_offset':info['bytes']})
        started=time.monotonic();next_progress=started+15
        with log.open('ab',buffering=0) as stream:
            p=subprocess.Popen([str(SFTP),*options(),'-P','22','-B','262144','-R','64','-b',str(batch),USER+'@'+HOST],
                               stdout=stream,stderr=stream,stdin=subprocess.DEVNULL)
            try:
                while p.poll() is None:
                    time.sleep(1)
                    if time.monotonic()>=next_progress:
                        n=remote('stat',part=part)['bytes']
                        elapsed=time.monotonic()-started
                        emit({'stage':'upload_progress','file':path.name,'bytes':n,'total_bytes':size,
                              'percent':round(100*n/size,1),'mib_per_second':round((n-info['bytes'])/max(elapsed,1)/1024**2,2)})
                        next_progress=time.monotonic()+15
            except BaseException:
                p.terminate();p.wait(timeout=15)
                raise
        if p.returncode:
            raise RuntimeError('SFTP failed; resumable .part preserved. '+log.read_text(encoding='utf-8',errors='replace')[-2000:])
    verified=remote('finalize',name=path.name,part=part,bytes=size,sha256=checksum)
    emit({'stage':'file_sha256_verified',**verified})
    return {'name':path.name,'bytes':size,'sha256':checksum,'remote_verified':True,'skipped_existing':False}


def main():
    matches=[]
    for line in KNOWN.read_text(encoding='ascii').splitlines():
        if line.startswith('#') or not line.strip():continue
        host,kind,value,*_=line.split()
        if host==HOST and kind=='ssh-ed25519':
            actual='SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(value)).digest()).decode().rstrip('=')
            matches.append(actual)
    if matches!=[FINGERPRINT]:
        raise RuntimeError('Pinned host key differs from user-provided deployment fingerprint')
    receipt=LOCAL/'transfer_receipt.json'
    prior=json.loads((LOCAL/'package_receipt.json').read_text(encoding='utf-8'))
    for archive in prior['archives']:
        p=LOCAL/archive['archive']
        if p.stat().st_size!=archive['bytes'] or digest(p)!=archive['sha256']:
            raise RuntimeError('Source archive changed: '+str(p))
    initial=remote('preflight')
    if initial['user']!=USER or initial['directory']!=REMOTE or initial['free_bytes']<10*1024**3:
        raise RuntimeError('Remote user, directory, or free-space check failed: '+json.dumps(initial))
    emit({'stage':'preflight_passed',**initial})
    state={'package_id':PACKAGE,'host':HOST,'user':USER,'port':22,'remote_directory':REMOTE,
           'host_fingerprint':FINGERPRINT,'started_utc':datetime.now(timezone.utc).isoformat(),
           'state':'transferring','files':[],'remote_preflight':initial}
    def save():receipt.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save()
    try:
        names=['SHA256SUMS','verify_deployment_package.py','README.zh-CN.md','receiver_setup.zh-CN.md',
               'package_receipt.json','models_and_references.zip','holdout_images.zip']
        for name in names:
            state['files'].append(transfer(LOCAL/name));save()
        state['remote_archive_verification']=remote('verify_packages')
        state['state']='complete_verified'
        state['completed_utc']=datetime.now(timezone.utc).isoformat()
        state['total_payload_bytes']=sum(r['bytes'] for r in state['files'])
        save()
        transfer(receipt)
        emit({'stage':'complete_verified','receipt':str(receipt.relative_to(ROOT)),
              'remote_directory':REMOTE,'total_payload_bytes':state['total_payload_bytes'],
              'archive_verification':state['remote_archive_verification']})
    except BaseException as exc:
        state['state']='interrupted_or_failed';state['error']=str(exc);save();raise


if __name__=='__main__':
    main()
