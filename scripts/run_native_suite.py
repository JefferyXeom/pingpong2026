"""Sequential durable queue: train, evaluate, convert and benchmark each experiment."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train_native_experiment import atomic

ROOT=Path(__file__).resolve().parents[1]
SUITE='native_resolution_suite_v1'
DIRECTORY=ROOT/'reports/experiments'/SUITE


def main():
    global SUITE,DIRECTORY
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',default='native_resolution_suite_v1')
    parser.add_argument('--launch',action='store_true',help='Detach the queue with persistent logs')
    args=parser.parse_args()
    if Path(args.suite).name!=args.suite or not args.suite.replace('_','').isalnum():
        raise ValueError('Invalid suite ID')
    SUITE=args.suite
    DIRECTORY=ROOT/'reports/experiments'/SUITE
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    import psutil
    process_file=DIRECTORY/'process.json'
    if process_file.exists():
        prior=json.loads(process_file.read_text())
        try:
            process=psutil.Process(prior['pid'])
            if process.pid!=os.getpid() and abs(process.create_time()-prior['create_time'])<1:
                raise RuntimeError('Native suite already running: '+str(prior['pid']))
        except psutil.NoSuchProcess:
            pass
    if args.launch:
        log_name='native_suite' if SUITE=='native_resolution_suite_v1' else SUITE
        with (ROOT/f'logs/{log_name}.stdout.log').open('ab',buffering=0) as out,(ROOT/f'logs/{log_name}.stderr.log').open('ab',buffering=0) as err:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--suite',SUITE],cwd=ROOT,
                stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS|subprocess.CREATE_NO_WINDOW)
        print(json.dumps({'launched_pid':process.pid,'progress':str(DIRECTORY/'progress.json')}),flush=True)
        return
    import msvcrt
    lock=(DIRECTORY/'queue.lock').open('a+b')
    lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    gpu_lock=(ROOT/'logs/native_gpu_queue.lock').open('a+b')
    gpu_lock.write(b'0');gpu_lock.flush();gpu_lock.seek(0)
    msvcrt.locking(gpu_lock.fileno(),msvcrt.LK_NBLCK,1)
    atomic(process_file,{'pid':os.getpid(),'create_time':psutil.Process().create_time()})
    suite=json.loads((ROOT/f'configs/{SUITE}.json').read_text(encoding='utf-8'))
    state={'suite_id':SUITE,'state':'running','pid':os.getpid(),'completed':[]}

    def execute(stage,command):
        state.update({'stage':stage,'command':command,'updated_utc':datetime.now(timezone.utc).isoformat()})
        atomic(DIRECTORY/'progress.json',state)
        print(json.dumps({'stage':stage,'command':command}),flush=True)
        stamp=int(time.time())
        with (ROOT/'logs'/f'{stage}_{stamp}.stdout.log').open('wb',buffering=0) as out,(ROOT/'logs'/f'{stage}_{stamp}.stderr.log').open('wb',buffering=0) as err:
            child=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL)
            state['child_pid']=child.pid
            atomic(DIRECTORY/'progress.json',state)
            result=child.wait()
        if result:
            raise RuntimeError(f'{stage} failed with exit code {result}; inspect timestamped logs')
        state['completed'].append(stage)
        atomic(DIRECTORY/'progress.json',state)

    try:
        for run in suite['runs']:
            identifier=run['id']
            result=ROOT/'reports/training'/identifier/'result.json'
            if not result.exists():
                command=['scripts/train_native_experiment.py','--suite',SUITE,'--run',identifier]
                if (ROOT/'runs/detect'/identifier/'weights/last.pt').exists():
                    command.append('--resume')
                execute(identifier+'_train',command)
            selected=json.loads(result.read_text())
            evaluation=identifier+'_conversion_v1'
            execute(identifier+'_evaluate',['scripts/evaluate_conversion_matrix.py','--experiment',evaluation,
                '--weights',selected['best'],'--resume'])
            execute(identifier+'_benchmark',['scripts/rebenchmark_conversion_matrix.py','--experiment',evaluation])
            execute(identifier+'_summary',suite.get('summary_command',['scripts/summarize_native_suite.py']))
        state['state']='complete'
        state['stage']='all_training_and_evaluation_complete'
    except Exception as exc:
        state['state']='failed'
        state['error']=repr(exc)
        raise
    finally:
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        atomic(DIRECTORY/'progress.json',state)
        lock.close()
        gpu_lock.close()


if __name__=='__main__':
    main()
