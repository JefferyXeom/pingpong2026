"""Concise live queue/training status and observed epoch timing, no GPU work."""
import csv
import argparse
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',default='native_resolution_suite_v1')
    args=parser.parse_args()
    if Path(args.suite).name!=args.suite or not args.suite.replace('_','').isalnum():
        raise ValueError('Invalid suite ID')
    import psutil
    config=json.loads((ROOT/'configs'/f'{args.suite}.json').read_text(encoding='utf-8'))
    suite=ROOT/'reports/experiments'/args.suite
    queue=json.loads((suite/'progress.json').read_text())
    process=json.loads((suite/'process.json').read_text())
    try:
        alive=abs(psutil.Process(process['pid']).create_time()-process['create_time'])<1
    except psutil.NoSuchProcess:
        alive=False
    result={'queue_state':queue['state'],'queue_alive':alive,'stage':queue['stage'],'runs':[]}
    for run in config['runs']:
        identifier=run['id']
        directory=ROOT/'reports/training'/identifier
        item={'id':identifier,'state':'queued'}
        progress=directory/'progress.json'
        if progress.exists():
            state=json.loads(progress.read_text())
            complete=(directory/'result.json').exists()
            item.update({'state':'complete' if complete else 'training',
                'epochs_completed':state.get('epochs_completed',0),'epoch_in_progress':None if complete else state.get('epoch_in_progress'),
                'selected_epoch':state.get('selected_epoch'),'selected_val_F1':state.get('selected_val_center',{}).get('center5_f1'),
                'tensor_bchw':state.get('observed_tensor_bchw'),'updated_utc':state.get('updated_utc')})
            csv_path=ROOT/'runs/detect'/identifier/'results.csv'
            if csv_path.exists():
                rows=list(csv.DictReader(csv_path.open(encoding='utf-8',newline='')))
                times=[float(row['time']) for row in rows]
                deltas=[b-a for a,b in zip(times,times[1:]) if b>a]
                if deltas:
                    seconds=statistics.median(deltas[-5:])
                    item.update({'observed_recent_epoch_seconds':seconds,
                        'remaining_hours_if_max100':0 if complete else seconds*max(config['train']['epochs']-len(rows),0)/3600})
        elif (directory/'preflight.json').exists():
            item['preflight_passed']=True
        result['runs'].append(item)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
