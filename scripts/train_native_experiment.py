"""Run one configured native-resolution experiment with durable center-based progress."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--suite',default='native_resolution_suite_v1',help='Configuration ID, defaults to historical native suite')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight',action='store_true',help='Verify native tensors and center validation without optimizing weights')
    args=parser.parse_args()
    if Path(args.suite).name!=args.suite or not args.suite.replace('_','').isalnum():
        raise ValueError('Invalid suite ID')
    suite=json.loads((ROOT/'configs'/f'{args.suite}.json').read_text(encoding='utf-8'))
    run=next(r for r in suite['runs'] if r['id']==args.run)
    if Path(run['id']).name!=run['id']:
        raise ValueError('Invalid run ID')
    prepared=json.loads((ROOT/suite.get('data_preparation','reports/datasets/native_resolution_suite_v1/data_preparation.json')).read_text(encoding='utf-8'))
    data=prepared['datasets'][run['dataset']]
    directory=ROOT/'reports/training'/run['id']
    directory.mkdir(parents=True,exist_ok=True)
    from ultralytics import YOLO
    import torch
    from native_center_trainer import NativeCenterTrainer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    run_dir=ROOT/'runs/detect'/run['id']
    if args.resume and (directory/'result.json').exists():
        print('Completed experiment; nothing to resume.',flush=True)
        return
    if run_dir.exists() and not (args.resume or args.preflight):
        raise RuntimeError('Existing run; use explicit --resume after reviewing progress')
    weights=run_dir/'weights/last.pt' if args.resume else ROOT/run['pretrained']
    if not weights.is_file():
        raise FileNotFoundError(weights)
    with weights.open('rb') as stream:
        source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    settings={**suite['train'],'data':str(ROOT/data['yaml']),'project':str(ROOT/'runs/detect'),
              'name':run['id'],'exist_ok':False}
    metadata={'suite_id':suite['suite_id'],'run':run,'settings':settings,'data':data,
              'source_weights':str(weights),'source_sha256':source_hash,'original_width_height':[1280,1024],
              'training_height_width':[1024,1280],'validation_precision':'FP32',
              'fitness':'full-val center5 F1 at fixed conf=.25; ties latest checkpoint',
              'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
              'started_utc':datetime.now(timezone.utc).isoformat()}
    if args.preflight:
        trainer=NativeCenterTrainer(overrides={**settings,'model':str(weights),'name':run['id']+'_preflight','workers':0,'plots':False})
        initialization=YOLO(str(weights)).model
        trainer.model=trainer.get_model(cfg=initialization.yaml,weights=initialization,verbose=False).to('cuda')
        trainer.model.names={0:'ball'}
        dataset=trainer.build_dataset(str(ROOT/data['train']['list']),'train',8)
        sample=dataset.collate_fn([dataset[0],dataset[1]])
        trainer.preprocess_batch(sample)
        vdataset=trainer.build_dataset(str(ROOT/data['val']['list']),'val',8)
        assert all(tuple(shape)==(1024,1280) for shape in vdataset.batch_shapes)
        result={'passed':True,'run':run['id'],'sample_train_tensor':list(sample['img'].shape),
                'val_shapes':vdataset.batch_shapes[:1].tolist(),'full_train_images':len(dataset),'val_images':len(vdataset)}
        atomic(directory/'preflight.json',result)
        print(json.dumps(result),flush=True)
        return
    if not args.resume:
        atomic(directory/'configuration.json',metadata)
    (directory/'README.md').write_text('# '+run['id']+'\n\n训练1280×1024，完整验证按原图5像素球心F1选模及早停。配置与数据量见configuration.json，30秒进度见progress.json，逐轮指标见epoch_metrics.jsonl，完成后见result.json。原始args/CSV/权重在runs/detect同名目录。\n',encoding='utf-8')
    state=json.loads((directory/'progress.json').read_text()) if args.resume and (directory/'progress.json').exists() else {}
    state.update({'state':'starting','run_id':run['id'],'pid':os.getpid(),'max_epochs':settings['epochs'],
                  'batches_seen_this_process':0,'data':data})
    last_progress=0
    started=time.time()

    def save_progress(trainer,force=False):
        nonlocal last_progress
        if shutil.disk_usage(ROOT).free<5.5*1024**3:
            raise RuntimeError('Training stopped to retain 5GiB disk reserve')
        now=time.time()
        if not force and now-last_progress<30:
            return
        last_progress=now
        state.update({'state':'training','updated_utc':datetime.now(timezone.utc).isoformat(),'epoch_in_progress':trainer.epoch+1,
                      'training_batch_actual':trainer.batch_size,'observed_tensor_bchw':getattr(trainer,'observed_train_shape',None),
                      'elapsed_seconds_this_process':now-started,'free_gib':shutil.disk_usage(ROOT).free/1024**3})
        atomic(directory/'progress.json',state)

    def batch_end(trainer):
        state['batches_seen_this_process']+=1
        save_progress(trainer)

    def epoch_end(trainer):
        rows=list(csv.DictReader(Path(trainer.csv).open(encoding='utf-8',newline='')))
        if not rows:
            return
        completed=int(float(rows[-1]['epoch']))
        final=trainer.epoch+1!=completed
        if not final:
            state.update({'epochs_completed':completed,'last_metrics':trainer.metrics,
                          'center_validation':trainer.validator.center_result})
            if trainer.fitness==trainer.best_fitness:
                state.update({'selected_epoch':completed,'selected_val_center':trainer.validator.center_result})
            with (directory/'epoch_metrics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'epoch':completed,'metrics':trainer.metrics,'center':trainer.validator.center_result,'fitness':trainer.fitness})+'\n')
        save_progress(trainer,True)
        print(json.dumps({'native_epoch_complete':completed,'selected_epoch':state.get('selected_epoch'),
                          'center':state.get('center_validation'),'final_best_validation':final}),flush=True)

    model=YOLO(str(weights))
    model.add_callback('on_train_batch_end',batch_end)
    model.add_callback('on_fit_epoch_end',epoch_end)
    model.train(trainer=NativeCenterTrainer,**({'resume':True,'device':0} if args.resume else settings))
    state['state']='complete'
    state['completed_utc']=datetime.now(timezone.utc).isoformat()
    atomic(directory/'progress.json',state)
    best=run_dir/'weights/best.pt'
    with best.open('rb') as stream:
        checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    atomic(directory/'result.json',{'run':run,'source_configuration':'configuration.json',
           'epochs_completed':state['epochs_completed'],'selected_epoch':state['selected_epoch'],
           'selected_validation':state['selected_val_center'],'best':best.relative_to(ROOT).as_posix(),
           'best_sha256':checksum,'last':(run_dir/'weights/last.pt').relative_to(ROOT).as_posix(),
           'selection':'center5 F1, conf=.25, complete v1 val; test not used','completed_utc':state['completed_utc']})


if __name__=='__main__':
    main()
