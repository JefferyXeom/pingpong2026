"""Resumable full-data precision/speed matrix for native-size and 960-long-side inference.

Native TensorRT engines have no Ultralytics metadata wrapper. Every backend uses
the same CPU resize/RGB normalization and restores coordinates to original pixels.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
DLL_HANDLES = []
if os.name == 'nt':
    for directory in [Path(sys.prefix)/'Lib/site-packages'/p for p in
                      ('torch/lib', 'tensorrt_libs', 'tensorrt_cu12_libs', 'onnxruntime/capi')]:
        if directory.is_dir():
            DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')

OUT = ROOT/'artifacts/center_conversion_matrix_v1'
REPORT = ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_v1.json'
SHAPES = [(768, 960), (1024, 1280)]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def guard():
    if shutil.disk_usage(ROOT).free < 8*1024**3:
        raise RuntimeError('Evaluation stopped to preserve disk reserve.')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def progress(stage, **details):
    message = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **details}
    atomic_json(REPORT.parent/'progress.json' if REPORT.parent.name!='center_conversion_matrix_v1' else REPORT.parent/'center_conversion_matrix_progress.json', message)
    print(json.dumps(message), flush=True)


def export_onnx(shape):
    import onnx
    from ultralytics import YOLO
    directory = OUT/f'h{shape[0]}_w{shape[1]}'
    directory.mkdir(exist_ok=True)
    path = directory/'weights.onnx'
    if path.exists():
        onnx.checker.check_model(str(path))
        return path
    source = OUT/'source.pt'
    local = directory/'weights.pt'
    if not local.exists():
        shutil.copy2(source, local)
    progress('export_onnx', height=shape[0], width=shape[1])
    exported = YOLO(str(local)).export(format='onnx', imgsz=shape, batch=2,
        dynamic=True, quantize=32, nms=False, simplify=True, opset=17, device=0)
    assert Path(exported)==path
    onnx.checker.check_model(str(path))
    return path


def build_engine(onnx_path, shape, batch, precision):
    import tensorrt as trt
    target = onnx_path.parent/f'trt_fp{precision}_b{batch}.engine'
    metadata_path = target.with_suffix('.build.json')
    if target.exists() and metadata_path.exists():
        return target
    guard()
    progress('build_engine', height=shape[0], width=shape[1], batch=batch, precision=precision)
    started = time.time()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2*1024**3)
    config.builder_optimization_level = 3
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    fixed_shape = (batch, 3, *shape)
    profile.set_shape('images', fixed_shape, fixed_shape, fixed_shape)
    actual=profile.get_shape('images')
    if not actual or any(tuple(s)!=fixed_shape for s in actual):
        raise RuntimeError(f'TensorRT profile did not retain requested shape: {actual}')
    config.add_optimization_profile(profile)
    constraints = []
    if precision==16:
        config.set_flag(trt.BuilderFlag.FP16)
        # Keep only the final head probability sigmoid in FP32; backbone SiLU
        # sigmoids must remain eligible for fusion/reduced precision.
        indices=[int(m.group(1)) for layer in network for m in [re.search(r'/model\.(\d+)/',layer.name)] if m]
        head=f'/model.{max(indices)}/' if indices else '/'
        for layer in network:
            if head in layer.name and layer.type==trt.LayerType.ACTIVATION and 'sigmoid' in layer.name.lower():
                layer.precision=trt.float32
                for i in range(layer.num_outputs):
                    layer.set_output_type(i, trt.float32)
                constraints.append(layer.name)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    cache_path=OUT/f'timing_fp{precision}.cache'
    cache=config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b'')
    config.set_timing_cache(cache, False)
    serialized=builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT build failed')
    temporary=target.with_suffix('.engine.partial')
    temporary.write_bytes(bytes(serialized))
    temporary.replace(target)
    cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    atomic_json(metadata_path, {'shape':fixed_shape, 'precision':precision, 'tf32':False,
        'fp32_sigmoid_layers':constraints, 'workspace_gib':2, 'builder_optimization_level':3,
        'seconds':time.time()-started, 'onnx_sha256':digest(onnx_path),
        'engine_sha256':digest(target), 'format':'native TensorRT, no metadata header',
        'portability':'Windows RTX3090 evaluation only; rebuild on target Ubuntu/2060 SUPER.'})
    del serialized, parser, network, config, builder
    gc.collect()
    return target


class Runner:
    def __init__(self, backend, shape, batch, path=None, source_weights=None):
        import torch
        self.backend, self.shape, self.batch = backend, shape, batch
        self.input=torch.empty((batch,3,*shape),device='cpu' if backend=='onnx_cpu_fp32' else 'cuda',dtype=torch.float32)
        self.output=None
        self.graph=None
        if backend=='pytorch_fp32':
            from ultralytics import YOLO
            self.wrapper=YOLO(str(source_weights or OUT/'source.pt'))
            self.model=self.wrapper.model.to('cuda').float().eval()
            self.model.end2end=True
            self.model.fuse(verbose=False)
            self.model.model[-1].export=True
        elif backend=='onnx_cpu_fp32':
            import onnxruntime as ort
            options=ort.SessionOptions()
            options.intra_op_num_threads=2
            self.session=ort.InferenceSession(str(path),sess_options=options,
                providers=['CPUExecutionProvider'])
        else:
            import tensorrt as trt
            self.logger=trt.Logger(trt.Logger.WARNING)
            self.runtime=trt.Runtime(self.logger)
            self.engine=self.runtime.deserialize_cuda_engine(path.read_bytes())
            self.context=self.engine.create_execution_context()
            if not self.context.set_input_shape('images',tuple(self.input.shape)):
                raise RuntimeError('Engine profile shape mismatch')
            self.output_names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                               if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))==trt.TensorIOMode.OUTPUT]
            assert len(self.output_names)==1
            name=self.output_names[0]
            assert self.engine.get_tensor_dtype('images')==trt.float32
            assert self.engine.get_tensor_dtype(name)==trt.float32
            self.output=torch.empty(tuple(self.context.get_tensor_shape(name)),device='cuda',dtype=torch.float32)
            self.context.set_tensor_address('images',self.input.data_ptr())
            self.context.set_tensor_address(name,self.output.data_ptr())

    def forward(self):
        import torch
        if self.graph is not None:
            self.graph.replay()
            return self.output
        if self.backend=='pytorch_fp32':
            self.output=self.model(self.input)
        elif self.backend=='onnx_cpu_fp32':
            self.output=torch.from_numpy(self.session.run(None,{'images':self.input.numpy()})[0])
        else:
            if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError('TensorRT enqueue failed')
        return self.output

    def benchmark(self, samples=200, warmup=30):
        import numpy as np
        import torch
        self.input.uniform_(0,1)
        if self.backend=='onnx_cpu_fp32':
            for _ in range(3):
                self.forward()
            elapsed=[]
            for _ in range(30):
                start=time.perf_counter()
                self.forward()
                elapsed.append((time.perf_counter()-start)*1000)
            return {'cpu_batch_ms_mean':float(np.mean(elapsed)),
                'cpu_batch_ms_p50_p95_p99':np.percentile(elapsed,[50,95,99]).tolist(),
                'iterations':30,'warmup_iterations':3,
                'scope':'CPU numerical reference only, two CPU threads. Not included in GPU speedup claims. ORT GPU installed for CUDA13 is incompatible with current CUDA12 stack; environment unchanged.'}
        for _ in range(warmup):
            self.forward()
            if warmup>=200:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        def measure():
            elapsed=[]
            wall_started=time.perf_counter()
            for _ in range(samples):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                self.forward()
                end.record()
                end.synchronize()
                elapsed.append(start.elapsed_time(end))
            wall=time.perf_counter()-wall_started
            return {'iterations':samples,'cuda_event_batch_ms_mean':float(np.mean(elapsed)),
                    'cuda_event_batch_ms_p50_p95_p99':np.percentile(elapsed,[50,95,99]).tolist(),
                    'cuda_event_images_per_second':self.batch*1000/float(np.mean(elapsed)),
                    'synchronized_loop_images_per_second':samples*self.batch/wall}
        result={'ordinary':measure(), 'warmup_iterations':warmup,
                'scope':'Preallocated device input/output, excludes image decode, resize, normalization, H2D/D2H and postprocess. All on RTX3090; no target GPU projection.'}
        if self.backend!='onnx_cpu_fp32':
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.forward()
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                self.forward()
            self.graph=graph
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(10):
                self.forward()
            torch.cuda.synchronize()
            result['cuda_graph']=measure()
        return result


def preprocess(record, shape):
    import cv2
    import numpy as np
    image=cv2.imread(str(ROOT/'datasets/pingpong_v1'/record['image']))
    if image is None or image.shape[:2]!=(1024,1280):
        raise RuntimeError('Unexpected source image')
    # Both target aspect ratios exactly match source; no padding, cropping or distortion.
    if image.shape[:2]!=shape:
        image=cv2.resize(image,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
    rgb=np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))
    return rgb.astype(np.float32)/255


def infer_dataset(runner, records, cache_path, case):
    import numpy as np
    import torch
    predictions=[]
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0,len(records),64):
            chunk=records[offset:offset+64]
            arrays=list(pool.map(lambda r:preprocess(r,runner.shape),chunk))
            for index in range(0,len(arrays),runner.batch):
                batch=arrays[index:index+runner.batch]
                valid=len(batch)
                batch+= [batch[-1]]*(runner.batch-valid)
                runner.input.copy_(torch.from_numpy(np.stack(batch)))
                raw=runner.forward().detach().cpu().numpy().copy()
                if raw.shape!=(runner.batch,300,6) or not np.isfinite(raw).all():
                    raise RuntimeError('Invalid backend output')
                for detections in raw[:valid]:
                    selected=detections[detections[:,4]>=.001].copy()
                    selected[:,[0,2]]*=1280/runner.shape[1]
                    selected[:,[1,3]]*=1024/runner.shape[0]
                    selected[:,[0,2]]=np.clip(selected[:,[0,2]],0,1280)
                    selected[:,[1,3]]=np.clip(selected[:,[1,3]],0,1024)
                    assert (selected[:,5]==0).all()
                    predictions.append(selected)
            if offset%256==0:
                progress('evaluate_images',case=case,completed=min(offset+64,len(records)),total=len(records))
    seconds=time.perf_counter()-started
    np.savez_compressed(cache_path, **{f'f{i}':p for i,p in enumerate(predictions)})
    return predictions, {'seconds':seconds,'images_per_second':len(records)/seconds,
        'scope':'Offline PNG decode + 2 CPU resize workers + RGB/FP32 normalization + H2D + inference + D2H/filter. Includes chunk scheduling; excludes camera, network/video decode, stereo matching and final metric computation. Not optimized streaming.'}


def truth(record):
    import numpy as np
    boxes=[]
    for line in record['label'].splitlines():
        _,x,y,w,h=map(float,line.split())
        boxes.append([(x-w/2)*1280,(y-h/2)*1024,(x+w/2)*1280,(y+h/2)*1024])
    return np.asarray(boxes).reshape(-1,4)


def box_iou_numpy(left,right):
    import numpy as np
    inter=np.maximum(0,np.minimum(left[:,None,2:],right[None,:,2:])-np.maximum(left[:,None,:2],right[None,:,:2])).prod(axis=-1)
    union=np.prod(left[:,2:]-left[:,:2],axis=-1)[:,None]+np.prod(right[:,2:]-right[:,:2],axis=-1)[None,:]-inter
    return inter/np.maximum(union,1e-9)


def score_predictions(records,predictions):
    import numpy as np
    from diagnose_ball_centers import match_pairs
    from ultralytics.utils.metrics import ap_per_class
    thresholds=(.05,.25,.5)
    totals={t:defaultdict(list) for t in thresholds}
    all_tp, all_conf, all_cls, targets=[],[],[],[]
    for record,detections in zip(records,predictions):
        gt=truth(record)
        iou=box_iou_numpy(gt,detections[:,:4])
        tp=np.zeros((len(detections),10),dtype=bool)
        for j,limit in enumerate(np.linspace(.5,.95,10)):
            for a,b,_ in match_pairs(iou,limit,maximize=True):
                tp[b,j]=True
        all_tp.append(tp)
        all_conf.append(detections[:,4])
        all_cls.append(detections[:,5])
        targets.extend([0]*len(gt))
        for threshold,acc in totals.items():
            pred=detections[detections[:,4]>=threshold,:4]
            distance=np.linalg.norm((gt[:,None,:2]+gt[:,None,2:])/2-(pred[None,:,:2]+pred[None,:,2:])/2,axis=-1)
            pairs5=match_pairs(distance,5)
            pairs10=match_pairs(distance,10)
            acc['gt'].append(len(gt)); acc['pred'].append(len(pred))
            acc['tp5'].append(len(pairs5)); acc['tp10'].append(len(pairs10))
            acc['missed_frames'].append(int(len(pairs5)<len(gt)))
            acc['extra_frames'].append(int(len(pred)>len(pairs5)))
            for a,b,d in pairs10:
                acc['errors'].append(d)
                acc['width_ratio'].append((pred[b,2]-pred[b,0])/(gt[a,2]-gt[a,0]))
                acc['height_ratio'].append((pred[b,3]-pred[b,1])/(gt[a,3]-gt[a,1]))
    metrics={}
    for threshold,acc in totals.items():
        gt,pred,tp5=sum(acc['gt']),sum(acc['pred']),sum(acc['tp5'])
        p,r=tp5/max(pred,1),tp5/gt
        metrics[str(threshold)]={'gt':gt,'predictions':pred,'center5_matches':tp5,
            'center5_precision':p,'center5_recall':r,'center5_f1':2*p*r/max(p+r,1e-9),
            'center10_matches':sum(acc['tp10']),
            'matched_error_px_p50_p95_p99':np.percentile(acc['errors'],[50,95,99]).tolist() if acc['errors'] else None,
            'width_ratio_median':float(np.median(acc['width_ratio'])) if acc['width_ratio'] else None,
            'height_ratio_median':float(np.median(acc['height_ratio'])) if acc['height_ratio'] else None,
            'missed_frames_center5':sum(acc['missed_frames']),'extra_frames_center5':sum(acc['extra_frames'])}
    ap=ap_per_class(np.concatenate(all_tp),np.concatenate(all_conf),np.concatenate(all_cls),np.asarray(targets),plot=False,names={0:'ball'})[5]
    return {'images':len(records),'thresholds':metrics,'box_mAP50':float(ap[:,0].mean()),'box_mAP50_95':float(ap.mean()),
        'box_AP_note':'Ultralytics AP integration, greedy descending-IoU unique matching, conf floor .001, NMS-free max300. Centers are separate supplemental metrics; original boxes unchanged.'}


def compare_outputs(records,reference,candidate,threshold=.25):
    import numpy as np
    from diagnose_ball_centers import match_pairs
    shifts=[]; score_gaps=[]; unmatched_ref=unmatched_new=lost_gt=gained_gt=0
    examples=[]
    for record,ref,new in zip(records,reference,candidate):
        ref=ref[ref[:,4]>=threshold]; new=new[new[:,4]>=threshold]
        distance=np.linalg.norm((ref[:,None,:2]+ref[:,None,2:4])/2-(new[None,:,:2]+new[None,:,2:4])/2,axis=-1)
        pairs=match_pairs(distance,10)
        unmatched_ref+=len(ref)-len(pairs); unmatched_new+=len(new)-len(pairs)
        for a,b,d in pairs:
            shifts.append(d); score_gaps.append(abs(float(ref[a,4]-new[b,4])))
        gt=truth(record)
        matched_sets=[]
        for det in (ref,new):
            d=np.linalg.norm((gt[:,None,:2]+gt[:,None,2:])/2-(det[None,:,:2]+det[None,:,2:4])/2,axis=-1)
            matched_sets.append({a for a,_,_ in match_pairs(d,5)})
        lost=matched_sets[0]-matched_sets[1]; gained=matched_sets[1]-matched_sets[0]
        lost_gt+=len(lost); gained_gt+=len(gained)
        if lost or gained:
            examples.append({'image':record['image'],'lost_gt_indices':sorted(lost),'gained_gt_indices':sorted(gained)})
    return {'threshold':threshold,'matching_radius_px':10,
        'matched_prediction_pairs':len(shifts),'prediction_center_shift_px_p50_p95_p99_max':np.percentile(shifts,[50,95,99,100]).tolist() if shifts else None,
        'matched_score_abs_gap_p50_p95_p99_max':np.percentile(score_gaps,[50,95,99,100]).tolist() if score_gaps else None,
        'unmatched_reference_predictions':unmatched_ref,'unmatched_candidate_predictions':unmatched_new,
        'new_gt_misses_at_5px':lost_gt,'recovered_gt_at_5px':gained_gt,'changed_gt_examples':examples[:30]}


def main():
    global OUT, REPORT
    import cv2
    import numpy as np
    import torch
    import ultralytics
    import onnxruntime as ort
    import tensorrt as trt
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--shape',choices=['960','1280','all'],default='all')
    parser.add_argument('--weights',default='models/pingpong_yolo26n_center_v1/weights.pt')
    parser.add_argument('--experiment',default='center_conversion_matrix_v1',help='Independent artifact/report batch ID')
    args=parser.parse_args()
    if Path(args.experiment).name!=args.experiment or not re.fullmatch(r'[a-zA-Z0-9_-]+',args.experiment):
        raise ValueError('Invalid experiment ID')
    OUT=ROOT/'artifacts'/args.experiment
    REPORT=ROOT/'reports/evaluation'/args.experiment/f'{args.experiment}.json'
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    guard()
    OUT.mkdir(exist_ok=args.resume)
    source=(ROOT/args.weights).resolve()
    if not source.is_relative_to(ROOT) or not source.is_file():
        raise ValueError('Weights must exist inside workspace')
    if (OUT/'source.pt').exists():
        assert digest(source)==digest(OUT/'source.pt')
    else:
        shutil.copy2(source,OUT/'source.pt')
    records=[json.loads(line) for line in (ROOT/'datasets/pingpong_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    records=[r for r in records if r['split'] in ('val','test')]
    # Preserve split order; batches below are throughput experiments, not certified synchronized camera pairs.
    records=sorted(records,key=lambda r:(r['split'],r['image']))
    atomic_json(OUT/'image_order.json',[{'image':r['image'],'split':r['split'],'image_sha256':r['image_sha256']} for r in records])
    report=json.loads(REPORT.read_text(encoding='utf-8')) if args.resume and REPORT.exists() else {
        'weights_sha256':digest(source),'source_weights':str(source.relative_to(ROOT)),
        'experiment':args.experiment,'gpu':torch.cuda.get_device_name(0),
        'versions':{'torch':torch.__version__,'cuda':torch.version.cuda,'ultralytics':ultralytics.__version__,'onnxruntime':ort.__version__,'tensorrt':trt.__version__},
        'image_count':len(records),'cases':{},'limits':['RTX3090 Windows only; no 2060 SUPER throughput guarantee.',
            'Val/test fixed; no training, threshold tuning or model selection in this matrix.',
            'Batch2 groups offline images; no proof of camera synchronization, stereo depth or actual stream throughput.',
            'Input preprocessing is fixed per case; source training resolution is recorded in the training experiment.',
            'No verified no-ball evaluation data. Matched-error quantiles exclude unmatched balls; counts reported separately.']}
    shapes=[s for s in SHAPES if args.shape=='all' or s[1]==int(args.shape)]
    with torch.inference_mode():
        for shape in shapes:
            onnx_path=export_onnx(shape)
            # Accuracy reference is FP32 per shape and batch. Extra FP32 TRT at b2 isolates reduced-precision effects.
            experiments=[('pytorch_fp32',1),('pytorch_fp32',2),('onnx_cpu_fp32',2),
                         ('tensorrt_fp32',2),('tensorrt_fp16',1),('tensorrt_fp16',2)]
            for backend,batch in experiments:
                case=f'h{shape[0]}_w{shape[1]}_{backend}_b{batch}'
                if case in report['cases']:
                    progress('skip_completed',case=case)
                    continue
                path=onnx_path
                if backend.startswith('tensorrt'):
                    path=build_engine(onnx_path,shape,batch,16 if backend.endswith('16') else 32)
                progress('load_backend',case=case)
                runner=Runner(backend,shape,batch,path)
                benchmark=runner.benchmark()
                cache=onnx_path.parent/(case+'_predictions.npz')
                predictions,wall=infer_dataset(runner,records,cache,case)
                item={'backend':backend,'batch':batch,'input_height_width':shape,'benchmark':benchmark,
                      'offline_pipeline':wall,'prediction_cache':str(cache.relative_to(ROOT)),
                      'preprocessing':'cv2 INTER_LINEAR only at 960; native 1280 unchanged; BGR->RGB CHW FP32 /255; no padding; outputs in 1280x1024 pixels.',
                      'splits':{}}
                for split in ('val','test'):
                    indices=[i for i,r in enumerate(records) if r['split']==split]
                    item['splits'][split]=score_predictions([records[i] for i in indices],[predictions[i] for i in indices])
                if backend!='pytorch_fp32' or batch==2:
                    ref_case=f'h{shape[0]}_w{shape[1]}_pytorch_fp32_b{batch if backend!="pytorch_fp32" else 1}'
                    ref_cache=ROOT/report['cases'][ref_case]['prediction_cache']
                    with np.load(ref_cache) as ref:
                        reference=[ref[f'f{i}'] for i in range(len(records))]
                    item['against_pytorch']=compare_outputs(records,reference,predictions)
                report['cases'][case]=item
                atomic_json(REPORT,report)
                progress('case_complete',case=case,
                    val_center5_f1=item['splits']['val']['thresholds']['0.25']['center5_f1'],
                    test_center5_f1=item['splits']['test']['thresholds']['0.25']['center5_f1'])
                del runner,predictions
                gc.collect()
                torch.cuda.empty_cache()
    report['expected_cases']=len(shapes)*6
    report['state']='complete' if len(report['cases'])>=len(shapes)*6 else 'partial'
    atomic_json(REPORT,report)
    progress(report['state'],completed_cases=len(report['cases']))


if __name__=='__main__':
    main()
