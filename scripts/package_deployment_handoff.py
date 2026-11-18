"""Package audited weights, ONNX, full holdouts and cached reference predictions.

Uses ZIP64 without duplicating source images into a staging dataset. No inference,
training, engine construction, network transfer, credentials or training images.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import zipfile

import numpy as np
from PIL import Image
from audit_git_content import git_binary
from evaluate_deployment_predictions import load_predictions, score, compare
from verify_deployment_package import verify

ROOT=Path(__file__).resolve().parents[1]
PACKAGE='deployment_handoff_20261118_v1'
OUT=ROOT/'transfers'/PACKAGE
MODELS=[('B_n_merged_v1','native_v1_additional_yolo26n_1280'),
        ('C_s_merged_v1','native_v1_additional_yolo26s_1280'),
        ('D_n_merged_v2','native_merged_v2_baiduout_yolo26n_1280'),
        ('E_s_merged_v2','native_merged_v2_baiduout_yolo26s_1280')]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def as_json(value):
    return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


class Part:
    def __init__(self,name):
        self.path=OUT/(name+'.zip')
        self.partial=OUT/(name+'.zip.partial')
        if self.path.exists() or self.partial.exists():
            raise FileExistsError('Existing package is preserved: '+str(self.path))
        self.archive=zipfile.ZipFile(self.partial,'w',compression=zipfile.ZIP_STORED,allowZip64=True)
        self.files=[]

    def add_bytes(self,name,data):
        self.archive.writestr(name,data)
        self.files.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})

    def add_file(self,name,source,expected=None):
        size=source.stat().st_size
        value=hashlib.sha256()
        with source.open('rb') as stream, self.archive.open(name,'w',force_zip64=True) as target:
            for block in iter(lambda:stream.read(4*1024**2),b''):
                value.update(block); target.write(block)
        checksum=value.hexdigest()
        if expected and checksum!=expected:
            raise ValueError('Source checksum changed: '+str(source))
        self.files.append({'path':name,'bytes':size,'sha256':checksum})
        return checksum

    def finish(self,part):
        manifest={'package_id':PACKAGE,'part':part,'files':self.files}
        self.archive.writestr('file_manifest.json',as_json(manifest))
        self.archive.close()
        self.partial.replace(self.path)
        return {'archive':self.path.name,'bytes':self.path.stat().st_size,'sha256':digest(self.path),'files':len(self.files)}


def quick_subset(records):
    groups=defaultdict(list)
    for i,r in enumerate(records):
        groups[(r['split'],r['camera'],r['objects'])].append(i)
    rng=random.Random(20261118)
    for values in groups.values():
        rng.shuffle(values)
    chosen=[]
    while len(chosen)<128:
        for key in sorted(groups):
            if groups[key] and len(chosen)<128:
                chosen.append(groups[key].pop())
    return sorted(chosen)


def main():
    if shutil.disk_usage(ROOT).free<10*1024**3:
        raise RuntimeError('Insufficient disk reserve for full holdout ZIP')
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.glob('*.zip*')):
        raise FileExistsError('Package version already exists or is in progress; do not overwrite')
    code_commit=subprocess.check_output([git_binary(),'-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    source_records=[json.loads(line) for line in (ROOT/'datasets/pingpong_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    lookup={r['image']:r for r in source_records}
    order=read(f'artifacts/{MODELS[0][1]}_conversion_v1/image_order.json')
    records=[]
    for index,item in enumerate(order):
        r=lookup[item['image']]
        assert r['split'] in ['val','test'] and item['image_sha256']==r['image_sha256']
        gt=[]
        for line in r['label'].splitlines():
            _,x,y,w,h=map(float,line.split())
            gt.append([(x-w/2)*1280,(y-h/2)*1024,(x+w/2)*1280,(y+h/2)*1024])
        records.append({'index':index,'image':r['image'],'label':r['label_path'],'annotation':r['annotation'],
                        'split':r['split'],'recording':r['batch'],'camera':'left' if '__left_' in r['image'] else 'right',
                        'width':r['width'],'height':r['height'],'objects':r['objects'],'gt_xyxy_px':gt,
                        'image_sha256':r['image_sha256'],'annotation_sha256':r['annotation_sha256']})
    assert Counter(r['split'] for r in records)=={'val':1272,'test':830} and len(records)==2102
    indices=quick_subset(records)
    protocol={'conf':.25,'center_match_radius_px':5,'localization_error_match_radius_px':10,
              'matching':'greedy ascending Euclidean original-pixel center distance; each GT/prediction used once',
              'classes':{'0':'ball'},'input_bchw':['B',3,1024,1280],'input_dtype':'float32',
              'preprocessing':'cv2.imread BGR -> RGB -> contiguous CHW float32 /255; native no resize/padding/crop',
              'output_shape':['B',300,6],'output_fields':['x1','y1','x2','y2','confidence','class_id'],
              'output_dtype':'float32 (including mixed FP16 engine)','nms':False,
              'reference_cache':'Nx6 per image, conf floor .001, boxes clipped to original [0,1280]x[0,1024]; NPZ f0..f2101 maps image_order.json',
              'quick128_cache':'For new quick-only predictions NPZ f0..f127 maps quick128_indices.json in listed order',
              'conversion_shift':'same model/shape/batch versus source PyTorch FP32; conf.25, prediction pairing<=10px; val+test',
              'test_usage':'Frozen historical holdouts; do not use for training, INT8 calibration, threshold or ROI tuning',
              'stereo_usage':'Images are not a verified synchronized stereo acceptance set. Batch2 in historical cache is not proof of temporal pairing.',
              'target':'Ubuntu / RTX2060 Super, actual environment and Turing-compatible TensorRT version must be checked',
              'not_completed':['target GPU accuracy/latency','camera synchronization/calibration','stereo association','INT8'],
              'coordinate_convention':'historical VOC: center=((min+max)/2-1)/size, extent=(max-min)/size'}
    core=Part('models_and_references')
    image_part=Part('holdout_images')
    core.add_bytes('testdata/image_order.json',as_json(records))
    core.add_bytes('testdata/quick128_indices.json',as_json(indices))
    for split in ['val','test']:
        core.add_bytes(f'testdata/{split}.txt',('\n'.join(r['image'] for r in records if r['split']==split)+'\n').encode())
    core.add_bytes('testdata/quick128.txt',('\n'.join(records[i]['image'] for i in indices)+'\n').encode())
    core.add_bytes('configs/inference_contract.json',as_json(protocol))
    for path in ['verify_deployment_package.py','evaluate_deployment_predictions.py','build_deployment_engine.py','package_deployment_handoff.py']:
        core.add_file('tools/'+path,ROOT/'scripts'/path)
    core.add_file('configs/deployment_target.json',ROOT/'configs/deployment_target.json')
    models=[]
    verification=[]
    import onnx
    for model_id,run in MODELS:
        print(json.dumps({'stage':'model_and_cached_metrics','model':model_id}),flush=True)
        base=f'artifacts/{run}_conversion_v1'
        result=read(f'reports/training/{run}/result.json')
        configuration=read(f'reports/training/{run}/configuration.json')
        matrix=read(f'reports/evaluation/{run}_conversion_v1/{run}_conversion_v1.json')
        assert matrix['state']=='complete' and matrix['weights_sha256']==result['best_sha256']
        assert read(base+'/image_order.json')==order
        model_path=f'models/{model_id}'
        core.add_file(model_path+'/best.pt',ROOT/result['best'],result['best_sha256'])
        onnx_path=ROOT/base/'h1024_w1280/weights.onnx'
        graph=onnx.load(str(onnx_path),load_external_data=False)
        assert not any(onnx.external_data_helper.uses_external_data(t) for t in graph.graph.initializer), 'ONNX external tensors require explicit packaging'
        onnx.checker.check_model(str(onnx_path))
        build=read(base+'/h1024_w1280/trt_fp16_b2.build.json')
        onnx_sha=core.add_file(model_path+'/model.onnx',onnx_path,build['onnx_sha256'])
        for filename in ['trt_fp32_b2.build.json','trt_fp16_b1.build.json','trt_fp16_b2.build.json']:
            core.add_file(model_path+'/source_build/'+filename,ROOT/base/'h1024_w1280'/filename)
        for filename in ['configuration.json','result.json','epoch_metrics.jsonl']:
            core.add_file(model_path+'/training/'+filename,ROOT/f'reports/training/{run}'/filename)
        for filename in ['args.yaml','results.csv']:
            core.add_file(model_path+'/training/'+filename,ROOT/f'runs/detect/{run}'/filename)
        cases=[]
        ref_predictions={}
        for key,case in matrix['cases'].items():
            if case['input_height_width']!=[1024,1280]:
                continue
            path=ROOT/base/'h1024_w1280'/f'{key}_predictions.npz'
            predictions=load_predictions(path,len(records))
            ref_predictions[key]=predictions
            metrics=score(records,predictions)
            for split in ['val','test']:
                source=case['splits'][split]['thresholds']['0.25']
                assert metrics[split]['gt']==source['gt']
                assert metrics[split]['tp']==source['center5_matches'] and metrics[split]['predictions']==source['predictions']
                assert np.allclose(metrics[split]['error_p50_p95_p99_px'],source['matched_error_px_p50_p95_p99'],atol=1e-7,rtol=0)
            reference=f'references/{model_id}/{key}_predictions.npz'
            core.add_file(reference,path)
            cases.append({'key':key,'backend':case['backend'],'batch':case['batch'],'reference_predictions':reference,
                          'expected':metrics,'expected_quick128':score([records[i] for i in indices],[predictions[i] for i in indices]),
                          'source_against_pytorch':case.get('against_pytorch'),
                          'source_gpu_benchmark':case['benchmark'], 'benchmark_gpu':'Windows RTX3090, not target performance'})
        assert len(cases)==6
        for case in cases:
            if case['source_against_pytorch']:
                reference=ref_predictions[f'h1024_w1280_pytorch_fp32_b{case["batch"]}']
                compared=compare(records,reference,ref_predictions[case['key']])
                expected=case['source_against_pytorch']
                assert compared['new_gt_misses_at_5px']==expected['new_gt_misses_at_5px']
                assert compared['recovered_gt_at_5px']==expected['recovered_gt_at_5px']
                assert np.allclose(compared['shift_p50_p95_p99_max_px'],expected['prediction_center_shift_px_p50_p95_p99_max'],atol=1e-7,rtol=0)
        model={'id':model_id,'run_id':run,'architecture':configuration['run']['architecture'],
               'dataset':configuration['run']['dataset'],'counts':{s:{k:v for k,v in configuration['data'][s].items() if k in ['images','objects']} for s in ['train','val','test']},
               'selected_epoch':result['selected_epoch'],'epochs_completed':result['epochs_completed'],
               'weights':model_path+'/best.pt','weights_sha256':result['best_sha256'],
               'onnx':model_path+'/model.onnx','onnx_sha256':onnx_sha,'source_versions':matrix['versions'],
               'source_gpu':matrix['gpu'],'cases':cases,
               'target_validation_completed':False,'priority':'first target candidate' if model_id.startswith('B_') else 'comparison candidate'}
        core.add_bytes(model_path+'/model_card.json',as_json(model))
        models.append(model)
        verification.append({'id':model_id,'weights_and_onnx_hashes_verified':True,'onnx_checker_passed':True,
                             'all_six_cached_native_cases_reproduce_gt_and_shift_metrics':True})
        del ref_predictions
    # Stream source images straight into the ZIP; avoid a second copied dataset.
    for i,(r,item) in enumerate(zip(records,order),1):
        original=lookup[r['image']]
        source=ROOT/'datasets/pingpong_v1'/r['image']
        with Image.open(source) as image:
            image.load()
            assert image.size==(1280,1024)
        image_part.add_file(r['image'],source,r['image_sha256'])
        label=(ROOT/'datasets/pingpong_v1'/r['label']).read_text(encoding='utf-8')
        assert label.strip()==original['label'].strip()
        core.add_bytes('testdata/'+r['label'],label.encode('utf-8'))
        core.add_file('testdata/'+r['annotation'],ROOT/'datasets/pingpong_v1'/r['annotation'],r['annotation_sha256'])
        if i%200==0:
            print(json.dumps({'stage':'original_images_hash_decode_zip','completed':i,'total':len(records)}),flush=True)
        if shutil.disk_usage(ROOT).free<5.5*1024**3:
            raise RuntimeError('Stop to preserve 5GiB disk reserve')
    image_part.add_bytes('image_order.json',as_json(records))
    image_part.add_bytes('README.txt',b'Full frozen val/test originals only. Match image_order with the models_and_references package. Not synchronized stereo pairs.\n')
    package={'package_id':PACKAGE,'created_utc':datetime.now(timezone.utc).isoformat(),
             'source_repository':'https://github.com/JefferyXeom/pingpong2026','source_commit':code_commit,
             'source_note':'Exact packaging, evaluator and builder source snapshots are embedded and hashed; source_commit records repository state before packaging additions.',
             'models':models,'protocol':protocol,'test_images':len(records),
             'counts':{s:{'images':sum(r['split']==s for r in records),'objects':sum(r['objects'] for r in records if r['split']==s)} for s in ['val','test']},
             'quick128_counts':dict(Counter(records[i]['split'] for i in indices)),
             'model_verification':verification,'includes_training_images':False,'includes_windows_engines':False,
             'includes_credentials':False,'network_transfer_state':'not_started; receiver not configured'}
    core.add_bytes('package.json',as_json(package))
    core.add_bytes('requirements-check.txt',b'# Hash verification: Python standard library only\n# Cached metric reproduction: numpy\n# Engine builder: compatible TensorRT10 Python API on target CUDA/driver\n# Do not install Windows environment_freeze blindly on Ubuntu/Turing\n')
    core.add_file('source_environment/windows_environment_freeze.txt',ROOT/'reports/environment/windows_rtx3090/environment_freeze.txt')
    text=readme()
    core.add_bytes('README.zh-CN.md',text.encode('utf-8'))
    core_result=core.finish('models_and_references')
    image_result=image_part.finish('holdout_images')
    (OUT/'README.zh-CN.md').write_text(text,encoding='utf-8')
    shutil.copy2(ROOT/'scripts/verify_deployment_package.py',OUT/'verify_deployment_package.py')
    sums=''.join(r['sha256']+'  '+r['archive']+'\n' for r in [core_result,image_result])
    (OUT/'SHA256SUMS').write_text(sums,encoding='utf-8')
    report={'package_id':PACKAGE,'archives':[core_result,image_result],'models':[{k:v for k,v in m.items() if k in ['id','run_id','selected_epoch','weights_sha256','onnx_sha256']} for m in models],
            'full_holdout_images':len(records),'quick_images':len(indices),'cached_native_cases':sum(len(m['cases']) for m in models),
            'test_counts':package['counts'],'source_commit':code_commit,'source_files_modified':False,
            'model_verification':verification,'network_transfer_state':'waiting_for_receiver_configuration'}
    # Read back every archived member and compare SHA256 before declaring ready.
    report['archive_verification']=[verify(OUT/r['archive']) for r in report['archives']]
    report['free_disk_gib']=shutil.disk_usage(ROOT).free/1024**3
    report['state']='ready_for_transfer'
    (OUT/'package_receipt.json').write_bytes(as_json(report))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def readme():
    return '''# 训练端→部署端模型与测试包（2026-11-18）

两个ZIP属于同一版本deployment_handoff_20261118_v1。先用SHA256SUMS核对整包，再用随附Python工具逐文件验证，之后分别解压到core和testdata目录。包内不含私钥、Token、训练图片或Windows/3090 engine。

## 内容

- models_and_references.zip：B旧n第24轮、C旧s第30轮、D新n第12轮、E新s第22轮；各含best.pt、原图动态ONNX、源构建参数、实际args/训练CSV/模型卡。B为目标机第一验证候选，其他保留对照，不预设新模型全面更好。
- 同模型/原图下的PT FP32 b1/b2、ORT CPU FP32 b2、TRT FP32 b2、TRT FP16 b1/b2参考预测，共24份完整缓存；package.json保存精确val/test球心指标、定位分位误差和转换位移。
- holdout_images.zip：完整固定val1272图/1272球、test830图/859球，共2102张1280×1024原PNG；YOLO标签和原始XML在core/testdata。没有确认无球负样本，也不是经过验证的同步左右帧序列。
- quick128.txt/quick128_indices.json：从上述图片按split、左右目、目标数分层抽128张，仅用于流程检查，无额外重复图片。此子集不替代完整验收，也不得用于选阈值或INT8校准。

## 部署端检查与解压

```bash
sha256sum -c SHA256SUMS
python3 verify_deployment_package.py models_and_references.zip
python3 verify_deployment_package.py holdout_images.zip
mkdir -p core testdata
python3 -m zipfile -e models_and_references.zip core
python3 -m zipfile -e holdout_images.zip testdata
python3 core/tools/verify_deployment_package.py core
python3 core/tools/verify_deployment_package.py testdata
```

校验失败不得使用。先保证接收目录有约8GiB空间（存放压缩包与解压文件，另预留引擎/软件空间）；ZIP不包含整个Git仓库，代码通过git pull取得。package.json中的source_commit与包内工具快照/哈希用于追溯。

## 精度核验

先仅用numpy复算缓存，确认评估工具工作正常：

```bash
python3 core/tools/evaluate_deployment_predictions.py --core core --model B_n_merged_v1 --case h1024_w1280_tensorrt_fp16_b2 --subset quick128
python3 core/tools/evaluate_deployment_predictions.py --core core --model B_n_merged_v1 --case h1024_w1280_tensorrt_fp16_b2
```

原图推理输入[B,3,1024,1280]、RGB/CHW/FP32除255，原图不缩放/补边/裁切。输出[B,300,6]为xyxy/conf/class0=ball；免NMS，保留多个候选。FP16 engine接口仍是FP32。目标端新预测恢复到原图坐标、裁到[0,1280]×[0,1024]，保存Nx6、conf>=.001，每图NPZ字段f0..fN-1；顺序必须与core/testdata/image_order.json一致，quick-only为quick128_indices.json列出的顺序。

```bash
python3 core/tools/evaluate_deployment_predictions.py --core core --model B_n_merged_v1 --case h1024_w1280_tensorrt_fp16_b2 --predictions target_b2.npz --output target_b2_metrics.json
```

完整固定conf=.25，5px贪心一对一球心P/R/F1和TP/FP/FN；定位误差中位/P95/P99仅对10px内GT匹配，不含漏检。转换位移相对同模型/尺寸/batch的源PT FP32，合并val+test、预测配对10px，包含可能共同误检；新增5px漏匹配单列。框IoU不作为主要排名。包内所有24组合的参考指标和转换位移均已从现有缓存重算核对，未重跑模型。

## 目标机TensorRT构建

部署机先核实RTX2060 Super、驱动/CUDA/Ubuntu、相机采集模式和TensorRT对Turing SM7.5的支持。Windows freeze只供追溯，不能直接照搬安装。提供的参考构建脚本使用TensorRT10 API；其他主版本必须适配，不能默认兼容。已在训练端做语法/配置审查，未在目标GPU执行。

```bash
python3 core/tools/build_deployment_engine.py --onnx core/models/B_n_merged_v1/model.onnx --output target_engines/B_fp16_b2.engine --batch 2 --precision fp16
python3 core/tools/build_deployment_engine.py --onnx core/models/B_n_merged_v1/model.onnx --output target_engines/B_fp16_b1.engine --batch 1 --precision fp16
```

固定min=opt=max、workspace2GiB、优化级别3、禁TF32、FP16检测头相关Sigmoid留FP32，写构建JSON/哈希；原生engine无Ultralytics元数据头。动态ONNX不意味着固定engine可以任意改尺寸/batch。构建成功不等于精度和实时性已通过；目标机需要独立推理程序生成预测/测速，包内工具不会自动接相机。

batch2离线图片不保证同步左右目；实际串流需同次曝光配对。先完成用户已有硬件同步/标定，再做极线+共享主球轨迹关联，歧义不强行输出实测深度。240图/秒目标尚未在目标机验收，应测完整流程延迟/积压/丢帧，不能只看3090 GPU速度。INT8未执行，不用val/test做校准。
'''


if __name__=='__main__':
    main()
