"""Export a checkpoint to FP32 ONNX and FP16 TensorRT, then run real images."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
DLL_HANDLES = []
if os.name == 'nt':
    site = Path(sys.prefix) / 'Lib/site-packages'
    dll_dirs = [site / 'torch/lib', site / 'tensorrt_libs', site / 'tensorrt_cu12_libs', site / 'nvidia/cuda_runtime/bin']
    for directory in dll_dirs:
        if directory.is_dir():
            DLL_HANDLES.append(os.add_dll_directory(str(directory)))
            os.environ['PATH'] = str(directory) + os.pathsep + os.environ.get('PATH', '')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='runs/detect/smoke_yolo26n_960/weights/best.pt')
    parser.add_argument('--name', default='smoke_yolo26n_960')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--external-nms', action='store_true', help='Export the conventional head instead of the YOLO26 NMS-free head')
    args = parser.parse_args()
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError('At least 8 GiB free required before engine export.')
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import tensorrt as trt
    import ultralytics
    from ultralytics import YOLO
    weights = (ROOT / args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    output = ROOT / 'artifacts' / args.name
    output.mkdir(parents=True, exist_ok=False)
    fp32_weights = output / 'model_fp32.pt'
    fp16_weights = output / 'model_fp16.pt'
    shutil.copy2(weights, fp32_weights)
    shutil.copy2(weights, fp16_weights)
    nms_mode = None if args.external_nms else False
    report = {'source_weights': str(weights), 'imgsz': args.imgsz, 'batch': 1, 'dynamic': False, 'nms_free': not args.external_nms, 'versions': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'tensorrt': trt.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'cuda': torch.version.cuda}, 'gpu': torch.cuda.get_device_name(0), 'engine_portability': 'Built on Windows for this GPU/software stack. Rebuild and validate on the target Ubuntu server.', 'accuracy_scope': 'Conversion checks validate export/inference. Detection quality and quantization accuracy need evaluation on held-out data using the trained baseline.'}
    onnx_path = Path(YOLO(str(fp32_weights)).export(format='onnx', imgsz=args.imgsz, batch=1, dynamic=False, simplify=True, opset=17, quantize=32, nms=nms_mode, device=0))
    onnx.checker.check_model(str(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    report['onnx'] = {'path': str(onnx_path), 'inputs': [{'name': value.name, 'shape': value.shape, 'type': value.type} for value in session.get_inputs()], 'outputs': [{'name': value.name, 'shape': value.shape, 'type': value.type} for value in session.get_outputs()], 'checker_passed': True}
    # Use identical square preprocessing and the same detection head for a raw FP32 check.
    import cv2
    from ultralytics.data.augment import LetterBox
    sample = sorted((ROOT / 'datasets/pingpong_v1/images/test').glob('*.png'))[0]
    pixels = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False)(image=cv2.imread(str(sample)))
    pixels = np.ascontiguousarray(pixels[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255
    reference = YOLO(str(weights))
    reference.model.end2end = not args.external_nms
    reference.model.float().eval().cpu().fuse(verbose=False)
    with torch.inference_mode():
        raw_reference = reference.model(torch.from_numpy(pixels))
        if isinstance(raw_reference, (tuple, list)):
            raw_reference = raw_reference[0]
        raw_reference = raw_reference.detach().cpu().numpy()
    raw_onnx = session.run(None, {session.get_inputs()[0].name: pixels})[0]
    assert raw_reference.shape == raw_onnx.shape
    assert np.isfinite(raw_reference).all() and np.isfinite(raw_onnx).all()
    score_gap = float(np.max(np.abs(raw_reference[..., 4] - raw_onnx[..., 4]))) if not args.external_nms else float(np.max(np.abs(raw_reference[:, 4:] - raw_onnx[:, 4:])))
    assert score_gap < 0.005, 'ONNX FP32 scores differ unexpectedly from PyTorch'
    report['onnx_raw_comparison'] = {'image': str(sample), 'shape': list(raw_onnx.shape), 'scores_max_absolute_difference': score_gap, 'all_values_finite': True, 'max_absolute_element_difference': float(np.max(np.abs(raw_reference - raw_onnx))), 'coordinate_comparison_note': 'TopK ties can reorder low-confidence rows; full model accuracy is evaluated after baseline training.'}
    del reference
    report_path = output / 'conversion_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    del session
    engine = Path(YOLO(str(fp16_weights)).export(format='engine', imgsz=args.imgsz, batch=1, dynamic=False, simplify=True, opset=17, quantize=16, nms=nms_mode, workspace=2, device=0))
    report['engine'] = {'path': str(engine), 'precision': 'FP16', 'size_bytes': engine.stat().st_size}
    data = ROOT / 'datasets/pingpong_v1'
    records = [json.loads(line) for line in (data / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    heldout = [r for r in records if r['split'] in ('val', 'test')]
    samples = [heldout[int(i * len(heldout) / 8)] for i in range(8)]
    paths = [str(data / r['image']) for r in samples]
    predictions = {}
    speed = {}
    for backend, file, device in [('pytorch', weights, 0), ('onnx_cpu', onnx_path, 'cpu'), ('tensorrt_fp16', engine, 0)]:
        model = YOLO(str(file), task='detect')
        model.predict(paths[0], imgsz=args.imgsz, batch=1, device=device, nms=nms_mode, rect=False, conf=0.05, verbose=False)
        results = model.predict(paths, imgsz=args.imgsz, batch=1, device=device, nms=nms_mode, rect=False, conf=0.05, verbose=False)
        arrays = [r.boxes.data.cpu().numpy() for r in results]
        assert all(np.isfinite(a).all() for a in arrays), backend + ' produced non-finite outputs'
        predictions[backend] = [a.tolist() for a in arrays]
        speed[backend] = {'mean_inference_ms': float(np.mean([r.speed['inference'] for r in results])), 'detections_per_image': [len(r.boxes) for r in results], 'finite_outputs': True}
        del model
        torch.cuda.empty_cache()
    report['inference_checks'] = speed
    report['sample_images'] = [r['image'] for r in samples]
    report['predictions'] = predictions
    report['verification_limit'] = 'Conf=0.05 real-image inference and finite outputs checked on all three backends. Full precision/recall and conversion agreement must be evaluated using the trained baseline.'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'predictions'}, indent=2), flush=True)

if __name__ == '__main__':
    main()
