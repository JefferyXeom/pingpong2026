"""Build a fixed-shape native TensorRT10 engine on the deployment GPU.

Run only after verifying the model and target software's Turing support.
Preserves source experiment settings; target execution has not been tested here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, choices=[1,2], required=True)
    parser.add_argument('--precision', choices=['fp32','fp16'], default='fp16')
    args = parser.parse_args()
    import tensorrt as trt
    if trt.__version__.split('.')[0] != '10':
        raise RuntimeError('This reference builder uses TensorRT10 precision constraints; do not silently migrate to another major API')
    if args.output.exists() or args.output.with_suffix('.build.json').exists():
        raise FileExistsError('Choose a new engine name; existing results are preserved')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if args.precision == 'fp16' and not builder.platform_has_fast_fp16:
        raise RuntimeError('Target reports no fast FP16 support')
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network,logger)
    if not onnx_parser.parse_from_file(str(args.onnx.resolve())):
        raise RuntimeError('\n'.join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
    if network.num_inputs != 1 or network.get_input(0).name != 'images':
        raise RuntimeError('Unexpected input contract')
    if network.get_input(0).dtype != trt.float32 or network.num_outputs != 1 or network.get_output(0).dtype != trt.float32:
        raise RuntimeError('Expected FP32 input/output interfaces even for mixed FP16 execution')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2*1024**3)
    config.builder_optimization_level = 3
    config.clear_flag(trt.BuilderFlag.TF32)
    shape = (args.batch,3,1024,1280)
    profile = builder.create_optimization_profile()
    if not profile.set_shape('images',shape,shape,shape):
        raise RuntimeError('Unable to create fixed shape profile')
    config.add_optimization_profile(profile)
    constraints=[]
    if args.precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        ids=[int(m.group(1)) for layer in network for m in [re.search(r'/model\.(\d+)/',layer.name)] if m]
        if not ids:
            raise RuntimeError('Unable to identify YOLO detection head')
        head=f'/model.{max(ids)}/'
        for layer in network:
            if head in layer.name and layer.type==trt.LayerType.ACTIVATION and 'sigmoid' in layer.name.lower():
                layer.precision=trt.float32
                for i in range(layer.num_outputs):
                    layer.set_output_type(i,trt.float32)
                constraints.append(layer.name)
        if not constraints:
            raise RuntimeError('No FP32 head Sigmoid constraints found; review graph before changing policy')
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    started=time.perf_counter()
    serialized=builder.build_serialized_network(network,config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')
    runtime=trt.Runtime(logger)
    engine=runtime.deserialize_cuda_engine(serialized)
    context=engine.create_execution_context()
    if not context.set_input_shape('images',shape):
        raise RuntimeError('Built engine rejected requested shape')
    names=[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    outputs=[name for name in names if engine.get_tensor_mode(name)==trt.TensorIOMode.OUTPUT]
    if len(outputs)!=1 or tuple(context.get_tensor_shape(outputs[0]))!=(args.batch,300,6):
        raise RuntimeError('Unexpected output contract')
    if any(engine.get_tensor_dtype(name)!=trt.float32 for name in names):
        raise RuntimeError('Engine I/O dtype changed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+'.partial')
    temporary.write_bytes(bytes(serialized)); temporary.replace(args.output)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],capture_output=True,text=True)
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'onnx_sha256':digest(args.onnx),'engine_sha256':digest(args.output),
              'shape':shape,'precision':args.precision,'io_dtype':'FP32','tf32':False,'workspace_gib':2,
              'builder_optimization_level':3,'fp32_sigmoid_layers':constraints,'tensorrt':trt.__version__,
              'gpu_and_driver':gpu.stdout.strip(),'seconds':time.perf_counter()-started,
              'format':'native TensorRT without Ultralytics metadata header','inference_accuracy_validated':False}
    args.output.with_suffix('.build.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
