"""Repeat GPU-only timings after all engines exist, with longer warmup and measurement."""
import gc
import argparse
import json
from pathlib import Path

from evaluate_conversion_matrix import ROOT, OUT, REPORT, Runner, atomic_json, progress


def main():
    global OUT, REPORT
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',default='center_conversion_matrix_v1')
    args=parser.parse_args()
    if Path(args.experiment).name!=args.experiment:
        raise ValueError('Invalid experiment ID')
    OUT=ROOT/'artifacts'/args.experiment
    REPORT=ROOT/'reports/evaluation'/args.experiment/f'{args.experiment}.json'
    import evaluate_conversion_matrix as matrix
    matrix.OUT, matrix.REPORT=OUT, REPORT
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    report=json.loads(REPORT.read_text(encoding='utf-8'))
    if report.get('state')!='complete':
        raise RuntimeError('Wait until the full precision matrix completes.')
    with torch.inference_mode():
        for case,item in report['cases'].items():
            if item['backend']=='onnx_cpu_fp32' or item.get('timing_repeated_after_all_builds'):
                continue
            progress('final_benchmark',case=case,warmup=200,iterations=1000)
            shape=tuple(item['input_height_width'])
            directory=OUT/f'h{shape[0]}_w{shape[1]}'
            path=directory/f'trt_fp{16 if item["backend"].endswith("16") else 32}_b{item["batch"]}.engine'
            runner=Runner(item['backend'],shape,item['batch'],path,source_weights=OUT/'source.pt')
            item['initial_benchmark']=item['benchmark']
            # TensorRT warns that the legacy default stream may add implicit
            # synchronizations. Use a dedicated stream for both GPU backends.
            benchmark_stream=torch.cuda.Stream()
            benchmark_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(benchmark_stream):
                item['benchmark']=runner.benchmark(samples=1000,warmup=200)
            torch.cuda.current_stream().wait_stream(benchmark_stream)
            item['benchmark']['execution_stream']='dedicated non-default CUDA stream; identical policy for PyTorch and TensorRT'
            item['timing_repeated_after_all_builds']=True
            atomic_json(REPORT,report)
            del runner
            gc.collect();torch.cuda.empty_cache()
    progress('complete',completed_cases=len(report['cases']),final_benchmarks_complete=True)


if __name__=='__main__':
    main()
