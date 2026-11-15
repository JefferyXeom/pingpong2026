"""Check completed evaluation artifacts without repeating inference or modifying data."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    report=json.loads((ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_v1.json').read_text(encoding='utf-8'))
    assert report['state']=='complete' and len(report['cases'])==12
    assert report['image_count']==2102
    with (ROOT/'models/pingpong_yolo26n_center_v1/weights.pt').open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==report['weights_sha256']
    checked=[]
    for key,case in report['cases'].items():
        assert tuple(case['input_height_width']) in ((768,960),(1024,1280))
        for split,images,balls in (('val',1272,1272),('test',830,859)):
            item=case['splits'][split]
            assert item['images']==images
            for metrics in item['thresholds'].values():
                assert metrics['gt']==balls
                assert 0<=metrics['center5_matches']<=metrics['center10_matches']<=balls
                assert all(math.isfinite(metrics[k]) and 0<=metrics[k]<=1 for k in
                           ('center5_precision','center5_recall','center5_f1'))
                assert all(math.isfinite(v) and 0<=v<=10 for v in metrics['matched_error_px_p50_p95_p99'])
        with np.load(ROOT/case['prediction_cache']) as cache:
            assert len(cache.files)==2102
        if case['backend']!='onnx_cpu_fp32':
            assert case['timing_repeated_after_all_builds']
            assert case['benchmark']['warmup_iterations']==200
            for mode in ('ordinary','cuda_graph'):
                assert case['benchmark'][mode]['iterations']==1000
                assert case['benchmark'][mode]['cuda_event_batch_ms_mean']>0
        checked.append(key)
    examples=json.loads((ROOT/'reports/evaluation/center_conversion_matrix_v1/conversion_original_examples/index.json').read_text(encoding='utf-8'))
    from PIL import Image
    assert len(examples)==8
    for example in examples:
        with Image.open(ROOT/example['file']) as image:
            assert image.size==(1280,1024)
    result={'passed':True,'checked_cases':checked,'images_per_case':2102,
            'final_gpu_iterations_per_mode':1000,'original_resolution_examples':8,
            'weights_sha256':report['weights_sha256']}
    (ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
