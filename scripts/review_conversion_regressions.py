"""Explain individual FP16 regressions using already saved prediction caches."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    from evaluate_conversion_matrix import truth
    report=json.loads((ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_v1.json').read_text(encoding='utf-8'))
    order=json.loads((ROOT/'artifacts/center_conversion_matrix_v1/image_order.json').read_text(encoding='utf-8'))
    positions={r['image']:i for i,r in enumerate(order)}
    manifest={r['image']:r for r in map(json.loads,(ROOT/'datasets/pingpong_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    changes=[]
    for key,case in report['cases'].items():
        if case['backend']!='tensorrt_fp16':
            continue
        shape=case['input_height_width']
        reference=report['cases'][f'h{shape[0]}_w{shape[1]}_pytorch_fp32_b{case["batch"]}']
        with np.load(ROOT/reference['prediction_cache']) as ref, np.load(ROOT/case['prediction_cache']) as new:
            for example in case['against_pytorch']['changed_gt_examples']:
                record=manifest[example['image']]
                gt=truth(record)
                centers=(gt[:,:2]+gt[:,2:])/2
                item={'case':key,**example,'closest_predictions':[]}
                for index in sorted(set(example['lost_gt_indices']+example['gained_gt_indices'])):
                    detail={'gt_index':index,'gt_center':centers[index].tolist()}
                    for label,cache in (('pytorch',ref),('tensorrt_fp16',new)):
                        predictions=cache[f'f{positions[example["image"]]}']
                        distances=np.linalg.norm((predictions[:,:2]+predictions[:,2:4])/2-centers[index],axis=1)
                        indices=np.argsort(distances)[:3]
                        detail[label]=[{'center_distance_px':float(distances[i]),'confidence':float(predictions[i,4]),
                                        'xyxy':predictions[i,:4].tolist()} for i in indices]
                    item['closest_predictions'].append(detail)
                changes.append(item)
    output=ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_regressions.json'
    output.write_text(json.dumps(changes,indent=2),encoding='utf-8')
    print(json.dumps(changes,indent=2))


if __name__=='__main__':
    main()
