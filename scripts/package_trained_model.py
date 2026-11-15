"""Preserve a center-localization candidate chosen on validation, with honest evaluation notes."""
import json
from pathlib import Path
import shutil

from audit_additional_data import ROOT, sha256


def main():
    validation = json.loads((ROOT/'reports/evaluation/baseline_yolo26n_960/center_selection_v1/baseline_full_val_centers.json').read_text(encoding='utf-8'))
    testing = json.loads((ROOT/'reports/evaluation/baseline_yolo26n_960/center_selection_v1/baseline_full_test_centers.json').read_text(encoding='utf-8'))
    analysis = json.loads((ROOT/'reports/training/baseline_yolo26n_960/baseline_training_analysis.json').read_text(encoding='utf-8'))
    chosen = validation['selected_for_center_application']
    assert analysis['normal_completion_report_exists']
    selected_val = next(r for r in validation['evaluations'] if r['weights']==chosen['weights'])
    selected_test = next(r for r in testing['evaluations'] if r['weights']==chosen['weights'])
    target = ROOT/'models/pingpong_yolo26n_center_v1'
    target.mkdir(exist_ok=False)
    source = ROOT/chosen['weights']
    destination = target/'weights.pt'
    checksum = sha256(source)
    shutil.copy2(source,destination)
    assert sha256(destination)==checksum
    card = {'model':'YOLO26n', 'task':'Single-class ball detection / 2D center localization',
            'weights':'weights.pt', 'sha256':checksum, 'source_checkpoint':chosen['weights'],
            'source_epoch':analysis['completed_epochs'] if source.stem=='last' else analysis['best_epoch'],
            'selection':chosen, 'validation':selected_val, 'test':selected_test,
            'training_dataset':'datasets/pingpong_v1', 'training_images':6618, 'validation_images':1272, 'test_images':830,
            'training_epochs_completed':analysis['completed_epochs'], 'input_long_side':960,
            'training_input_shape_height_width':[960,960], 'training_rect':False, 'training_multi_scale':0,
            'evaluation_preprocessing':{'imgsz':960,'batch':16,'rect':True,'nms':False},
            'class_names':{'0':'ball'},
            'limitations':['Ball-center metrics use distance to original annotation centers, not physical 3D ground truth.',
                'Box AP remains low, consistent with substantial predicted/annotated size differences. Original labels were preserved.',
                'No verified no-ball examples; eye/watch reflection false positives require separate testing.',
                'No temporal tracking, stereo synchronization, triangulation or target-Ubuntu latency assessment.',
                'Rectangular FP16 follow-up is recorded separately; fixed-square TensorRT and INT8 still require validation.'],
            'normal_completion_note':'Finished 28 epochs under patience=25; final best/last checkpoints are inference weights with optimizer removed by Ultralytics. New training can initialize from these weights, but cannot resume their stripped optimizer state.'}
    (target/'model_card.json').write_text(json.dumps(card,indent=2),encoding='utf-8')
    vs,ts=selected_val['scores']['0.25'],selected_test['scores']['0.25']
    text=f'''# 乒乓球 YOLO26n：球心定位候选 v1

训练已正常完成 {analysis['completed_epochs']} 轮，因连续 25 轮框 mAP 未提升而早停。数据为 pingpong_v1：6,618 张训练、1,272 张验证、830 张测试。新增历史备份和换背景图片未加入本轮。

原始图片宽×高为1280×1024；训练网络输入宽×高为960×960（imgsz=960、rect=False、multi_scale=0），batch=16，启用AMP混合精度训练。推理尺寸另见正式评估协议。

候选由完整验证集在固定 conf=0.25、原图球心距离不超过 5 像素的 F1 选择，来自第 {card['source_epoch']} 轮 `{source.name}`。测试集没有参与权重或阈值选择。它与 Ultralytics 按框 mAP 选择的第 3 轮 best.pt 的用途评价不同，两份原始训练权重均保留。

| 完整数据集 | 球心精确率 | 球心召回率 | 球心 F1 | 匹配误差中位数 | 匹配误差 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 验证 1,272 张 | {vs['center5_precision']:.2%} | {vs['center5_recall']:.2%} | {vs['center5_f1']:.2%} | {vs['center10_matched_error_px_median_p95'][0]:.2f} px | {vs['center10_matched_error_px_median_p95'][1]:.2f} px |
| 测试 830 张 | {ts['center5_precision']:.2%} | {ts['center5_recall']:.2%} | {ts['center5_f1']:.2%} | {ts['center10_matched_error_px_median_p95'][0]:.2f} px | {ts['center10_matched_error_px_median_p95'][1]:.2f} px |

球心精确率/召回率使用 5 像素距离阈值；误差分位数统计 10 像素以内成功匹配的球，不包括未匹配目标。坐标都在原始 1280×1024 图片上计算。它们是补充指标，不能替代或改称标准框 mAP。

验证集中预测框宽、高的中位数约为对应标注框的 {vs['center10_matched_pred_gt_width_ratio_median']:.2f}、{vs['center10_matched_pred_gt_height_ratio_median']:.2f} 倍。即使球心准确，框尺寸不同仍会让 IoU 很低。没有自动放大预测框或修改验证标签来提高分数，需要统一后续标注规范。

推理示例，在工作区根目录的 Python 环境执行：

```python
from ultralytics import YOLO
model = YOLO('models/pingpong_yolo26n_center_v1/weights.pt')
results = model.predict('your_image.png', imgsz=960, conf=0.25,
                        nms=False, rect=True, device=0)
for result in results:
    centers_xy = result.boxes.xywh[:, :2]  # 原图坐标；可能有多个候选
```

此模型包记录选模时的PyTorch球心结果。后续矩形FP16正式评估、各批次模型及数据量见[训练与评估总览](../../docs/training_and_evaluation_overview.zh-CN.md)。包内权重保持原样；固定方形引擎、INT8、无球专项、双目三维与目标Ubuntu性能需单独验证。

训练正常结束后，Ultralytics 已剥离优化器；这里的权重可用于推理和新一轮微调的初始化，不能直接作为含完整优化器状态的断点续训。
'''
    (target/'README.zh-CN.md').write_text(text,encoding='utf-8')
    (ROOT/'reports/evaluation/baseline_yolo26n_960/center_selection_v1/trained_model_selection.json').write_text(json.dumps(card,indent=2),encoding='utf-8')
    print(json.dumps({'model_directory':str(target),'weights_sha256':checksum,'source_epoch':card['source_epoch'],
                      'validation_conf_025':vs,'test_conf_025':ts},indent=2),flush=True)


if __name__=='__main__':
    main()
