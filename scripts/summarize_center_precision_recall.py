"""Publish center-only P/R tables from recorded metrics, without inference or tuning."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = 'native_resolution_suite_v1'
OUT = ROOT / 'reports/experiments' / SUITE
DOC = ROOT / 'docs/experiments/native_center_precision_recall.zh-CN.md'
LABELS = {'baseline_yolo26n_960': '旧n/v1，第28轮',
          'native_v1_yolo26n_1280': 'A：n/v1，第33轮',
          'native_v1_additional_yolo26n_1280': 'B：n/合并，第24轮',
          'native_v1_additional_yolo26s_1280': 'C：s/合并，第30轮'}


def main():
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    rows = []
    for original in summary['rows']:
        reference = next(r for r in summary['rows'] if r['training_run'] == original['training_run']
                         and r['input_height'] == original['input_height'] and r['input_width'] == original['input_width']
                         and r['batch'] == original['batch'] and r['backend'] == 'pytorch_fp32')
        for split in ('val', 'test'):
            tp, gt, predictions = [original[f'{split}_{key}'] for key in ('center5_matches', 'gt', 'predictions')]
            p, recall = original[f'{split}_center5_precision'], original[f'{split}_center5_recall']
            assert abs(p - tp / max(predictions, 1)) < 1e-12
            assert abs(recall - tp / gt) < 1e-12
            rows.append({'model': LABELS[original['training_run']], 'training_run': original['training_run'],
                         'selected_epoch': original['selected_epoch'], 'weights': original['weights'],
                         'weights_sha256': original['weights_sha256'],
                         'input_width': original['input_width'], 'input_height': original['input_height'],
                         'backend': original['backend'], 'batch': original['batch'], 'split': split,
                         'conf': 0.25, 'center_match_radius_original_px': 5,
                         'gt': gt, 'predictions': predictions, 'tp': tp, 'fp': predictions-tp, 'fn': gt-tp,
                         'center_precision': p, 'center_recall': recall,
                         'precision_delta_pp_vs_same_batch_pt': 100*(p-reference[f'{split}_center5_precision']),
                         'recall_delta_pp_vs_same_batch_pt': 100*(recall-reference[f'{split}_center5_recall']),
                         'center_f1': original[f'{split}_center5_f1'],
                         'matched10_error_p50_original_px': original[f'{split}_error_p50_px'],
                         'matched10_error_p95_original_px': original[f'{split}_error_p95_px'],
                         'matched10_error_p99_original_px': original[f'{split}_error_p99_px']})
    assert len(rows) == 96 and len({(r['training_run'], r['input_width'], r['backend'], r['batch'], r['split']) for r in rows}) == 96
    with (OUT / 'center_precision_recall.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'center_precision_recall.json').write_text(json.dumps({
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Recorded conf=.25 metrics only; no inference or threshold tuning',
        'primary_metrics': ['center_precision', 'center_recall', 'center_f1',
                            'matched10_error_p50_original_px', 'matched10_error_p95_original_px', 'matched10_error_p99_original_px'], 'no_iou_matching': True,
        'evaluated_precisions': ['FP32', 'TensorRT mixed FP16'], 'int8_evaluated': False,
        'case_count': 48, 'split_rows': len(rows), 'rows': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 各推理版本的球心 precision / recall', '',
        '球心precision、recall、F1和定位误差中位数/P95/P99都是本项目重要的部署比较指标。本页侧重P/R，误差和转换位移另见完整结果；CSV同时保留F1和误差分位数。本页不按框IoU判断正确与否，也不按框mAP排名。', '',
        '本表来自已完成的48项评估，各包含val/test，因此CSV共96行。没有重新训练、推理或调整阈值。新三组训练仍按当时的完整val球心F1选best，历史选模方式未改变。', '',
        '[球心专用CSV](../../reports/experiments/native_resolution_suite_v1/center_precision_recall.csv) · [球心专用JSON](../../reports/experiments/native_resolution_suite_v1/center_precision_recall.json) · [完整训练配置与结果](native_resolution_suite_v1_results.zh-CN.md)', '',
        '## 统计口径', '',
        '置信度conf≥0.25；预测中心和标注中心在原图坐标中距离≤5px，并做一对一贪心最近匹配，即计为TP。precision=TP/全部预测数，recall=TP/全部标注球数；FP=预测数−TP，FN=标注球数−TP。框的大小和重叠面积不参与这些正确匹配判断。', '',
        '5px是本次统一评估半径，不代表应用已经接受5px定位误差；定位误差分布另保留。位置偏差超过5px的预测可能同时产生FP和FN。当前没有已确认无球测试数据，本表precision不能代替反光眼睛/手表等无球视频的专项误报评估。', '',
        '全部原图为宽1280×高1024。1280档不缩放/补边/裁剪；960档仅等比缩小至960×768，再将预测坐标还原到原图。val为1,272张/1,272球，test为830张/859球。表中P/R顺序固定为precision / recall。', '',
        '## 已实测的版本', '',
        'PyTorch FP32（b1/b2）、ONNX Runtime CPU FP32（b2）、TensorRT FP32（b2）、TensorRT混合FP16（b1/b2）。FP16使用半精度计算并保留部分FP32操作；它与INT8量化不同。**INT8尚未转换、校准或评估，没有可报告的INT8 P/R**。各后端来自对应同一训练权重；主表按同权重、同尺寸、同batch比较。', '']
    for width, batch, title in [(1280, 2, '原图1280×1024，batch2（优先比较）'),
                               (1280, 1, '原图1280×1024，batch1'),
                               (960, 2, '缩小至960×768，batch2'),
                               (960, 1, '缩小至960×768，batch1')]:
        lines.extend(['## ' + title, '',
                      '| 模型 | 推理版本 | val P / R | test P / R | test TP / FP / FN | test ΔP / ΔR（百分点） |',
                      '| --- | --- | --- | --- | --- | --- |'])
        for r in rows:
            if r['input_width'] != width or r['batch'] != batch or r['split'] != 'test':
                continue
            val = next(v for v in rows if v['training_run'] == r['training_run'] and v['backend'] == r['backend']
                       and v['batch'] == batch and v['input_width'] == width and v['split'] == 'val')
            lines.append(f'| {r["model"]} | {r["backend"]} | {val["center_precision"]:.2%} / {val["center_recall"]:.2%} | '
                         f'{r["center_precision"]:.2%} / {r["center_recall"]:.2%} | {r["tp"]} / {r["fp"]} / {r["fn"]} | '
                         f'{r["precision_delta_pp_vs_same_batch_pt"]:+.3f} / {r["recall_delta_pp_vs_same_batch_pt"]:+.3f} |')
        lines.append('')
    lines.extend(['## 读取结论', '',
        '原图batch2下，B的TRT FP16测试P=94.87%、R=96.86%；C为P=95.45%、R=97.67%。本次B/C的这两项与各自PT FP32一致，逐目标比较也没有新增5px匹配失败。C有更高测试P/R，B在完整val上precision更高而recall更低，需要结合实际误报与漏检代价和目标机速度取舍。', '',
        'FP16并非所有组合都完全无影响：例如B原图batch2的val precision有所下降，recall保持；C原图batch1的test多1个额外预测，使precision小幅下降。应分别看P与R及计数，不能只凭F1接近就说没有变化。', '',
        'FP16可能改变坐标和置信度；即使中心位置仍正确，置信度跨过0.25也可能影响P/R。此处所有比较保持阈值不变，没有为量化版本单独调阈值。部署验收尚未指定P/R最低门槛，不自动将这些结果视为验收通过。', '',
        '复现本页：`& ./.venv/Scripts/python.exe scripts/summarize_center_precision_recall.py`。只读取已有结果，不运行模型。'])
    DOC.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'cases': 48, 'split_rows': len(rows), 'primary': ['center_precision', 'center_recall'], 'int8_evaluated': False}))


if __name__ == '__main__':
    main()
