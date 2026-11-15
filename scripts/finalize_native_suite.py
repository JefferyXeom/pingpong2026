"""Verify completed cached experiments and publish results; no training or inference."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SUITE = 'native_resolution_suite_v1'
OUT = ROOT / 'reports/experiments' / SUITE
DOC = ROOT / 'docs/experiments/native_resolution_suite_v1_results.zh-CN.md'
LABELS = ['旧n / v1 / 960训练', 'A：n / v1 / 原图训练',
          'B：n / 合并 / 原图训练', 'C：s / 合并 / 原图训练']


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def local_time(iso):
    return datetime.fromisoformat(iso).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')


def table(lines, headers, rows):
    lines.extend(['', '| ' + ' | '.join(headers) + ' |',
                  '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    lines.append('')


def main():
    import numpy as np
    queue = read(f'reports/experiments/{SUITE}/progress.json')
    assert queue['state'] == 'complete', 'Suite must finish before final publication'
    from summarize_native_suite import main as refresh
    refresh()
    summary = read(f'reports/experiments/{SUITE}/summary.json')
    assert len(summary['rows']) == 48
    runs = summary['runs']
    training, matrices, checked_hashes = [], [], []
    verified_cases = gpu_cases = engines = 0
    reference_order = None
    for index, record in enumerate(runs):
        identifier = record['id']
        experiment = 'center_conversion_matrix_v1' if index == 0 else identifier + '_conversion_v1'
        artifacts = ROOT / 'artifacts' / experiment
        matrix = read(f'reports/evaluation/{experiment}/{experiment}.json')
        assert matrix['state'] == 'complete' and len(matrix['cases']) == 12
        assert matrix['image_count'] == 2102
        assert digest(artifacts / 'source.pt') == matrix['weights_sha256']
        checked_hashes.append(str((artifacts / 'source.pt').relative_to(ROOT)))
        order = read(f'artifacts/{experiment}/image_order.json')
        if reference_order is None:
            reference_order = order
        assert order == reference_order, 'Holdout order/content must match all models'
        expected = {f'h{h}_w{w}_{backend}_b{b}' for h, w in [(768, 960), (1024, 1280)]
                    for backend, b in [('pytorch_fp32', 1), ('pytorch_fp32', 2),
                    ('onnx_cpu_fp32', 2), ('tensorrt_fp32', 2),
                    ('tensorrt_fp16', 1), ('tensorrt_fp16', 2)]}
        assert set(matrix['cases']) == expected
        for key, case in matrix['cases'].items():
            height, width = case['input_height_width']
            folder = artifacts / f'h{height}_w{width}'
            with np.load(folder / f'{key}_predictions.npz', allow_pickle=False) as cache:
                assert set(cache.files) == {f'f{i}' for i in range(2102)}, key
            for split, images, gt in [('val', 1272, 1272), ('test', 830, 859)]:
                stats = case['splits'][split]
                assert stats['images'] == images and stats['thresholds']['0.25']['gt'] == gt
            if case['backend'] != 'onnx_cpu_fp32':
                assert case['timing_repeated_after_all_builds']
                assert case['benchmark']['warmup_iterations'] == 200
                for mode in ['ordinary', 'cuda_graph']:
                    assert case['benchmark'][mode]['iterations'] == 1000
                gpu_cases += 1
            if case['backend'].startswith('tensorrt'):
                precision = 16 if case['backend'].endswith('16') else 32
                prefix = f'trt_fp{precision}_b{case["batch"]}'
                build = read(str((folder / f'{prefix}.build.json').relative_to(ROOT)))
                assert build['shape'] == [case['batch'], 3, height, width]
                for path, checksum in [(folder / f'{prefix}.engine', build['engine_sha256']),
                                       (folder / 'weights.onnx', build['onnx_sha256'])]:
                    assert digest(path) == checksum, str(path)
                    checked_hashes.append(str(path.relative_to(ROOT)))
                engines += 1
            verified_cases += 1
        matrices.append(matrix)
        if index == 0:
            continue
        config = read(f'reports/training/{identifier}/configuration.json')
        result = record['training_result']
        progress = record['training_progress']
        with (ROOT / f'runs/detect/{identifier}/results.csv').open(encoding='utf-8', newline='') as stream:
            history = list(csv.DictReader(stream))
        assert len(history) == result['epochs_completed']
        assert digest(ROOT / result['best']) == result['best_sha256'] == matrix['weights_sha256']
        assert digest(ROOT / config['source_weights']) == config['source_sha256']
        checked_hashes.extend([result['best'], config['source_weights']])
        assert result['epochs_completed'] - result['selected_epoch'] == 20
        assert progress['training_batch_actual'] == 8
        epochs = [json.loads(line) for line in (ROOT / f'reports/training/{identifier}/epoch_metrics.jsonl').read_text().splitlines()]
        assert len(epochs) == len(history)
        selected = next(row for row in epochs if row['epoch'] == result['selected_epoch'])
        assert selected['fitness'] == max(row['fitness'] for row in epochs)
        training.append({'id': identifier, 'label': LABELS[index], 'configuration': config, 'result': result,
                         'csv_training_hours': float(history[-1]['time']) / 3600,
                         'wall_hours': (datetime.fromisoformat(result['completed_utc']) - datetime.fromisoformat(config['started_utc'])).total_seconds() / 3600,
                         'batches_per_epoch': (config['data']['train']['images'] + 7) // 8,
                         'last_batch': config['data']['train']['images'] % 8 or 8})

    def row(index, backend='pytorch_fp32', width=1280, batch=2):
        return next(r for r in summary['rows'] if r['training_run'] == runs[index]['id']
                    and r['backend'] == backend and r['input_width'] == width and r['batch'] == batch)

    verification = {'passed': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
                    'queue_complete': True, 'matrix_cases': verified_cases, 'prediction_caches': verified_cases,
                    'same_holdout_image_order_and_hash_records': True, 'images_per_case': 2102,
                    'gpu_cases_with_200_warmup_1000_measurements_both_modes': gpu_cases,
                    'engine_hashes_checked': engines, 'hash_verified_paths': sorted(set(checked_hashes)),
                    'three_training_results_verified_against_csv_and_epoch_fitness': True,
                    'free_disk_gib': shutil.disk_usage(ROOT).free / 1024**3,
                    'scope': 'Existing files, hashes, metadata and cached predictions only; no training/inference repeated.'}
    assert verification['free_disk_gib'] >= 5
    write_json(OUT / 'completion_verification.json', verification)
    write_json(OUT / 'final_analysis.json', {'completed_beijing': local_time(queue['updated_utc']),
        'training': training, 'native_fp32_batch2': [row(i) for i in range(4)],
        'native_trt_fp16_batch2': [row(i, 'tensorrt_fp16') for i in range(4)], 'verification': verification})

    lines = ['# 原图训练与转换评估：最终结果（2026-11-10）', '',
        f'全部完成于北京时间 **{local_time(queue["updated_utc"])}**。三组训练均正常早停，训练、转换精度矩阵、最终GPU复测全部完成；含旧基线共48个组合，新模型占36项。', '',
        '建议将 **B：YOLO26n＋v1/additional_v1合并数据＋1280×1024原图＋TensorRT FP16** 作为目标机优先验证候选，同时保留C（YOLO26s）比较漏检与计算预算。B验证F1略高、已匹配定位P95更小且更快；C在本次test上漏检更少，但P95更大、推理更慢。此为部署候选建议，没有用test重新选择checkpoint或阈值；尚未定版部署。', '',
        '[配置及48组合总表](native_resolution_suite_v1.zh-CN.md) · [可筛选CSV](../../reports/experiments/native_resolution_suite_v1/all_combinations.csv) · [最终分析JSON](../../reports/experiments/native_resolution_suite_v1/final_analysis.json) · [完整性核验](../../reports/experiments/native_resolution_suite_v1/completion_verification.json)', '',
        '**球心precision、recall、F1及定位误差中位数/P95/P99均是重要部署指标。** 各后端完整val/test P/R、TP/FP/FN及转换变化见[球心P/R专表](native_center_precision_recall.zh-CN.md)，定位误差与转换位移一并比较。框IoU/mAP保留历史存档，不作为本项目部署排名依据。', '',
        '## 训练配置与实际完成情况', '',
        '任务为单类别`0: ball`检测，关注二维球心。三组均从对应官方预训练n/s独立初始化，不接着旧模型微调。原图宽×高1280×1024；实际训练/验证张量为B×3×1024×1280，无缩放、补边或裁剪。imgsz=1280、rect=True、multi_scale=0；batch实际均为8。']
    table(lines, ['实验', 'train图片 / 球框', '每轮批数 / 末批', '完成 / 上限轮', '选中epoch', '验证F1', 'CSV训练耗时'],
        [(t['label'], f'{t["configuration"]["data"]["train"]["images"]:,} / {t["configuration"]["data"]["train"]["objects"]:,}',
          f'{t["batches_per_epoch"]} / {t["last_batch"]}', f'{t["result"]["epochs_completed"]} / 100',
          t['result']['selected_epoch'], f'{t["result"]["selected_validation"]["center5_f1"]:.2%}', f'{t["csv_training_hours"]:.3f}小时') for t in training])
    lines += ['CSV耗时包含逐轮训练/验证，排除后续转换及测速；三组共约' + f'{sum(t["csv_training_hours"] for t in training):.2f}小时。配置起止墙钟另见final_analysis.json。三组均在最佳球心F1之后20轮无严格提升而早停；评估用best.pt，不能用最终last.pt或progress中最后一轮指标代替。完成后progress里的epoch_in_progress可能保留结束回调的下一轮值，完成轮数以result.json/CSV为准。', '',
        '共同设置：MuSGD，lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、warmup3轮、nbs64；AMP训练、FP32完整验证、关闭TF32；seed20261109、deterministic=True、workers2、cache=False。关闭mosaic、几何scale/translate/旋转/透视、mixup和copy_paste，保留水平翻转.5及HSV .015/.7/.4；max_det300、nms=False。最多100轮、patience20，以conf=.25、完整val的原图5px球心F1选best和早停，平分保存最新权重。仍优化YOLO检测损失，未增加球心损失。', '',
        '固定val为1,272张/1,272球，test为830张/859球。合并训练含additional_v1的10,786张/14,042球，其中2,997张白背景变体继承已知v1 train原图；未向val/test添加历史数据。本批没有已确认无球负样本，缺标注图片没有当负样本。源数据保持不动，派生列表引用已有副本，不再次复制图片。', '',
        '## 测试协议与原图FP32定位结果', '',
        '所有模型用同一完整val/test。主阈值conf=.25，原图坐标距离≤5px一对一贪心最近匹配计算P/R/F1。误差中位数和P95仅对≤10px成功匹配计算，不包含未匹配目标；必须同时看未匹配球数。这里的“5px未匹配”包含没有预测及定位偏差过大的情况，“额外预测”也可能是定位不匹配，不等同于无球场景误检率。']
    table(lines, ['模型', 'test P', 'test R', 'test F1', '匹配 / 859', '5px未匹配球', '额外预测', '中位误差px', 'P95 px'],
        [(LABELS[i], f'{r["test_center5_precision"]:.2%}', f'{r["test_center5_recall"]:.2%}', f'{r["test_center5_f1"]:.2%}',
          r['test_center5_matches'], 859-r['test_center5_matches'], r['test_predictions']-r['test_center5_matches'],
          f'{r["test_error_p50_px"]:.3f}', f'{r["test_error_p95_px"]:.3f}') for i in range(4) for r in [row(i)]])
    lines += ['B相对A增加训练数据后，test匹配由828升至832、额外预测由57降至45，F1约提升0.90个百分点；C相对B多匹配7个球、额外预测少5个，F1高约0.70个百分点，但其已匹配球的中位/P95误差更大。不同模型成功匹配的球集合不同，因此单看误差分位数不能证明逐球定位优劣，也未做多随机种子显著性评估。', '',
        'A与旧原图推理基线相比有改善，但旧模型训练输入960×960、增强/随机种子和按框选模策略也不同，不能把差异单独归因于分辨率。旧960档F1为94.93%，与A原图94.95%相近。', '',
        '## 转换精度：同一模型权重，不额外训练', '',
        '各模型均测试两档输入：1280×1024不缩放，960×768等比缩至75%；无补边、无裁剪。每档包含PT FP32 b1/b2、ORT CPU FP32 b2、TRT FP32 b2、TRT FP16 b1/b2。ORT为CPU数值参考，不用于和GPU公平比速。FP32关闭TF32；FP16为混合精度，头部Sigmoid保留FP32。INT8尚未执行。', '',
        '下表为原图输入；位移和新增未匹配统计覆盖val+test共2,102张，比较同模型、同尺寸、同batch的PyTorch FP32参考。FP32本次b1/b2主指标一致。新增未匹配是逐目标比较，不能仅根据F1不变判断为0。']
    table(lines, ['模型', 'TRT FP16 batch', 'val P / R', 'test P / R', 'test F1', 'test中位 / P95 px', '位移P95 / max px', '新增5px未匹配 / 恢复'],
        [(LABELS[i], b, f'{r["val_center5_precision"]:.2%} / {r["val_center5_recall"]:.2%}',
          f'{r["test_center5_precision"]:.2%} / {r["test_center5_recall"]:.2%}',
          f'{r["test_center5_f1"]:.2%}', f'{r["test_error_p50_px"]:.3f} / {r["test_error_p95_px"]:.3f}',
          f'{c["prediction_center_shift_px_p50_p95_p99_max"][1]:.3f} / {c["prediction_center_shift_px_p50_p95_p99_max"][3]:.3f}',
          f'{c["new_gt_misses_at_5px"]} / {c["recovered_gt_at_5px"]}')
         for i in range(4) for b in [1, 2] for r in [row(i, 'tensorrt_fp16', batch=b)]
         for c in [matrices[i]['cases'][f'h1024_w1280_tensorrt_fp16_b{b}']['against_pytorch']]])
    lines += ['“位移P95 / max px”比较同一图片上PyTorch FP32与转换后端的预测球心，而“定位误差”比较预测球心与标注球心。位移为两预测坐标的欧氏距离；P95为95百分位，max为本次已匹配预测对中的最大值，单位均为原图像素。此处覆盖val+test，仅将两边conf≥.25、中心距离≤10px的一对一匹配预测对纳入位移统计，无法配对的输出另计，可能包含共同误检的预测。位移小不能单独证明定位正确。', '',
        '例如B原图FP16 batch2相对FP32的2,151对匹配预测，位移中位数0.075px、P95为0.194px、P99为0.255px、max为0.458px；但它相对标注的test定位误差中位数为1.205px、P95为2.485px，两者不是同一个统计。这里的max不保证所有未来画面或未匹配预测都有这个上限。', '',
        'B/C原图FP16 b2本次没有新增5px未匹配，测试P/R/F1保持一致；B验证F1从98.82%降至98.74%，原因是额外预测增加，不能描述为“输出完全相同”。位移P95约0.2px不表示所有球误差≤0.2px，也不表示置信度完全稳定。完整置信度差、未匹配预测和改变目标的路径保存在对应矩阵against_pytorch。', '',
        '## 原图GPU推理速度（RTX3090，batch2）', '',
        '最终复测使用独立非默认CUDA stream，预热200次、普通及CUDA Graph各测1000次。批次毫秒为CUDA event均值；图/秒为包含调用与同步开销的墙钟吞吐。输入已在显存，不含图片解码、预处理、H2D/D2H、后处理、相机、网络、双目匹配和三维计算。Graph是额外执行优化，需部署实现。离线batch2不保证同步左右目。']
    table(lines, ['模型', '后端', '普通批次ms', '普通图/s', 'Graph批次ms', 'Graph图/s'],
        [(LABELS[i], backend, f'{r["ordinary_batch_ms"]:.3f}', f'{r["ordinary_images_per_second"]:.1f}',
          f'{r["cuda_graph_batch_ms"]:.3f}', f'{r["cuda_graph_images_per_second"]:.1f}')
         for i in range(4) for backend in ['pytorch_fp32', 'tensorrt_fp32', 'tensorrt_fp16'] for r in [row(i, backend)]])
    b, c = row(2, 'tensorrt_fp16'), row(3, 'tensorrt_fp16')
    lines += [f'B从PT FP32到TRT FP16普通batch2耗时降低约{100*(1-b["ordinary_batch_ms"]/row(2)["ordinary_batch_ms"]):.1f}%，同口径约{row(2)["ordinary_batch_ms"]/b["ordinary_batch_ms"]:.2f}倍；C的TRT FP16批次耗时是B的{c["ordinary_batch_ms"]/b["ordinary_batch_ms"]:.2f}倍。以上全部为3090实测，不能按比例外推2060 Super。目标120对/秒=240图/秒，平均每对到达间隔8.33ms；目前没有目标卡端到端达标证据。', '',
        '## 降分辨率是否值得？']
    table(lines, ['原图训练模型', '原图test F1', '960×768 test F1', '原图TRT FP16 ms', '960 TRT FP16 ms'],
        [(LABELS[i], f'{row(i)["test_center5_f1"]:.2%}', f'{row(i, width=960)["test_center5_f1"]:.2%}',
          f'{row(i,"tensorrt_fp16")["ordinary_batch_ms"]:.3f}', f'{row(i,"tensorrt_fp16",width=960)["ordinary_batch_ms"]:.3f}') for i in range(1, 4)])
    lines += ['960×768是缩小全图，不是ROI裁切。原图训练模型缩小推理后的检出损失明显，现阶段优先保留1280×1024。此轮没有测试ROI；不能由缩放结果推断裁切一定更快或更准。', '',
        '## 历史指标存档与评估边界', '',
        '<details>', '<summary>历史框mAP记录（不作为本项目部署排名依据）</summary>']
    table(lines, ['模型', 'test框mAP50', 'test框mAP50:95'],
        [(LABELS[i], f'{row(i)["test_box_mAP50"]:.3%}', f'{row(i)["test_box_mAP50_95"]:.3%}') for i in range(4)])
    lines += ['框mAP与球心P/R衡量不同的目标，保留原始数值供追溯。旧数据已发现标注框和预测框尺度不一致，原有标注未改写；新增数据也未做全面人工质量审查。部署比较以球心P/R为主。', '', '</details>', '',
        '二维像素误差不是三维精度。', '',
        '新增历史数据做了精确内容去重、已知变体父图split核对，但未知历史录制与保留集的场地/序列独立性未被证明。旧test已经多次用于报告，后续应补充新录制且独立保留的测试集。每配置仅一次训练，不能据此宣称微小差异具有统计显著性。反光眼睛/手表/白色物体仍需要有球及已确认无球视频评估；没有负样本专项结果。', '',
        '## 权重、转换文件及复现入口']
    table(lines, ['模型', '选中权重', 'SHA256'], [(t['label'], '`'+t['result']['best']+'`', '`'+t['result']['best_sha256']+'`') for t in training])
    lines += ['每个模型的配置和源预训练权重SHA保存在`reports/training/<run_id>/configuration.json`，实际参数在`runs/detect/<run_id>/args.yaml`。同名评估位于`reports/evaluation/<run_id>_conversion_v1/`。导出产物位于`artifacts/<run_id>_conversion_v1/h1024_w1280/`，包含weights.onnx及trt_fp16_b2.engine等。', '',
        'ONNX为opset17、dynamic=True、simplify=True、免NMS；TensorRT每个engine的min=opt=max固定batch/高宽，workspace2GiB、优化级别3。ONNX支持动态维度不代表这些固定engine可以任意改输入；原生engine无Ultralytics文件头。应把PT/ONNX及构建参数迁至Ubuntu，核对Turing兼容版本并在2060 Super重建engine，不直接沿用3090/Windows engine。', '',
        '训练环境：Windows RTX3090 24GiB，Python3.12.3，torch2.11.0+cu128/CUDA12.8，Ultralytics8.4.207，TensorRT10.16.1.11，ONNX1.23.2，ORT1.31.1 CPU参考。精度评估预处理为OpenCV线性缩放（原图档跳过）、BGR转RGB、CHW、FP32/255，输出还原原图坐标。', '',
        '仅刷新/核验现有结果及图表，不重新训练或推理：', '',
        '```powershell', r'& .\.venv\Scripts\python.exe scripts/finalize_native_suite.py', '```', '',
        '## 训练曲线与精度/速度对照', '',
        '曲线来自逐轮完整val及训练CSV；右下角仅比较原图TRT FP16 batch2，横轴为纯GPU批次耗时。红色参考线8.33ms表示120对/秒的平均到达间隔，不是本项目已经验证的端到端延迟。', '',
        '![训练与原图评估对照](../../reports/experiments/native_resolution_suite_v1/native_results.png)', '',
        '[可导出SVG图](../../reports/experiments/native_resolution_suite_v1/native_results.svg)']
    DOC.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    plot(training, row)
    from build_report_catalog import main as catalog
    catalog()
    print(json.dumps({'passed': True, 'cases': verified_cases, 'gpu_cases': gpu_cases, 'engines': engines,
                      'completed_beijing': local_time(queue['updated_utc']), 'report': str(DOC.relative_to(ROOT))}, ensure_ascii=False))


def plot(training, row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    names = ['A: n / v1', 'B: n / merged', 'C: s / merged']
    colors = ['#168a83', '#e08025', '#7262b4']
    for item, name, color in zip(training, names, colors):
        epochs = [json.loads(line) for line in (ROOT / f'reports/training/{item["id"]}/epoch_metrics.jsonl').read_text().splitlines()]
        x = [e['epoch'] for e in epochs]
        axes[0, 0].plot(x, [100*e['center']['center5_f1'] for e in epochs], label=name, color=color)
        selected = item['result']['selected_epoch']
        axes[0, 0].scatter([selected], [100*item['result']['selected_validation']['center5_f1']], color=color, marker='*', s=110)
        axes[0, 1].plot(x, [e['center']['matched_error_px_p50_p95_p99'][1] for e in epochs], color=color, label=name)
        with (ROOT / f'runs/detect/{item["id"]}/results.csv').open(encoding='utf-8') as stream:
            history = list(csv.DictReader(stream))
        axes[1, 0].plot(x, [float(e['train/box_loss']) for e in history], color=color, label=name)
    for i, label in enumerate(['Old n / v1 / 960 train'] + names):
        r = row(i, 'tensorrt_fp16')
        axes[1, 1].scatter(r['ordinary_batch_ms'], r['test_center5_f1']*100, s=80, color=(['#64748b']+colors)[i])
        axes[1, 1].annotate(label, (r['ordinary_batch_ms'], r['test_center5_f1']*100), xytext=(7, 5), textcoords='offset points', fontsize=8)
    axes[1, 1].axvline(8.333, color='#b44646', linestyle='--', label='120 pairs/s arrival interval')
    axes[1, 1].set_xlim(2.7, 9.5)
    specs = [('Full-val center F1 (5 px, conf=.25)', 'Epoch', 'F1 (%)'),
             ('Full-val matched-center P95 (10 px matches)', 'Epoch', 'Original-image pixels'),
             ('Training box loss', 'Epoch', 'Box loss'),
             ('Native 1280x1024 / TRT FP16 / batch 2 / RTX3090', 'GPU batch time (ms), excludes transfers', 'Test center F1 (%)')]
    for ax, (title, xlabel, ylabel) in zip(axes.flat, specs):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(OUT / 'native_results.png', dpi=160)
    fig.savefig(OUT / 'native_results.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
