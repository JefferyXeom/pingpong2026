"""Verify and publish the Baidu follow-up suite using existing files only."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def digest(path):
    with (ROOT / path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def table(lines, headers, rows):
    lines.extend(['', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
    lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    lines.append('')


def percent(value):
    return f'{value:.2%}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='native_merged_v2_baiduout_20261113')
    args = parser.parse_args()
    if Path(args.suite).name != args.suite or not args.suite.replace('_', '').isalnum():
        raise ValueError('Invalid suite ID')
    suite_id = args.suite
    suite = read(f'configs/{suite_id}.json')
    out = Path('reports/experiments') / suite_id
    queue = read(out / 'progress.json')
    assert queue['state'] == 'complete'
    assert len(queue['completed']) == 4 * len(suite['runs'])
    # Refresh after the queue's final status write; the original summary preceded it.
    from summarize_followup_suite import main as refresh
    import sys
    prior_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__)), '--suite', suite_id]
        refresh()
    finally:
        sys.argv = prior_argv
    summary = read(out / 'summary.json')
    old = read('reports/experiments/native_resolution_suite_v1/summary.json')
    old_order = read('artifacts/center_conversion_matrix_v1/image_order.json')
    prep = read(suite['data_preparation'])
    import numpy as np
    training, rows, matrices, checked = [], [], {}, set()
    gpu_count = engines = caches = 0
    expected = {f'h{h}_w{w}_{backend}_b{b}' for h, w in [(768, 960), (1024, 1280)]
                for backend, b in [('pytorch_fp32', 1), ('pytorch_fp32', 2), ('onnx_cpu_fp32', 2),
                                   ('tensorrt_fp32', 2), ('tensorrt_fp16', 1), ('tensorrt_fp16', 2)]}
    for record in summary['runs']:
        run_id = record['id']
        directory = Path('reports/training') / run_id
        config = read(directory / 'configuration.json')
        result = record['result']
        progress = record['progress']
        with (ROOT / f'runs/detect/{run_id}/results.csv').open(encoding='utf-8', newline='') as stream:
            history = list(csv.DictReader(stream))
        epochs = [json.loads(line) for line in (ROOT / directory / 'epoch_metrics.jsonl').read_text().splitlines()]
        assert progress['state'] == 'complete' and progress['training_batch_actual'] == 8
        assert len(history) == len(epochs) == result['epochs_completed']
        best = [e for e in epochs if e['fitness'] == max(t['fitness'] for t in epochs)][-1]
        assert best['epoch'] == result['selected_epoch']
        assert result['epochs_completed'] - result['selected_epoch'] == 20
        assert all(e['center']['actual_shapes_height_width'] == [[1024, 1280]] for e in epochs)
        assert progress['observed_tensor_bchw'] == [1, 3, 1024, 1280]
        assert digest(result['best']) == result['best_sha256']
        assert digest(config['source_weights']) == config['source_sha256']
        checked.update([result['best'], config['source_weights']])
        for split in ['train', 'val', 'test']:
            data = config['data'][split]
            # Preparation hashes UTF-8 text before Windows newline translation.
            assert hashlib.sha256((ROOT / data['list']).read_text(encoding='utf-8').encode('utf-8')).hexdigest() == data['sha256']
            assert len((ROOT / data['list']).read_text().splitlines()) == data['images']
            if split != 'train':
                previous_data = old['runs'][1]['training_progress']['data'][split]
                assert data['sha256'] == previous_data['sha256']
                assert (ROOT / data['list']).read_bytes() == (ROOT / previous_data['list']).read_bytes()
        t = {'id': run_id, 'architecture': record['architecture'], 'configuration': config, 'result': result,
             'csv_training_hours': float(history[-1]['time']) / 3600,
             'wall_hours': (datetime.fromisoformat(result['completed_utc']) - datetime.fromisoformat(config['started_utc'])).total_seconds() / 3600,
             'batches_per_epoch': (config['data']['train']['images'] + 7) // 8,
             'last_batch': config['data']['train']['images'] % 8 or 8}
        training.append(t)
        experiment = run_id + '_conversion_v1'
        artifacts = Path('artifacts') / experiment
        matrix = read(f'reports/evaluation/{experiment}/{experiment}.json')
        matrices[run_id] = matrix
        assert matrix['state'] == 'complete' and set(matrix['cases']) == expected
        assert matrix['image_count'] == 2102
        assert read(artifacts / 'image_order.json') == old_order
        assert digest(artifacts / 'source.pt') == result['best_sha256'] == matrix['weights_sha256']
        checked.add(str(artifacts / 'source.pt'))
        for key, case in matrix['cases'].items():
            height, width = case['input_height_width']
            folder = artifacts / f'h{height}_w{width}'
            with np.load(ROOT / folder / f'{key}_predictions.npz', allow_pickle=False) as cache:
                assert set(cache.files) == {f'f{i}' for i in range(2102)}
                assert all(cache[name].ndim == 2 and cache[name].shape[1] == 6 for name in cache.files)
            caches += 1
            if case['backend'] != 'onnx_cpu_fp32':
                assert case['timing_repeated_after_all_builds']
                assert case['benchmark']['warmup_iterations'] == 200
                assert all(case['benchmark'][m]['iterations'] == 1000 for m in ['ordinary', 'cuda_graph'])
                gpu_count += 1
            if case['backend'].startswith('tensorrt'):
                precision = 16 if case['backend'].endswith('16') else 32
                prefix = f'trt_fp{precision}_b{case["batch"]}'
                build = read(folder / f'{prefix}.build.json')
                assert build['shape'] == [case['batch'], 3, height, width]
                for name, checksum in [(f'{prefix}.engine', build['engine_sha256']), ('weights.onnx', build['onnx_sha256'])]:
                    assert digest(folder / name) == checksum
                    checked.add(str(folder / name))
                engines += 1
            comparison = case.get('against_pytorch', {})
            shifts = comparison.get('prediction_center_shift_px_p50_p95_p99_max', [None] * 4)
            row = {'training_run': run_id, 'architecture': record['architecture'], 'training_dataset': record['dataset'],
                   'training_height': 1024, 'training_width': 1280, 'weights': result['best'],
                   'weights_sha256': matrix['weights_sha256'], 'selected_epoch': result['selected_epoch'],
                   'case': key, 'input_height': height, 'input_width': width, 'backend': case['backend'], 'batch': case['batch'],
                   'timing_final': case.get('timing_repeated_after_all_builds', False),
                   'new_gt_misses': comparison.get('new_gt_misses_at_5px'),
                   'recovered_gt': comparison.get('recovered_gt_at_5px'),
                   'backend_shift_p50': shifts[0], 'backend_shift_p95': shifts[1],
                   'backend_shift_p99': shifts[2], 'backend_shift_max': shifts[3],
                   'matched_prediction_pairs': comparison.get('matched_prediction_pairs'),
                   'unmatched_reference_predictions': comparison.get('unmatched_reference_predictions'),
                   'unmatched_candidate_predictions': comparison.get('unmatched_candidate_predictions')}
            for split, images, gt in [('val', 1272, 1272), ('test', 830, 859)]:
                stats = case['splits'][split]
                m = stats['thresholds']['0.25']
                assert stats['images'] == images and m['gt'] == gt
                for field in ['gt', 'predictions', 'center5_matches', 'center5_precision', 'center5_recall', 'center5_f1']:
                    row[f'{split}_{field}'] = m[field]
                row[f'{split}_fp'] = m['predictions'] - m['center5_matches']
                row[f'{split}_fn'] = m['gt'] - m['center5_matches']
                for p, value in zip([50, 95, 99], m['matched_error_px_p50_p95_p99']):
                    row[f'{split}_error_p{p}_px'] = value
                ref = matrix['cases'][f'h{height}_w{width}_pytorch_fp32_b{case["batch"]}']['splits'][split]['thresholds']['0.25']
                for metric in ['precision', 'recall', 'f1']:
                    row[f'{split}_{metric}_change_pp'] = 100 * (m[f'center5_{metric}'] - ref[f'center5_{metric}'])
            for mode in ['ordinary', 'cuda_graph']:
                b = case['benchmark'].get(mode, {})
                row[f'{mode}_batch_ms'] = b.get('cuda_event_batch_ms_mean')
                row[f'{mode}_batch_ms_p95'] = b.get('cuda_event_batch_ms_p50_p95_p99', [None] * 3)[1]
                row[f'{mode}_images_per_second'] = b.get('synchronized_loop_images_per_second')
            row['offline_pipeline_images_per_second'] = case['offline_pipeline']['images_per_second']
            rows.append(row)
    assert len(rows) == 24 and caches == 24 and gpu_count == 20 and engines == 12
    free = shutil.disk_usage(ROOT).free / 1024**3
    assert free >= 5
    verification = {'passed': True, 'checked_utc': datetime.now(timezone.utc).isoformat(), 'queue_complete': True,
                    'completed_stages': len(queue['completed']), 'matrix_cases': len(rows), 'prediction_caches': caches,
                    'images_per_case': 2102, 'same_holdout_image_order_and_hash_records_as_previous_suite': True,
                    'gpu_cases_with_200_warmup_1000_measurements_both_modes': gpu_count, 'engine_hashes_checked': engines,
                    'training_results_verified_against_csv_and_epoch_fitness': 2,
                    'hash_verified_paths': sorted(checked), 'free_disk_gib': free,
                    'scope': 'Existing metadata, hashes and cached array integrity only; no inference or training repeated; source-image bytes not rehashed.'}
    write(out / 'completion_verification.json', verification)
    completed = datetime.fromisoformat(queue['updated_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    write(out / 'final_analysis.json', {'suite_id': suite_id, 'completed_beijing': completed,
         'training': training, 'rows': rows, 'previous_suite': old['suite_id'], 'verification': verification})
    # Shared columns permit filtering all six trained weight choices in one table.
    combined = [{**r, 'suite_id': old['suite_id']} for r in old['rows']] + [{**r, 'suite_id': suite_id} for r in rows]
    for filename, data in [('all_combinations.csv', rows), ('all_models_combinations.csv', combined)]:
        fields = list(dict.fromkeys(k for r in data for k in r))
        with (ROOT / out / filename).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)

    def row(size, backend='tensorrt_fp16', width=1280, batch=2):
        return next(r for r in rows if r['architecture'] == f'YOLO26{size}' and r['backend'] == backend
                    and r['input_width'] == width and r['batch'] == batch)

    def previous(size, backend='tensorrt_fp16'):
        return next(r for r in old['rows'] if r['training_run'] == f'native_v1_additional_yolo26{size}_1280'
                    and r['backend'] == backend and r['input_width'] == 1280 and r['batch'] == 2)

    doc = Path('docs/experiments') / f'{suite_id}_results.zh-CN.md'
    lines = ['# Baidu新增数据：原图 n/s 训练与转换最终报告', '',
        '查看数据标注本身：[Baidu标注质量可视化](../../reports/datasets/pingpong_baidu_out_v1/label_quality_review.zh-CN.md)，含96张新增/已用样本的标注框、球心和无损局部图；未修改数据或训练结果。','',
        f'本轮全部完成于北京时间 **{completed}**（Asia/Shanghai，UTC+08:00）。两个模型均正常早停，训练、全量转换评估和最终GPU复测全部完成。8个队列阶段、24个推理组合、48行val/test球心结果；每组合评估2,102张图片。', '',
        '**新增数据提高了测试召回率，但增加了额外预测，测试F1未全面超过旧版。** 不能将新模型直接视为全面升级。本轮s的测试定位P95较旧s改善，n/s仍需与旧B/C一起保留，在独立新录制与目标机上评估。阈值仍固定.25；未用test重选权重或调阈值。', '',
        f'[本轮48行球心表](../../{out.as_posix()}/center_results.csv) · [24组合完整表](../../{out.as_posix()}/all_combinations.csv) · [含历史共72组合](../../{out.as_posix()}/all_models_combinations.csv) · [核验记录](../../{out.as_posix()}/completion_verification.json) · [精确结果JSON](../../{out.as_posix()}/final_analysis.json)', '',
        '## 数据版本、去重与保留集', '',
        '源目录 `E:/BaiduNetdiskDownload/pingpongdata----out` 共有18,086张PNG及同数XML，约24.92GiB。经文件SHA256与解码RGB像素SHA256比对：7,089张已在additional_v1 train中使用，标注一致；10,997张有效新图含13,912个球框，新备份命名为 `pingpong_baidu_out_v1`，图片约15.20GiB。桌面、D盘历史目录和E盘源文件保持不动，新合并列表引用已有备份图片，未复制整份合并数据。']
    table(lines, ['数据组成 / 集合', '图片数', '球框数', '用途'], [
        ['v1 train', '6,618', '8,139', '原始训练图'], ['additional_v1', '10,786', '14,042', '含2,997张白背景变体'],
        ['baidu_out_v1新增', '10,997', '13,912', '仅加入train'], ['merged_v2 train', '28,401', '36,093', 'n/s均完整使用'],
        ['固定val：20240527_3', '1,272', '1,272', '选模与早停；左1122 / 右150'],
        ['固定test：20240527_2', '830', '859', '训练后报告；左629 / 右201']])
    lines += ['val/test列表与上一批逐字节相同，转换评估图片顺序及图片哈希记录也一致；同一录制左右目留在同集合。新增图仅加入train。精确内容去重不能证明跨场地、跨录制或近似图独立；新目录的后缀变体不能全部当独立采集，未知历史派生关系仍待核实。旧test已多次报告，下一次验收应补充独立新录制。', '',
        f'数据审计：[摘要](../../reports/datasets/pingpong_baidu_out_v1/audit_summary.json)、[逐图决策](../../reports/datasets/pingpong_baidu_out_v1/audit_decisions.jsonl)、[实际列表及SHA](../../{suite["data_preparation"]})。本轮没有已确认无球负样本，缺标签图片没有当负样本使用。', '',
        '## 实际训练配置与停止条件', '',
        '单类别 `0: ball`，YOLO26n与YOLO26s分别从官方预训练初始化，使用完整merged_v2重新训练；没有接着旧B/C的优化器续训。**原图宽1280×高1024，实际训练/验证张量B×3×1024×1280，无缩放、补边、裁剪**；imgsz=1280、rect=True、multi_scale=0。训练batch=8，每轮3551批，末批1张。progress最后张量[1,3,1024,1280]是末批；preflight的[2,3,1024,1280]是两图检查样本，都不能当正式batch。']
    table(lines, ['模型 / 本轮简称', '运行ID', '完成 / 上限epoch', '球心best epoch', '选中时val F1', 'CSV训练时间 / 墙钟时间'], [
        [t['architecture'] + ' / ' + ('D' if i == 0 else 'E'), t['id'], f'{t["result"]["epochs_completed"]} / 100',
         t['result']['selected_epoch'], percent(t['result']['selected_validation']['center5_f1']),
         f'{t["csv_training_hours"]:.3f}h / {t["wall_hours"]:.3f}h'] for i, t in enumerate(training)])
    lines += [f'CSV训练/逐轮验证共 **{sum(t["csv_training_hours"] for t in training):.3f}小时**，不包含之后导出、TRT构建及测速。完整队列墙钟约{(datetime.fromisoformat(queue["updated_utc"])-datetime.fromisoformat(training[0]["configuration"]["started_utc"])).total_seconds()/3600:.3f}小时。两个模型在best之后连续20轮无严格球心F1提升而早停；最多100轮。使用完整val、conf=.25、原图5px球心F1选best，平分保留最新权重。评估使用best.pt，last.pt仅作续训/追溯。', '',
        'MuSGD：lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、warmup3、nbs64；训练AMP、验证FP32，关闭TF32；seed20261109、deterministic=True、workers2、cache=False。关闭mosaic、scale、translate、旋转、shear、perspective、mixup、copy_paste与上下翻转；保留fliplr=.5、HSV=.015/.7/.4；max_det300、nms=False、save_period=-1、plots=False、device0。仍优化YOLO检测损失，未新增球心损失。', '',
        '训练回调验证与独立转换矩阵分开保存：n选中时val F1为98.9416%，独立PT矩阵val为98.9020%（同一best权重、固定.25）；两条执行路径的边界输出有小差异，不能混用为转换退化。下文所有后端变化均以独立矩阵同尺寸/同batch PT结果为参考。s选模与矩阵F1本轮一致。', '',
        '## 评估协议', '',
        '每个模型12项：原图1280×1024和全图缩小960×768，各包含PT FP32 b1/b2、ONNX Runtime CPU FP32 b2、TRT FP32 b2、TRT FP16 b1/b2。没有裁切。conf=.25；在原图坐标以≤5px一对一贪心最近匹配计算P/R/F1，TP为成功匹配、FP为额外预测、FN为未匹配标注球。FP/FN可能包括定位偏差导致的匹配失败，并不等于纯无球误检数/模型完全没输出。', '',
        '定位误差是预测球心到GT标注球心的欧氏距离；中位/P95/P99仅统计≤10px成功匹配，需同时看TP/FP/FN。不同模型匹配集合可不同，不能仅靠分位数证明逐球都更准。二维像素误差不等于三维精度。框IoU/mAP保存在原始矩阵，本报告不按其排名。', '',
        '## 原图球心精度：所有batch2后端', '']
    for split, title in [('val', '验证集1272图 / 1272球'), ('test', '测试集830图 / 859球')]:
        lines += ['### ' + title]
        table(lines, ['模型', '后端', 'P', 'R', 'F1', 'TP / FP / FN', '中位 / P95 / P99 px'], [
            [r['architecture'], r['backend'], *[percent(r[f'{split}_center5_{m}']) for m in ['precision', 'recall', 'f1']],
             f'{r[f"{split}_center5_matches"]} / {r[f"{split}_fp"]} / {r[f"{split}_fn"]}',
             ' / '.join(f'{r[f"{split}_error_p{p}_px"]:.3f}' for p in [50, 95, 99])]
            for size in ['n', 's'] for backend in ['pytorch_fp32', 'onnx_cpu_fp32', 'tensorrt_fp32', 'tensorrt_fp16'] for r in [row(size, backend)]])
    lines += ['PT FP32、ORT CPU FP32、TRT FP32在本轮原图主指标一致，仍可能有微小坐标差。ORT使用CPU两线程作为数值参考，不能与GPU速度作公平比较。FP16是半精度/混合精度推理；本轮未做INT8校准、训练或评估，暂无INT8结果。', '',
        '## 转换位移、精度变化与新增漏检', '',
        '这里的位移比较同权重、同尺寸、同batch的PT FP32预测与转换预测，覆盖val+test 2102张，仅对两侧conf≥.25且预测中心≤10px的匹配预测对统计，也可能含共同误检；无法配对输出另计。P95为95百分位，max为本次已匹配预测对的最大位移，不是未来所有画面的误差上限。它们与相对GT的定位误差是两种统计。']
    table(lines, ['模型', 'FP16 batch', 'test ΔP / ΔR / ΔF1（百分点）', '位移中位 / P95 / P99 / max px', '新增5px漏匹配 / 恢复', '未配对PT / TRT输出'], [
        [r['architecture'], b, ' / '.join(f'{r[f"test_{m}_change_pp"]:+.4f}' for m in ['precision', 'recall', 'f1']),
         ' / '.join(f'{r[k]:.4f}' for k in ['backend_shift_p50', 'backend_shift_p95', 'backend_shift_p99', 'backend_shift_max']),
         f'{r["new_gt_misses"]} / {r["recovered_gt"]}', f'{r["unmatched_reference_predictions"]} / {r["unmatched_candidate_predictions"]}']
        for size in ['n', 's'] for b in [1, 2] for r in [row(size, batch=b)]])
    lines += ['新增漏匹配与恢复是逐GT对照，不能只从总召回率推断；完整置信度差、变化目标路径和两侧未配对输出保存在各矩阵against_pytorch中。FP16量化舍入会改变坐标和阈值附近置信度，指标略升或略降都不代表重新训练了模型。', '',
        '## 同协议与上一版合并模型比较', '',
        '旧B/C分别为上一版合并数据（v1＋additional_v1）的n/s，train17404图/22181球；本轮D/E增加10997图。共同官方预训练、seed和训练策略不变，但每轮优化步数、最佳epoch及样本分布都变化，不能单凭一次结果确定误检增加的原因。下表原图TRT FP16 batch2、test，阈值均.25。']
    table(lines, ['模型版本', 'P / R / F1', 'TP / FP / FN', '中位 / P95 / P99 px', '本机普通GPU批次ms'], [
        [f'{size} / {version}', ' / '.join(percent(r[f'test_center5_{m}']) for m in ['precision', 'recall', 'f1']),
         f'{r["test_center5_matches"]} / {r["test_predictions"]-r["test_center5_matches"]} / {859-r["test_center5_matches"]}',
         ' / '.join(f'{r[f"test_error_p{p}_px"]:.3f}' for p in [50, 95, 99]), f'{r["ordinary_batch_ms"]:.3f}']
        for size in ['n', 's'] for version, r in [('旧合并v1', previous(size)), ('新合并v2', row(size))]])
    for size in ['n', 's']:
        r, before = row(size), previous(size)
        lines += [f'- {size}：test P / R / F1变化为' + ' / '.join(f'{100*(r[f"test_center5_{m}"]-before[f"test_center5_{m}"]):+.3f}' for m in ['precision', 'recall', 'f1']) + '个百分点；新增匹配' + str(r['test_center5_matches']-before['test_center5_matches']) + '个球。']
    lines += ['', '本轮val F1提高而test F1下降，表明增量收益没有在现有test上全面体现；并非“数据越多必然越好”。旧n继续作为速度优先目标机候选保留，新s的定位尾部和召回值得进一步对照；没有依据直接替换全部旧模型。各配置仅单次训练，微小差异未做多seed显著性检验。', '',
        '## RTX3090最终纯GPU速度', '',
        '最终复测使用独立非默认CUDA stream；预热200次，普通执行和CUDA Graph分别1000次。批次ms为CUDA event均值，图/s为包含调用及同步开销的墙钟吞吐；表中P95是批次耗时P95。输入输出已在显存，排除解码、预处理、H2D/D2H、后处理、相机、网络、双目匹配和三维计算。Graph需部署程序实现。']
    table(lines, ['模型', '后端', 'batch', '普通均值 / P95 ms', '普通图/s', 'Graph均值 ms / 图/s'], [
        [r['architecture'], backend, b, f'{r["ordinary_batch_ms"]:.3f} / {r["ordinary_batch_ms_p95"]:.3f}',
         f'{r["ordinary_images_per_second"]:.1f}', f'{r["cuda_graph_batch_ms"]:.3f} / {r["cuda_graph_images_per_second"]:.1f}']
        for size in ['n', 's'] for backend, b in [('pytorch_fp32', 2), ('tensorrt_fp32', 2), ('tensorrt_fp16', 1), ('tensorrt_fp16', 2)] for r in [row(size, backend, batch=b)]])
    for size in ['n', 's']:
        r, pt = row(size), row(size, 'pytorch_fp32')
        lines += [f'- {size}：同batch2从PT FP32到TRT FP16普通GPU耗时降低{100*(1-r["ordinary_batch_ms"]/pt["ordinary_batch_ms"]):.1f}%，约{pt["ordinary_batch_ms"]/r["ordinary_batch_ms"]:.2f}倍；离线PNG解码流水线实测{r["offline_pipeline_images_per_second"]:.1f}图/s，后者不是已优化相机串流性能。']
    lines += ['', '目标120对/秒=240图/秒，每对平均间隔8.33ms；离线batch2并不保证图片来自同一时刻的左右目。上述3090 GPU结果不能证明Ubuntu/2060 Super端到端达标，也不能按硬件比例可靠外推。当前尚无目标机测试结果。', '',
        '## 降分辨率对照（TRT FP16 batch2）']
    table(lines, ['模型', '输入宽×高', 'test P / R / F1', '中位 / P95 / P99 px', '普通GPU ms / 图/s'], [
        [r['architecture'], f'{width}×{r["input_height"]}', ' / '.join(percent(r[f'test_center5_{m}']) for m in ['precision', 'recall', 'f1']),
         ' / '.join(f'{r[f"test_error_p{p}_px"]:.3f}' for p in [50, 95, 99]), f'{r["ordinary_batch_ms"]:.3f} / {r["ordinary_images_per_second"]:.1f}']
        for size in ['n', 's'] for width in [1280, 960] for r in [row(size, width=width)]])
    lines += ['960×768是全图等比缩到75%，没有ROI裁切；其原图坐标误差已还原。缩小推理降低计算时间，同时损失球心检出指标，仍优先原图1280×1024。本轮没有裁切收益实测，不能据此保证ROI一定适用。', '',
        '## 权重身份、转换配置与复现', '']
    table(lines, ['模型', '评估使用的best.pt', 'SHA256'], [[t['architecture'], f'[{t["result"]["best"]}](../../{t["result"]["best"]})', f'`{t["result"]["best_sha256"]}`'] for t in training])
    table(lines, ['模型', '官方预训练初始化', '源SHA256'], [[t['architecture'], t['configuration']['run']['pretrained'], f'`{t["configuration"]["source_sha256"]}`'] for t in training])
    lines += [f'完整固定配置：[suite JSON](../../configs/{suite_id}.json)。每模型实际参数保存在runs/detect/同名目录args.yaml，逐轮训练CSV在results.csv；配置及源哈希在reports/training/同名目录configuration.json，球心曲线在epoch_metrics.jsonl，选模与结束时间在result.json。模型包models/pingpong_yolo26n_merged_v2_baiduout和对应s目录仅引用原始best，不重复存储权重。', '',
        'ONNX opset17、dynamic=True、simplify=True、免NMS；TensorRT每个engine的min=opt=max固定[B,3,H,W]，workspace2GiB、builder优化级别3、关闭TF32；FP16 mixed，检测头相关Sigmoid保留FP32（准确节点清单见build.json）。每模型2份ONNX、6份engine。原生engine没有Ultralytics文件头，动态ONNX不代表现有固定engine可任意改尺寸或batch。', '',
        '转换文件位于artifacts/<run_id>_conversion_v1/h1024_w1280（以及h768_w960），包含weights.onnx、trt_fp16_b2.engine、对应.build.json及逐图NPZ缓存；source.pt在该实验artifacts根目录。部署应迁移PT/ONNX与构建参数，在Ubuntu/2060 Super核对Turing兼容软件并重建engine，不能直接沿用Windows/3090 engine。', '',
        '精度矩阵预处理：OpenCV INTER_LINEAR仅用于960档，原图档不缩放；BGR→RGB、CHW、FP32除255，无padding；结果还原到1280×1024坐标。训练环境Windows RTX3090 24GiB、Python3.12.3、torch2.11.0+cu128/CUDA12.8、Ultralytics8.4.207、TensorRT10.16.1.11、ONNX1.23.2、ORT1.31.1 CPU参考；版本原始记录在每个matrix的versions。', '',
        '实际队列顺序是每模型训练→完整矩阵→最终复测→汇总，再进入下个模型。下面列出已执行命令的模板用于学习与追溯，**本轮已完成，无需重新启动**：', '', '```powershell',
        f'& ./.venv/Scripts/python.exe scripts/run_native_suite.py --suite {suite_id} --launch',
        f'& ./.venv/Scripts/python.exe scripts/train_native_experiment.py --suite {suite_id} --run <run_id>',
        '& ./.venv/Scripts/python.exe scripts/evaluate_conversion_matrix.py --experiment <run_id>_conversion_v1 --weights runs/detect/<run_id>/weights/best.pt --resume',
        '& ./.venv/Scripts/python.exe scripts/rebenchmark_conversion_matrix.py --experiment <run_id>_conversion_v1', '```', '',
        '只刷新现有报告、核验缓存和生成曲线（不训练、不推理）：', '', '```powershell',
        f'& ./.venv/Scripts/python.exe scripts/finalize_followup_suite.py --suite {suite_id}', '```', '',
        f'本次核验通过：24份缓存完整、12份engine及4份ONNX哈希一致、20个GPU组合完成200预热/每模式1000次测试、2组训练轮数与选模一致、holdout列表与历史一致。核验时D盘空余{free:.2f}GiB，满足≥5GiB要求。未重新逐字节读取源目录图片；源数据审计以此前逐图记录为准。', '',
        '## 曲线与对照', '',
        f'![训练球心曲线与原图FP16对照](../../{out.as_posix()}/native_results.png)', '',
        f'[导出SVG](../../{out.as_posix()}/native_results.svg)', '',
        '## 下一阶段边界', '',
        '反光眼睛、手表及白色小物体专项仍缺已确认无球数据。应采集真实相机连续片段，人工确认无球后创建同名空YOLO标签文件；包含真球的困难画面需完整标注全部球，不能把漏标真球作为负样本。按录制/场景整体分train、val和新独立test，避免相邻帧及左右目泄漏；训练加入困难负样本，保留无球测试集统计每图/每分钟误检及有球片段召回。当前模型的FP不能替代这项专项结论。', '',
        '目前没有INT8、目标卡端到端、相机串流或三维定位的完成证据。下一步先在目标卡以原图、batch1/2、TRT FP16测真实流水线；若做INT8，校准只用train代表样本，再以固定协议比较球心P/R/F1、分位误差和新增漏检。']
    (ROOT / doc).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    plot(training, row, previous, out)
    registry = read('reports/experiment_registry.json')
    for t in training:
        identifier = t['id']; r = t['result']; c = t['configuration']
        model_id = identifier + '_best'
        upsert(registry['models'], {'id': model_id, 'architecture': t['architecture'], 'path': r['best'],
               'epoch': r['selected_epoch'], 'sha256': r['best_sha256'], 'purpose': 'merged_v2球心best；Baidu增量原图转换矩阵唯一源权重'})
        upsert(registry['training_runs'], {'id': identifier, 'architecture': t['architecture'],
               'initialization': c['run']['pretrained'], 'initialization_sha256': c['source_sha256'],
               'epochs_completed': r['epochs_completed'], 'selected_center_epoch': r['selected_epoch'], 'selected_model_id': model_id,
               'epochs_requested': 100, 'patience': 20, 'batch': 8, 'imgsz': 1280, 'input_shape_height_width': [1024, 1280],
               'rect': True, 'multi_scale': 0, 'amp': True, 'counts': c['data'], 'actual_args': f'runs/detect/{identifier}/args.yaml',
               'reports': f'reports/training/{identifier}', 'selection': r['selection']})
        experiment = identifier + '_conversion_v1'
        upsert(registry['evaluation_batches'], {'id': experiment, 'model_ids': [model_id], 'weights_sha256': r['best_sha256'],
               'val_images': 1272, 'test_images': 830, 'case_count': 12, 'state': 'complete',
               'reports': f'reports/evaluation/{experiment}', 'artifacts': f'artifacts/{experiment}'})
        package = Path('models') / ('pingpong_' + t['architecture'].lower() + '_merged_v2_baiduout')
        card = read(package / 'model_card.json')
        card.update({'dataset_counts': c['data'], 'training_configuration': f'reports/training/{identifier}/configuration.json',
                     'actual_args': f'runs/detect/{identifier}/args.yaml', 'final_report': doc.as_posix(), 'evaluation_state': 'complete',
                     'primary_native_batch2': [row(t['architecture'][-1], b) for b in ['pytorch_fp32', 'tensorrt_fp16']],
                     'int8_evaluated': False})
        write(package / 'model_card.json', card)
        (ROOT / package / 'README.zh-CN.md').write_text(f'# {package.name}\n\n来自`{identifier}`，merged_v2完整28401张train/36093球，原图1280×1024，batch8。训练{r["epochs_completed"]}轮，球心F1选中第{r["selected_epoch"]}轮；PT/ORT/TRT矩阵已完成。权重引用原best.pt，SHA见[模型卡](model_card.json)。\n\n[本轮最终报告](../../{doc.as_posix()})含全部配置、val/test P/R/F1、定位误差、转换位移及测速。没有INT8结果；目标2060 Super未验证。\n', encoding='utf-8')
    registry['as_of_date'] = datetime.now(timezone.utc).date().isoformat()
    registry['followup_suites'][suite_id]['final_report'] = doc.as_posix()
    registry['followup_suites'][suite_id]['state'] = 'complete'
    registry['datasets']['pingpong_baidu_out_v1'] = prep['increment']
    registry['datasets']['pingpong_merged_v2'] = {'split_counts_and_lists': prep['datasets']['pingpong_merged_v2'],
                                               'data_preparation': suite['data_preparation']}
    write('reports/experiment_registry.json', registry)
    from build_report_catalog import main as catalog
    catalog()
    print(json.dumps({'passed': True, 'cases': len(rows), 'combined_cases': len(combined), 'report': doc.as_posix(),
                      'free_disk_gib': free, 'completed': completed}, ensure_ascii=False))


def upsert(items, entry):
    for i, old in enumerate(items):
        if old['id'] == entry['id']:
            items[i] = entry
            return
    items.append(entry)


def plot(training, row, previous, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for t in training:
        epochs = [json.loads(line) for line in (ROOT / f'reports/training/{t["id"]}/epoch_metrics.jsonl').read_text().splitlines()]
        for ax, field in [(axes[0, 0], 'center5_f1'), (axes[0, 1], 'error')]:
            y = [e['center']['matched_error_px_p50_p95_p99'][1] if field == 'error' else 100 * e['fitness'] for e in epochs]
            ax.plot([e['epoch'] for e in epochs], y, label=t['architecture'])
            ax.axvline(t['result']['selected_epoch'], alpha=.3, linestyle='--')
            ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend()
    axes[0, 0].set_title('Full-val center F1 (5px, conf=.25)'); axes[0, 0].set_ylabel('F1 (%)')
    axes[0, 1].set_title('Full-val matched localization P95'); axes[0, 1].set_ylabel('Original-image pixels')
    for size in ['n', 's']:
        for version, r, marker in [('v1', previous(size), 'o'), ('v2', row(size), '^')]:
            label = f'{size} / merged {version}'
            axes[1, 0].scatter(100*r['test_center5_recall'], 100*r['test_center5_precision'], label=label, marker=marker, s=65)
            axes[1, 1].scatter(r['ordinary_batch_ms'], r['test_error_p95_px'], label=label, marker=marker, s=65)
    axes[1, 0].set(xlabel='Test recall (%)', ylabel='Test precision (%)', title='Native TRT FP16 b2: precision / recall')
    axes[1, 1].set(xlabel='RTX3090 ordinary batch ms', ylabel='Test localization P95 (px)', title='Native TRT FP16 b2: speed / localization')
    for ax in axes[1]:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Baidu follow-up / native 1280x1024 / existing cached results')
    for suffix in ['png', 'svg']:
        fig.savefig(ROOT / out / f'native_results.{suffix}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
