"""Render all available baseline/native model x input x backend combinations honestly."""
import csv
from datetime import datetime,timezone
import json
from pathlib import Path

from train_native_experiment import atomic

ROOT=Path(__file__).resolve().parents[1]
SUITE='native_resolution_suite_v1'


def main():
    suite=json.loads((ROOT/f'configs/{SUITE}.json').read_text(encoding='utf-8'))
    data=json.loads((ROOT/f'reports/datasets/{SUITE}/data_preparation.json').read_text(encoding='utf-8'))
    directory=ROOT/'reports/experiments'/SUITE
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    records=[]
    batches=[('baseline_yolo26n_960','center_conversion_matrix_v1',None)]+[(r['id'],r['id']+'_conversion_v1',r) for r in suite['runs']]
    for identifier,evaluation,run in batches:
        training=ROOT/'reports/training'/identifier
        result_path=training/'result.json'
        progress=training/'progress.json'
        result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
        state='historical_complete' if run is None else 'training_complete' if result else json.loads(progress.read_text()).get('state') if progress.exists() else 'queued'
        if progress.exists() and state=='starting' and json.loads(progress.read_text()).get('batches_seen_this_process',0)>0:
            state='training'
        records.append({'id':identifier,'run':run,'state':state,'training_result':result,
                        'training_progress':json.loads(progress.read_text()) if progress.exists() else None})
        path=ROOT/'reports/evaluation'/evaluation/f'{evaluation}.json'
        if not path.exists():
            continue
        report=json.loads(path.read_text(encoding='utf-8'))
        for key,case in report['cases'].items():
            comparison=case.get('against_pytorch',{})
            row={'training_run':identifier,'architecture':run['architecture'] if run else 'YOLO26n',
                 'training_dataset':run['dataset'] if run else 'pingpong_v1',
                 'training_height':1024 if run else 960,'training_width':1280 if run else 960,
                 'weights':report.get('source_weights',result['best'] if result else 'models/pingpong_yolo26n_center_v1/weights.pt'),
                 'weights_sha256':report['weights_sha256'],'selected_epoch':result['selected_epoch'] if result else 28 if run is None else None,
                 'case':key,'input_height':case['input_height_width'][0],'input_width':case['input_height_width'][1],
                 'backend':case['backend'],'batch':case['batch'],
                 'timing_final':bool(case.get('timing_repeated_after_all_builds')),
                 'new_gt_misses':comparison.get('new_gt_misses_at_5px'),
                 'backend_shift_p95':(comparison.get('prediction_center_shift_px_p50_p95_p99_max') or [None]*4)[1]}
            for split in ('val','test'):
                metrics=case['splits'][split]['thresholds']['0.25']
                for name in ('gt','predictions','center5_matches','center5_precision','center5_recall','center5_f1'):
                    row[split+'_'+name]=metrics[name]
                for name,value in zip(('p50','p95','p99'),metrics['matched_error_px_p50_p95_p99'] or [None]*3):
                    row[f'{split}_error_{name}_px']=value
                row[split+'_box_mAP50']=case['splits'][split]['box_mAP50']
                row[split+'_box_mAP50_95']=case['splits'][split]['box_mAP50_95']
            for mode in ('ordinary','cuda_graph'):
                benchmark=case['benchmark'].get(mode,{})
                row[mode+'_batch_ms']=benchmark.get('cuda_event_batch_ms_mean')
                row[mode+'_images_per_second']=benchmark.get('synchronized_loop_images_per_second')
            rows.append(row)
    generated=datetime.now(timezone.utc).isoformat()
    atomic(directory/'summary.json',{'suite_id':SUITE,'generated_utc':generated,'runs':records,'completed_matrix_cases':len(rows),'rows':rows})
    if rows:
        with (directory/'all_combinations.csv').open('w',encoding='utf-8-sig',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    lines=['# 原图训练三组对照：配置、进度及结果','',
        '本页由scripts/summarize_native_suite.py从实际进度/结果更新。未完成项保留为排队或训练中，不以计划配置冒充结果。当前队列状态见[progress.json](../../reports/experiments/native_resolution_suite_v1/progress.json)。',
        '', '本次汇总生成时间UTC：'+generated+'。阶段结束后队列自动更新本页，训练中的30秒进度以各训练progress.json为准。',
        '', '## 模型与数据','',
        '| 训练实验 | 模型 | train图片 / 球框 | val图片 / 球框 | test图片 / 球框 | 训练输入宽×高 | 状态 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for record in records[1:]:
        run=record['run'];counts=data['datasets'][run['dataset']]
        lines.append(f'| {run["id"]} | {run["architecture"]} | {counts["train"]["images"]:,} / {counts["train"]["objects"]:,} | 1,272 / 1,272 | 830 / 859 | 1280×1024 | {record["state"]} |')
    lines+=['', '三组均从对应型号的官方预训练权重独立初始化，不接着旧第28轮模型训练。第二、三组加入additional_v1全部10,786张，包括2,997张已知train原图的白背景变体；native指空间分辨率，不表示排除换背景图。原有v1 val/test固定不变，新增只加入train。精确哈希去重与已知原图split已核验；未知历史录制不能保证场地/序列独立。此轮旧test已多次用于报告，后续最终泛化结论应补充新保留录制。',
        '', '## 训练和选模设置','',
        '原始图片1280×1024；训练/验证张量固定高1024×宽1280，无缩放、无补边。imgsz=1280、rect=True、多尺度关闭；自定义训练器消除默认验证额外padding，并对每个实际batch断言尺寸。batch计划为8（若库因OOM自动调整，记录实际batch）。完整验证用FP32；训练AMP=True。',
        '', '最多100轮、patience=20；以完整val上conf=0.25、原图5像素球心F1保存best并早停，平分保存最新checkpoint。验证不使用test。优化训练仍为YOLO检测损失，没有新增自定义球心损失；球心指标用于选模和停止。框mAP另报，匹配误差分位数只涵盖10像素内检出，漏检同时报告。',
        '', '三组相同设置：MuSGD，lr0=.01，lrf=.01，momentum=.937，weight_decay=.0005，warmup3轮，nbs64；seed20261109、workers2、cache=False、AMP、确定性设置。关闭mosaic、随机几何scale/translate/旋转、mixup/copy_paste；保留fliplr=.5及HSV .015/.7/.4。详情见configs/native_resolution_suite_v1.json，各次实际args.yaml和configuration.json。',
        '', '与旧960×960基线相比，增强、种子和按球心早停也改变，因此不能将变化单独解释为分辨率的因果收益。三组新实验之间采用相同策略。D盘保留至少5GiB，训练低于5.5GiB/转换低于8GiB停止，不复制图片、不逐轮堆积checkpoint。',
        '', '## 推理组合','',
        '每个新选中模型都优先报告1280×1024原图推理，另用960×768作比较：PyTorch FP32 b1/b2、ONNX Runtime CPU FP32 b2、TensorRT FP32 b2、FP16 b1/b2，共12项/模型；加上旧基线12项，完整表计划48项。ORT CPU仅作数值参考，不用于宣称GPU提速。每项全量val/test；同尺寸/同batch比较转换球心位移、置信度与新增漏检。',
        '', 'ONNX是模型格式；ONNX Runtime是跨平台执行引擎；TensorRT针对NVIDIA GPU优化，也可读取ONNX。TensorRT同精度下通常有更好的GPU优化机会，但无普遍速度保证，ORT也能使用TensorRT provider。本项目以TensorRT原生engine测GPU，不能把ORT CPU对TRT GPU当作后端公平测速。',
        '', '不预设ROI裁切能满足需求。本轮使用全视野原图推理；ROI需要证明覆盖全部球运动、坐标偏移正确、丢失可重捕获，并测完整流程收益才考虑。对全图切多块再推理未必节省总计算。全部速度来自RTX3090，不能当作目标2060 Super实测。',
        '', '## 已有结果（batch2）','',
        '| 训练实验 | 推理宽×高 | 后端 | test球心P | test球心R | test球心F1 | test中位误差px | test P95 px | 普通批次ms | 最终复测 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        if row['batch']!=2:
            continue
        ms='CPU参考' if row['ordinary_batch_ms'] is None else f'{row["ordinary_batch_ms"]:.3f}'
        lines.append(f'| {row["training_run"]} | {row["input_width"]}×{row["input_height"]} | {row["backend"]} | {row["test_center5_precision"]:.2%} | {row["test_center5_recall"]:.2%} | {row["test_center5_f1"]:.2%} | {row["test_error_p50_px"]:.3f} | {row["test_error_p95_px"]:.3f} | {ms} | {row["timing_final"]} |')
    lines+=['', '[全部组合CSV](../../reports/experiments/native_resolution_suite_v1/all_combinations.csv) · [完整JSON](../../reports/experiments/native_resolution_suite_v1/summary.json)',
        '', '每个训练reports/training/<run_id>/有configuration.json、progress.json、epoch_metrics.jsonl和完成后的result.json；每个转换矩阵位于reports/evaluation/<run_id>_conversion_v1。队列日志logs/native_suite.stdout.log和stderr.log，子阶段日志带时间戳。已完成后可重新执行汇总脚本，不重新训练或推理。']
    if (ROOT/'docs/experiments/native_resolution_suite_v1_results.zh-CN.md').exists():
        lines[2:2]=['**三组已全部完成。训练耗时、选中epoch、转换损失、速度及候选建议见[最终结果分析](native_resolution_suite_v1_results.zh-CN.md)。**','']
    if len(rows)==48:
        from summarize_center_precision_recall import main as center_report
        center_report()
        lines[2:2]=['主要部署比较：[各版本球心precision / recall专表](native_center_precision_recall.zh-CN.md)，含同batch转换前后P/R变化、TP/FP/FN，IoU不作为主判据。','']
    lines += ['', '## 查看与恢复','',
        '```powershell',
        '# 只读：队列存活、轮数、最佳验证指标、实际张量与实测轮耗时',
        '& ./.venv/Scripts/python.exe scripts/check_native_suite_status.py',
        '# 从当前结果刷新文档/CSV及目录导航，不启动训练',
        '& ./.venv/Scripts/python.exe scripts/summarize_native_suite.py',
        '# 队列不存在时启动/恢复；若队列已在运行，脚本拒绝重复启动',
        '& ./.venv/Scripts/python.exe scripts/run_native_suite.py --launch',
        '```',
        '', '队列按每组训练→完整两尺寸转换评估→统一复测速率→汇总顺序执行，然后进入下一组；没有并发占用GPU训练和测速。未完成训练存在last.pt时续训，完成训练凭result.json跳过；转换矩阵按缓存续跑。异常时记录failed与错误并停止，不静默跳过某组。Windows重启后需重新执行队列启动命令。']
    out=ROOT/'docs/experiments'
    out.mkdir(exist_ok=True)
    (out/f'{SUITE}.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from build_report_catalog import main as update_catalog
    update_catalog()
    print(json.dumps({'runs':[{'id':r['id'],'state':r['state']} for r in records],'cases':len(rows)}),flush=True)


if __name__=='__main__':
    main()
