"""Render the precision/speed matrix as CSV, Markdown and standalone figures."""
import csv
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    source=json.loads((ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_v1.json').read_text(encoding='utf-8'))
    rows=[]
    for key,case in source['cases'].items():
        benchmark=case['benchmark']
        row={'case':key,'height':case['input_height_width'][0],'width':case['input_height_width'][1],
             'backend':case['backend'],'batch':case['batch']}
        for split in ('val','test'):
            metrics=case['splits'][split]['thresholds']['0.25']
            for field in ('center5_precision','center5_recall','center5_f1'):
                row[f'{split}_{field}']=metrics[field]
            for i,quantile in enumerate(('p50','p95','p99')):
                row[f'{split}_center_error_{quantile}_px']=metrics['matched_error_px_p50_p95_p99'][i]
            row[f'{split}_box_map50']=case['splits'][split]['box_mAP50']
            row[f'{split}_box_map50_95']=case['splits'][split]['box_mAP50_95']
        for mode in ('ordinary','cuda_graph'):
            timing=benchmark.get(mode,{})
            row[f'{mode}_batch_ms']=timing.get('cuda_event_batch_ms_mean')
            row[f'{mode}_images_per_second']=timing.get('synchronized_loop_images_per_second')
        row['offline_pipeline_images_per_second']=case['offline_pipeline']['images_per_second']
        comparison=case.get('against_pytorch',{})
        shift=comparison.get('prediction_center_shift_px_p50_p95_p99_max')
        row['backend_center_shift_p95_px']=shift[1] if shift else None
        row['new_gt_misses_vs_pytorch']=comparison.get('new_gt_misses_at_5px')
        row['recovered_gt_vs_pytorch']=comparison.get('recovered_gt_at_5px')
        rows.append(row)
    table=ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_summary.csv'
    with table.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary={'state':source.get('state','running'),'gpu':source['gpu'],'completed_cases':len(rows),
             'precision':[], 'same_batch_speedups':[], 'native_size_comparison':[]}
    lines=['# 正式模型转换、原图精度与速度评估','',
           f'测试 GPU：{source["gpu"]}，Windows。当前完成 {len(rows)}/12 项。',
           '','验证集 1,272 张、测试集 830 张，固定权重及 conf=0.25。球心指标使用原图距离≤5像素；误差分位数针对10像素内成功匹配，漏检另报。',
           '', '## 完整测试集球心结果','',
           '| 输入宽×高 | 后端 | batch | 精确率 | 召回率 | F1 | 误差中位数 px | P95 px |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["width"]}×{row["height"]} | {row["backend"]} | {row["batch"]} | {row["test_center5_precision"]:.2%} | {row["test_center5_recall"]:.2%} | {row["test_center5_f1"]:.2%} | {row["test_center_error_p50_px"]:.3f} | {row["test_center_error_p95_px"]:.3f} |')
    small=source['cases'].get('h768_w960_pytorch_fp32_b2')
    native=source['cases'].get('h1024_w1280_pytorch_fp32_b2')
    if small and native:
        a=small['splits']['test']['thresholds']['0.25']
        b=native['splits']['test']['thresholds']['0.25']
        lines+=['',f'当前960训练权重直接切换到原图推理后，测试球心F1变化为{100*(b["center5_f1"]-a["center5_f1"]):+.3f}个百分点，误差P95变化为{b["matched_error_px_p50_p95_p99"][1]-a["matched_error_px_p50_p95_p99"][1]:+.3f}原图像素。输入像素数增加77.8%，本次没有观察到精度收益。这不代表原图尺寸训练或微调没有价值；本次没有进行此类训练。']
    lines+=['','## 本机纯 GPU 耗时','',
            '输入已在显存。全部引擎构建完成后复测：普通执行200次预热、1000次测量；Graph捕获前另行预热，捕获后10次回放预热、1000次测量。PyTorch和TensorRT均使用独立非默认stream，FP32禁用TF32。初次30/200结果保存在initial_benchmark。GPU event 平均批次耗时与同步调用的墙钟吞吐分开列出；后者包含调用和同步开销。CUDA Graph 结果单独展示，不与普通执行混称。',
            '', '| 输入宽×高 | 后端 | batch | 普通批次 ms | 普通同步吞吐 图/s | Graph批次 ms | Graph同步吞吐 图/s |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    def fmt(value):
        return '—' if value is None else f'{value:.3f}'
    for row in rows:
        if row['ordinary_batch_ms'] is None:
            continue
        lines.append(f'| {row["width"]}×{row["height"]} | {row["backend"]} | {row["batch"]} | {fmt(row["ordinary_batch_ms"])} | {fmt(row["ordinary_images_per_second"])} | {fmt(row["cuda_graph_batch_ms"])} | {fmt(row["cuda_graph_images_per_second"])} |')
    for shape in ((768,960),(1024,1280)):
        for batch in (1,2):
            base=source['cases'].get(f'h{shape[0]}_w{shape[1]}_pytorch_fp32_b{batch}')
            candidate=source['cases'].get(f'h{shape[0]}_w{shape[1]}_tensorrt_fp16_b{batch}')
            if not base or not candidate:
                continue
            speed={}
            for mode in ('ordinary','cuda_graph'):
                before=base['benchmark'][mode]['cuda_event_batch_ms_mean']
                after=candidate['benchmark'][mode]['cuda_event_batch_ms_mean']
                speed[mode]={'speedup':before/after,'time_reduction_fraction':1-after/before}
            summary['same_batch_speedups'].append({'input_height_width':shape,'batch':batch,**speed})
        for key,case in source['cases'].items():
            if tuple(case['input_height_width'])==shape and 'against_pytorch' in case:
                summary['precision'].append({'case':key,**case['against_pytorch']})
    for backend in ('pytorch_fp32','onnx_cpu_fp32','tensorrt_fp32','tensorrt_fp16'):
        base=source['cases'].get(f'h768_w960_{backend}_b2')
        native=source['cases'].get(f'h1024_w1280_{backend}_b2')
        if not base or not native:
            continue
        changes={}
        for split in ('val','test'):
            before=base['splits'][split]['thresholds']['0.25']
            after=native['splits'][split]['thresholds']['0.25']
            changes[split]={'center5_recall_change_pp':100*(after['center5_recall']-before['center5_recall']),
                'center5_f1_change_pp':100*(after['center5_f1']-before['center5_f1']),
                'median_error_change_px':after['matched_error_px_p50_p95_p99'][0]-before['matched_error_px_p50_p95_p99'][0],
                'p95_error_change_px':after['matched_error_px_p50_p95_p99'][1]-before['matched_error_px_p50_p95_p99'][1]}
        summary['native_size_comparison'].append({'backend':backend,**changes})
    lines+=['','## FP16转换的定位变化','',
        '以下在完整验证集和测试集合计2,102张上，与同尺寸、同batch的PyTorch FP32逐图比较。新增漏检按真值球、原图5像素匹配计算；即使总召回相同，也需检查新增漏检与恢复检出是否抵消。',
        '', '| 输入宽×高 | batch | 位移中位数 px | 位移P95 px | 最大位移 px | 新增漏检球 | 恢复检出球 | 未匹配参考/转换预测 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,case in source['cases'].items():
        if case['backend']!='tensorrt_fp16':
            continue
        change=case['against_pytorch']
        shift=change['prediction_center_shift_px_p50_p95_p99_max']
        lines.append(f'| {case["input_height_width"][1]}×{case["input_height_width"][0]} | {case["batch"]} | {shift[0]:.4f} | {shift[1]:.4f} | {shift[3]:.4f} | {change["new_gt_misses_at_5px"]} | {change["recovered_gt_at_5px"]} | {change["unmatched_reference_predictions"]}/{change["unmatched_candidate_predictions"]} |')
    regression_path=ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_regressions.json'
    if regression_path.exists():
        regressions=json.loads(regression_path.read_text(encoding='utf-8'))
        for change in regressions:
            for detail in change['closest_predictions']:
                if detail['gt_index'] not in change['lost_gt_indices']:
                    continue
                before=detail['pytorch'][0]
                after=detail['tensorrt_fp16'][0]
                lines+=['',f'新增漏检逐图复查：`{change["case"]}`，`{change["image"]}`。最近候选置信度由{before["confidence"]:.6f}变为{after["confidence"]:.6f}，对应球心误差由{before["center_distance_px"]:.3f}变为{after["center_distance_px"]:.3f}像素。本例候选仍在5像素内，但FP16置信度低于固定0.25阈值，导致过滤后漏检。未为消除此例而调整阈值；详细候选见reports/evaluation/center_conversion_matrix_v1/center_conversion_regressions.json。']
    lines+=['','## 标准框指标与漏检数','',
        '球心指标不能替代标准框mAP。当前原始标注框通常比预测球体框宽松，因此同时保留以下结果；评估没有调整原始标签。中心误差分位数只覆盖10像素内匹配成功的球。',
        '', '| 输入宽×高 | 后端（batch2） | 框mAP50 | 框mAP50:95 | 测试球总数 | 5像素内检出 | 10像素内检出 | 预测总数 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key,case in source['cases'].items():
        if case['batch']!=2:
            continue
        split=case['splits']['test']
        m=split['thresholds']['0.25']
        lines.append(f'| {case["input_height_width"][1]}×{case["input_height_width"][0]} | {case["backend"]} | {split["box_mAP50"]:.2%} | {split["box_mAP50_95"]:.2%} | {m["gt"]} | {m["center5_matches"]} | {m["center10_matches"]} | {m["predictions"]} |')
    lines+=['','## 结果的适用范围','',
        '全部原图档输入为1280×1024，没有空间缩放；960档为等比例960×768，不补边。图像归一化和坐标还原对所有后端一致。',
        '', 'ONNX使用CPU作为数值参考：本机ORT GPU包需要CUDA13，而PyTorch为CUDA12.8，未改变训练环境。CPU耗时不用于GPU加速比。',
        '', '离线PNG全流程速度包含解码、CPU预处理和传输，不包含相机、真实视频传输或双目关联；它不是优化后的串流吞吐上限。batch2未保证实际同步左右目，仅用于批处理对比。',
        '', '全部速度为RTX3090本机数据，不能据此认证2060 Super达到240张图/秒。原生TensorRT引擎需在目标Ubuntu/GPU重建。',
        '', '目前无已确认的无球数据，不能评估眼睛/手表专项误检。球心误差基于原标签，不能直接作为三维定位误差。',
        '', '均匀抽取验证集4张、测试集4张的原图局部对比见[图片](../reports/evaluation/center_conversion_matrix_v1/conversion_original_image_crops.jpg)：左列为完整画面缩略图，后三列为同一原图48×48区域放大4倍。绿色为原始标注框，紫色为预测框，黄色为预测球心；三列依次为PyTorch 960、PyTorch原图、TensorRT FP16原图。原图评估本身使用完整1280×1024图像，不使用这些展示裁剪作为网络输入。8张保持1280×1024的完整标注图保存在reports/evaluation/center_conversion_matrix_v1/conversion_original_examples，index.json记录源图、哈希和配置。',
        '', '完整数值、后端新增漏检、预测位移及缓存路径见JSON；便于筛选的表格见同名summary.csv。']
    (ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (ROOT/'docs/conversion_evaluation_results.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    selected=[r for r in rows if r['batch']==2]
    labels=[f'{r["width"]}x{r["height"]}\n{r["backend"].replace("tensorrt","TRT").replace("pytorch","PT").replace("onnx_cpu","ONNX CPU")}' for r in selected]
    x=np.arange(len(selected))
    axes[0].bar(x,[100*r['test_center5_f1'] for r in selected]);axes[0].set_ylabel('Test center F1 (%)')
    axes[1].bar(x,[r['test_center_error_p95_px'] for r in selected]);axes[1].set_ylabel('Matched center error P95 (original pixels)')
    axes[2].bar(x,[r['cuda_graph_images_per_second'] or 0 for r in selected]);axes[2].set_ylabel('RTX3090 graph synchronized loop (images/s)')
    for ax in axes:
        ax.set_xticks(x,labels,rotation=60,ha='right',fontsize=7);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Full held-out evaluation / batch2 / conf=0.25; ONNX CPU has no GPU speed bar')
    fig.tight_layout()
    fig.savefig(ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_comparison.png',dpi=160)
    plt.close(fig)
    print(json.dumps({'state':summary['state'],'completed_cases':len(rows),'speedups':summary['same_batch_speedups'],
                      'native_size_comparison':summary['native_size_comparison']},indent=2),flush=True)


if __name__=='__main__':
    main()
