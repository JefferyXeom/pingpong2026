"""Track versioned follow-up training and native center metrics without new inference."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from train_native_experiment import atomic

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',required=True)
    args=parser.parse_args()
    if Path(args.suite).name!=args.suite or not args.suite.replace('_','').isalnum():
        raise ValueError('Invalid suite ID')
    suite=read(ROOT/'configs'/f'{args.suite}.json')
    preparation=read(ROOT/suite['data_preparation'])
    destination=ROOT/'reports/experiments'/args.suite
    destination.mkdir(parents=True,exist_ok=True)
    records=[];rows=[]
    for run in suite['runs']:
        directory=ROOT/'reports/training'/run['id']
        result=read(directory/'result.json') if (directory/'result.json').exists() else None
        progress=read(directory/'progress.json') if (directory/'progress.json').exists() else None
        evaluation=run['id']+'_conversion_v1'
        report=ROOT/'reports/evaluation'/evaluation/f'{evaluation}.json'
        record={'id':run['id'],'architecture':run['architecture'],'dataset':run['dataset'],
                'state':'training_complete' if result else (progress['state'] if progress else 'queued'),
                'result':result,'progress':progress,'evaluation':'queued'}
        if result:
            package=ROOT/'models'/('pingpong_'+run['architecture'].lower()+'_merged_v2_baiduout')
            package.mkdir(parents=True,exist_ok=True)
            card={'model_id':package.name,'run_id':run['id'],'architecture':run['architecture'],'classes':{'0':'ball'},
                'dataset_version':'pingpong_merged_v2','increment_version':'pingpong_baidu_out_v1',
                'training_input_width_height':[1280,1024],'preferred_inference_width_height':[1280,1024],
                'weights':result['best'],'weights_sha256':result['best_sha256'],'selected_epoch':result['selected_epoch'],
                'epochs_completed':result['epochs_completed'],'selection':result['selection'],
                'evaluation_report':report.relative_to(ROOT).as_posix(),'deployment_status':'not validated on target2060Super'}
            atomic(package/'model_card.json',card)
            (package/'README.zh-CN.md').write_text('# '+package.name+'\n\n模型来自'+run['id']+'，数据版本pingpong_merged_v2，增量pingpong_baidu_out_v1。实际训练输入1280×1024，球心F1选中第'+str(result['selected_epoch'])+'轮。权重通过model_card.json引用原始best.pt，不重复复制。完整数据量、参数、评估状态见docs/experiments/'+args.suite+'.zh-CN.md；目标2060 Super尚未验证。\n',encoding='utf-8')
        if report.exists():
            matrix=read(report)
            record['evaluation']=matrix.get('state','running')
            for key,case in matrix['cases'].items():
                for split in ['val','test']:
                    m=case['splits'][split]['thresholds']['0.25']
                    ref_key=f'h{case["input_height_width"][0]}_w{case["input_height_width"][1]}_pytorch_fp32_b{case["batch"]}'
                    ref=matrix['cases'][ref_key]['splits'][split]['thresholds']['0.25']
                    errors=m['matched_error_px_p50_p95_p99'] or [None]*3
                    rows.append({'run':run['id'],'selected_epoch':result['selected_epoch'],'weights_sha256':matrix['weights_sha256'],
                        'case':key,'backend':case['backend'],'batch':case['batch'],'input_width':case['input_height_width'][1],
                        'input_height':case['input_height_width'][0],'split':split,'conf':.25,'match_radius_px':5,
                        'gt':m['gt'],'tp':m['center5_matches'],'fp':m['predictions']-m['center5_matches'],
                        'fn':m['gt']-m['center5_matches'],'precision':m['center5_precision'],'recall':m['center5_recall'],'f1':m['center5_f1'],
                        'error_p50_px':errors[0],'error_p95_px':errors[1],'error_p99_px':errors[2],
                        'precision_change_pp':100*(m['center5_precision']-ref['center5_precision']),
                        'recall_change_pp':100*(m['center5_recall']-ref['center5_recall']),
                        'ordinary_batch_ms':case['benchmark'].get('ordinary',{}).get('cuda_event_batch_ms_mean'),
                        'gpu_timing_final':case.get('timing_repeated_after_all_builds',False),
                        'new_misses_both_splits':case.get('against_pytorch',{}).get('new_gt_misses_at_5px')})
        records.append(record)
    if not (destination/'progress.json').exists():
        atomic(destination/'progress.json',{'suite_id':args.suite,'state':'queued','stage':'not_launched'})
    queue=read(destination/'progress.json')
    payload={'suite_id':args.suite,'updated_utc':datetime.now(timezone.utc).isoformat(),
             'queue':queue,'data_preparation':suite['data_preparation'],'runs':records,'center_metric_rows':rows}
    atomic(destination/'summary.json',payload)
    if rows:
        with (destination/'center_results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Baidu增量数据与原图训练：pingpong_merged_v2','',
        '数据本身的框选质量可查看[Baidu标注质量可视化](../../reports/datasets/pingpong_baidu_out_v1/label_quality_review.zh-CN.md)：96张样本、原图框/球心、无损局部放大及12页拼图；没有模型预测，未修改标注。','',
        '来源审计日期2026-11-13。本页由scripts/summarize_followup_suite.py依据实际产物更新，未完成结果不提前宣称完成。', '',
        '增量来源为E:/BaiduNetdiskDownload/pingpongdata----out。按文件SHA256及解码RGB像素SHA256比对所有v1/additional图片；有效新图独立复制到datasets/pingpong_baidu_out_v1。已用图不重复加入，有标注冲突、缺标注、疑似保留集派生图均排除。源目录和此前训练/评估不修改。', '',
        '## 数据身份与数量','',
        '| 项目 | 图片 | 球框 |','| --- | --- | --- |',
        f'| 原始发现目录 | {preparation["source_audit"]["images"]:,} | 以逐图审计为准 |',
        f'| 本次有效唯一新增 | {preparation["increment"]["images"]:,} | {preparation["increment"]["objects"]:,} |']
    counts=preparation['datasets']['pingpong_merged_v2']
    for split in ['train','val','test']:
        lines.append(f'| 合并v2 / {split} | {counts[split]["images"]:,} | {counts[split]["objects"]:,} |')
    lines+=['','原有验证集20240527_3（1272图/1272球）与测试集20240527_2（830图/859球）保持不变；新图只加train。已知文件名父图候选均在train，但历史录制/背景派生关系和跨场地独立性不能仅由哈希证明；源图中的后缀变体不可当作全部独立采集。缺标签及空标注不自动当作无球负样本。', '',
        '[源目录重复/新增统计](../../reports/datasets/pingpong_baidu_out_v1/audit_summary.json) · [逐图决策](../../reports/datasets/pingpong_baidu_out_v1/audit_decisions.jsonl) · [数据清单与哈希](../../'+suite['data_preparation']+')','',
        '## 训练配置与模型命名','',
        '保持与此前原图实验相同的策略：YOLO26n后接YOLO26s顺序训练；对应官方预训练分别初始化，全量合并数据训练，并非恢复已结束B/C的优化器。这样可与旧B/C比较新增数据影响。实际输入宽1280×高1024，张量B×3×1024×1280，不缩放/补边/裁剪；batch8（末批按数据余数）、最多100轮，完整val conf=.25且原图5px球心F1选best，连续20轮无严格提升早停。训练AMP、验证FP32、关闭TF32。', '',
        'MuSGD lr0=.01、lrf=.01、momentum=.937、weight_decay=.0005、warmup3、nbs64；seed20261109与此前相同；workers2、cache=False、deterministic=True；mosaic/几何scale/translate/旋转/mixup/copy_paste关闭；fliplr=.5、HSV .015/.7/.4。save_period=-1，仅best/last；预留至少5GiB磁盘。完整实际设置和权重SHA见各configuration.json及args.yaml。', '',
        '| 训练ID | 架构 | 状态 | 完成epoch | 选中epoch | 转换评估 |','| --- | --- | --- | --- | --- | --- |']
    for r in records:
        lines.append(f'| {r["id"]} | {r["architecture"]} | {r["state"]} | {r["result"]["epochs_completed"] if r["result"] else "—"} | {r["result"]["selected_epoch"] if r["result"] else "—"} | {r["evaluation"]} |')
    lines+=['','每组训练完成后依次进行两尺寸/12项PT、ORT CPU FP32、TRT FP32/FP16完整评估和GPU最终复测，然后下一模型。P/R/F1与定位中位数/P95/P99均为重点；新增漏检和转换位移同样保留。INT8未纳入本次任务。3090速度不代表2060 Super达标。','',
        '## 已有原图batch2评估','',
        '| 模型 | 后端 | 集合 | 球心P | 球心R | F1 | 中位 / P95 / P99 px |','| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        if r['input_width']==1280 and r['batch']==2:
            lines.append(f'| {r["run"]} | {r["backend"]} | {r["split"]} | {r["precision"]:.2%} | {r["recall"]:.2%} | {r["f1"]:.2%} | {r["error_p50_px"]:.3f} / {r["error_p95_px"]:.3f} / {r["error_p99_px"]:.3f} |')
    previous_summary=read(ROOT/'reports/experiments/native_resolution_suite_v1/summary.json')
    lines+=['','## 相对上一版合并模型（原图batch2、test）','',
        '| 新模型 | 后端 | 旧版P / R / F1 | 新版P / R / F1 | 旧版P95 / 新版P95 px |',
        '| --- | --- | --- | --- | --- |']
    for r in rows:
        if r['input_width']!=1280 or r['batch']!=2 or r['split']!='test':
            continue
        size='n' if 'yolo26n' in r['run'] else 's'
        old=next(item for item in previous_summary['rows'] if item['training_run']==f'native_v1_additional_yolo26{size}_1280'
            and item['input_width']==1280 and item['batch']==2 and item['backend']==r['backend'])
        lines.append(f'| {r["run"]} | {r["backend"]} | {old["test_center5_precision"]:.2%} / {old["test_center5_recall"]:.2%} / {old["test_center5_f1"]:.2%} | '
            f'{r["precision"]:.2%} / {r["recall"]:.2%} / {r["f1"]:.2%} | {old["test_error_p95_px"]:.3f} / {r["error_p95_px"]:.3f} |')
    lines+=['','没有评估行即表示尚未执行，不用计划数值代替结果。置信度.25，5px一对一中心匹配算P/R/F1；误差分位数只含10px内匹配，必须同时看FN；框IoU不作为主判据。','',
        '## 查看状态','', '```powershell',
        f'& ./.venv/Scripts/python.exe scripts/check_native_suite_status.py --suite {args.suite}',
        f'& ./.venv/Scripts/python.exe scripts/summarize_followup_suite.py --suite {args.suite}',
        '# 队列停止且无独立训练进程时才恢复：',
        f'& ./.venv/Scripts/python.exe scripts/run_native_suite.py --suite {args.suite} --launch',
        '```','',
        f'[队列状态](../../reports/experiments/{args.suite}/progress.json) · [汇总JSON](../../reports/experiments/{args.suite}/summary.json)', '',
        '各训练进度reports/training/<run_id>/progress.json，30秒刷新；完成后result.json记录权重路径与SHA。模型卡在完成训练后生成于models/pingpong_yolo26n_merged_v2_baiduout及对应s目录，引用原始best.pt节省空间。']
    doc=ROOT/'docs/experiments'/f'{args.suite}.zh-CN.md'
    doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    registry=ROOT/'reports/experiment_registry.json'
    registered=read(registry)
    registered.setdefault('followup_suites',{})[args.suite]={'configuration':f'configs/{args.suite}.json',
        'data_preparation':suite['data_preparation'],'summary':str((destination/'summary.json').relative_to(ROOT)),
        'runs':[{k:r[k] for k in ['id','architecture','dataset','state','evaluation']} for r in records],
        'documentation':str(doc.relative_to(ROOT))}
    atomic(registry,registered)
    from build_report_catalog import main as catalog
    catalog()
    print(json.dumps({'suite':args.suite,'runs':[{'id':r['id'],'state':r['state']} for r in records],'completed_metric_rows':len(rows)}),flush=True)


if __name__=='__main__':
    main()
