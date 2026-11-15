"""Build human and machine indexes for report batches; does not run experiments."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BATCHES={
    'environment/windows_rtx3090': ('训练机环境', '初始环境快照；不以文件移动时间推断安装时间。',
        'Windows / RTX3090，供所有现有实验参考；不是Ubuntu/2060 Super环境。'),
    'datasets/pingpong_v1': ('首轮原始数据整理 pingpong_v1', '数据版本v1',
        '桌面yolov5data；8,720张、10,270球框，实际用于smoke和baseline。'),
    'datasets/pingpong_additional_v1': ('新增历史数据备份', '2026-11-08整理记录',
        'D:/yolov5-master/VOCData及history_VOCOdata；10,786张、14,042球框；未参与旧960基线，新原图合并实验通过独立列表使用。'),
    'training/smoke_yolo26n_960': ('小规模训练流程验证', '运行ID smoke_yolo26n_960；独立1轮',
        'YOLO26n，192张train、64张val，batch8；只验证训练链路。'),
    'training/baseline_yolo26n_960': ('正式基线训练', '运行ID baseline_yolo26n_960；2026-11-08完成',
        'YOLO26n，6,618张train、1,272张val；从24轮恢复，28轮早停。result中的最终验证对应best.pt第3轮，不能当作last.pt第28轮指标。'),
    'evaluation/baseline_yolo26n_960/diagnostics_epoch24': ('第24轮中断阶段诊断', '2026-11-08；恢复训练之前',
        '使用保存的best第3轮、last第24轮快照；训练子集384张、验证子集256张；不使用test。这里的last.pt不是最终第28轮。'),
    'evaluation/baseline_yolo26n_960/center_selection_v1': ('训练结束后的完整球心评估与选模', '2026-11-08；完成28轮之后',
        '完整val1,272张对比best第3轮和last第28轮；选定last后，完整test830张只评估last。'),
    'evaluation/smoke_conversion_v1': ('小规模权重的转换流程验证', '早期流程批次；不是正式模型精度评估',
        '使用smoke第1轮best.pt；方形960×960，抽样8张，比较传统NMS与免NMS导出；原始转换报告在artifacts的两个smoke目录。'),
    'evaluation/center_conversion_matrix_v1': ('正式模型的尺寸与转换精度/速度评估', '2026-11-09完成；12项',
        '使用pingpong_yolo26n_center_v1，即baseline第28轮last.pt；全量val1,272张＋test830张；960×768与原图1280×1024，PyTorch/ONNX/TensorRT FP32/FP16。'),
}
PURPOSES={
    'label_quality_gallery.html':'Baidu原始标注交互浏览：新增/已用筛选、标注框/球心开关、无损局部放大及待复查清单导出；图片内嵌可离线打开。',
    'label_quality_review.zh-CN.md':'Baidu框选质量可视化入口、抽样规则、坐标约定和逐页拼图导航。',
    'label_quality_samples.json':'框选质量抽样的逐图来源、哈希、原始XML和训练标注像素坐标；不含模型预测。',
    'label_geometry_summary.json':'全源Baidu标注几何统计与抽样可视化验证；几何合法不代表人工框选准确。',
    'source_inventory.json':'2026-11-13 Baidu源目录文件数、扩展名、体积；源文件只读。',
    'source_audit.jsonl':'Baidu逐图文件/像素哈希、实际尺寸、XML标注验证记录。',
    'existing_pixel_hashes.jsonl':'此前v1和additional全部图片的解码RGB像素哈希，识别重新编码副本。',
    'audit_decisions.jsonl':'Baidu逐图已用/新增/排除决策、旧集合匹配与文件名父图候选。',
    'audit_summary.json':'Baidu目录与旧数据重复数量、有效唯一新增数量和来源限制。',
    'source_preview.jpg':'Baidu有效新图与文件名父图候选的带标注预览和球心局部放大。',
    'source_preview.json':'Baidu新增预览的图片来源、父图候选和文件名前缀数量。',
    'center_results.csv':'增量训练各模型的val/test球心P/R/F1、TP/FP/FN、中位/P95/P99和转换变化。',
    'center_precision_recall.csv':'球心专表：48个推理组合×val/test共96行，P/R、TP/FP/FN、同batch转换变化，不含IoU指标。',
    'center_precision_recall.json':'球心P/R专表的完整精确数值、配置、源权重哈希及尚无INT8结果的说明。',
    'completion_verification.json':'本目录批次完成核验：组合数、权重/engine/ONNX哈希、缓存、保留集顺序、训练选模和最终测速次数。',
    'final_analysis.json':'本目录批次实际训练耗时、选中权重、原图FP32/FP16结果及完成核验。',
    'native_results.png':'本目录批次训练曲线及原图TensorRT FP16精度/速度对照，RTX3090。',
    'native_results.svg':'训练和原图精度/速度对照的可导出矢量图。',
    'yolo26s_pretrained_download.json':'YOLO26s官方预训练权重下载地址、字节数及SHA256。',
    'data_preparation.json':'原图实验数据列表、图片/球框数量、哈希、变体和来源检查。',
    'configuration.json':'本次训练参数、初始化权重哈希、数据范围及原图输入尺寸。',
    'preflight.json':'正式优化前的实际训练张量、数据量及验证尺寸检查。',
    'epoch_metrics.jsonl':'逐轮完整val球心、框指标和误差，包含用于早停的fitness。',
    'result.json':'训练完成记录、最佳球心epoch、权重路径与SHA256；仅完成后生成。',
    'progress.json':'训练/评估/队列状态；实际批次张量及轮数用于判断进展。',
    'process.json':'队列进程ID与创建时间，用于防重复启动。',
    'summary.json':'旧基线及三组新实验全部可用组合汇总，保留未完成状态。',
    'all_combinations.csv':'训练模型×数据×推理尺寸×后端×精度×batch的已有结果。',
    'all_models_combinations.csv':'历史48项加本轮24项，共72项模型/尺寸/后端/batch组合；suite_id区分批次。',
    'queue.lock':'操作系统持有的互斥锁；文件存在不等于队列仍运行。',
    'environment_freeze.txt':'pip依赖版本冻结；平台、GPU及CUDA信息另见实验总览。',
    'dataset_inventory.json':'源目录文件数量、扩展名、容量和初始盘点。',
    'dataset_detail.json':'源数据详细目录、标签类别与原有Python环境探测。',
    'dataset_preparation.json':'独立副本准备记录、筛选决策和逐项来源。',
    'dataset_verification.json':'整理后图片、标签、清单和集合隔离校验。',
    'dataset_preview.jpg':'首轮数据集抽样画面预览。',
    'pairing_review.json':'原图与标注配对、缺失标注/图片复查。',
    'archive_inventory.json':'历史ZIP清单与缺失配对复查，未重复解压图片。',
    'additional_data_audit.json':'新增目录逐文件配对、哈希去重、标注一致性详单，体积较大。',
    'additional_data_summary.json':'新增源数据扫描与可补充图片数量汇总。',
    'additional_data_review.json':'新增数据分组、异常状态与抽样复查摘要。',
    'additional_data_provenance.json':'新增原图、重复图片、ZIP及换背景版本的来源关系。',
    'additional_dataset_verification.json':'新增独立备份的文件、哈希、标签校验。',
    'additional_data_preview.jpg':'新增数据抽样画面；不代表参与训练。',
    'smoke_yolo26n_960_result.json':'smoke一轮训练完成报告及框指标；不能作为最终检测效果。',
    'baseline_process.json':'最后一次启动的PID、命令及日志位置；历史记录不代表进程仍运行。',
    'baseline_launch_history.jsonl':'初次启动与断点恢复的追加历史。',
    'baseline_yolo26n_960_progress.json':'训练最终状态及最后一次回调；final_best_validation对应best。',
    'baseline_yolo26n_960_result.json':'训练正常结束报告；返回的是best.pt第3轮的最终框验证。',
    'baseline_interruption_review.json':'恢复前停在24轮的状态快照；后来用户确认手动中断，已恢复完成。',
    'baseline_training_analysis.json':'28轮训练过程、数据量、耗时、最佳/末轮指标分析。',
    'baseline_learning_curves.png':'正式基线损失与框指标曲线。',
    'baseline_diagnostics.json':'第3/24轮快照在384张train与256张val上的框指标及检测头对比。',
    'baseline_center_diagnostics.json':'同一第3/24轮诊断子集的球心和框尺度分析。',
    'baseline_prediction_crops.jpg':'第24轮诊断阶段的原始标签/预测局部对照。',
    'train_val_domain_review.json':'各录制批次的框尺度、球心位置和32张抽样亮度；未使用test。',
    'train_val_domain_preview.jpg':'训练与验证录制批次抽样对照。',
    'baseline_full_val_centers.json':'完整val上第3/28轮球心对比及按固定阈值选模结果。',
    'baseline_full_val_centers_best_frames.json':'完整val、第3轮best的逐帧计数与匹配误差。',
    'baseline_full_val_centers_last_frames.json':'完整val、第28轮last的逐帧计数与匹配误差。',
    'baseline_full_test_centers.json':'选模后仅第28轮last在完整test上的球心结果。',
    'baseline_full_test_centers_last_frames.json':'完整test、第28轮last逐帧计数与误差。',
    'trained_model_selection.json':'候选模型来源、SHA256、选择依据和初始完整球心评估。',
    'conversion_comparison.json':'两个smoke转换批次的预测空间一致性；不是正式模型P/R。',
    'center_conversion_matrix_v1.json':'正式模型12项完整精度、转换差异、初测/复测速率主报告。',
    'center_conversion_matrix_progress.json':'此转换矩阵最终完成状态；12项及最终复测均完成。',
    'center_conversion_matrix_summary.json':'转换矩阵的后端误差、同batch提速和原图变化摘要。',
    'center_conversion_matrix_summary.csv':'12项指标的可筛选表格，包含val/test球心及框AP。',
    'center_conversion_matrix_verification.json':'权重哈希、12项缓存、测速次数与原分辨率示例校验。',
    'center_conversion_matrix_comparison.png':'batch2球心F1、P95和RTX3090 Graph吞吐可视化。',
    'center_conversion_regressions.json':'FP16新增漏检逐图复查，含候选置信度和球心距离。',
    'conversion_original_image_crops.jpg':'固定均匀抽取8张的局部放大对照；不是评估输入裁剪。',
    'best.pt':'诊断用第3轮best快照，保留当时checkpoint状态；不是当前球心候选。',
    'last.pt':'诊断用第24轮last快照；不是训练结束的第28轮last。',
    'train_subset.txt':'诊断训练子集384张图片路径（每个训练批次128张）。',
    'val_subset.txt':'诊断验证子集256张图片路径。',
    'train_subset.yaml':'上述训练诊断子集的Ultralytics数据配置。',
    'val_subset.yaml':'上述验证诊断子集的Ultralytics数据配置。',
    'index.json':'8张均匀抽样原分辨率标注图的源图片、哈希和推理配置。',
}


def main():
    if (ROOT/'reports/datasets/pingpong_baidu_out_v1').exists():
        BATCHES['datasets/pingpong_baidu_out_v1']=('Baidu新增数据审计','2026-11-13；源目录只读','E:/BaiduNetdiskDownload/pingpongdata----out；字节/解码像素去重及逐图标注核验。')
    for followup_path in sorted((ROOT/'configs').glob('native_merged_v2_*.json')):
        followup=json.loads(followup_path.read_text(encoding='utf-8'))
        identifier=followup['suite_id']
        if (ROOT/'reports/datasets'/identifier).exists():
            BATCHES['datasets/'+identifier]=('合并v2数据列表','版本pingpong_merged_v2','v1原train＋additional_v1＋Baidu有效新图；原val/test固定。')
        if (ROOT/'reports/experiments'/identifier).exists():
            BATCHES['experiments/'+identifier]=('合并v2原图训练队列','2026-11-13独立批次','按n、s顺序训练/评估；不覆盖旧原图三组实验。')
        for run in followup['runs']:
            directory=ROOT/'reports/training'/run['id']
            if directory.exists():
                BATCHES['training/'+run['id']]=(run['architecture']+'合并v2原图训练',run['id'],
                    '数据pingpong_merged_v2，原图1280×1024、batch8、最多100轮、球心F1早停20轮。')
            evaluation=run['id']+'_conversion_v1'
            if (ROOT/'reports/evaluation'/evaluation).exists():
                BATCHES['evaluation/'+evaluation]=(run['architecture']+'合并v2转换评估',evaluation,
                    '对应球心best，全val1272/test830，两尺寸12项；P/R/F1和定位误差为重点。')
    config=ROOT/'configs/native_resolution_suite_v1.json'
    if config.exists():
        BATCHES['datasets/native_resolution_suite_v1']=('原图实验派生列表','2026-11-09；不复制图片','v1 train6618张，合并train17404张；val1272/test830固定；合并含2997背景变体。')
        BATCHES['experiments/native_resolution_suite_v1']=('原图实验队列与组合汇总','实时状态见progress.json','顺序训练n/v1、n/合并、s/合并；1280×1024，球心F1选模，原图推理优先。')
        for run in json.loads(config.read_text(encoding='utf-8'))['runs']:
            if (ROOT/'reports/training'/run['id']).exists():
                BATCHES['training/'+run['id']]=(run['architecture']+'原图训练','运行ID '+run['id'],'数据 '+run['dataset']+'；batch8、最多100轮、球心patience20；实际状态见进度。')
                result=ROOT/'reports/training'/run['id']/'result.json'
                if result.exists():
                    completed=json.loads(result.read_text(encoding='utf-8'))
                    BATCHES['training/'+run['id']]=(run['architecture']+'原图训练','已完成；运行ID '+run['id'],
                        f'数据 {run["dataset"]}；1280×1024、batch8；{completed["epochs_completed"]}轮早停，评估使用第{completed["selected_epoch"]}轮球心best。')
            evaluation=run['id']+'_conversion_v1'
            if (ROOT/'reports/evaluation'/evaluation).exists():
                BATCHES['evaluation/'+evaluation]=(run['architecture']+'原图模型转换评估',evaluation,'此训练球心best；全val1272/test830；两尺寸、6后端/batch组合每档。')
    catalog=[]
    index=['# 报告分类与批次导航','',
           '先看[训练与评估总览](../docs/training_and_evaluation_overview.zh-CN.md)，再按下表打开对应批次。`reports`根目录仅放导航、实验登记和整理校验；报告实体按用途及实验ID保存。',
           '', '| 目录 | 用途与批次 | 模型、数据及范围 |','| --- | --- | --- |']
    if (ROOT/'docs/experiments/native_resolution_suite_v1_results.zh-CN.md').exists():
        index[2:2]=['最新已完成：[原图三组训练与转换最终结果](../docs/experiments/native_resolution_suite_v1_results.zh-CN.md)，48项组合含旧基线。','']
    followup_final=ROOT/'docs/experiments/native_merged_v2_baiduout_20261113_results.zh-CN.md'
    if followup_final.exists():
        index[2:4]=['最新已完成：[Baidu新增数据原图n/s最终报告](../docs/experiments/native_merged_v2_baiduout_20261113_results.zh-CN.md)，本轮24项；[含历史72组合](experiments/native_merged_v2_baiduout_20261113/all_models_combinations.csv)。旧三组实验保留原批次。','']
    for directory,(title,stage,scope) in BATCHES.items():
        target=ROOT/'reports'/directory
        lines=[f'# {title}','','批次：'+stage+'。','','范围：'+scope,'',
               '本目录内的历史报告保留当时结论；最新状态以[总览]('+('../'*(len(Path(directory).parts)+1))+'docs/training_and_evaluation_overview.zh-CN.md)为准。',
               '', '| 文件 | 用途 |','| --- | --- |']
        for path in sorted(target.rglob('*')):
            if not path.is_file() or path.name=='README.md':
                continue
            purpose=PURPOSES.get(path.name)
            if purpose is None and path.name.startswith('label_quality_contact_') and path.suffix=='.jpg':
                purpose='Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。'
            if purpose is None and path.suffix=='.json' and path.stem.endswith('_conversion_v1'):
                purpose='此模型的完整尺寸/后端矩阵主报告、源权重哈希与全量指标。'
            if purpose is None and path.suffix.lower()=='.jpg' and 'conversion_original_examples' in path.parts:
                purpose='原图1280×1024标注示例，TensorRT FP16 batch2；源图片见同目录index.json。'
            if purpose is None:
                raise ValueError('Missing report description: '+str(path))
            relative=path.relative_to(target).as_posix()
            lines.append(f'| [{relative}]({relative}) | {purpose} |')
            catalog.append({'path':path.relative_to(ROOT).as_posix(),'batch':directory,
                            'purpose':purpose,'scope':scope,'bytes':path.stat().st_size})
        (target/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        index.append(f'| [{directory}]({directory}/README.md) | {title}；{stage} | {scope} |')
    index+=['','`runs/`保存Ultralytics原始args.yaml、results.csv、训练图和权重；`artifacts/`保存ONNX、TensorRT、预测缓存与构建参数；`logs/`保存执行日志。这里的报告通过路径引用这些产物，不重复复制。',
        '', '查找原则：先确定实验ID，再确定权重来源和数据范围。`baseline_diagnostics/last.pt`是第24轮快照；`runs/detect/baseline_yolo26n_960/weights/last.pt`及当前模型包是第28轮。',
        '', '新增训练使用`reports/training/<run_name>/`；新增评估使用`reports/evaluation/<experiment_id>/`。`evaluate_centers.py`提供`--report-dir`，新实验应显式选择新目录。每个新批次应记录源权重/SHA256、数据版本、split、图片数、输入尺寸、阈值、后端/精度、日期与限制，登记到实验总览。',
        '', '[机器可读文件目录](catalog.json) · [实验登记](experiment_registry.json) · [本次路径迁移记录](report_layout_migration.json) · [整理校验](report_layout_verification.json)']
    (ROOT/'reports/README.md').write_text('\n'.join(index)+'\n',encoding='utf-8')
    (ROOT/'reports/catalog.json').write_text(json.dumps({'files':catalog,'count':len(catalog)},indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'batches':len(BATCHES),'indexed_files':len(catalog)}))


if __name__=='__main__':
    main()
