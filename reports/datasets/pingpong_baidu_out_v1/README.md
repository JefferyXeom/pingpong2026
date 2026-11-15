# Baidu新增数据审计

批次：2026-11-13；源目录只读。

范围：E:/BaiduNetdiskDownload/pingpongdata----out；字节/解码像素去重及逐图标注核验。

本目录内的历史报告保留当时结论；最新状态以[总览](../../../docs/training_and_evaluation_overview.zh-CN.md)为准。

| 文件 | 用途 |
| --- | --- |
| [audit_decisions.jsonl](audit_decisions.jsonl) | Baidu逐图已用/新增/排除决策、旧集合匹配与文件名父图候选。 |
| [audit_summary.json](audit_summary.json) | Baidu目录与旧数据重复数量、有效唯一新增数量和来源限制。 |
| [existing_pixel_hashes.jsonl](existing_pixel_hashes.jsonl) | 此前v1和additional全部图片的解码RGB像素哈希，识别重新编码副本。 |
| [label_geometry_summary.json](label_geometry_summary.json) | 全源Baidu标注几何统计与抽样可视化验证；几何合法不代表人工框选准确。 |
| [label_quality_contact_01.jpg](label_quality_contact_01.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_02.jpg](label_quality_contact_02.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_03.jpg](label_quality_contact_03.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_04.jpg](label_quality_contact_04.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_05.jpg](label_quality_contact_05.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_06.jpg](label_quality_contact_06.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_07.jpg](label_quality_contact_07.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_08.jpg](label_quality_contact_08.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_09.jpg](label_quality_contact_09.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_10.jpg](label_quality_contact_10.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_11.jpg](label_quality_contact_11.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_contact_12.jpg](label_quality_contact_12.jpg) | Baidu标注质量抽样拼图：全图标注框及第一目标局部放大，序号对应交互页；不是模型预测。 |
| [label_quality_gallery.html](label_quality_gallery.html) | Baidu原始标注交互浏览：新增/已用筛选、标注框/球心开关、无损局部放大及待复查清单导出；图片内嵌可离线打开。 |
| [label_quality_review.zh-CN.md](label_quality_review.zh-CN.md) | Baidu框选质量可视化入口、抽样规则、坐标约定和逐页拼图导航。 |
| [label_quality_samples.json](label_quality_samples.json) | 框选质量抽样的逐图来源、哈希、原始XML和训练标注像素坐标；不含模型预测。 |
| [source_audit.jsonl](source_audit.jsonl) | Baidu逐图文件/像素哈希、实际尺寸、XML标注验证记录。 |
| [source_inventory.json](source_inventory.json) | 2026-11-13 Baidu源目录文件数、扩展名、体积；源文件只读。 |
| [source_preview.jpg](source_preview.jpg) | Baidu有效新图与文件名父图候选的带标注预览和球心局部放大。 |
| [source_preview.json](source_preview.json) | Baidu新增预览的图片来源、父图候选和文件名前缀数量。 |
