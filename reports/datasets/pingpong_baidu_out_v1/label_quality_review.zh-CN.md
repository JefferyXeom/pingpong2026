# Baidu数据标注质量可视化

**交互浏览入口：[label_quality_gallery.html](label_quality_gallery.html)。下载该单个HTML后用浏览器打开即可，全部样本图已嵌入，无需访问训练机E盘。VS Code可先打开下方JPG快速预览。**

本次抽取96张：新增48张、此前已用48张。随机种子20261115。每组32张按文件名前缀分层随机抽取，再补充小框、大框、长条框、多球/边缘目标；重叠样本只保留一次。

绿色为实际训练标注框，红色为标注框几何中心；不是模型预测。浏览页可搜索文件名、筛选新增/已用、选择同图不同球、切换框/中心显示，并查看原始XML坐标、训练坐标和源路径。无损局部PNG直接从备份原PNG截取，整数倍最近邻放大，适合检查模糊、框松紧和中心偏差；全图JPEG用于观察场景，JPG拼图也仅是快速预览。

训练坐标沿用项目历史VOC约定：center=(min+max)/2−1，宽高=max−min。浏览页同时列出原始XML值；不要将两种坐标显示差异误认为新改了标签。所有样本图已核对内容哈希及XML→训练坐标；未修改原图、标注、split、模型或训练结果。

可在交互页标记“待复查”并导出JSON，该操作仅导出复查清单，不修改标签。重点检查：球是否漏标/多标、框是否覆盖可见球、模糊拖影是否采用一致标注标准、框几何中心是否与期望球心一致。几何合法不代表人工框选正确，本抽样含针对性极值，不能估计全体错误率。

初步目视提示：在已查看的第1、5、6页中，可见多处模糊球以及手持/靠近球拍的球；一些框与可见亮斑接近，但模糊轮廓下的几何中心和期望物理球心仍需人工判断。应在交互页关闭框/球心后比较原始像素，逐目标检查，不能由抽样推断全体准确率或认定本轮F1变化由标注造成。新增标注球框宽/高的中位数均为13px；本项目关心像素级定位，少量像素的中心偏移也值得复查。

[全源标注几何统计](label_geometry_summary.json) · [样本来源与全部框坐标](label_quality_samples.json)

| 拼图页 | 样本编号 |
| --- | --- |
| [label_quality_contact_01.jpg](label_quality_contact_01.jpg) | 1–8 |
| [label_quality_contact_02.jpg](label_quality_contact_02.jpg) | 9–16 |
| [label_quality_contact_03.jpg](label_quality_contact_03.jpg) | 17–24 |
| [label_quality_contact_04.jpg](label_quality_contact_04.jpg) | 25–32 |
| [label_quality_contact_05.jpg](label_quality_contact_05.jpg) | 33–40 |
| [label_quality_contact_06.jpg](label_quality_contact_06.jpg) | 41–48 |
| [label_quality_contact_07.jpg](label_quality_contact_07.jpg) | 49–56 |
| [label_quality_contact_08.jpg](label_quality_contact_08.jpg) | 57–64 |
| [label_quality_contact_09.jpg](label_quality_contact_09.jpg) | 65–72 |
| [label_quality_contact_10.jpg](label_quality_contact_10.jpg) | 73–80 |
| [label_quality_contact_11.jpg](label_quality_contact_11.jpg) | 81–88 |
| [label_quality_contact_12.jpg](label_quality_contact_12.jpg) | 89–96 |
