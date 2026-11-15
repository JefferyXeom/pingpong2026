"""Publish a read-only, deterministic label review with lossless original crops."""
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/datasets/pingpong_baidu_out_v1'
SEED = 20261115


def encoded(im, fmt):
    stream = io.BytesIO()
    im.save(stream, format=fmt, **({'quality': 92, 'subsampling': 0} if fmt == 'JPEG' else {}))
    return 'data:image/' + ('jpeg' if fmt == 'JPEG' else 'png') + ';base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def geometry(r):
    return [[x*r['width'], y*r['height'], w*r['width'], h*r['height']] for _, x, y, w, h in r['boxes']]


def main():
    records = [json.loads(line) for line in (OUT / 'audit_decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    backups = {}
    for dataset in ['pingpong_baidu_out_v1', 'pingpong_additional_v1']:
        for line in (ROOT / 'datasets' / dataset / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            backups[r['image_sha256']] = (dataset, r)
    chosen, reasons = {}, {}

    def select(r, reason):
        key = r['source_image']
        chosen[key] = r
        reasons.setdefault(key, []).append(reason)

    rng = random.Random(SEED)
    groups = [('new', 'new_train_candidate'), ('used', 'already_present_bytes')]
    statistics = {}
    for group, decision in groups:
        pool = [r for r in records if r['decision'] == decision]
        prefixes = sorted({Path(r['source_image']).stem.split('_')[0] for r in pool})
        # 32 random examples per provenance group; balance filename prefixes.
        for i, prefix in enumerate(prefixes):
            candidates = [r for r in pool if Path(r['source_image']).stem.split('_')[0] == prefix]
            count = 32 // len(prefixes) + (i < 32 % len(prefixes))
            for r in rng.sample(candidates, min(count, len(candidates))):
                select(r, 'random_prefix_stratified')
        for r in sorted(pool, key=lambda r: min(w*h for _, _, w, h in geometry(r)))[:4]:
            select(r, 'smallest_box_area')
        for r in sorted(pool, key=lambda r: max(w*h for _, _, w, h in geometry(r)), reverse=True)[:4]:
            select(r, 'largest_box_area')
        for r in sorted(pool, key=lambda r: max(max(w/h, h/w) for _, _, w, h in geometry(r)), reverse=True)[:4]:
            select(r, 'elongated_box')
        for r in sorted(pool, key=lambda r: (-r['objects'], min(min(x-w/2, y-h/2, r['width']-x-w/2, r['height']-y-h/2) for x,y,w,h in geometry(r))))[:4]:
            select(r, 'multiple_objects_or_edge')
        boxes = np.asarray([g for r in pool for g in geometry(r)])
        statistics[group] = {'images': len(pool), 'objects': len(boxes), 'object_count_histogram': dict(Counter(r['objects'] for r in pool)),
                             'filename_prefix_counts': dict(Counter(Path(r['source_image']).stem.split('_')[0] for r in pool)),
                             'quantile_order': ['min', 'p05', 'p50', 'p95', 'max'],
                             'box_width_px': np.percentile(boxes[:, 2], [0,5,50,95,100]).tolist(),
                             'box_height_px': np.percentile(boxes[:, 3], [0,5,50,95,100]).tolist()}
    samples, visual_tiles = [], []
    for i, r in enumerate(chosen.values(), 1):
        dataset, backup = backups[r['sha256']]
        image_path = ROOT / 'datasets' / dataset / backup['image']
        annotation = ROOT / 'datasets' / dataset / backup['annotation']
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == r['sha256']
        xml_boxes = []
        for obj in ET.parse(annotation).getroot().findall('object'):
            box = obj.find('bndbox')
            v = [float(box.findtext(k)) for k in ['xmin', 'ymin', 'xmax', 'ymax']]
            xml_boxes.append({'class': obj.findtext('name'), 'xmin_ymin_xmax_ymax': v})
        assert len(xml_boxes) == r['objects']
        xywh = geometry(r)
        for (cx, cy, w, h), xml in zip(xywh, xml_boxes):
            x0,y0,x1,y1 = xml['xmin_ymin_xmax_ymax']
            assert np.allclose([cx,cy,w,h], [(x0+x1)/2-1,(y0+y1)/2-1,x1-x0,y1-y0], atol=1e-7)
        with Image.open(image_path) as im:
            original = im.convert('RGB')
        assert original.size == (r['width'], r['height'])
        crops, crop_images = [], []
        for cx, cy, w, h in xywh:
            side = max(96, int(np.ceil(max(w,h))) + 32)
            left = max(0, min(original.width-side, int(round(cx-side/2))))
            top = max(0, min(original.height-side, int(round(cy-side/2))))
            crop = original.crop((left, top, left+side, top+side))
            crops.append({'left': left, 'top': top, 'side': side, 'image': encoded(crop, 'PNG')})
            crop_images.append(crop)
        group = 'new' if r['decision'] == 'new_train_candidate' else 'used'
        sample = {'id': i, 'name': Path(r['source_image']).name, 'group': group, 'prefix': Path(r['source_image']).stem.split('_')[0],
                  'reason': reasons[r['source_image']], 'source_image': r['source_image'], 'source_annotation': r['annotation'],
                  'backup_image': image_path.relative_to(ROOT).as_posix(), 'backup_annotation': annotation.relative_to(ROOT).as_posix(),
                  'sha256': r['sha256'], 'width': r['width'], 'height': r['height'], 'boxes_xywh_px': xywh,
                  'xml_boxes': xml_boxes, 'full': encoded(original, 'JPEG'), 'crops': crops}
        samples.append(sample)
        # Contact sheet: context and first object crop with identical training labels.
        tile = Image.new('RGB', (800, 390), '#18212c')
        context = original.resize((400,320), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(context)
        for j,(cx,cy,w,h) in enumerate(xywh):
            d.rectangle(((cx-w/2)/3.2,(cy-h/2)/3.2,(cx+w/2)/3.2,(cy+h/2)/3.2), outline='#00ff8c', width=1)
            d.text((cx/3.2+5,cy/3.2-10), str(j+1), fill='#ffff00')
        tile.paste(context,(0,40))
        crop = crop_images[0].resize((320,320), Image.Resampling.NEAREST)
        d = ImageDraw.Draw(crop)
        c = crops[0]; scale = 320/c['side']
        for cx,cy,w,h in xywh:
            d.rectangle(((cx-w/2-c['left'])*scale,(cy-h/2-c['top'])*scale,(cx+w/2-c['left'])*scale,(cy+h/2-c['top'])*scale), outline='#00ff8c', width=1)
            px,py = (cx-c['left'])*scale,(cy-c['top'])*scale
            for line in [(px-9,py,px-4,py),(px+4,py,px+9,py),(px,py-9,px,py-4),(px,py+4,px,py+9)]:
                d.line(line, fill='#ff5959', width=1)
        tile.paste(crop,(430,40))
        d=ImageDraw.Draw(tile)
        d.text((8,8), f'#{i:03d} {group.upper()} {sample["name"]}', fill='white')
        d.text((8,365), f'{len(xywh)} boxes | ' + ', '.join(sample['reason']), fill='white')
        visual_tiles.append(tile)
    pages=[]
    for offset in range(0,len(samples),8):
        page=Image.new('RGB',(1600,4*390),'#18212c')
        for j,tile in enumerate(visual_tiles[offset:offset+8]):
            page.paste(tile,((j%2)*800,(j//2)*390))
        name=f'label_quality_contact_{offset//8+1:02d}.jpg'
        page.save(OUT/name, quality=94, subsampling=0)
        pages.append(name)
    metadata = [{k:v for k,v in s.items() if k not in ['full','crops']} for s in samples]
    payload={'seed': SEED, 'samples': samples, 'statistics': statistics}
    template=(ROOT/'scripts/baidu_label_gallery.html').read_text(encoding='utf-8')
    html=template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
    (OUT/'label_quality_gallery.html').write_text(html,encoding='utf-8')
    summary={'source_images':len(records), 'sample_images':len(samples), 'seed':SEED, 'statistics':statistics,
             'sampling':'32 random per new/used group balanced by filename prefix, plus 4 each smallest/largest/elongated/multiple-or-edge; deduplicate selected images.',
             'sample_groups':dict(Counter(s['group'] for s in samples)), 'source_and_labels_modified':False,
             'model_predictions_used':False, 'sampled_image_hashes_and_xml_conversion_verified':True,
             'rendering':'Context is JPEG92 full resolution; embedded object crops are lossless PNG at original pixels; HTML zoom uses nearest-neighbor. Static sheets are JPEG previews.',
             'coordinates':'Training convention: center=(VOC_min+VOC_max)/2-1, extent=VOC_max-VOC_min; raw XML shown separately.',
             'limits':'Geometric validity is not visual annotation correctness; sample is partly targeted, not a population error-rate estimate.',
             'contact_sheets':pages}
    for name,data in [('label_quality_samples.json',metadata),('label_geometry_summary.json',summary)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Baidu数据标注质量可视化','',
           '**交互浏览入口：[label_quality_gallery.html](label_quality_gallery.html)。下载该单个HTML后用浏览器打开即可，全部样本图已嵌入，无需访问训练机E盘。VS Code可先打开下方JPG快速预览。**','',
           f'本次抽取{len(samples)}张：新增{summary["sample_groups"].get("new",0)}张、此前已用{summary["sample_groups"].get("used",0)}张。随机种子{SEED}。每组32张按文件名前缀分层随机抽取，再补充小框、大框、长条框、多球/边缘目标；重叠样本只保留一次。','',
           '绿色为实际训练标注框，红色为标注框几何中心；不是模型预测。浏览页可搜索文件名、筛选新增/已用、选择同图不同球、切换框/中心显示，并查看原始XML坐标、训练坐标和源路径。无损局部PNG直接从备份原PNG截取，整数倍最近邻放大，适合检查模糊、框松紧和中心偏差；全图JPEG用于观察场景，JPG拼图也仅是快速预览。','',
           '训练坐标沿用项目历史VOC约定：center=(min+max)/2−1，宽高=max−min。浏览页同时列出原始XML值；不要将两种坐标显示差异误认为新改了标签。所有样本图已核对内容哈希及XML→训练坐标；未修改原图、标注、split、模型或训练结果。','',
           '可在交互页标记“待复查”并导出JSON，该操作仅导出复查清单，不修改标签。重点检查：球是否漏标/多标、框是否覆盖可见球、模糊拖影是否采用一致标注标准、框几何中心是否与期望球心一致。几何合法不代表人工框选正确，本抽样含针对性极值，不能估计全体错误率。','',
           '初步目视提示：在已查看的第1、5、6页中，可见多处模糊球以及手持/靠近球拍的球；一些框与可见亮斑接近，但模糊轮廓下的几何中心和期望物理球心仍需人工判断。应在交互页关闭框/球心后比较原始像素，逐目标检查，不能由抽样推断全体准确率或认定本轮F1变化由标注造成。新增标注球框宽/高的中位数均为13px；本项目关心像素级定位，少量像素的中心偏移也值得复查。','',
           '[全源标注几何统计](label_geometry_summary.json) · [样本来源与全部框坐标](label_quality_samples.json)','', '| 拼图页 | 样本编号 |','| --- | --- |']
    for i,name in enumerate(pages):
        lines.append(f'| [{name}]({name}) | {i*8+1}–{min((i+1)*8,len(samples))} |')
    (OUT/'label_quality_review.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'samples':len(samples),'groups':summary['sample_groups'],'contact_sheets':len(pages),
                      'gallery':str((OUT/'label_quality_gallery.html').relative_to(ROOT)),
                      'statistics':statistics},ensure_ascii=False))


if __name__=='__main__':
    main()
