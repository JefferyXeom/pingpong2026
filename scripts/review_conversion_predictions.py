"""Save original-image crops for uniformly sampled size/backend comparisons."""
import json
from pathlib import Path

from evaluate_conversion_matrix import ROOT, OUT


def main():
    import numpy as np
    from PIL import Image, ImageDraw
    report=json.loads((ROOT/'reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_v1.json').read_text(encoding='utf-8'))
    order=json.loads((OUT/'image_order.json').read_text(encoding='utf-8'))
    manifest=[json.loads(line) for line in (ROOT/'datasets/pingpong_v1/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    by_image={r['image']:r for r in manifest}
    configurations=[('h768_w960_pytorch_fp32_b2','PT 960'),('h1024_w1280_pytorch_fp32_b2','PT native'),
                    ('h1024_w1280_tensorrt_fp16_b2','TRT FP16 native')]
    if any(key not in report['cases'] for key,_ in configurations):
        raise RuntimeError('Wait for all required cases.')
    cached={key:np.load(ROOT/report['cases'][key]['prediction_cache']) for key,_ in configurations}
    indices=[]
    for split in ('val','test'):
        available=[i for i,r in enumerate(order) if r['split']==split]
        indices.extend(available[int((j+.5)*len(available)/4)] for j in range(4))
    examples=ROOT/'reports/evaluation/center_conversion_matrix_v1/conversion_original_examples'
    examples.mkdir(exist_ok=True)
    example_index=[]
    preview=Image.new('RGB',(960,8*210),'#202020')
    for row,index in enumerate(indices):
        record=by_image[order[index]['image']]
        with Image.open(ROOT/'datasets/pingpong_v1'/record['image']) as opened:
            image=opened.convert('RGB')
        boxes=[]
        for line in record['label'].splitlines():
            _,x,y,w,h=map(float,line.split())
            boxes.append([(x-w/2)*1280,(y-h/2)*1024,(x+w/2)*1280,(y+h/2)*1024])
        gt=np.asarray(boxes)
        original=image.copy()
        d=ImageDraw.Draw(original)
        for box in gt:
            d.rectangle(tuple(box),outline='lime',width=2)
        native_predictions=cached[configurations[-1][0]][f'f{index}']
        for detection in native_predictions[native_predictions[:,4]>=.25]:
            box=detection[:4]
            d.rectangle(tuple(box),outline='magenta',width=1)
            x,y=(box[:2]+box[2:])/2
            d.line((x-4,y,x+4,y),fill='yellow',width=1)
            d.line((x,y-4,x,y+4),fill='yellow',width=1)
        d.text((8,8),'TRT FP16 native 1280x1024; GT=green, prediction=magenta, center=yellow',fill='yellow',stroke_width=1,stroke_fill='black')
        example_path=examples/f'{row+1:02d}_{record["split"]}_native_fp16.jpg'
        original.save(example_path,quality=95)
        example_index.append({'file':str(example_path.relative_to(ROOT)), 'source_image':record['image'],
                              'image_sha256':order[index]['image_sha256'],'split':record['split'],
                              'size':[1280,1024],'case':configurations[-1][0],'confidence':.25})
        full=image.resize((240,192))
        d=ImageDraw.Draw(full)
        for box in gt:
            d.rectangle(tuple(box*.1875),outline='lime',width=1)
        preview.paste(full,(0,row*210))
        cx,cy=(gt[0,:2]+gt[0,2:])/2
        left,top=round(cx)-24,round(cy)-24
        for column,(key,label) in enumerate(configurations,1):
            crop=image.crop((left,top,left+48,top+48)).resize((192,192))
            d=ImageDraw.Draw(crop)
            for box in gt:
                coordinates=(box-np.array([left,top,left,top]))*4
                d.rectangle(tuple(coordinates),outline='lime',width=2)
            predictions=cached[key][f'f{index}']
            predictions=predictions[predictions[:,4]>=.25]
            for box in predictions[:,:4]:
                coordinates=(box-np.array([left,top,left,top]))*4
                if coordinates[2]>=0 and coordinates[3]>=0 and coordinates[0]<192 and coordinates[1]<192:
                    d.rectangle(tuple(coordinates),outline='magenta',width=2)
                    center=(coordinates[:2]+coordinates[2:])/2
                    d.ellipse((center[0]-2,center[1]-2,center[0]+2,center[1]+2),fill='yellow')
            preview.paste(crop,(column*240,row*210))
            ImageDraw.Draw(preview).text((column*240+3,row*210+194),label,fill='white')
        ImageDraw.Draw(preview).text((3,row*210+194),record['split']+' / uniform sample',fill='white')
    path=ROOT/'reports/evaluation/center_conversion_matrix_v1/conversion_original_image_crops.jpg'
    preview.save(path,quality=93)
    (examples/'index.json').write_text(json.dumps(example_index,indent=2),encoding='utf-8')
    for cache in cached.values():
        cache.close()
    print(path)


if __name__=='__main__':
    main()
