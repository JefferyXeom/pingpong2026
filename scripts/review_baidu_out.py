"""Preview audited unique images and their filename relatives, without changing data."""
from collections import Counter
import json
from pathlib import Path
from PIL import Image, ImageDraw

from audit_baidu_out import ROOT, REPORT, old_records


def main():
    records=[json.loads(line) for line in (REPORT/'audit_decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    accepted=[r for r in records if r['decision']=='new_train_candidate']
    if not accepted:
        return
    previous={(r['dataset'],r['image']):r for r in old_records()}
    examples=[accepted[min(len(accepted)-1, int((i+.5)*len(accepted)/6))] for i in range(6)]
    preview=Image.new('RGB',(1280,6*300),'#20242a')
    metadata=[]
    for index,r in enumerate(examples):
        items=[(Path(r['source_image']),r['boxes'],'NEW '+Path(r['source_image']).name)]
        parent=r['filename_parent_candidates'][0] if r['filename_parent_candidates'] else None
        if parent:
            old=previous[(parent['dataset'],parent['image'])]
            items.append((Path(old['absolute_image']),old['old_boxes'],'OLD TRAIN '+Path(old['source_image']).name))
        for col,(path,boxes,title) in enumerate(items):
            with Image.open(path) as im:
                tile=im.convert('RGB').resize((360,288))
            draw=ImageDraw.Draw(tile)
            for _,x,y,w,h in boxes:
                draw.rectangle(((x-w/2)*360,(y-h/2)*288,(x+w/2)*360,(y+h/2)*288),outline='#00ff72',width=2)
            preview.paste(tile,(col*640,index*300))
            main_draw=ImageDraw.Draw(preview)
            main_draw.text((col*640+365,index*300+8),title[:34],fill='white')
            with Image.open(path) as im:
                if boxes:
                    _,x,y,w,h=boxes[0]
                    cx,cy=round(x*im.width),round(y*im.height)
                    crop=im.convert('RGB').crop((cx-40,cy-40,cx+40,cy+40)).resize((240,240))
                    d=ImageDraw.Draw(crop);d.line((110,120,130,120),fill='#ff5555',width=1);d.line((120,110,120,130),fill='#ff5555',width=1)
                    preview.paste(crop,(col*640+375,index*300+40))
        metadata.append({'new_image':r['source_image'],'filename_parent_candidate':parent,
                         'relation':'Filename candidate only; not proof of augmentation; inspect visual context.'})
    preview.save(REPORT/'source_preview.jpg',quality=92)
    (REPORT/'source_preview.json').write_text(json.dumps({'new_examples':metadata,
        'new_prefix_counts':dict(Counter(Path(r['source_image']).stem.split('_')[0] for r in accepted)),
        'note':'Original image with label box at left; first-ball80px crop enlarged at right. No label edits.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'preview':str((REPORT/'source_preview.jpg').relative_to(ROOT)),'examples':len(examples)}))


if __name__=='__main__':
    main()
