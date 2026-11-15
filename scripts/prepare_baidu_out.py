"""Copy only audited new images, retain original annotations, version merged training lists."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from audit_baidu_out import ROOT, REPORT, VERSION, old_records
from audit_additional_data import sha256, txt_label, labels_agree

SUITE = 'native_merged_v2_baiduout_20261113'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def copy_verified(source, target, checksum):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != checksum:
            raise RuntimeError('Existing destination differs: '+str(target))
        return
    temporary = target.with_suffix(target.suffix+'.copying')
    shutil.copy2(source, temporary)
    if sha256(temporary) != checksum:
        raise RuntimeError('Copied content differs: '+str(target))
    temporary.replace(target)


def main():
    summary = json.loads((REPORT/'audit_summary.json').read_text(encoding='utf-8'))
    decisions = [json.loads(line) for line in (REPORT/'audit_decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    fresh = [r for r in decisions if r['decision']=='new_train_candidate']
    if not fresh:
        print(json.dumps({'new_images':0,'training_needed':False}), flush=True)
        return
    previous = old_records()
    assert not {r['sha256'] for r in fresh} & {r['image_sha256'] for r in previous}
    assert all(p['split']=='train' for r in fresh for p in r['filename_parent_candidates'])
    base = ROOT/'datasets'/VERSION
    required = sum(r['bytes'] for r in fresh if not (base/'images/train'/f'{r["sha256"][:16]}__{Path(r["source_image"]).name}').exists())
    if shutil.disk_usage(ROOT).free-required < 8*1024**3:
        raise RuntimeError('Insufficient disk to copy new images and retain conversion/training reserve')
    manifest=[]
    for index,r in enumerate(fresh,1):
        if shutil.disk_usage(ROOT).free < 8*1024**3:
            raise RuntimeError('Stop to preserve disk reserve')
        filename=r['sha256'][:16]+'__'+Path(r['source_image']).name
        image=base/'images/train'/filename
        label=base/'labels/train'/Path(filename).with_suffix('.txt')
        annotation=base/'annotations/original'/Path(filename).with_suffix(Path(r['annotation']).suffix)
        copy_verified(Path(r['source_image']),image,r['sha256'])
        annotation_hash=sha256(Path(r['annotation']))
        copy_verified(Path(r['annotation']),annotation,annotation_hash)
        label.parent.mkdir(parents=True,exist_ok=True)
        content=''.join('0 '+' '.join(f'{v:.10f}' for v in box[1:])+'\n' for box in r['boxes'])
        if label.exists() and label.read_text(encoding='utf-8')!=content:
            raise RuntimeError('Existing label differs: '+str(label))
        label.write_text(content,encoding='utf-8')
        assert labels_agree(txt_label(label),r['boxes']),str(label)
        manifest.append({'source_image':r['source_image'],'source_annotation':r['annotation'],
            'image':image.relative_to(base).as_posix(),'label':label.relative_to(base).as_posix(),
            'annotation':annotation.relative_to(base).as_posix(),'image_sha256':r['sha256'],
            'pixel_sha256':r['pixel_sha256'],'annotation_sha256':annotation_hash,'split':'train',
            'width':r['width'],'height':r['height'],'objects':r['objects'],'boxes':r['boxes'],
            'filename_parent_candidates':r['filename_parent_candidates'],
            'provenance_limit':'Historical source; filename relatives retained in train; capture independence not proven.'})
        if index%250==0:
            print(json.dumps({'copied_verified_new':index,'total_new':len(fresh)}),flush=True)
    with (base/'manifest.jsonl').open('w',encoding='utf-8') as stream:
        for r in manifest:
            stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    counts={'version':VERSION,'images':len(manifest),'objects':sum(r['objects'] for r in manifest),
            'image_bytes':sum(r['bytes'] for r in fresh),'source_files_modified':False,'negative_images':0,
            'split':'train-only','duplicates_not_copied':summary['images']-len(fresh),
            'audit':'reports/datasets/'+VERSION+'/audit_summary.json'}
    write_json(base/'dataset_summary.json',counts)
    (base/'README.md').write_text('# '+VERSION+'\n\nOnly unique verified training candidates from E:/BaiduNetdiskDownload/pingpongdata----out, audited 2026-11-13. Existing byte/pixel duplicates are excluded. See manifest.jsonl for source/hash/labels/filename parent candidates. Original source is unchanged. Not a standalone heldout dataset.\n',encoding='utf-8')
    datasets=ROOT/'configs/datasets'/SUITE
    datasets.mkdir(parents=True,exist_ok=True)
    data={}
    for split in ['train','val','test']:
        originals=[r for r in previous if r['effective_split']==split]
        items=[(Path(r['absolute_image']),len(r['old_boxes'])) for r in originals]
        if split=='train':
            items += [(base/r['image'],r['objects']) for r in manifest]
        content=''.join(p.as_posix()+'\n' for p,_ in items)
        path=datasets/f'pingpong_merged_v2_{split}.txt'
        if path.exists() and path.read_text(encoding='utf-8')!=content:
            raise RuntimeError('Existing versioned data list differs')
        path.write_text(content,encoding='utf-8')
        if split!='train':
            old_path=ROOT/f'configs/datasets/native_resolution_suite_v1/native_v1_{split}.txt'
            assert sorted(content.splitlines())==sorted(old_path.read_text(encoding='utf-8').splitlines())
        data[split]={'images':len(items),'objects':sum(n for _,n in items),'list':path.relative_to(ROOT).as_posix(),
                     'sha256':hashlib.sha256(content.encode()).hexdigest()}
    yaml=datasets/'pingpong_merged_v2.yaml'
    yaml.write_text('\n'.join(f'{split}: {(ROOT/data[split]["list"]).as_posix()}' for split in ['train','val','test'])+'\nnames:\n  0: ball\n',encoding='utf-8')
    data['yaml']=yaml.relative_to(ROOT).as_posix()
    preparation={'suite_id':SUITE,'dataset_version':'pingpong_merged_v2','increment_version':VERSION,
        'source_audit':summary,'increment':counts,'datasets':{'pingpong_merged_v2':data},
        'holdouts_unchanged':True,'source_files_modified':False,'disk_free_gib':shutil.disk_usage(ROOT).free/1024**3}
    preparation_path=ROOT/'reports/datasets'/SUITE/'data_preparation.json'
    write_json(preparation_path,preparation)
    (preparation_path.parent/'README.md').write_text('# pingpong_merged_v2 derived training lists\n\nBase v1 train + additional_v1 + audited Baidu unique increment. Original val/test retained unchanged. data_preparation.json stores counts, hashes, and audit provenance. No duplicates from prior versions are counted as new.\n',encoding='utf-8')
    old_suite=json.loads((ROOT/'configs/native_resolution_suite_v1.json').read_text(encoding='utf-8'))
    suite={**old_suite,'suite_id':SUITE,'date':'2026-11-13','objective':'Add audited novel Baidu source images to merged dataset, compare n/s at native resolution',
        'data_preparation':preparation_path.relative_to(ROOT).as_posix(),
        'summary_command':['scripts/summarize_followup_suite.py','--suite',SUITE],
        'initialization_policy':'Official pretrained per architecture; full merged training, not resume of completed B/C optimizer. Same settings/seed for prior comparison.',
        'additional_policy':'All previous v1/additional train plus new unique validated Baidu images; duplicates excluded; native1280x1024; no verified negatives.',
        'runs':[{'id':f'native_merged_v2_baiduout_yolo26{size}_1280','architecture':f'YOLO26{size}',
                 'pretrained':f'models/yolo26{size}.pt','dataset':'pingpong_merged_v2'} for size in ['n','s']]}
    config_path=ROOT/'configs'/f'{SUITE}.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=suite:
        raise RuntimeError('Existing experiment configuration differs')
    write_json(config_path,suite)
    print(json.dumps({'increment':counts,'merged_counts':data,'suite':SUITE,'free_disk_gib':preparation['disk_free_gib']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
