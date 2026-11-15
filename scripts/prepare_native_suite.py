"""Create audited, image-copy-free data lists for the three native-size experiments."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SUITE='native_resolution_suite_v1'


def read_manifest(name):
    return [json.loads(line) for line in (ROOT/'datasets'/name/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]


def main():
    original=read_manifest('pingpong_v1')
    additional=read_manifest('pingpong_additional_v1')
    hashes={r['image_sha256']:r['split'] for r in original}
    assert len(hashes)==len(original)==8720
    assert len(additional)==10786
    assert len({r['image_sha256'] for r in additional})==len(additional)
    assert not set(hashes)&{r['image_sha256'] for r in additional}
    counts=Counter()
    for dataset,records in (('pingpong_v1',original),('pingpong_additional_v1',additional)):
        for r in records:
            assert (r['width'],r['height'])==(1280,1024)
            image=ROOT/'datasets'/dataset/r['image']
            label=image.with_suffix('.txt')
            label=Path(str(label).replace('\\images\\','\\labels\\'))
            assert image.is_file() and label.is_file(),str(image)
            lines=label.read_text(encoding='utf-8').strip().splitlines()
            expected=r['label'].strip().splitlines() if dataset=='pingpong_v1' else [' '.join(map(str,b)) for b in r['boxes']]
            assert len(lines)==len(expected)>0
            for line,wanted in zip(lines,expected):
                a,b=list(map(float,line.split())),list(map(float,wanted.split()))
                assert len(a)==len(b)==5 and a[0]==b[0]==0
                assert all(abs(x-y)<1e-7 for x,y in zip(a,b))
            if dataset=='pingpong_additional_v1':
                parents=r['active_parent_matches_for_review']
                assert all(parent['split']=='train' for parent in parents),r['image']
                if r['background_variant']:
                    assert len(parents)==1 and parents[0]['split']=='train'
                counts[r['filename_family']]+=1
    train=[('pingpong_v1',r) for r in original if r['split']=='train']
    val=[('pingpong_v1',r) for r in original if r['split']=='val']
    test=[('pingpong_v1',r) for r in original if r['split']=='test']
    directory=ROOT/'configs/datasets'/SUITE
    directory.mkdir(parents=True,exist_ok=True)
    report={'suite_id':SUITE,'original_files_modified':False,'images_copied':0,
            'original_size_width_height':[1280,1024],
            'additional_families':dict(counts),'datasets':{},
            'limits':['Unknown historical recording families are not proven independent of v1 recordings.',
                      'Exact SHA256 disjointness and known-parent training isolation verified.',
                      'Additional includes 2997 white-background variants; not 10786 independent original captures.']}
    for name,training in (('native_v1',train),('native_v1_additional',train+[('pingpong_additional_v1',r) for r in additional])):
        data={}
        for split,items in (('train',training),('val',val),('test',test)):
            path=directory/f'{name}_{split}.txt'
            content=''.join((ROOT/'datasets'/ds/r['image']).as_posix()+'\n' for ds,r in items)
            if path.exists() and path.read_text(encoding='utf-8')!=content:
                raise RuntimeError('Existing data list differs: '+str(path))
            path.write_text(content,encoding='utf-8')
            data[split]={'images':len(items),'objects':sum(len(r['label'].splitlines()) if ds=='pingpong_v1' else r['objects'] for ds,r in items),
                         'list':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(content.encode()).hexdigest()}
        yaml=directory/f'{name}.yaml'
        yaml.write_text('\n'.join(f'{split}: {(ROOT/data[split]["list"]).as_posix()}' for split in ('train','val','test'))+'\nnames:\n  0: ball\n',encoding='utf-8')
        data['yaml']=yaml.relative_to(ROOT).as_posix()
        report['datasets'][name]=data
    destination=ROOT/'reports/datasets'/SUITE
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'data_preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (destination/'README.md').write_text('# 原图训练对照实验数据\n\n配置见configs/native_resolution_suite_v1.json；data_preparation.json记录数据数量、列表哈希、背景变体与来源限制。列表引用现有图片和标签，不复制图片。v1原始val/test固定不变。\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
