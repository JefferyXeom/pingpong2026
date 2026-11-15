"""Later opt-in: add old background replacements for TRAINING originals only.

Creates data_background.yaml; the original data.yaml and split stay unchanged.
"""
import json
from pathlib import Path
import shutil
from PIL import Image
from prepare_dataset import SOURCE, ROOT, digest, copy_verified

DATA = ROOT / 'datasets/pingpong_v1'

def main():
    if not (DATA / 'dataset_summary.json').is_file() or (DATA / 'PREPARATION_IN_PROGRESS').exists():
        raise RuntimeError('Original dataset is not ready.')
    output_yaml = DATA / 'data_background.yaml'
    if output_yaml.exists():
        raise RuntimeError('Background dataset variant already exists; review before running again.')
    records = [json.loads(line) for line in (DATA / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    candidates = [r for r in records if r['split'] == 'train' and r.get('future_background_source')]
    expected = sum((SOURCE / r['future_background_source']).stat().st_size for r in candidates)
    if shutil.disk_usage(ROOT).free - expected < 8 * 1024**3:
        raise RuntimeError('Insufficient space for background copies and the 5 GiB floor.')
    listing = ['./' + r['image'] for r in records if r['split'] == 'train']
    added = []
    for record in candidates:
        source = SOURCE / record['future_background_source']
        with Image.open(source) as image:
            if image.size != (record['width'], record['height']):
                raise RuntimeError('Background variant dimensions differ: ' + str(source))
            image.load()
        name = Path(record['image']).name
        image_relative = Path('images/background_train') / name
        label_relative = Path('labels/background_train') / (Path(name).stem + '.txt')
        checksum = digest(source)
        copy_verified(source, DATA / image_relative, checksum)
        (DATA / label_relative).parent.mkdir(parents=True, exist_ok=True)
        (DATA / label_relative).write_text(record['label'], encoding='utf-8')
        listing.append('./' + image_relative.as_posix())
        added.append({'source': str(source), 'original_image': record['image'], 'image': image_relative.as_posix(), 'image_sha256': checksum, 'split': 'train'})
    (DATA / 'train_with_background.txt').write_text('\n'.join(listing) + '\n', encoding='utf-8')
    output_yaml.write_text('train: train_with_background.txt\nval: images/val\ntest: images/test\nnames:\n  0: ball\n', encoding='utf-8')
    (DATA / 'background_manifest.json').write_text(json.dumps(added, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'training_originals': len(listing) - len(added), 'background_training_images': len(added), 'data_yaml': str(output_yaml)}, indent=2))

if __name__ == '__main__':
    main()
