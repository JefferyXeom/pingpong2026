"""One-time, resumable report relocation with path updates and a migration manifest.

Only reports and their textual references move; checkpoints and evaluation numbers
are not regenerated. Run with --apply after reviewing the printed plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
GROUPS={
    'environment/windows_rtx3090': ['environment_freeze.txt'],
    'datasets/pingpong_v1': [
        'dataset_inventory.json','dataset_detail.json','dataset_preparation.json',
        'dataset_verification.json','dataset_preview.jpg','pairing_review.json','archive_inventory.json'],
    'datasets/pingpong_additional_v1': [
        'additional_data_audit.json','additional_data_summary.json','additional_data_review.json',
        'additional_data_provenance.json','additional_dataset_verification.json','additional_data_preview.jpg'],
    'training/smoke_yolo26n_960': ['smoke_yolo26n_960_result.json'],
    'training/baseline_yolo26n_960': [
        'baseline_process.json','baseline_launch_history.jsonl','baseline_yolo26n_960_progress.json',
        'baseline_yolo26n_960_result.json','baseline_interruption_review.json',
        'baseline_training_analysis.json','baseline_learning_curves.png'],
    'evaluation/baseline_yolo26n_960/diagnostics_epoch24': [
        'baseline_diagnostics','baseline_diagnostics.json','baseline_center_diagnostics.json',
        'baseline_prediction_crops.jpg','train_val_domain_review.json','train_val_domain_preview.jpg'],
    'evaluation/baseline_yolo26n_960/center_selection_v1': [
        'baseline_full_val_centers.json','baseline_full_val_centers_best_frames.json',
        'baseline_full_val_centers_last_frames.json','baseline_full_test_centers.json',
        'baseline_full_test_centers_last_frames.json','trained_model_selection.json'],
    'evaluation/smoke_conversion_v1': ['conversion_comparison.json'],
    'evaluation/center_conversion_matrix_v1': [
        'center_conversion_matrix_v1.json','center_conversion_matrix_progress.json',
        'center_conversion_matrix_summary.json','center_conversion_matrix_summary.csv',
        'center_conversion_matrix_verification.json','center_conversion_matrix_comparison.png',
        'center_conversion_regressions.json','conversion_original_image_crops.jpg','conversion_original_examples'],
}


def mapping():
    return {f'reports/{name}':f'reports/{directory}/{name}'
            for directory,names in GROUPS.items() for name in names}


def rewrite(text,paths):
    if 'reports' not in text:
        return text
    for old,new in sorted(paths.items(),key=lambda pair:-len(pair[0])):
        name=old.removeprefix('reports/')
        if name not in text:
            continue
        for separator in ('/','\\','\\\\'):
            text=text.replace(old.replace('/',separator),new.replace('/',separator))
        # Also handle ROOT / 'reports' / 'literal_name' source expressions.
        parent=new.rsplit('/',1)[0]
        pattern=r"(['\"])reports\1(\s*/\s*)(['\"])"+re.escape(name)+r"\3"
        text=re.sub(pattern,lambda m:f'{m[1]}{parent}{m[1]}{m[2]}{m[3]}{name}{m[3]}',text)
    return text


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    paths=mapping()
    report_root=(ROOT/'reports').resolve()
    for old,new in paths.items():
        source,target=ROOT/old,ROOT/new
        assert source.resolve().is_relative_to(report_root) and target.resolve().is_relative_to(report_root)
        if source.exists() and target.exists():
            raise RuntimeError(f'Both source and destination exist: {old}, {new}')
        if not source.exists() and not target.exists():
            raise FileNotFoundError(old)
    print(json.dumps(paths,indent=2))
    if not args.apply:
        return
    originals=[]
    for old,new in paths.items():
        source,target=ROOT/old,ROOT/new
        if source.exists():
            originals.extend({'old_path':str(p.relative_to(ROOT)),
                              'new_path':str((target/p.relative_to(source)).relative_to(ROOT)) if source.is_dir() else new,
                              'sha256_before':hashlib.sha256(p.read_bytes()).hexdigest()}
                             for p in (source.rglob('*') if source.is_dir() else [source]) if p.is_file())
            target.parent.mkdir(parents=True,exist_ok=True)
            source.rename(target)
    roots=['scripts','docs','models','configs','reports','artifacts','runs/diagnostics','datasets/pingpong_additional_v1']
    texts=[ROOT/'README.md']
    for relative in roots:
        texts.extend(p for p in (ROOT/relative).rglob('*') if p.is_file() and
                     p.suffix.lower() in {'.py','.ps1','.md','.json','.jsonl','.yaml','.yml','.txt','.csv'} and
                     p.name not in {'manifest.jsonl','image_order.json','migrate_report_layout.py','report_layout_migration.json'})
    updated=[]
    for path in texts:
        original=path.read_bytes()
        try:
            text=original.decode('utf-8')
        except UnicodeDecodeError:
            continue
        result=rewrite(text,paths)
        if result!=text:
            path.write_bytes(result.encode('utf-8'))
            updated.append(str(path.relative_to(ROOT)))
    for item in originals:
        item['sha256_after']=hashlib.sha256((ROOT/item['new_path']).read_bytes()).hexdigest()
        item['content_changed_for_path_references']=item['sha256_before']!=item['sha256_after']
    manifest={'operation':'relocate reports and update textual path references only',
              'files':originals,'updated_reference_files':updated,'path_mapping':paths}
    output=ROOT/'reports/report_layout_migration.json'
    if originals or not output.exists():
        output.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'relocated_files':len(originals),'updated_reference_files':len(updated)}))


if __name__=='__main__':
    main()
