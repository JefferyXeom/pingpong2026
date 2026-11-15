"""Validate relocation, report links and checkpoint identities without inference."""
import ast
import hashlib
import json
from pathlib import Path
import re

from migrate_report_layout import mapping, rewrite

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    migration=json.loads((ROOT/'reports/report_layout_migration.json').read_text(encoding='utf-8'))
    missing=[]
    binary_count=0
    for item in migration['files']:
        current=ROOT/item['new_path']
        assert current.is_file(),current
        assert not (ROOT/item['old_path']).exists(),item['old_path']
        if current.suffix.lower() in {'.pt','.png','.jpg'}:
            assert digest(current)==item['sha256_before'],current
            binary_count+=1
    main_report='reports/evaluation/center_conversion_matrix_v1/center_conversion_matrix_v1.json'
    original=next(item for item in migration['files'] if item['new_path']==main_report)
    assert digest(ROOT/main_report)==original['sha256_before'],'Formal matrix data changed during organization'
    for path in (ROOT/'scripts').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    checked_json=0
    for path in (ROOT/'reports').rglob('*.json'):
        json.loads(path.read_text(encoding='utf-8-sig'))
        checked_json+=1
    root_files={p.name for p in (ROOT/'reports').iterdir() if p.is_file()}
    assert root_files<={'README.md','catalog.json','experiment_registry.json','report_layout_migration.json','report_layout_verification.json'},root_files
    candidates=[ROOT/'README.md']
    for directory in ('scripts','docs','models','configs'):
        candidates.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and
                          p.suffix in {'.py','.ps1','.md','.json','.yaml'} and
                          p.name not in {'migrate_report_layout.py','verify_report_layout.py'})
    stale=[]
    for path in candidates:
        content=path.read_text(encoding='utf-8-sig')
        if rewrite(content,mapping())!=content:
            stale.append(str(path.relative_to(ROOT)))
    assert not stale,stale
    markdown=[ROOT/'README.md',*(ROOT/'docs').rglob('*.md'),*(ROOT/'reports').rglob('*.md'),*(ROOT/'models').rglob('*.md')]
    checked_links=0
    for path in markdown:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            target=target.split('#')[0].strip('<>')
            resolved=(path.parent/target).resolve()
            if resolved==ROOT/'reports/report_layout_verification.json':
                continue
            if not resolved.exists():
                missing.append({'file':str(path.relative_to(ROOT)),'target':target})
            checked_links+=1
    assert not missing,missing
    registry=json.loads((ROOT/'reports/experiment_registry.json').read_text(encoding='utf-8'))
    for model in registry['models']:
        assert digest(ROOT/model['path'])==model['sha256'],model['id']
    for case in registry['evaluation_batches']:
        assert (ROOT/case['reports']).is_dir()
    result={'passed':True,'relocated_files':len(migration['files']),
            'binary_files_sha256_unchanged':binary_count,'formal_matrix_sha256_unchanged':True,
            'json_reports_parsed':checked_json,'local_markdown_links_checked':checked_links,
            'legacy_script_doc_model_references':stale,'model_hashes_checked':len(registry['models']),
            'note':'Organization and metadata checks only; no training or inference repeated.'}
    (ROOT/'reports/report_layout_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
