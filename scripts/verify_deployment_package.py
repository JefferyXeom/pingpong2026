"""Verify an archive before extraction, or verify an extracted package; stdlib only."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def checksum(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024**2), b''):
        value.update(block)
    return value.hexdigest()


def validate_name(name):
    path = PurePosixPath(name)
    if '\\' in name or path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError('Unsafe package path: ' + name)


def verify(target):
    if target.is_file():
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive members')
            for name in names:
                validate_name(name)
            manifest = json.loads(archive.read('file_manifest.json'))
            expected = {r['path'] for r in manifest['files']} | {'file_manifest.json'}
            if set(names) != expected:
                raise ValueError('Archive member list differs from manifest')
            for i, item in enumerate(manifest['files'], 1):
                info = archive.getinfo(item['path'])
                if info.file_size != item['bytes']:
                    raise ValueError('Size mismatch: ' + item['path'])
                with archive.open(item['path']) as stream:
                    if checksum(stream) != item['sha256']:
                        raise ValueError('Hash mismatch: ' + item['path'])
                if i % 250 == 0:
                    print(json.dumps({'checked': i, 'total': len(manifest['files'])}), flush=True)
    else:
        manifest = json.loads((target / 'file_manifest.json').read_text(encoding='utf-8'))
        for item in manifest['files']:
            validate_name(item['path'])
            path = target / item['path']
            if not path.resolve().is_relative_to(target.resolve()) or path.stat().st_size != item['bytes']:
                raise ValueError('Invalid extracted member: ' + item['path'])
            with path.open('rb') as stream:
                if checksum(stream) != item['sha256']:
                    raise ValueError('Hash mismatch: ' + item['path'])
    result = {'passed': True, 'package_id': manifest['package_id'], 'part': manifest['part'],
              'files': len(manifest['files']), 'bytes': sum(r['bytes'] for r in manifest['files'])}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    verify(parser.parse_args().path)
