"""Download only the requested YOLO26s official pretrained initialization."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'models/yolo26s.pt'
    url='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt'
    if not target.exists():
        temporary=target.with_suffix('.pt.partial')
        with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as out:
            while data:=response.read(1024*1024):
                out.write(data)
        if temporary.stat().st_size<1024*1024:
            raise RuntimeError('Unexpected pretrained download size')
        temporary.replace(target)
    with target.open('rb') as stream:
        checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    report={'source_url':url,'path':target.relative_to(ROOT).as_posix(),'sha256':checksum,'bytes':target.stat().st_size}
    out=ROOT/'reports/environment/windows_rtx3090/yolo26s_pretrained_download.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
