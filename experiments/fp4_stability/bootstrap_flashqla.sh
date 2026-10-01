#!/usr/bin/env bash
# Install an isolated experimental backend without changing the locked environment.
set -euo pipefail
cd /workspace/gleipnir
source .cache-runtime.env
revision=da06429d54b0f577de0a638f451ac8f0b395e0ac
target=.cache/kernels/flashqla-da06429
mkdir -p .cache/kernels/sources "$target"
.venv/bin/python - <<'PY'
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

revision = 'da06429d54b0f577de0a638f451ac8f0b395e0ac'
archive = Path('.cache/kernels/sources/flashqla-' + revision + '.tar.gz')
url = 'https://codeload.github.com/QwenLM/FlashQLA/tar.gz/' + revision
if not archive.exists():
    urllib.request.urlretrieve(url, archive)
with tarfile.open(archive) as source:
    source.extractall('.cache/kernels/sources', filter='data')
Path('.cache/kernels/flashqla-da06429/install_manifest.json').write_text(
    json.dumps(dict(revision=revision, source_url=url,
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    tilelang='0.1.12', tvm_ffi='0.1.11'), indent=2) + '\n')
PY
uv pip install --python .venv/bin/python --no-deps \
  --target "$target" tilelang==0.1.12 apache-tvm-ffi==0.1.11
QLA_VERSION_SUFFIX=+da06429 uv pip install --python .venv/bin/python \
  --no-deps --no-build-isolation --target "$target" \
  ".cache/kernels/sources/FlashQLA-$revision"
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

target = Path('.cache/kernels/flashqla-da06429')
path = target / 'install_manifest.json'
manifest = json.loads(path.read_text())
manifest['package_sha256'] = {
    str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((target / 'flash_qla').rglob('*'))
    if p.is_file() and p.suffix in {'.py', '.csv'}
}
path.write_text(json.dumps(manifest, indent=2) + '\n')
PY
