"""Download and verify the fixed public TabICLv2 checkpoint; never train a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'jingang/TabICL'
REVISION = '4dcd344ece2c00be9e831fdd35bed57b5ad83e19'
FILENAME = 'tabicl-classifier-v2-20260212.ckpt'
SHA256 = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
URL = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def setup(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / FILENAME
    if target.exists():
        if digest(target) != SHA256:
            raise ValueError(f'Existing checkpoint has the wrong SHA-256: {target}; preserve it for inspection and remove it explicitly before retrying')
        return target
    # An interrupted or corrupt download cannot replace a verified checkpoint.
    fd, temporary = tempfile.mkstemp(prefix='.tabicl-download-', dir=destination)
    try:
        with os.fdopen(fd, 'wb') as output, urlopen(URL, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if digest(temporary) != SHA256:
            raise ValueError('Downloaded checkpoint failed the pinned SHA-256 check')
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'results/round7/foundation/checkpoints')
    args = parser.parse_args()
    path = setup(args.directory)
    print(json.dumps(dict(path=str(path), repository=REPOSITORY, revision=REVISION,
                         filename=FILENAME, sha256=digest(path), bytes=path.stat().st_size), indent=2))


if __name__ == '__main__':
    main()
