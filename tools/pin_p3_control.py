#!/usr/bin/env python3
"""Pin the retained P3 FP32 oracle without regenerating any tensors."""
import argparse
import hashlib
import json
from pathlib import Path

NAME = "p3b-python.safetensors"
SHA256 = "e6c122b87f78be8312ba6ed5eed63514789b1aed9a3cb9823666edda5a018994"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root
    if not (root / "manifest.json").is_file():
        root = root / "first-song"
    with (root / NAME).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == SHA256, (NAME, digest, SHA256)
    entry = json.loads((root / "p3b-python.json").read_text())["files"][NAME]
    assert entry["sha256"] == digest and entry["bytes"] == (root / NAME).stat().st_size
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if NAME in manifest["files"] or args.check:
        assert manifest["files"][NAME] == entry, "Pinned fixture metadata changed"
    else:
        manifest["files"][NAME] = entry
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
    print(f"{NAME}: bytes={entry['bytes']} sha256={digest}; manifest verified")


if __name__ == "__main__":
    main()
