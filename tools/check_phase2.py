#!/usr/bin/env python3
"""Validate P2a using the existing read-only YuE2 native ABC parser."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("--parser", type=Path, default=Path.home() / "yue2/YuE/skills/yue2-music/scripts/abc_tools.py")
    args = cli.parse_args()
    spec = importlib.util.spec_from_file_location("abc_tools", args.parser)
    abc_tools = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = abc_tools
    spec.loader.exec_module(abc_tools)
    metadata = json.loads((args.directory.parent / "sampling.json").read_text())
    reference = metadata["reference_plan"]["abc"]
    actual = (args.directory / "score.abc").read_text()
    for label, text in (("reference", reference), ("Rust", actual)):
        parsed = abc_tools.parse_abc(text)
        print(f"P2a {label} ABC parse PASS: " + ", ".join(f"{name}={len(v.bars)} measures/{len(v.notes)} notes" for name, v in parsed.voices.items()))
    sections = lambda text: [line.strip() for line in text.splitlines() if line.startswith("% ")]
    counts = lambda text: {field: sum(line.startswith(field + ":") for line in text.splitlines()) for field in ("K", "M")}
    print(f"P2a sections: Rust={sections(actual)}, reference={sections(reference)}")
    print(f"P2a key/meter line counts: Rust={counts(actual)}, reference={counts(reference)}")
    assert sections(actual) == sections(reference), (sections(actual), sections(reference))
    assert counts(actual) == counts(reference), (counts(actual), counts(reference))
    print(f"P2a section tags={sections(actual)}; key/meter line counts={counts(actual)}: exact")


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    main()
