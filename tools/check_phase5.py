#!/usr/bin/env python3
"""Verify native Rust artifacts through Python's public plan/result loaders."""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from yue2.pipeline import SymbolicPlan
from yue2.protocol import CODEC_SIZE, token_prefixes
from yue2.storage import identity, verify_result
from yue2.tokenization_yue2 import YuE2TextTokenizer
from dump_reference import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--abc-file", type=Path)
    parser.add_argument("--reference", type=Path, default=Path.home() / "yue2/outputs/first-song")
    parser.add_argument("--audio-export", action="store_true")
    parser.add_argument("--make-edit", action="store_true")
    args = parser.parse_args()
    directory = args.directory
    if args.make_edit:
        original = (directory / "plan/score.abc").read_bytes()
        edited, changes = re.subn(rb"(?m)^Q:1/4=([0-9]+)$",
            lambda match: b"Q:1/4=" + str(int(match[1]) + 4).encode(), original)
        assert changes == 1
        edited = edited.replace(b"\n", b"\r\n")
        assert edited != original
        (directory / "edited.abc").write_bytes(edited)
        print(f"P5 edit prepared: {len(original)} -> {len(edited)} bytes")
        return
    if args.audio_export:
        from safetensors.numpy import load_file
        reference = load_file(str(Path.home() / "work/yue2-rs-fixtures/first-song/p4/audio.safetensors"))["audio"][0].T.clip(-1, 1)
        for name, subtype, tolerance in [("audio.wav", "FLOAT", 0), ("audio.flac", "PCM_24", 2**-24)]:
            info = sf.info(directory / name)
            actual, sr = sf.read(directory / name, dtype="float32", always_2d=True)
            assert sr == 48000 and actual.shape == reference.shape and info.subtype == subtype
            error = np.max(np.abs(actual - reference))
            assert error <= tolerance, error
            print(f"P5 native {name}: {info.frames} frames, {sr} Hz, {info.channels} channels, {subtype}; roundtrip_max={error:.12f}")
        return
    plan = SymbolicPlan.load(directory)
    tokenizer = YuE2TextTokenizer(snapshot("YuE2-3B") / "qwen.tiktoken")
    assert token_prefixes(plan.request, tokenizer, plan.abc_ids) == plan.prefix
    assert np.load(directory / "prefix.npy", allow_pickle=False).dtype == np.int32
    assert np.load(directory / "abc_tokens.npy", allow_pickle=False).dtype == np.int32
    if plan.abc is not None:
        assert tokenizer.decode(plan.abc_ids) == plan.abc
        spec = importlib.util.spec_from_file_location("abc_tools", Path.home() / "yue2/YuE/skills/yue2-music/scripts/abc_tools.py")
        abc_tools = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = abc_tools
        spec.loader.exec_module(abc_tools)
        score = abc_tools.parse_abc(plan.abc)
        print(f"P5 ABC parse PASS: {len(score.voices)} voices; {len(plan.abc_ids)} ABC tokens; prefix={len(plan.prefix)}")
    if args.abc_file:
        assert (directory / "score.abc").read_bytes() == args.abc_file.read_bytes()
        assert plan.request.abc == args.abc_file.read_bytes().decode("utf-8")
        assert plan.timing["output_tokens"] == 0 and plan.timing["seconds"] == 0
        assert plan.timing["external_prefix_tokens"] == len(plan.abc_ids)
        print("P5 render PASS: edited ABC bytes/prefix preserved; zero generated ABC tokens")
    if args.plan_only:
        expected = {"score.abc", "plan.json", "plan_manifest.json", "prefix.npy", "abc_tokens.npy"}
        if plan.abc is None:
            expected.remove("score.abc")
        assert {p.name for p in directory.iterdir()} == expected
        print(f"P5 plan PASS: Python SymbolicPlan.load verified all {len(expected)} files")
        return
    request = json.loads((directory / "request.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    expected_identity = identity({"request": request, "config": config, "weights": result["weights"]})
    verify_result(directory, expected_identity)
    assert request == plan.request.to_dict()
    reference = json.loads((args.reference / "result.json").read_text())
    expected = set(reference["artifacts"]) | {"result.json"}
    if plan.abc is None:
        expected.remove("score.abc")
    assert {p.name for p in directory.iterdir()} == expected
    assert set(result["artifacts"]) == expected - {"result.json"}
    assert result["weights"] == reference["weights"]
    codec = np.load(directory / "semantic.npy", allow_pickle=False)
    latent = np.load(directory / "latent.npy", allow_pickle=False)
    assert codec.ndim == 1 and codec.dtype == np.int32 and codec.size > 0
    assert 0 <= codec.min() <= codec.max() < CODEC_SIZE
    assert latent.dtype == np.float32 and latent.shape == (len(codec), 64) and np.isfinite(latent).all()
    info = sf.info(directory / "audio.flac")
    audio, sr = sf.read(directory / "audio.flac", dtype="float32", always_2d=True)
    assert info.format == "FLAC" and info.subtype == "PCM_24" and sr == 48000
    assert audio.shape == (len(codec) * 1920 - 64, 2)
    assert np.isfinite(audio).all() and np.abs(audio).max() <= 1 and np.std(audio) > 0
    assert result["sample_rate"] == sr and result["audio_seconds"] == len(audio) / sr
    assert result["truncated"]["abc"] == plan.truncated
    assert result["timing"]["abc"] == plan.timing
    assert config["generation"]["ode_steps"] == 32
    print(f"P5 artifacts PASS: exact Python file set ({len(expected)} files), hashes and request/config/weight identity verified")
    print(f"P5 audio PASS: {len(codec)} codec frames ({codec.min()}..{codec.max()}), latents={latent.shape} {latent.dtype}; {len(audio)} stereo samples, {len(audio)/sr:.9f} s, peak={np.abs(audio).max():.9f}")
    print("P5 timing " + json.dumps(result["timing"], sort_keys=True))


if __name__ == "__main__":
    main()
