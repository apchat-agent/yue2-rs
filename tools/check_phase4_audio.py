#!/usr/bin/env python3
"""P4 artifact writer/checker only; consumes Rust audio, never decodes a VAE.

Match pipeline.py: clamp to [-1,1], FLOAT WAV and PCM_24 FLAC. Native Rust
storage and pipeline/CLI integration belong to Phase 5.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from safetensors.numpy import load_file


def snr(actual, reference):
    actual, reference = actual.astype(np.float64), reference.astype(np.float64)
    error = np.sum((actual - reference) ** 2)
    return float(10 * np.log10(np.sum(reference ** 2) / error)) if error else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "work/yue2-rs-fixtures/first-song")
    args = parser.parse_args()
    audio = load_file(args.root / "p4/audio.safetensors")["audio"]
    reference = load_file(args.root / "vae.safetensors")["audio"][0].T
    assert audio.ndim == 3 and audio.shape[:2] == (1, 2) and audio.dtype == np.float32
    audio = audio[0].T.copy()
    assert np.isfinite(audio).all()
    raw_snr = snr(audio[:192000], reference)
    assert raw_snr >= 40, raw_snr
    clipped = np.clip(audio, -1, 1)
    print(f"P4b Rust raw first-4s SNR={raw_snr:.6f} dB; peak={np.abs(audio).max():.9f}; clipped_values={np.count_nonzero(audio != clipped)}")
    for extension, subtype in (("wav", "FLOAT"), ("flac", "PCM_24")):
        path = args.root / f"p4/audio.{extension}"
        sf.write(path, clipped, 48000, subtype=subtype)
        actual, rate = sf.read(path, dtype="float32", always_2d=True)
        info = sf.info(path)
        assert rate == 48000 and actual.shape == audio.shape and info.subtype == subtype
        delta = np.abs(actual - clipped).max()
        assert delta <= (0 if extension == "wav" else 2 ** -23), delta
        score = snr(actual[:192000], reference)
        assert score >= 40, score
        print(f"P4b {path.name}: {info.frames} frames, {rate} Hz, {info.channels} channels, {info.subtype}; first-4s SNR={score:.6f} dB; roundtrip_max={delta:.12f}; bytes={path.stat().st_size}")
    size = sum(p.stat().st_size for p in args.root.rglob("*") if p.is_file())
    assert size <= 1_500_000_000, size
    print(f"Total recursive fixture/artifact bytes: {size:,} <= 1,500,000,000")


if __name__ == "__main__":
    main()
