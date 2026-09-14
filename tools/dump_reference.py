#!/usr/bin/env python3
"""Offline, read-only YuE2 0.1.6 oracle; artifacts are never model weights.

Run with the reference venv, PYTHONDONTWRITEBYTECODE=1 and CUDA_VISIBLE_DEVICES=0.
The original sampled ABC is preserved; --greedy adds a separate eager oracle.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HOME", str(Path.home() / "yue2/hf"))
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from safetensors.torch import save_file
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.modeling_vae import DecoderBlock, YuE2VAE
from yue2.nar import CachedNAR, song_chunks
from yue2.protocol import GenerationConfig, SongRequest, token_prefixes, negative_prefix
from yue2.sampling import generate_tokens
from yue2.tokenization_yue2 import YuE2TextTokenizer


PROBES = [
    "Neon fades along the lane", "We will sing beyond the night", "I'm here; you're free.",
    "[Verse]", "[Chorus]\n", "[Bridge]\nHold on!", "X:1\nT:City Lights\n",
    "M:4/4\nL:1/8\nQ:1/4=88\nK:C\n", '"Am"A2 B2 c2 e2 |',
    "w: Let the day come in-to view", "|: CDEF GABc :|", "z4 | [CEG]4 |]",
    "1234567890", "  light\tand rain\n\n", "<abc>\n</abc>",
    "Café, déjà vu — lumière", "你好，世界！", "夜の歌と朝の光", "Привет, музыка!", "🎵 Sing 🌙\n",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(name):
    parent = Path(os.environ["HF_HOME"]) / "hub" / ("models--m-a-p--" + name)
    ref = parent / "refs/main"
    if ref.is_file():
        return parent / "snapshots" / ref.read_text().strip()
    paths = sorted((parent / "snapshots").glob("*"))
    if len(paths) != 1:
        raise ValueError(f"Specify a directory: ambiguous snapshots in {parent}")
    return paths[0]


class Writer:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.files = {}

    def tensors(self, name, tensors):
        tensors = {k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}
        path = self.directory / name
        save_file(tensors, path)
        self.files[name] = {
            "bytes": path.stat().st_size, "sha256": sha256(path),
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
        }
        print(f"{name}: {path.stat().st_size:,} bytes; {len(tensors)} tensors", flush=True)
        assert sum(f["bytes"] for f in self.files.values()) <= 1_500_000_000

    def json(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def ids(values):
    return torch.tensor(list(values), dtype=torch.int64)


def dump_tokens(writer, tokenizer, source):
    arrays = {key: ids(np.load(source / filename, allow_pickle=False).tolist()) for key, filename in
              (("prefix", "prefix.npy"), ("abc_ids", "abc_tokens.npy"), ("semantic_ids", "semantic.npy"))}
    cases = []
    for index, name in enumerate(("first-song", "song2", "song3")):
        directory = source if index == 0 else source.parent / name
        request = SongRequest(**json.loads((directory / "request.json").read_text()))
        if index == 0:
            example = SongRequest(**json.loads((Path.home() / "yue2/YuE/examples/song.json").read_text()))
            assert request == example
        abc = np.load(directory / "abc_tokens.npy", allow_pickle=False).tolist()
        saved = np.load(directory / "prefix.npy", allow_pickle=False).tolist()
        prompt = token_prefixes(request, tokenizer)
        assert token_prefixes(request, tokenizer, abc) == saved
        stem = f"request.{index}"
        arrays.update({stem + ".prompt": ids(prompt), stem + ".prefix": ids(saved), stem + ".abc": ids(abc),
                       stem + ".text": ids(tokenizer.encode(request.text())),
                       stem + ".negative": ids(negative_prefix(request, tokenizer, abc))})
        cases.append({"name": name, "request": request.to_dict(), "text": request.text(), "stem": stem})
        print(f"{name}: generation prompt={len(prompt)}, saved semantic prefix={len(saved)}, ABC={len(abc)}", flush=True)
    for index, probe in enumerate(PROBES):
        arrays[f"probe.{index:02}"] = ids(tokenizer.encode(probe))
        assert tokenizer.decode(tokenizer.encode(probe)) == probe
    specials = tokenizer._enc._special_tokens
    arrays["special_ids"] = ids(specials.values())
    writer.tensors("tokens.safetensors", arrays)
    writer.json("tokens.json", {"probes": PROBES, "special_tokens": specials, "requests": cases,
                               "normalization_probe": {"text": "Cafe\u0301", "normalized": "Café",
                                                       "ids": tokenizer.encode("Cafe\u0301")}})
    return arrays, SongRequest(**cases[0]["request"])


@torch.inference_mode()
def dump_ar(writer, model, arrays):
    # Follow TASK's literal prefix + abc_ids, selecting the first 64 positions.
    sequence = torch.cat((arrays["prefix"], arrays["abc_ids"]))[:64][None].cuda()
    captured = {"input_ids": sequence.cpu()}
    hooks = []
    for name, module in (("embedding", model.model.embed_tokens), ("layer.0", model.model.layers[0]),
                         ("layer.13", model.model.layers[13]), ("final_norm", model.model.norm)):
        def capture(_module, _args, output, name=name):
            captured[name] = output.float().cpu()
        hooks.append(module.register_forward_hook(capture))
    try:
        captured["logits"] = model(sequence, use_cache=False).logits.float().cpu()
    finally:
        for hook in hooks:
            hook.remove()
    writer.tensors("ar_logits.safetensors", captured)


@torch.inference_mode()
def dump_greedy(writer, model, arrays, request):
    sampling = dataclasses.replace(GenerationConfig().abc, temperature=0, max_tokens=64)
    collected = []
    handle = model.lm_head.register_forward_hook(lambda _m, _a, out: collected.append(out[:, -1].float().cpu()))
    try:
        tokens, _, truncated = generate_tokens(model, arrays["request.0.prompt"].tolist(), sampling,
                                               request.seed, "abc", use_cuda_graph=False)
    finally:
        handle.remove()
    assert len(tokens) == 64 and truncated
    sampled = arrays["abc_ids"][:64].tolist()
    same = sum(a == b for a, b in zip(tokens, sampled))
    first = next((i for i, (a, b) in enumerate(zip(tokens, sampled)) if a != b), None)
    writer.tensors("greedy.safetensors", {"ids": ids(tokens), "logits": torch.cat(collected)})
    writer.json("greedy.json", {"sampling": dataclasses.asdict(sampling), "seed": request.seed,
                                "steps": 64, "execution": "eager", "sampled_reference_matches": same,
                                "first_sampled_reference_mismatch": first})
    print(f"greedy oracle: 64 steps; original sampled ABC matches={same}/64; first difference={first}", flush=True)


@torch.inference_mode()
def trace_chunk(model, chunk, steps):
    """Observe the installed CachedNAR.solve; do not replace its solver.

    Each step saves both velocity calls' inputs/outputs, the raw scalar times,
    and the updated state (the next velocity input, or solve's final result).
    These are all loop intermediates, without multi-GB per-layer activations.
    """
    engine = CachedNAR(model, chunk)
    tensors = {"ar_tokens": ids(chunk.ar_tokens), "noise": chunk.noise.clone()}
    velocity = engine.velocity
    call = 0

    def capture(state, raw_t):
        nonlocal call
        step, half = divmod(call, 2)
        stem = f"step.{step:02}"
        tensors[stem + (".state" if half == 0 else ".mid")] = state.cpu().clone()
        tensors[stem + (".raw_t" if half == 0 else ".raw_mid")] = torch.tensor(raw_t, dtype=torch.float64)
        result = velocity(state, raw_t)
        tensors[stem + (".first" if half == 0 else ".second")] = result.cpu().clone()
        if step > 0 and half == 0:
            tensors[f"step.{step - 1:02}.next"] = state.cpu().clone()
        call += 1
        return result

    engine.velocity = capture
    try:
        tensors["latents"] = engine.solve(steps)
        tensors[f"step.{steps - 1:02}.next"] = tensors["latents"].to(engine.dtype)
        assert call == 2 * steps
        dt = 1.0 / steps
        for step in range(steps):
            stem = f"step.{step:02}"
            # Validate captured loop arithmetic on the same device and dtype.
            state, first, second = (tensors[stem + suffix].cuda() for suffix in (".state", ".first", ".second"))
            assert torch.equal((state - first * (dt / 2)).cpu(), tensors[stem + ".mid"])
            assert torch.equal((state - second * dt).cpu(), tensors[stem + ".next"])
    finally:
        engine.close()
    return tensors


def dump_nar(writer, model, arrays, request, two_chunks):
    prefix, codec = arrays["prefix"].tolist(), arrays["semantic_ids"].tolist()
    native = song_chunks(prefix, codec, request.seed)
    tensors, cases = {}, []
    variants = [("native", 24576, native)]
    if len(native) < 2 and two_chunks:
        # The saved 1,484-frame song has only ONE native chunk. Preserve it and
        # add an explicit context override; never present artificial cuts as native.
        context = len(prefix) + 3 + 2 * ((len(codec) + 1) // 2)
        variants.append(("two_chunk", context, song_chunks(prefix, codec, request.seed, context)))
    for name, context, chunks in variants:
        ranges = []
        start = 0
        for i, chunk in enumerate(chunks[:2]):
            stem = f"{name}.{i}"
            print(f"NAR {stem}: context={context}, frames={len(chunk.noise)}, AR={len(chunk.ar_tokens)}", flush=True)
            traced = trace_chunk(model, chunk, 32)
            tensors.update({stem + "." + key: value for key, value in traced.items()})
            ranges.append([start, start + len(chunk.noise)])
            start += len(chunk.noise)
        cases.append({"name": name, "context": context, "native_total_chunks": len(chunks), "ranges": ranges})
    writer.tensors("nar.safetensors", tensors)
    writer.json("nar.json", {"seed": request.seed, "steps": 32, "method": "midpoint", "attention": "sdpa",
                             "cases": cases, "noise": "one full-song CPU float32 torch.randn draw before slicing",
                             "intermediates": "all solve-loop states, midpoint states, velocities and raw scalar times"})


@torch.inference_mode()
def dump_vae(writer, vae_dir, source):
    vae = YuE2VAE.from_pretrained(vae_dir, decoder_only=True, device="cuda:0", local_files_only=True)
    latents = torch.from_numpy(np.load(source / "latent.npy", allow_pickle=False)).float()
    z = latents.T[None].contiguous()
    # Decode the full reference with the release's halo/crop boundary treatment.
    audio = vae.decode_tiled(z, core_frames=1024, halo_frames=16)[..., :4 * 48000]
    tensors = {"latents": latents, "audio": audio, "slice": z[..., :64]}
    hooks = []
    for index, (name, module) in enumerate((n, m) for n, m in vae.named_modules() if isinstance(m, DecoderBlock)):
        def capture(_module, _args, output, index=index):
            tensors[f"block.{index}"] = output.float().cpu()
        hooks.append(module.register_forward_hook(capture))
    try:
        tensors["slice_audio"] = vae.decode(z[..., :64]).cpu()
    finally:
        for hook in hooks:
            hook.remove()
    assert len(hooks) == 6
    writer.tensors("vae.safetensors", tensors)
    writer.json("vae.json", {"sample_rate": 48000, "audio_layout": "B,C,samples; unclipped",
                             "latents_layout": "frames,64", "slice_layout": "B,64,frames",
                             "core_frames": 1024, "halo_frames": 16, "block_order": "decoder.layers.1 through .6"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path.home() / "work/yue2-rs-fixtures/first-song")
    parser.add_argument("--source", type=Path, default=Path.home() / "yue2/outputs/first-song")
    parser.add_argument("--model-dir", type=Path, default=snapshot("YuE2-3B"))
    parser.add_argument("--vae-dir", type=Path, default=snapshot("YuE2-Vae"))
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--two-chunks", action="store_true", help="Also dump an explicit smaller-context two-chunk case")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=0; the other GPUs are shared")
    started = time.perf_counter()
    torch.set_num_threads(12)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(28 * 2**30 / total, 0)
    torch.cuda.reset_peak_memory_stats()
    writer = Writer(args.output)
    tokenizer = YuE2TextTokenizer(args.model_dir / "qwen.tiktoken")
    arrays, request = dump_tokens(writer, tokenizer, args.source)
    model = YuE2ForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                          dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
    dump_ar(writer, model, arrays)
    if args.greedy:
        dump_greedy(writer, model, arrays, request)
    dump_nar(writer, model, arrays, request, args.two_chunks)
    del model
    torch.cuda.empty_cache()
    dump_vae(writer, args.vae_dir, args.source)
    weights = {name: {"path": str(directory / "model.safetensors"),
                      "sha256": sha256(directory / "model.safetensors")} for name, directory in
               (("model", args.model_dir), ("vae", args.vae_dir))}
    manifest = {"format_version": 1, "seed": request.seed, "weights": weights, "files": writer.files,
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "yue2-infer", "safetensors", "tiktoken")},
                "source_sha256": {p.name: sha256(p) for p in sorted(args.source.glob("*.npy"))},
                "python_modules_sha256": {p.name: sha256(p) for p in sorted(Path(__import__("yue2").__file__).parent.glob("*.py"))},
                "model_dtype": "bfloat16", "vae_dtype": "float32", "ar_execution": "eager sdpa",
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    writer.json("manifest.json", manifest)
    size = sum(p.stat().st_size for p in args.output.iterdir() if p.is_file())
    assert size <= 1_500_000_000, size
    print(f"Total fixture bytes: {size:,} <= 1,500,000,000", flush=True)
    for name, entry in weights.items():
        print(f"{name} model.safetensors sha256: {entry['sha256']}", flush=True)
    print(f"Peak CUDA allocated/reserved: {manifest['peak_cuda_allocated_bytes']:,}/{manifest['peak_cuda_reserved_bytes']:,} bytes", flush=True)
    print(f"P0 dump PASS ({time.perf_counter() - started:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
