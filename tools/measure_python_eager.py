#!/usr/bin/env python3
"""Offline P2c fixed-plan or P5 end-to-end eager measurement on GPU 0."""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import torch
from dump_reference import snapshot
from yue2.pipeline import YuE2Pipeline, SymbolicPlan
from yue2.protocol import GenerationConfig, SongRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=0")
    source = Path.home() / "yue2/outputs/first-song"
    output = args.output or Path.home() / "work/yue2-rs-fixtures/first-song/p2-eager"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty measurement directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads((source / "plan.json").read_text())
    request = SongRequest(**saved["request"])
    config = GenerationConfig.from_dict(json.loads((source / "config.json").read_text())["generation"])
    torch.set_num_threads(12)
    start = time.perf_counter()
    with YuE2Pipeline.from_pretrained(snapshot("YuE2-3B"), vae=snapshot("YuE2-Vae"), device="cuda:0",
                      backend="torch-eager", generation_config=config,
                      memory_budget_gib=24, local_files_only=True, progress=False) as pipe:
        if args.end_to_end:
            torch.cuda.reset_peak_memory_stats()
            result = pipe(**request.to_dict())
            assert result.timing["abc"]["execution"] == result.timing["semantic"]["execution"] == "eager"
            result.save_artifacts(output / "song")
            measurement = {"timing": result.timing, "audio_seconds": len(result.audio) / result.sample_rate,
                "wall_including_integrity_and_storage_seconds": time.perf_counter() - start,
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                "note": "Cold pipeline; torch-eager AR (sdpa), native CachedNAR sdpa, tiled FP32 VAE; no CUDA graphs"}
            (output / "measurement.json").write_text(json.dumps(measurement, indent=2) + "\n")
            print(json.dumps(measurement, indent=2), flush=True)
            return
        pipe._load_model()  # Loading excluded from the package's synchronized stage timers.
        torch.cuda.reset_peak_memory_stats()
        plan = pipe.plan(request=request)
        reference = SymbolicPlan(request, saved["abc"], saved["abc_ids"], saved["prefix"],
                                 saved["timing"], saved["truncated"])
        semantic = pipe.generate_semantic(reference)
        assert plan.timing["execution"] == semantic.timing["execution"] == "eager"
        result = {"request": request.to_dict(), "generation": config.to_dict(),
                  "semantic_plan": "exact immutable first-song Python plan (same as Rust P2 gate)",
                  "abc": plan.timing, "semantic": semantic.timing,
                  "truncated": {"abc": plan.truncated, "semantic": semantic.truncated},
                  "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
        (output / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
        (output / "plan.json").write_text(json.dumps(dataclasses.asdict(plan), indent=2) + "\n")
        (output / "semantic.json").write_text(json.dumps(dataclasses.asdict(semantic), indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
