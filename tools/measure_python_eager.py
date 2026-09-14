#!/usr/bin/env python3
"""One P2c eager measurement, with the same saved plan used by the Rust gate."""
import dataclasses
import json
import os
from pathlib import Path

import torch
from dump_reference import snapshot
from yue2.pipeline import YuE2Pipeline, SymbolicPlan
from yue2.protocol import GenerationConfig, SongRequest


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=0")
    source = Path.home() / "yue2/outputs/first-song"
    output = Path.home() / "work/yue2-rs-fixtures/first-song/p2-eager"
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads((source / "plan.json").read_text())
    request = SongRequest(**saved["request"])
    config = GenerationConfig.from_dict(json.loads((source / "config.json").read_text())["generation"])
    torch.set_num_threads(12)
    with YuE2Pipeline(snapshot("YuE2-3B"), snapshot("YuE2-Vae"), device="cuda:0",
                      backend="torch-eager", generation_config=config,
                      memory_budget_gib=24, verify_hashes=False, progress=False) as pipe:
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
