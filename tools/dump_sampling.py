"""Phase 2 oracle supplement, invoked only via dump_reference.py."""
import dataclasses
import json

import torch
from yue2.protocol import Sampling, ABC_END, MUSIC_END, CODEC_OFFSET, VOCAB_SIZE
from yue2.sampling import distribution


def dump_sampling(writer, source):
    tensors, cases = {}, []
    # Distinct scores avoid ambiguous sort ties except the dedicated top-k case.
    for index, (phase, legacy, changes, step) in enumerate([
        ("abc", False, {}, 0),
        ("abc", False, {"temperature": 0}, 40),
        ("abc", False, {"top_p": 1., "top_k": 2}, 40),
        ("abc", False, {"top_p": .01, "top_k": 7}, 40),
        ("semantic", False, {}, 0),
        ("semantic", False, {}, 200),
        ("semantic", True, {"temperature": .7, "top_p": .01}, 200),
        ("semantic", True, {"temperature": .7, "top_p": .95}, 200),
        ("semantic", True, {"temperature": 0, "repetition_penalty": .9}, 200),
        ("abc", False, {"temperature": 0, "penalty_window": 3}, 40),
        ("semantic", True, {"top_k": 184704, "top_p": 1.}, 200),
    ]):
        default = Sampling(temperature=.7, top_p=.9, top_k=30, repetition_penalty=1.005,
                           penalty_window=100, min_tokens=32, max_tokens=4096) if phase == "abc" else Sampling()
        sampling = dataclasses.replace(default, **changes)
        logits = torch.full((1, VOCAB_SIZE), -50., device="cuda", dtype=torch.bfloat16)
        start = 20 if phase == "abc" else CODEC_OFFSET
        logits[0, start:start + 12] = torch.tensor([9., 7.5, 6., 4.75, 3.25, 1., .25, -.5, -1.5, -3.25, -5., -7.], device="cuda")
        logits[0, ABC_END if phase == "abc" else MUSIC_END] = 8.5
        # Out-of-phase very large logits must be masked before filtering.
        logits[0, CODEC_OFFSET if phase == "abc" else 20] = 99.
        if index == 2:
            logits[0, start:start + 3] = 9.
            sampling = dataclasses.replace(sampling, repetition_penalty=1.)
        history = [start] * 101 + [start + 1, start + 7, start + 1, start + 7, start + 11]
        stem = f"case.{index}"
        tensors[stem + ".logits"] = logits
        tensors[stem + ".scores"] = distribution(logits, sampling, history, step, phase, legacy).float()
        cases.append({"stem": stem, "phase": phase, "legacy_off": legacy, "sampling": dataclasses.asdict(sampling),
                      "history": history, "step": step})
    conditional = torch.linspace(-8, 8, 1024, device="cuda").to(torch.bfloat16)
    unconditional = torch.linspace(4, -3, 1024, device="cuda").to(torch.bfloat16)
    tensors.update({"cfg.conditional": conditional, "cfg.unconditional": unconditional,
                    "cfg.logits": unconditional + 1.01 * (conditional - unconditional)})
    writer.tensors("sampling.safetensors", tensors)
    config = json.loads((source / "config.json").read_text())
    reference_plan = json.loads((source / "plan.json").read_text())
    writer.json("sampling.json", {"cases": cases, "cfg_scale": 1.01, "generation": config["generation"],
                                   "reference_plan": reference_plan, "files": writer.files})
    print(f"P2 sampling oracle: {len(cases)} distributions and BF16 CFG; original seed={reference_plan['request']['seed']}")
    size = sum(path.stat().st_size for path in writer.directory.rglob("*") if path.is_file())
    assert size <= 1_500_000_000, size
    print(f"Total fixture bytes including supplements: {size:,} <= 1,500,000,000")
